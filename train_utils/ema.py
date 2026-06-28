import os

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor


def local_shard(t):
    """Rank-local shard of an FSDP2 DTensor (shares storage, so in-place ops hit the live
    param); a plain tensor passes through unchanged."""
    return t.to_local() if isinstance(t, DTensor) else t


class LoraEMA:
    """EMA shadow over one named LoRA adapter, stored as a side dict (not in the module tree).

    Filters parameters by ``"lora_" in n and f".{adapter_name}." in n``. For PEFT-style
    runs use the adapter name passed to ``get_peft_model`` (often ``"default"``); for
    DMD StaticMultiLoraLinear use ``"student"`` so the fake critic adapter is excluded
    (DMD fake should track the current student, not a smoothed past).

    Each shadow tensor is a clone of the corresponding live param's local FSDP shard,
    so update is a local ``lerp_`` and ref-forward swap is a local ``copy_`` — no
    all-gathers per step, and shard layout matches by construction.

    Save path swaps the shadow into live params, gathers a full state_dict, and writes
    via ``save_peft_adapter`` so the artefact loads with ``load_peft_adapter`` (or any
    PEFT consumer). EMA resume from a previous run is not supported; cold-start
    initialises the shadow from current live LoRA values.
    """

    def __init__(self, transformer, decay: float, adapter_name: str = "default"):
        self.decay = decay
        self.adapter_name = adapter_name
        token = f".{adapter_name}."
        self.named_live = [
            (n, p) for n, p in transformer.named_parameters()
            if "lora_" in n and token in n and p.requires_grad
        ]
        if not self.named_live:
            raise RuntimeError(f"LoraEMA: no trainable lora_ params matched adapter '{adapter_name}'")
        # Shadow kept in float32: with decay=0.987 the per-step update is alpha≈0.013,
        # which bf16 mantissa would silently truncate over many steps.
        self.shadow = {n: local_shard(p.data).detach().to(torch.float32).clone() for n, p in self.named_live}

    @torch.no_grad()
    def update(self):
        alpha = 1.0 - self.decay
        for n, p in self.named_live:
            self.shadow[n].lerp_(local_shard(p.data).to(torch.float32), alpha)

    @torch.no_grad()
    def swap_in(self):
        saved = {}
        for n, p in self.named_live:
            shard = local_shard(p.data)
            saved[n] = shard.clone()
            shard.copy_(self.shadow[n].to(p.dtype))
        return saved

    @torch.no_grad()
    def swap_out(self, saved):
        for n, p in self.named_live:
            local_shard(p.data).copy_(saved[n])

    def save(self, save_dir: str, accelerator, transformer, transformer_unwrapped):
        """Gather full unsharded weights with EMA swapped in, write a PEFT adapter
        dir at ``{save_dir}/ema/adapter_model.safetensors`` (no per-adapter subdir;
        we only ever save the tracked adapter)."""
        from lora.static_multi_lora import save_peft_adapter

        saved = self.swap_in()
        try:
            full_sd = accelerator.get_state_dict(transformer, unwrap=False)
        finally:
            self.swap_out(saved)
        if accelerator.is_main_process:
            save_peft_adapter(
                transformer_unwrapped,
                save_dir,
                self.adapter_name,
                state_dict=full_sd,
                out_subdir="ema",
            )
        del full_sd


class FullFtEMA:
    """EMA over a full-FT student's params. Under FSDP2 each param is a sharded DTensor, so
    ``local_shard`` keeps every op on the rank-local slice (no collectives). Shadow in float32
    (decay~0.999 -> alpha 1e-3, below bf16 resolution). ``load_shadow_from`` re-seeds on resume."""

    def __init__(self, transformer, decay: float):
        self.decay = decay
        self.live = list(transformer.parameters())
        self.shadow = [local_shard(p.data).detach().to(torch.float32).clone() for p in self.live]

    @torch.no_grad()
    def update(self):
        alpha = 1.0 - self.decay
        for ema, p in zip(self.shadow, self.live):
            ema.lerp_(local_shard(p.data).to(torch.float32), alpha)

    @torch.no_grad()
    def reset_to_live(self):
        """Re-seed the shadow from the current live weights (e.g. to start EMA tracking
        at a later step rather than from the init weights)."""
        for ema, p in zip(self.shadow, self.live):
            ema.copy_(local_shard(p.data).to(torch.float32))

    @torch.no_grad()
    def swap_in(self):
        saved = []
        for p, ema in zip(self.live, self.shadow):
            shard = local_shard(p.data)
            saved.append(shard.clone())
            shard.copy_(ema.to(p.dtype))
        return saved

    @torch.no_grad()
    def swap_out(self, saved):
        for p, s in zip(self.live, saved):
            local_shard(p.data).copy_(s)

    @torch.no_grad()
    def load_shadow_from(self, fpath: str, fsdp_root, use_fsdp: bool):
        from safetensors.torch import load_file

        full_sd = load_file(fpath)
        saved = [local_shard(p.data).clone() for p in self.live]
        self.scatter_full_into_live(full_sd, fsdp_root, use_fsdp)
        for ema, p in zip(self.shadow, self.live):
            ema.copy_(local_shard(p.data).to(torch.float32))
        for p, s in zip(self.live, saved):
            local_shard(p.data).copy_(s)

    @staticmethod
    @torch.no_grad()
    def scatter_full_into_live(full_sd: dict, fsdp_root, use_fsdp: bool):
        """Scatter a full state dict into the live FSDP2 model via DCP ``set_model_state_dict``;
        every rank passes the full sd it loaded and takes its own shard."""
        if not use_fsdp:
            fsdp_root.load_state_dict(full_sd, strict=False)
            return
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        set_model_state_dict(
            fsdp_root,
            full_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False, strict=False),
        )

    def save(self, save_dir: str, accelerator, transformer, transformer_unwrapped):
        """Gather full unsharded weights with EMA swapped in, write
        ``{save_dir}/ema/diffusion_pytorch_model.safetensors`` + ``config.json``."""
        saved = self.swap_in()
        try:
            ema_sd = accelerator.get_state_dict(transformer, unwrap=False)
            if accelerator.is_main_process:
                ema_dir = os.path.join(save_dir, "ema")
                os.makedirs(ema_dir, exist_ok=True)
                transformer_unwrapped.save_config(ema_dir)
                cleaned = {
                    k.replace("module.", "").replace("._orig_mod", ""): v.contiguous()
                    for k, v in ema_sd.items()
                }
                save_file(cleaned, os.path.join(ema_dir, "diffusion_pytorch_model.safetensors"))
                del cleaned
            del ema_sd
        finally:
            self.swap_out(saved)

    @property
    def local_numel(self) -> int:
        return sum(s.numel() for s in self.shadow)
