import os

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor


def local_shard(t):
    """Rank-local shard of an FSDP2 DTensor (shares storage, so in-place ops hit the live
    param); a plain tensor passes through unchanged."""
    return t.to_local() if isinstance(t, DTensor) else t


def foreach_lerp_shadow(shadow: list, live: list, alpha: float):
    """`shadow.lerp_(live, alpha)` over parameter lists in a handful of batched kernels
    instead of one launch per tensor (~1000 for the full-FT student). Same-dtype pairs go
    through `_foreach_lerp_`; mismatched pairs (e.g. bf16 live vs fp32 shadow) fall back to
    the per-tensor upcast path, preserving the fp32 accumulation."""
    matched = [(s, t) for s, t in zip(shadow, live) if s.dtype == t.dtype]
    if matched:
        torch._foreach_lerp_([s for s, _ in matched], [t for _, t in matched], alpha)
    for s, t in zip(shadow, live):
        if s.dtype != t.dtype:
            s.lerp_(t.to(s.dtype), alpha)


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
        shadow = [self.shadow[n] for n, _ in self.named_live]
        live = [local_shard(p.data) for _, p in self.named_live]
        foreach_lerp_shadow(shadow, live, alpha)

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
        foreach_lerp_shadow(self.shadow, [local_shard(p.data) for p in self.live], alpha)

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
        """Scatter a full state dict into the live model; shard-aware under FSDP2."""
        if not use_fsdp:
            # Checkpoints store clean keys; a DDP-wrapped root expects "module."-prefixed
            # ones and strict=False would turn the whole load into a silent no-op.
            module = getattr(fsdp_root, "module", fsdp_root)
            loaded = {k for k in full_sd if k in dict(module.named_parameters())}
            if not loaded:
                raise ValueError("EMA shadow load matched zero parameter keys — key-prefix mismatch")
            module.load_state_dict(full_sd, strict=False)
            return
        from distributed import fsdp2_load_full_state

        fsdp2_load_full_state(fsdp_root, full_sd)

    def save(self, save_dir: str, accelerator, transformer, transformer_unwrapped, writes: list | None = None):
        """Gather full unsharded weights with EMA swapped in (collective), write
        ``{save_dir}/ema/diffusion_pytorch_model.safetensors`` + ``config.json`` on rank 0.
        With ``writes`` the write is appended as a closure for ``write_checkpoint_async``."""
        saved = self.swap_in()
        try:
            ema_sd = accelerator.get_state_dict(transformer, unwrap=False)
        finally:
            self.swap_out(saved)
        if not accelerator.is_main_process:
            return

        def write(out_dir: str):
            ema_dir = os.path.join(out_dir, "ema")
            os.makedirs(ema_dir, exist_ok=True)
            transformer_unwrapped.save_config(ema_dir)
            cleaned = {
                k.replace("module.", "").replace("._orig_mod", ""): v.contiguous()
                for k, v in ema_sd.items()
            }
            save_file(cleaned, os.path.join(ema_dir, "diffusion_pytorch_model.safetensors"))

        if writes is None:
            write(save_dir)
        else:
            writes.append(write)

    @property
    def local_numel(self) -> int:
        return sum(s.numel() for s in self.shadow)
