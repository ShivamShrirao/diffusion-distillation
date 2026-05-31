import os

import torch
from safetensors.torch import save_file


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
        self.shadow = {n: p.detach().to(torch.float32).clone() for n, p in self.named_live}

    @torch.no_grad()
    def update(self):
        alpha = 1.0 - self.decay
        for n, p in self.named_live:
            self.shadow[n].lerp_(p.data.to(torch.float32), alpha)

    @torch.no_grad()
    def swap_in(self):
        saved = {}
        for n, p in self.named_live:
            saved[n] = p.data.clone()
            p.data.copy_(self.shadow[n].to(p.dtype))
        return saved

    @torch.no_grad()
    def swap_out(self, saved):
        for n, p in self.named_live:
            p.data.copy_(saved[n])

    def save(self, save_dir: str, accelerator, transformer, transformer_unwrapped):
        """Gather full unsharded weights with EMA swapped in, write a PEFT adapter
        dir at ``{save_dir}/ema/adapter_model.safetensors`` (no per-adapter subdir;
        we only ever save the tracked adapter)."""
        from lora.static_multi_lora import save_peft_adapter

        saved = self.swap_in()
        try:
            if getattr(accelerator.state, "fsdp_plugin", None) is not None:
                accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
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
    """EMA shadow over an FSDP-wrapped student transformer's full parameters.

    Parallel to LoraEMA but operates on the per-shard FlatParameter list (rather than
    name-keyed adapter params), so each rank EMAs only its own local shard — no
    collectives per step, shard layout matches by construction.

    Shadow kept in float32: with decay=0.999 the per-step alpha is 1e-3, which bf16
    mantissa would silently truncate. Memory cost per rank ≈ fp32 size of the local
    FSDP shard (e.g. ~2.5 GB for a 5B-param student sharded across 8 H200s).

    EMA resume from a previous run is not supported; cold-start initialises the shadow
    from current live student weights.
    """

    def __init__(self, transformer, decay: float):
        self.decay = decay
        self.live = list(transformer.parameters())
        self.shadow = [p.data.detach().to(torch.float32).clone() for p in self.live]

    @torch.no_grad()
    def update(self):
        alpha = 1.0 - self.decay
        for ema, p in zip(self.shadow, self.live):
            ema.lerp_(p.data.to(torch.float32), alpha)

    @torch.no_grad()
    def swap_in(self):
        saved = [p.data.clone() for p in self.live]
        for p, ema in zip(self.live, self.shadow):
            p.data.copy_(ema.to(p.dtype))
        return saved

    @torch.no_grad()
    def swap_out(self, saved):
        for p, s in zip(self.live, saved):
            p.data.copy_(s)

    def save(self, save_dir: str, accelerator, transformer, transformer_unwrapped):
        """Gather full unsharded weights with EMA swapped in, write
        ``{save_dir}/ema/diffusion_pytorch_model.safetensors`` + ``config.json``."""
        saved = self.swap_in()
        try:
            if getattr(accelerator.state, "fsdp_plugin", None) is not None:
                accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
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
