import os

import torch


class LoraEMA:
    """EMA shadow over LoRA params, stored as a side dict (not in the module tree).

    Each shadow tensor is a clone of the corresponding live param's local FSDP shard,
    so update is a local ``lerp_`` and ref-forward swap is a local ``copy_`` — no
    all-gathers per step, and shard layout matches by construction.

    Save path gathers full-shape EMA tensors via ``accelerator.get_state_dict`` while
    the shadow is temporarily swapped into live params, then restores. EMA resume
    from a previous run is not supported; cold-start initialises the shadow from current
    live LoRA values.
    """

    def __init__(self, transformer, decay: float):
        self.decay = decay
        self.named_live = [
            (n, p) for n, p in transformer.named_parameters() if "lora_" in n and p.requires_grad
        ]
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

    def save(self, save_dir: str, accelerator, transformer):
        from safetensors.torch import save_file

        saved = self.swap_in()
        try:
            if getattr(accelerator.state, "fsdp_plugin", None) is not None:
                accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            full_sd = accelerator.get_state_dict(transformer, unwrap=False)
            full_sd = {
                k.replace("module.", "").replace("._orig_mod", ""): v for k, v in full_sd.items()
            }
            ema_sd = {k: v for k, v in full_sd.items() if "lora_" in k}
        finally:
            self.swap_out(saved)
        if accelerator.is_main_process and ema_sd:
            save_file(ema_sd, os.path.join(save_dir, "ema_lora.safetensors"))
