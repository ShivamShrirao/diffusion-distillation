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


def power_ema_decay(n_updates: int, cap: float, gamma: float) -> float:
    """EDM2 power EMA, capped: min(cap, (1 - 1/(n+1))^(gamma+1)); 0 on the first update (a copy).
    gamma <= 0 is a plain EMA at ``cap``."""
    if gamma <= 0:
        return cap
    return min(cap, (1.0 - 1.0 / (n_updates + 1)) ** (gamma + 1.0))


class FullFtEMA:
    """Power EMA over a full-FT student's params. Under FSDP2 each param is a sharded DTensor, so
    ``local_shard`` keeps every op on the rank-local slice (no collectives). Shadow in float32
    (decay~0.999 -> alpha 1e-3, below bf16 resolution). ``load_shadow_from`` re-seeds on resume;
    ``n_updates`` drives the power-EMA decay and is restored with it."""

    def __init__(self, transformer, decay: float, gamma: float = 0.0):
        self.cap, self.gamma = decay, gamma
        self.n_updates, self.decay = 0, 0.0
        self.live = list(transformer.parameters())
        self.shadow = [local_shard(p.data).detach().to(torch.float32).clone() for p in self.live]

    @torch.no_grad()
    def update(self):
        self.decay = power_ema_decay(self.n_updates, self.cap, self.gamma)
        foreach_lerp_shadow(self.shadow, [local_shard(p.data) for p in self.live], 1.0 - self.decay)
        self.n_updates += 1

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

    def save(self, save_dir: str, accelerator, transformer, transformer_unwrapped):
        """Gather full unsharded weights with EMA swapped in (collective), write
        ``{save_dir}/ema/diffusion_pytorch_model.safetensors`` + ``config.json`` on rank 0."""
        saved = self.swap_in()
        try:
            ema_sd = accelerator.get_state_dict(transformer, unwrap=False)
        finally:
            self.swap_out(saved)
        if not accelerator.is_main_process:
            return
        ema_dir = os.path.join(save_dir, "ema")
        os.makedirs(ema_dir, exist_ok=True)
        transformer_unwrapped.save_config(ema_dir)
        cleaned = {
            k.replace("module.", "").replace("._orig_mod", ""): v.contiguous()
            for k, v in ema_sd.items()
        }
        save_file(cleaned, os.path.join(ema_dir, "diffusion_pytorch_model.safetensors"))

    @property
    def local_numel(self) -> int:
        return sum(s.numel() for s in self.shadow)
