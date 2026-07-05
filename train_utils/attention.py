"""Attention-mask construction and backend selection for the training hot path.

Backends (``--attn_backend``):
  native — diffusers' SDPA path.
  fa3    — FlashAttention-3 (hub kernel) in varlen mode via diffusers' ``_flash_3_varlen_hub``.
           Fibo 2.0 builds its own key-padding masks (valid tokens first, padding last), which
           the varlen kernel turns into per-sequence lengths.

The trainers still pass a joint ``(B, 1, S, S)`` bias (text tokens first, then image tokens);
the Fibo 2.0 wrapper reads the text padding mask off it.
"""

import torch

from models.text_encoder import create_attention_matrix

FA3_BACKEND = "_flash_3_varlen_hub"  # diffusers' enum name for the FA3 hub varlen kernel


def build_attention_mask(prompt_attn_mask: torch.Tensor, latent_len: int, dtype: torch.dtype, backend: str):
    """Joint text+image attention bias from the (B, T) prompt mask; image tokens are
    always valid. Returns a ``(B, 1, S, S)`` additive bias for
    ``joint_attention_kwargs["attention_mask"]`` (the same for every backend)."""
    bsz = prompt_attn_mask.shape[0]
    latent_attn = torch.ones([bsz, latent_len], dtype=prompt_attn_mask.dtype, device=prompt_attn_mask.device)
    valid = torch.cat([prompt_attn_mask, latent_attn], dim=1)
    return create_attention_matrix(valid.to(dtype)).unsqueeze(1)


def replicate_attention_mask(mask, n: int):
    """Batch-replicate a mask for CFG-style batched forwards (e.g. [cond, uncond, dm])."""
    if mask is None or n == 1:
        return mask
    return torch.cat([mask] * n, dim=0)


def load_fa3_kernel():
    """Fetch the FA3 hub kernel. ``get_kernel`` with no revision works on older ``kernels``
    (<0.13); newer versions require an explicit revision and raise ``ValueError``."""
    from kernels import get_kernel

    repo = "kernels-community/flash-attn3"
    try:
        return get_kernel(repo)
    except ValueError:
        return get_kernel(repo, revision="main")


def warmup_fa3_kernel(accelerator):
    """Fetch the FA3 hub kernel once per node before any model selects the backend.

    Left ungated, every rank pulls the ~824MB kernel ``.so`` from the HF hub at once. The HF
    cache is node-local, so this gates on the *local* main process: local rank 0 on each node
    downloads first, the rest of that node then hit the warm cache. Returns the load error
    (caller falls back to native) or None on success; the error is caught INSIDE the
    local-main-first block so a failing rank still reaches the exit barrier and ranks never
    deadlock on mismatched collectives."""
    with accelerator.local_main_process_first():
        try:
            load_fa3_kernel()
            return None
        except Exception as e:
            return e


def diffusers_backend_name(backend: str) -> str:
    """Translate a ``--attn_backend`` value to the diffusers backend string accepted by
    ``model.set_attention_backend``."""
    if backend == "fa3":
        return FA3_BACKEND
    return backend
