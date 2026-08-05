"""Attention-mask construction and backend selection for the training hot path.

Backends (``--attn_backend``):
  native — diffusers' SDPA path.
  fa3    — FlashAttention-3 (hub kernel) in varlen mode via diffusers' ``_flash_3_varlen_hub``.
           Fibo 2.0 builds its own key-padding masks (valid tokens first, padding last), which
           the varlen kernel turns into per-sequence lengths.

Padded path: the trainers pass a joint ``(B, 1, S, S)`` bias (text tokens first, then image
tokens); the Fibo 2.0 wrapper reads the text padding mask off it.

Packed path (``--packed_varlen``): q/k/v are flat ``[tokens, heads, head_dim]`` streams and each
record attends only within its own segment (``varlen_attention``).
"""

import functools

import torch
import torch.nn.functional as F

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


VARLEN_BACKEND = "native"


def set_varlen_backend(backend: str):
    """Kernel for the packed path: ``fa3`` (hub kernel) or ``native`` (one SDPA per record)."""
    global VARLEN_BACKEND
    VARLEN_BACKEND = backend


def varlen_attention(q, k, v, seg_q, seg_k):
    """Attention of each query segment to its key segment; ``seg_*`` are ``Segments``."""
    if VARLEN_BACKEND == "fa3":
        dtype = q.dtype
        q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))  # the kernel takes 16-bit inputs only
        out = load_fa3_kernel().flash_attn_varlen_func(
            q, k, v, seg_q.cu_seqlens, seg_k.cu_seqlens, seg_q.max_len, seg_k.max_len
        )
        return (out[0] if isinstance(out, tuple) else out).to(dtype)  # some builds also return the LSE
    parts = zip(q.split(seg_q.lengths), k.split(seg_k.lengths), v.split(seg_k.lengths))
    return torch.cat([
        F.scaled_dot_product_attention(qi.transpose(0, 1), ki.transpose(0, 1), vi.transpose(0, 1)).transpose(0, 1)
        for qi, ki, vi in parts
    ])


@functools.cache
def load_fa3_kernel():
    """Fetch the FA3 hub kernel (once per process). ``get_kernel`` with no revision works on
    older ``kernels`` (<0.13); newer versions require an explicit revision and raise ``ValueError``."""
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
