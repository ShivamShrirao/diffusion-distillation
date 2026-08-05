"""FA3 varlen attention for the training hot path.

q/k/v are flat ``[tokens, heads, head_dim]`` streams and each record attends only within its own
segment (``varlen_attention``), through the FlashAttention-3 hub kernel in varlen mode. The rest of
Fibo 2.0's attention (the Perceiver) runs on diffusers' FA3 varlen backend (``FA3_BACKEND``).
"""

import functools

import torch

FA3_BACKEND = "_flash_3_varlen_hub"  # diffusers' enum name for the FA3 hub varlen kernel


def varlen_attention(q, k, v, seg_q, seg_k):
    """Attention of each query segment to its key segment; ``seg_*`` are ``Segments``."""
    dtype = q.dtype
    q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))  # the kernel takes 16-bit inputs only
    out = load_fa3_kernel().flash_attn_varlen_func(
        q, k, v, seg_q.cu_seqlens, seg_k.cu_seqlens, seg_q.max_len, seg_k.max_len
    )
    return (out[0] if isinstance(out, tuple) else out).to(dtype)  # some builds also return the LSE


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
    downloads first, the rest of that node then hit the warm cache."""
    with accelerator.local_main_process_first():
        load_fa3_kernel()
