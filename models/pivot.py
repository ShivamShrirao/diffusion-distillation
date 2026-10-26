"""Learned Pivot head for SMD: per-token coefficients a_m of the query path's bend.

The head reads the student's (detached) token stream at the input of one late main block (captured
by a forward pre-hook on the packed path), runs a few small varlen transformer blocks over each
record's [gist | image | context] tokens (context: an edit record's images to edit) and maps the
image tokens to ``terms`` coefficient maps of the student's token shape. Its output layer starts
at zero, so training starts from the straight path.
Training-only: replicated on every rank (not FSDP), with its own optimizer, unused at inference.
"""

import torch
from torch import nn

from models.transformer import PackedAttention, rope
from train_utils.attention import varlen_attention


class PivotBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.heads = dim // head_dim
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))

    def forward(self, x, seg, freqs):
        q, k, v = self.qkv(self.norm1(x)).unflatten(-1, (3, self.heads, -1)).unbind(1)
        x = x + self.out(varlen_attention(rope(q, freqs), rope(k, freqs), v, seg, seg).flatten(-2))
        return x + self.mlp(self.norm2(x))


class PivotHead(nn.Module):
    def __init__(self, in_dim: int, dim: int, head_dim: int, depth: int, terms: int, channels: int):
        super().__init__()
        self.terms, self.channels = terms, channels
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(PivotBlock(dim, head_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, terms * channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, stream: torch.Tensor, packed: PackedAttention, freqs: torch.Tensor, batch_size: int) -> torch.Tensor:
        """A main block's packed inputs -> (B, terms, N, channels) coefficients. Image tokens are the
        rows < B (gist and context tokens take the clean row B), record by record."""
        x = self.proj(stream.float())
        for block in self.blocks:
            x = block(x, packed.seg, freqs)
        out = self.head(self.norm(x[packed.rows < batch_size]))
        return out.view(batch_size, -1, self.terms, self.channels).transpose(1, 2)
