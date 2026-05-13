"""Typed batch structs."""
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class T2IBatch:
    pixel_values: torch.Tensor              # (B, C, H, W) — VAE latents
    captions: List[str]

    @classmethod
    def from_tuple(cls, tup):
        pixel_values, captions = tup
        return cls(pixel_values=pixel_values, captions=captions)
