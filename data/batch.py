"""Typed batch structs + caption parsing helpers."""

import json
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class T2IBatch:
    pixel_values: torch.Tensor  # (B, C, H, W) — VAE latents
    captions: List[str]

    @classmethod
    def from_tuple(cls, tup):
        pixel_values, captions = tup
        return cls(pixel_values=pixel_values, captions=captions)


def get_caption_short_description(caption):
    """Extract `short_description` from a structured caption; fall back to the raw text."""
    if isinstance(caption, dict):
        return caption.get("short_description", "")
    if not isinstance(caption, str):
        return str(caption)
    try:
        structured = json.loads(caption)
    except (json.JSONDecodeError, TypeError):
        return caption
    if isinstance(structured, dict):
        return structured.get("short_description", "")
    return caption
