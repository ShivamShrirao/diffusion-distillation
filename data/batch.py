"""Typed batch structs + caption parsing helpers."""

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class T2IBatch:
    pixel_values: torch.Tensor  # (B, C, H, W) — VAE latents
    captions: List[str]
    resolution: Optional[Tuple[int, int]] = None  # (height, width)

    @classmethod
    def from_tuple(cls, tup):
        pixel_values, captions, resolution = tup
        return cls(pixel_values=pixel_values, captions=captions, resolution=resolution)


def parse_caption(caption) -> dict | None:
    """A structured caption as a dict, or None for free text."""
    if isinstance(caption, dict):
        return caption
    try:
        structured = json.loads(caption)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(structured, dict) and "edit_instruction" in structured:  # an edit prompt: the caption of its result
        return parse_caption(structured.get("structured_caption"))
    return structured if isinstance(structured, dict) else None


def get_caption_text(caption) -> str | None:
    """The text the image should show: a structured caption's ``text_render`` entries, joined."""
    entries = (parse_caption(caption) or {}).get("text_render")
    if not isinstance(entries, list):
        return None
    texts = [e["text"].strip() for e in entries if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"].strip()]
    return " ".join(texts) or None


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
    if isinstance(structured, dict) and "edit_instruction" in structured:  # an edit prompt: the caption of its result
        return get_caption_short_description(structured.get("structured_caption") or "")
    if isinstance(structured, dict):
        return structured.get("short_description", "")
    return caption


def edit_prompt(caption, instruction: str) -> str:
    """Fibo 2.0's edit prompt (before its image markers): the instruction, with the caption of the result
    when the record has one."""
    if not caption:
        return instruction
    if isinstance(caption, str):
        try:
            caption = json.loads(caption)
        except json.JSONDecodeError:
            pass
    return json.dumps({"structured_caption": caption, "edit_instruction": instruction}, ensure_ascii=False)


def get_edit_instruction(caption) -> str | None:
    """The instruction of an edit prompt built by ``edit_prompt`` with a caption; None otherwise."""
    try:
        prompt = json.loads(caption)
    except (json.JSONDecodeError, TypeError):
        return None
    return prompt.get("edit_instruction") if isinstance(prompt, dict) else None


@dataclass
class EditBatch(T2IBatch):
    references: Optional[list] = None  # per record: (VAE crops, Qwen3-VL copies) of its images to edit; None = T2I

    @classmethod
    def from_tuple(cls, tup):
        pixel_values, captions, resolution, references = tup
        return cls(pixel_values=pixel_values, captions=captions, resolution=resolution, references=references)
