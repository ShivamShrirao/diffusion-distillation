"""Decoupled DMD/TDM on Fibo 2.0 edits.

Fibo 2.0 edits through reference images: Qwen3-VL reads the images to edit with the edit prompt,
and their VAE latents join every student, teacher and fake-critic forward as clean context tokens
(``context_latents``); the velocity covers the generated image only. Rollout, losses, EMA and
checkpointing are the T2I trainers'.

A run can mix edit and T2I data: every micro-step is one kind on every rank (``data/build.py``).
Edit steps take the main teacher; T2I steps take t2i_teacher on its coin.
"""

from __future__ import annotations

import json

from PIL import Image

from data.batch import EditBatch, get_edit_instruction
from data.build import build_dataloader_edit, preprocess_references
from models.vae import build_vae, encode_images
from trainers.base import logger
from trainers.dmd import DecoupledDMDT2ITrainer


class DecoupledDMDEditTrainer(DecoupledDMDT2ITrainer):
    """Decoupled DMD/TDM on edit and mixed edit + T2I records."""

    BATCH_CLS = EditBatch
    DATALOADER_BUILDER = staticmethod(build_dataloader_edit)

    def extra_setup(self):
        super().extra_setup()
        if self.vae is None:  # encodes the images to edit
            self.vae = build_vae(self.args, self.accelerator, self.weight_dtype)
        logger.info("DMD-edit: the images to edit join every forward as context tokens")

    def encode_references(self, crops) -> list:
        """One record's images to edit -> their (C, h, w) latents, normalized as the targets are."""
        device = self.accelerator.device
        return [encode_images(self.vae, crop.to(device, non_blocking=True))[0] for crop in crops]

    def prompt_images_for(self, batch: EditBatch):
        if not self.is_edit_batch(batch):
            return None
        return [images for _, images in batch.references]

    def context_references_for(self, batch: EditBatch):
        if not self.is_edit_batch(batch):
            return None
        return [self.encode_references(crops) for crops, _ in batch.references]

    def teacher_branch(self, batch, batch_size: int, device) -> dict:
        """The step kind picks the teacher: edit steps take the main teacher, T2I steps t2i_teacher on the
        rank-shared coin (drawn on T2I steps only), else the main teacher."""
        if self.is_edit_batch(batch):
            return self.main_branch(batch_size, device)
        return super().teacher_branch(batch, batch_size, device)

    def validation_inputs(self, caption: str, height: int, width: int):
        """An entry with ``images`` (paths of the images to edit) is an edit; its other keys are the edit
        prompt: ``edit_instruction`` and optionally ``structured_caption``."""
        try:
            entry = json.loads(caption)
        except json.JSONDecodeError:
            entry = None
        if not isinstance(entry, dict) or "images" not in entry:
            return super().validation_inputs(caption, height, width)
        paths = entry.pop("images")
        crops, images = preprocess_references([Image.open(p) for p in paths], height, width)
        return json.dumps(entry, ensure_ascii=False), [images], [self.encode_references(crops)]

    def log_caption_for(self, caption: str) -> str:
        return get_edit_instruction(caption) or super().log_caption_for(caption)
