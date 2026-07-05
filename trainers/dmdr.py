"""Decoupled DMD + DMDR: a frozen Qwen3-VL judge's pairwise reward on the student's x0
(arXiv 2511.13649)."""

from __future__ import annotations

import torch

from args.reward_args import add_dmdr_args
from data.batch import T2IBatch, get_caption_short_description
from models.vae import build_vae, decode_to_images
from reward.qwen3_reward import build_qwen_reward
from trainers.base import logger
from trainers.dmd import DecoupledDMDT2ITrainer


class DDMDRTrainer(DecoupledDMDT2ITrainer):
    """Decoupled DMD with the DMDR judge loss added to the student update."""

    PARSER_MIXINS = DecoupledDMDT2ITrainer.PARSER_MIXINS + [add_dmdr_args]

    def __init__(self):
        super().__init__()
        self.qwen_reward_model = None

    def extra_setup(self):
        super().extra_setup()
        # Rank-shared coin (base seed): every rank runs the sharded judge on the same steps.
        self.dmdr_rng = torch.Generator().manual_seed(self.args.seed + 3)
        # Per-rank pair-order draws: which image the judge sees first, per record.
        self.dmdr_order_rng = torch.Generator().manual_seed(self.args.seed + 7 + 1000 * self.accelerator.process_index)
        if self.args.dmdr_scale > 0:
            if self.vae is None:
                self.vae = build_vae(self.args, self.accelerator, self.weight_dtype)
            self.qwen_reward_model = build_qwen_reward(self.args, self.accelerator, self.weight_dtype)
            logger.info(f"DMDR enabled: scale={self.args.dmdr_scale} prob={self.args.dmdr_prob} judge={self.args.dmdr_judge}")

    def _reward_losses(self, batch: T2IBatch, setup: dict, dmd_latents: torch.Tensor):
        if self.qwen_reward_model is None:
            return None
        if torch.rand((), generator=self.dmdr_rng).item() >= self.args.dmdr_prob:
            return None
        height, width, device = setup["image"]
        weights = (setup["segments"] >= self.args.dmdr_min_segment).float().to(device)
        return self._qwen_reward_loss(dmd_latents, setup["real_latents"], batch.captions, weights, height, width)

    def _qwen_reward_loss(self, generated_latents, real_latents, captions, weights, height, width) -> dict:
        """Judge the decoded student x0 against the decoded real latents, each pair in a random
        order. The student's win is its P(yes) / (P(yes) + P(no)) as image 1, else P(no) / (...);
        the loss is ``dmdr_scale * mean(w * (1 - win))`` and reaches the student through the VAE
        decoder. ``weights`` (B,) zero out records that are not judged, so every rank still runs
        one judge forward of the same shape (the judge is FSDP-sharded)."""

        def decode(latents):
            return decode_to_images(
                self.vae, latents, height, width, self.vae_scale_factor, self.vae_config, self.args.do_patching
            ) * 255.0

        generated_images = decode(generated_latents)
        with torch.no_grad():
            real_images = decode(real_latents)
        prompts = [get_caption_short_description(c) for c in captions]
        first = (torch.rand(len(weights), generator=self.dmdr_order_rng) < 0.5).to(weights.device)
        pick = first.view(-1, 1, 1, 1)
        yes_no = self.qwen_reward_model.get_relative_distribution(
            prompts,
            torch.where(pick, generated_images, real_images),
            torch.where(pick, real_images, generated_images),
            prompt_ver=self.args.dmdr_prompt_ver,
        )
        yes_no = yes_no / yes_no.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        win = torch.where(first, yes_no[:, 0], yes_no[:, 1])
        loss = ((1.0 - win) * weights).sum() / len(weights) * self.args.dmdr_scale
        return {
            "loss": loss,
            "named_losses": {
                "dmdr_loss": loss.detach(),
                "dmdr_win": (win.detach() * weights).sum() / weights.sum().clamp_min(1),
            },
        }
