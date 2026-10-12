"""Decoupled DMD + DMDR: a frozen Qwen3-VL judge's pairwise reward on the student's x0
(arXiv 2511.13649)."""

from __future__ import annotations

import torch

from args.reward_args import add_dmdr_args
from data.batch import T2IBatch, get_caption_short_description, get_caption_text, get_edit_instruction
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
        self.dmdr_groups = [kind.strip() for kind in self.args.dmdr_groups.split(",")]
        # Per-rank text coin: a text record's check and its pair are judged in the same passes either way.
        self.dmdr_text_rng = torch.Generator().manual_seed(self.args.seed + 4 + 1000 * self.accelerator.process_index)
        # Per-rank pair-order draws: which image the judge sees first, per record.
        self.dmdr_order_rng = torch.Generator().manual_seed(self.args.seed + 7 + 1000 * self.accelerator.process_index)
        if self.args.dmdr_scale > 0 or self.args.dmdr_text_scale > 0:
            if self.vae is None:
                self.vae = build_vae(self.args, self.accelerator, self.weight_dtype)
            self.qwen_reward_model = build_qwen_reward(self.args, self.accelerator, self.weight_dtype)
            logger.info(f"DMDR enabled: scale={self.args.dmdr_scale} prob={self.args.dmdr_prob} judge={self.args.dmdr_judge}")

    def _reward_losses(self, batch: T2IBatch, setup: dict, dmd_latents: torch.Tensor):
        if self.qwen_reward_model is None:
            return None
        kind = "EDIT" if self.is_edit_batch(batch) else "T2I"  # every rank's step is the same kind
        if kind not in self.dmdr_groups or torch.rand((), generator=self.dmdr_rng).item() >= self.args.dmdr_prob:
            return None
        height, width, device = setup["image"]
        weights = (setup["segments"] >= self.args.dmdr_min_segment).float().to(device)
        text_mask = torch.tensor([
            self.args.dmdr_text_scale > 0
            and get_caption_text(c) is not None
            and torch.rand((), generator=self.dmdr_text_rng).item() < self.args.dmdr_text_prob
            for c in batch.captions
        ], device=device)
        return self._qwen_reward_loss(
            dmd_latents, setup["real_latents"], batch.captions, weights, text_mask, height, width, self.judge_references_for(batch)
        )

    def judge_references_for(self, batch):
        """Per record, its images to edit for the judge's pairwise edit prompt; None = the T2I pair prompt."""
        return None

    def _qwen_reward_loss(self, generated_latents, real_latents, captions, weights, text_mask, height, width, references=None) -> dict:
        """Judge the decoded student x0 against the decoded real latents.

        Pair: ``dmdr_scale * mean(w * (1 - win))``, win = the student's P(yes) / (P(yes) + P(no)).
        With ``dmdr_both_orders`` the pair is asked in both orders in one pass and the two wins
        averaged (the stock judge has a strong position bias); otherwise each pair is shown in a
        random order. Text check: records in ``text_mask`` instead take a one-image yes/no "is
        exactly this text rendered?", ``dmdr_text_scale * mean(w * (1 - P(yes)))``. The gradient
        reaches the student through the VAE decoder. ``weights`` (B,) zero out records that are not
        judged, so every rank runs the same judge forwards (the judge is FSDP-sharded). With ``references``
        (edit records: per record, its images to edit) the pair takes the pairwise edit prompt: the
        images to edit, the instruction, then the two candidates."""
        a = self.args

        def decode(latents):
            return decode_to_images(self.vae, latents, height, width, self.vae_scale_factor, self.vae_config, a.do_patching) * 255.0

        n = len(weights)
        generated_images = decode(generated_latents)
        with torch.no_grad():
            real_images = decode(real_latents)
        prompts = [get_caption_short_description(c) for c in captions]
        prompt_ver = a.dmdr_prompt_ver
        if references is not None:
            prompts, prompt_ver = [get_edit_instruction(c) or c for c in captions], "edit"
        pair_w = weights * (~text_mask).float()
        if a.dmdr_both_orders:
            first = torch.ones(n, dtype=torch.bool, device=weights.device)
            first = torch.cat([first, ~first])  # the student shown first, then second
            prompts = prompts * 2
            references = None if references is None else references * 2
            generated_images, real_images = torch.cat([generated_images] * 2), torch.cat([real_images] * 2)
        else:
            first = (torch.rand(n, generator=self.dmdr_order_rng) < 0.5).to(weights.device)
        pick = first.view(-1, 1, 1, 1)
        yes_no = self.qwen_reward_model.get_relative_distribution(
            prompts,
            torch.where(pick, generated_images, real_images),
            torch.where(pick, real_images, generated_images),
            prompt_ver=prompt_ver,
            refs=references,
        )
        yes_no = yes_no / yes_no.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        wins = torch.where(first, yes_no[:, 0], yes_no[:, 1])  # the student's win in each pair
        win = wins.view(-1, n).mean(0)
        loss = ((1.0 - win) * pair_w).sum() / n * a.dmdr_scale

        def weighted_mean(values, w):
            return (values.detach() * w).sum() / w.sum().clamp_min(1)

        first_w = pair_w.repeat(len(first) // n) * first.float()
        logs = {
            "dmdr_win": weighted_mean(win, pair_w),
            "dmdr_win_first": weighted_mean(wins, first_w),
            "dmdr_win_second": weighted_mean(wins, pair_w.repeat(len(first) // n) - first_w),
        }
        if a.dmdr_text_scale > 0:
            text_w = weights * text_mask.float()
            texts = [get_caption_text(c) or "" for c in captions]
            p_text = self.qwen_reward_model.get_text_probability(texts, generated_images[:n])
            loss = loss + ((1.0 - p_text) * text_w).sum() / n * a.dmdr_text_scale
            logs["dmdr_text"] = weighted_mean(p_text, text_w)
        return {"loss": loss, "named_losses": {"dmdr_loss": loss.detach(), **logs}}
