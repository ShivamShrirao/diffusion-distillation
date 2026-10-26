"""SMD trainer: segment matching on TDM segments, no fake critic (arXiv 2610.18472).

Same rollout, teachers, EMA and DMDR as the DMD trainers; the student update is ``compute_smd_loss``
(plus the DMDR judge loss when on). With ``--smd_pivot learned`` a Pivot head reads the student's
tokens at one late main block and bends the query path; it is replicated and stepped here.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from args.smd_args import add_smd_args
from data.batch import T2IBatch
from losses.smd import compute_smd_loss
from models.pivot import PivotHead
from train_utils.checkpoint import resolve_resume_checkpoint
from trainers.base import logger
from trainers.dmd_edit import DecoupledDMDEditTrainer
from trainers.dmdr import DDMDRTrainer


class SMDTrainer(DDMDRTrainer):
    PARSER_MIXINS = DDMDRTrainer.PARSER_MIXINS + [add_smd_args]
    has_fake_critic = False

    def __init__(self):
        super().__init__()
        self.pivot = None
        self.pivot_optimizer = None
        self.captured = None  # the Pivot head's inputs from the student's grad forward

    def extra_setup(self):
        super().extra_setup()
        a = self.args
        if self.route_views:
            raise ValueError("SMD supports only the per-step t2i_teacher_prob coin")
        if a.smd_pivot == "learned" and not a.packed_varlen:
            raise ValueError("smd_pivot learned needs --packed_varlen 1")
        a.dfake_gen_update_ratio = 1  # no critic: every step is a student update

    def configure_transformer(self):
        super().configure_transformer()
        if self.args.smd_pivot != "learned":
            return
        a, cfg = self.args, self.transformer.config
        self.pivot = PivotHead(
            cfg.dim, a.pivot_dim, cfg.dim // cfg.n_heads, a.pivot_blocks, a.pivot_terms, cfg.in_channels * cfg.patch_size**2
        ).to(self.accelerator.device)
        if dist.is_initialized():  # identical replicas: every rank starts from rank 0's init
            for p in self.pivot.parameters():
                dist.broadcast(p.data, src=0)
        block = self.transformer.layers[cfg.n_layers - a.pivot_feature_block_from_end]
        block.register_forward_pre_hook(self.capture_pivot_inputs)
        logger.info(f"Pivot head: {sum(p.numel() for p in self.pivot.parameters()):,} params")

    def capture_pivot_inputs(self, module, args):
        """Forward pre-hook on the student's feature block: keep the last grad forward's packed inputs."""
        if torch.is_grad_enabled() and self.captured is None:
            stream, packed, freqs = args[:3]
            self.captured = (stream.detach(), packed, freqs)

    def build_optimizer(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.args.learning_rate, **self.adamw_kwargs())
        self.optimizer = [optimizer]
        self.lr_scheduler = [self.build_scheduler(optimizer)]
        if self.pivot is not None:
            self.pivot_optimizer = torch.optim.AdamW(self.pivot.parameters(), lr=self.args.pivot_lr, **self.adamw_kwargs())

    def post_optimizer_setup(self):
        super().post_optimizer_setup()
        resume = resolve_resume_checkpoint(self.args)
        path = os.path.join(resume, "pivot_head.pt") if resume else None
        if self.pivot is not None and path and os.path.isfile(path):
            state = torch.load(path, map_location=self.accelerator.device, weights_only=True)
            self.pivot.load_state_dict(state["model"])
            self.pivot_optimizer.load_state_dict(state["optimizer"])
            logger.info(f"Loaded the Pivot head from {path}")

    def save_checkpoint(self):
        super().save_checkpoint()
        if self.pivot is not None and self.accelerator.is_main_process:
            path = os.path.join(self.args.checkpoint_local_path, f"checkpoint_{self.global_step:06d}", "pivot_head.pt")
            torch.save({"model": self.pivot.state_dict(), "optimizer": self.pivot_optimizer.state_dict()}, path)

    def pivot_coefficients(self, batch_size: int):
        if self.pivot is None:
            return None
        stream, packed, freqs = self.captured
        self.captured = None
        return self.pivot(stream, packed, freqs, batch_size)

    def pivot_update(self) -> float:
        """Average the replicated head's gradients over all ranks (zeros where a rank has none, so
        every rank joins every all-reduce), clip, step."""
        params = list(self.pivot.parameters())
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if dist.is_initialized():
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(params, self.args.max_grad_norm)
        self.pivot_optimizer.step()
        self.pivot_optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def training_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        self.captured = None
        return super().training_step(batch, fetch_time_sec=fetch_time_sec)

    def _generator_update(self, setup: dict, batch, branch: dict, named_losses):
        """Student step: SMD loss (plus the DMDR reward), backward, optimizer, Pivot head, EMA.

        Returns ``(smd_outputs, student_grad_norm)``."""
        eh, te_layers, patched_image_ids, joint_attn_kwargs = setup["cond"]
        dmd_latents = setup["rollout"][0]
        _, seg_lo, seg_hi = setup["tdm"]
        packed_stream, text_mask = setup["packed"]
        batch_size = dmd_latents.shape[0]
        null_embeds, null_layers, null_mask = self.null_conditioning_for(eh.shape[0], eh.shape[1], dtype=self.weight_dtype)
        smd_outputs = compute_smd_loss(
            student_input=setup["student_input"],
            generated_latents=dmd_latents,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            real_transformer=branch["teacher"],
            cond_hidden_states=eh,
            cond_layers=te_layers,
            uncond_hidden_states=null_embeds,
            uncond_layers=null_layers,
            img_ids=patched_image_ids,
            model_dtype=self.weight_dtype,
            args=self.args,
            real_guidance_scale=branch["guidance"],
            coefficients=self.pivot_coefficients(batch_size),
            generator=self.generator,
            packed_stream=packed_stream,
            text_mask_after_dropout=text_mask,
            uncond_text_mask=null_mask,
            context=joint_attn_kwargs.get("context_latents"),
        )
        smd_named = smd_outputs["named_losses"]
        named_losses.update(smd_named)
        self.record_loss_components({"loss": smd_outputs["loss"], "named_losses": smd_named})
        student_loss = smd_outputs["loss"]
        reward_outputs = self._reward_losses(batch, setup, dmd_latents)
        if reward_outputs is not None:
            reward_named = reward_outputs.get("named_losses") or {}
            named_losses.update(reward_named)
            self.record_loss_components({"loss": reward_outputs["loss"].detach(), "named_losses": reward_named})
            student_loss = student_loss + reward_outputs["loss"]
        self.accelerator.backward(student_loss)

        student_grad_norm = None
        if self.accelerator.sync_gradients:
            student_grad_norm = self.clip_grad_norm(self.transformer)
            self.optimizer[0].step()
            self.optimizer[0].zero_grad()
            if self.pivot is not None:
                named_losses["grad_norm_pivot"] = self.pivot_update()
        self._maybe_update_ema()
        return smd_outputs, student_grad_norm


class SMDEditTrainer(DecoupledDMDEditTrainer, SMDTrainer):
    """SMD on edit and mixed edit + T2I data (``trainers/dmd_edit.py``)."""
