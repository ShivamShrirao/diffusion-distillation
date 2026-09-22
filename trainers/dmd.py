"""Decoupled DMD trainers."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import wandb
from safetensors.torch import load_file, save_file

from args.base_args import add_base_args
from args.data_args import add_data_args
from args.dmd_args import add_decoupled_dmd_args
from data.batch import T2IBatch
from data.build import build_dataloader_t2i
from distributed import compile_transformer, fsdp2_set_reshard_after_forward, fsdp2_shard_model
from train_utils.ema import FullFtEMA
from train_utils.checkpoint import resolve_resume_checkpoint
from train_utils.packing import build_packed_layout, pack_t2i
from losses.dmd import add_flow_noise, compute_decoupled_dmd_loss, compute_fake_score_loss
from train_utils.attention import build_attention_mask, diffusers_backend_name
from models.transformer import Fibo2Transformer2DModel, build_transformer_config, load_transformer_init_weights
from models.vae import build_vae, decode_latents, decode_to_images
from schedulers.flow_matching import build_infer_scheduler, build_infer_timesteps
from schedulers.inference import predict_flow_step
from trainers.base import BaseTrainer, logger


def dmd_init_path(args, channel):
    """``{student,fake}`` subdir of resume ckpt, else ``--{channel}_init``."""
    resume = resolve_resume_checkpoint(args)
    if resume:
        return os.path.join(resume, channel)
    return getattr(args, f"{channel}_init", None)


class DecoupledDMDT2ITrainer(BaseTrainer):
    PARSER_MIXINS = [add_base_args, add_data_args, add_decoupled_dmd_args]
    BATCH_CLS = T2IBatch
    DATALOADER_BUILDER = staticmethod(build_dataloader_t2i)

    def __init__(self):
        super().__init__()
        self.infer_scheduler = None
        self.fake_transformer = None
        self.fake_transformer_unwrapped = None
        self.real_transformer = None
        self.real_transformer_unwrapped = None
        self.t2i_teacher = None
        self.ema_student = None

    def extra_setup(self):
        # Rank-uniform CPU RNG for per-step choices every rank must agree on (K, grad step).
        # Seeded from the base seed (NOT the per-rank seed): no collective, no device sync.
        self.shared_rng = torch.Generator().manual_seed(self.args.seed)
        self.infer_scheduler = build_infer_scheduler()
        a = self.args
        # Rank-shared teacher coin: every rank calls the same sharded teacher.
        self.t2i_teacher_rng = torch.Generator().manual_seed(a.seed + 5)
        if self.args.dmd_log_image_every > 0 or self.args.validate_every > 0:
            self.vae = build_vae(self.args, self.accelerator, self.weight_dtype)

    def configure_transformer(self):
        args = self.args
        logger.info("Using DMD full fine-tune mode: cloning student into fake + real transformers")
        self.transformer.requires_grad_(True)
        assert self.transformer.dtype == torch.float32

        transformer_config = build_transformer_config(args, self.vae_config, logger)
        with torch.device("meta"):
            self.fake_transformer = Fibo2Transformer2DModel.from_config(transformer_config)
            self.real_transformer = Fibo2Transformer2DModel.from_config(transformer_config)

        if not args.use_fsdp or self.accelerator.is_main_process:
            src_sd = self.transformer.state_dict()
            self.real_transformer.load_state_dict({k: v.detach().clone() for k, v in src_sd.items()}, assign=True)
            self.fake_transformer.load_state_dict({k: v.detach().clone() for k, v in src_sd.items()}, assign=True)
            del src_sd
            for t, adapter in ((self.transformer, "student"), (self.fake_transformer, "fake")):
                p = dmd_init_path(args, adapter)
                fp = os.path.join(p, "diffusion_pytorch_model.safetensors") if p else None
                if fp and os.path.exists(fp):
                    t.load_state_dict(load_file(fp), assign=True)
                    logger.info(f"Loaded full-FT '{adapter}' from {fp}")

        self.fake_transformer.requires_grad_(True)
        self.real_transformer.requires_grad_(False)
        if self.args.attn_backend != "native":
            self.fake_transformer.set_attention_backend(diffusers_backend_name(self.args.attn_backend))
            self.real_transformer.set_attention_backend(diffusers_backend_name(self.args.attn_backend))

        if args.gradient_checkpointing:
            self.fake_transformer.enable_gradient_checkpointing(args.gradient_checkpoint_first_n_blocks)
            self.real_transformer.enable_gradient_checkpointing(args.gradient_checkpoint_first_n_blocks)

        self.fake_transformer_unwrapped = self.fake_transformer
        self.real_transformer_unwrapped = self.real_transformer

        if args.t2i_teacher_prob > 0:
            if not args.t2i_teacher_path:
                raise ValueError("t2i_teacher_prob > 0 needs --t2i_teacher_path")
            with torch.device("meta"):
                self.t2i_teacher = Fibo2Transformer2DModel.from_config(transformer_config)
            if not args.use_fsdp or self.accelerator.is_main_process:
                load_transformer_init_weights(self.t2i_teacher, args.t2i_teacher_path, logger)
            self.t2i_teacher.requires_grad_(False)
            self.t2i_teacher.eval()
            if args.attn_backend != "native":
                self.t2i_teacher.set_attention_backend(diffusers_backend_name(args.attn_backend))
            if args.gradient_checkpointing:
                self.t2i_teacher.enable_gradient_checkpointing(args.gradient_checkpoint_first_n_blocks)
            logger.info(f"Loaded frozen t2i_teacher from {args.t2i_teacher_path} (prob={args.t2i_teacher_prob})")

        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Full-FT student trainable params: {n_trainable:,} / {n_total:,}")

    def shard_trainable_models(self):
        self.transformer = fsdp2_shard_model(self.transformer, self.accelerator)
        self.fake_transformer = fsdp2_shard_model(self.fake_transformer, self.accelerator)
        self.real_transformer = fsdp2_shard_model(self.real_transformer, self.accelerator)
        if self.t2i_teacher is not None:
            self.t2i_teacher = fsdp2_shard_model(self.t2i_teacher, self.accelerator)

    def build_optimizer(self):
        student_params = list(self.transformer.parameters())
        fake_params = list(self.fake_transformer.parameters())

        fake_lr = self.args.fake_lr if self.args.fake_lr is not None else self.args.learning_rate
        adamw_kwargs = self.adamw_kwargs()
        student_optimizer = torch.optim.AdamW(student_params, lr=self.args.learning_rate, **adamw_kwargs)
        fake_optimizer = torch.optim.AdamW(fake_params, lr=fake_lr, **adamw_kwargs)
        self.optimizer = [student_optimizer, fake_optimizer]
        self.lr_scheduler = [self.build_scheduler(student_optimizer), self.build_scheduler(fake_optimizer)]

    def post_optimizer_setup(self):
        if self.args.ema_decay > 0:
            self.ema_student = FullFtEMA(self.transformer, decay=self.args.ema_decay)
            logger.info(f"Full-FT student EMA enabled with decay={self.args.ema_decay}, tracking {self.ema_student.local_numel:,} local-shard params per rank")
            ema_init = dmd_init_path(self.args, "ema")
            ema_fp = os.path.join(ema_init, "diffusion_pytorch_model.safetensors") if ema_init else None
            resume = resolve_resume_checkpoint(self.args)
            # A checkpoint saved before ema_start_step holds an un-seeded shadow (still the
            # run's init weights); loading it would mark the EMA seeded and skip the
            # reset_to_live at ema_start_step, anchoring the EMA to those init weights.
            stale_shadow = resume is not None and int(os.path.basename(resume).split("_")[-1]) < self.args.ema_start_step
            if ema_fp and os.path.exists(ema_fp) and not stale_shadow:
                self.ema_student.load_shadow_from(ema_fp, self.transformer, bool(self.args.use_fsdp))
                # A loaded shadow IS the seed — without this flag the first eligible sync
                # step would reset_to_live() and silently discard the loaded EMA history.
                self._ema_seeded = True
                logger.info(f"Seeded EMA shadow from {ema_fp}")

        if self.args.full_torch_compile or self.args.regional_compile:
            kwargs = dict(regional=self.args.regional_compile, dynamic=bool(self.args.packed_varlen))
            compile_transformer(self.fake_transformer, self.fake_transformer_unwrapped, **kwargs)
            compile_transformer(self.real_transformer, self.real_transformer_unwrapped, **kwargs)
            if self.t2i_teacher is not None:
                compile_transformer(self.t2i_teacher, self.t2i_teacher, **kwargs)

    def is_edit_batch(self, batch) -> bool:
        references = getattr(batch, "references", None)
        has_references = references is not None and any(r is not None for r in references)
        if has_references:
            assert all(r is not None for r in references), "mixed edit/T2I microbatch"
        return has_references

    def context_references_for(self, batch) -> list | None:
        """Per record, the latents of its images to edit (Fibo 2.0's context tokens); None for T2I."""
        return None

    def prompt_images_for(self, batch) -> list | None:
        """Per record, its images to edit as the text encoder reads them; None for T2I."""
        return None

    def validation_inputs(self, caption: str, height: int, width: int):
        """``(prompt, text-encoder images, context latents)`` of one validation entry; T2I: the caption alone."""
        return caption, None, None

    def t2i_teacher_coin(self) -> bool:
        """Rank-shared coin: does this step's real score come from t2i_teacher?"""
        return torch.rand((), generator=self.t2i_teacher_rng).item() < self.args.t2i_teacher_prob

    def prepare_dmd_step(self, batch: T2IBatch) -> dict:
        """Shared step prologue for the DMD trainers: conditioning, packing, attention
        mask, per-step K / timestep grid, and initial noise. The trainer-specific rollout
        in ``training_step`` consumes the returned dict."""
        height, width = batch.resolution or self.args.resolution
        device = self.accelerator.device

        eh, te_layers, prompt_attn_mask, text_encoder_time = self.run_text_encoder(batch.captions, self.prompt_images_for(batch))
        pixel_values = batch.pixel_values.to(device=device, dtype=torch.float32, non_blocking=True)
        # Conditioning in the compute dtype up front: the loss helpers' .to(model_dtype)
        # casts (~60 tensors per call site) become no-ops.
        eh = eh.to(device=device, dtype=self.weight_dtype)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=self.weight_dtype)

        latents = (pixel_values - self.vae_config["shift_factor"]) * self.vae_config["scaling_factor"]
        context = self.context_references_for(batch)


        patched_latents, patched_image_ids = pack_t2i(latents, height, width, self.vae_scale_factor, self.args.do_patching, device)
        latents_seq_len = patched_latents.shape[1]
        text_mask = prompt_attn_mask.bool()
        mask_time = [0.0]
        attn_mask = packed_stream = None
        if self.args.packed_varlen:
            packed_stream = build_packed_layout(text_mask, patched_image_ids, self.transformer_unwrapped.config, context)
        elif not text_mask.all():
            mask_time = []
            with self.cuda_timer(mask_time):
                attn_mask = build_attention_mask(prompt_attn_mask, latents_seq_len, self.weight_dtype, self.args.attn_backend)
        joint_attn_kwargs = {"attention_mask": attn_mask}
        if context is not None:
            joint_attn_kwargs["context_latents"] = context

        # TDM steps-aware: sample the step-count K per iteration (shared RNG so FSDP ranks agree).
        num_inference_steps = self.args.dmd_num_inference_steps
        if self.args.tdm_step_list:
            k_choices = [int(x) for x in self.args.tdm_step_list.split(",") if x.strip()]
            num_inference_steps = k_choices[int(torch.randint(0, len(k_choices), (1,), generator=self.shared_rng).item())]
        infer_timesteps = build_infer_timesteps(self.infer_scheduler, num_inference_steps, latents_seq_len, device, mu=self.args.dmd_mu)

        generated_latents = torch.randn(patched_latents.shape, device=device, dtype=patched_latents.dtype, generator=self.generator)

        do_gen_update = (self.global_step + 1) % self.args.dfake_gen_update_ratio == 0

        gen_kwargs = dict(
            encoder_hidden_states=eh,
            text_encoder_layers=te_layers,
            img_ids=patched_image_ids,
            attn_kwargs=joint_attn_kwargs,
            latents_seq_len=latents_seq_len,
            packed_stream=packed_stream,
            text_mask_after_dropout=text_mask,
        )
        return dict(
            height=height,
            width=width,
            device=device,
            eh=eh,
            te_layers=te_layers,
            patched_latents=patched_latents,
            patched_image_ids=patched_image_ids,
            joint_attn_kwargs=joint_attn_kwargs,
            packed=[packed_stream, text_mask],
            infer_timesteps=infer_timesteps,
            num_inference_steps=num_inference_steps,
            generated_latents=generated_latents,
            do_gen_update=do_gen_update,
            gen_kwargs=gen_kwargs,
            text_encoder_time=text_encoder_time,
            mask_time=mask_time,
        )

    def training_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        prep = self.prepare_dmd_step(batch)
        device = prep["device"]
        infer_timesteps, gen_kwargs = prep["infer_timesteps"], prep["gen_kwargs"]
        generated_latents, do_gen_update = prep["generated_latents"], prep["do_gen_update"]
        patched_latents = prep["patched_latents"]

        # Segment mode (official TDM): full no-grad rollout, one random segment per SAMPLE.
        # Otherwise: one shared segment per step, rollout only up to it.
        per_sample = self.args.tdm_segment_renoise
        generate_time = []
        with self.cuda_timer(generate_time):
            bsz = generated_latents.shape[0]
            num_steps = len(infer_timesteps)
            if per_sample:
                # Fixed-size CPU draw keeps shared_rng in rank lockstep regardless of bsz.
                seg_idx = torch.randint(0, num_steps, (64,), generator=self.shared_rng)[:bsz]
            else:
                seg_idx = torch.randint(0, num_steps, (1,), generator=self.shared_rng).expand(bsz)
            idx_dev = seg_idx.to(device)
            t_vec = infer_timesteps[idx_dev]  # (B,) grad timesteps

            # PiD-style teacher forcing: with shared-RNG probability (rank-uniform — FSDP
            # forward counts must agree), skip the rollout and grad-forward from the GT
            # target flow-noised to the selected timestep instead of the student's own
            # trajectory state.
            teacher_forced = (
                self.args.teacher_force_prob > 0
                and torch.rand(1, generator=self.shared_rng).item() < self.args.teacher_force_prob
            )

            # Per-sample segments need the full rollout (and the forward count must
            # match across ranks — FSDP2 collectives); the shared segment rolls only up
            # to it. Keep the student's params gathered across the multi-forward rollout
            # instead of re-all-gathering the full model every step; weights don't
            # change inside the loop, and we reshard before the grad forward.
            n_roll = 0 if teacher_forced else (num_steps - 1 if per_sample else int(seg_idx[0]))
            hold_unsharded = n_roll > 1 and self.is_fsdp2()
            if hold_unsharded:
                fsdp2_set_reshard_after_forward(self.transformer, False)
            try:
                with torch.no_grad():
                    # Per-sample: select each sample's rollout input at its segment incrementally
                    # (sync-free, no K+1 trajectory stack). The latent fed into step i+1 is the
                    # output of step i; samples with seg_idx == 0 keep the initial latent.
                    if per_sample:
                        gen_input = generated_latents.clone()
                    for i, t in enumerate(infer_timesteps[:n_roll]):
                        pred = predict_flow_step(self.transformer, generated_latents, t, **gen_kwargs)
                        generated_latents = self.infer_scheduler.step(pred, t, generated_latents, return_dict=False)[0]
                        if per_sample:
                            gen_input = torch.where((idx_dev == i + 1).view(-1, 1, 1), generated_latents, gen_input)
            finally:
                if hold_unsharded:
                    fsdp2_set_reshard_after_forward(self.transformer, True, reshard_now=True)
            if teacher_forced:
                gen_input = add_flow_noise(patched_latents, t_vec, self.args.noise_max_t)
            elif not per_sample:
                gen_input = generated_latents

            with torch.set_grad_enabled(do_gen_update):
                pred = predict_flow_step(self.transformer, gen_input, t_vec, **gen_kwargs)
            gen_v = pred.detach() if per_sample else None
            sigma = (t_vec.float() / 1000.0).view(-1, 1, 1).to(device=device, dtype=gen_input.dtype)
            dmd_latents = gen_input - sigma * pred

        # Segment bounds: the selected step covers [seg_lo, seg_hi] in noise_max_t units
        # (seg_lo = 0 on the last step). Per-sample tensors in segment mode, floats otherwise.
        infer_ext = torch.cat([infer_timesteps.float(), infer_timesteps.new_zeros(1, dtype=torch.float32)])
        seg_hi = infer_ext[idx_dev]
        seg_lo = infer_ext[idx_dev + 1]
        if per_sample:
            grad_step_idx = seg_idx.float().mean().item()
        else:
            seg_hi, seg_lo = float(seg_hi[0]), float(seg_lo[0])
            grad_step_idx = int(seg_idx[0])

        return {
            "image": [prep["height"], prep["width"], device],
            "cond": [prep["eh"], prep["te_layers"], prep["patched_image_ids"], prep["joint_attn_kwargs"]],
            "rollout": [dmd_latents, do_gen_update, grad_step_idx],
            "teacher_forced": teacher_forced,
            "real_latents": patched_latents,
            "gen_v": gen_v,
            "segments": seg_idx,
            "packed": prep["packed"],
            "tdm": [prep["num_inference_steps"], seg_lo, seg_hi],
            "timings": [prep["text_encoder_time"], generate_time[0], prep["mask_time"][0], fetch_time_sec],
        }

    def teacher_branch(self, batch, batch_size: int, device) -> dict:
        """This step's real teacher and CA guidance scale: t2i_teacher (the main CFG) on a rank-shared
        coin, else the main teacher."""
        if self.t2i_teacher is not None and self.t2i_teacher_coin():
            return self.t2i_branch(self.t2i_teacher, batch_size, device)
        return self.main_branch(batch_size, device)

    def main_branch(self, batch_size: int, device) -> dict:
        """The main teacher with its CFG."""
        return {"teacher": self.real_transformer, "guidance": None}

    def t2i_branch(self, teacher, batch_size: int, device) -> dict:
        """``teacher`` with the main CFG."""
        return {**self.main_branch(batch_size, device), "teacher": teacher}

    def _reward_losses(self, batch: T2IBatch, setup: dict, dmd_latents: torch.Tensor):
        """Hook for a reward term on the student's x0 (DMDR); no-op in the base trainer."""
        return None

    def optimization_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        setup = self.training_step(batch, fetch_time_sec=fetch_time_sec)
        height, width, device = setup["image"]
        eh, te_layers, patched_image_ids, joint_attn_kwargs = setup["cond"]
        dmd_latents, do_gen_update, grad_step_idx = setup["rollout"]
        teacher_forced = setup["teacher_forced"]
        _, seg_lo, seg_hi = setup.get("tdm", [self.args.dmd_num_inference_steps, None, None])
        gen_v = setup.get("gen_v")
        packed_stream, text_mask = setup["packed"]
        text_encoder_time, generate_time, mask_time, fetch_time_sec = setup["timings"]
        student_optimizer, fake_optimizer = self.optimizer[0], self.optimizer[1]
        student_lr_scheduler, fake_lr_scheduler = self.lr_scheduler[0], self.lr_scheduler[1]

        named_losses = {}
        dmd_outputs = None
        dmd_loss = torch.tensor(0.0, device=device)
        student_grad_norm = None

        branch = self.teacher_branch(batch, dmd_latents.shape[0], device)
        real_module = branch["teacher"]
        fake_module = self.fake_transformer
        # 1) Student update — DMD loss (internal fake DM fwd + real teacher fwd).
        if do_gen_update:
            # Only the gen-update losses consume the null (uncond) embeddings; cached, so
            # the ~60-tensor pad+repeat runs once per (bsz, text-len, dtype) bucket.
            current_null_embeds, current_null_layers, current_null_mask = self.null_conditioning_for(
                eh.shape[0], eh.shape[1], dtype=self.weight_dtype
            )
            dmd_outputs = compute_decoupled_dmd_loss(
                generated_latents=dmd_latents,
                generated_v=gen_v,
                seg_lo=seg_lo,
                seg_hi=seg_hi,
                real_transformer=real_module,
                fake_transformer=fake_module,
                cond_hidden_states=eh,
                cond_layers=te_layers,
                uncond_hidden_states=current_null_embeds,
                uncond_layers=current_null_layers,
                img_ids=patched_image_ids,
                attn_kwargs=joint_attn_kwargs,
                model_dtype=self.weight_dtype,
                args=self.args,
                real_guidance_scale=branch["guidance"],
                packed_stream=packed_stream,
                text_mask_after_dropout=text_mask,
                uncond_text_mask=current_null_mask,
            )
            dmd_loss = dmd_outputs["loss"]
            dmd_named_losses = dmd_outputs.get("named_losses") or {}
            named_losses.update(dmd_named_losses)
            self.record_loss_components({"loss": dmd_loss, "named_losses": dmd_named_losses})

            student_loss = dmd_loss

            reward_outputs = self._reward_losses(batch, setup, dmd_latents)
            if reward_outputs is not None:
                reward_named = reward_outputs.get("named_losses") or {}
                named_losses.update(reward_named)
                self.record_loss_components({"loss": reward_outputs["loss"].detach(), "named_losses": reward_named})
                student_loss = student_loss + reward_outputs["loss"]

            self.accelerator.backward(student_loss)
            # Step only at the accumulation boundary: raw FSDP2 optimizers do not skip
            # non-sync micro-steps themselves, so ungated they bypass accumulation and
            # turn ga into `ga` updates per logged step.
            if self.accelerator.sync_gradients:
                student_grad_norm = self.clip_grad_norm(self.transformer)
                student_optimizer.step()
                student_optimizer.zero_grad()
            if self.accelerator.sync_gradients and self.ema_student is not None and self.global_step >= self.args.ema_start_step:
                # Start EMA tracking at ema_start_step: re-seed the shadow from the live student
                # on the first eligible step so it isn't anchored to the init weights.
                if not getattr(self, "_ema_seeded", False):
                    if hasattr(self.ema_student, "reset_to_live"):
                        self.ema_student.reset_to_live()
                    self._ema_seeded = True
                self.ema_student.update()
            dmd_latents = dmd_latents.detach()

        if self.accelerator.sync_gradients:
            student_lr_scheduler.step()

        # 2) Fake score forward (separate, full fake backbone on detached student latents).
        fake_outputs = compute_fake_score_loss(
            transformer=fake_module,
            generated_latents=dmd_latents,
            cond_hidden_states=eh,
            cond_layers=te_layers,
            img_ids=patched_image_ids,
            attn_kwargs=joint_attn_kwargs,
            model_dtype=self.weight_dtype,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            generated_v=gen_v,
            args=self.args,
            packed_stream=packed_stream,
            text_mask_after_dropout=text_mask,
        )
        fake_loss = fake_outputs["loss"]
        fake_named_losses = fake_outputs.get("named_losses") or {}
        named_losses.update(fake_named_losses)
        self.record_loss_components({"loss": fake_loss, "named_losses": fake_named_losses})

        guidance_loss = fake_loss

        self.accelerator.backward(guidance_loss)

        fake_grad_norm = None
        if self.accelerator.sync_gradients:
            fake_grad_norm = self.clip_grad_norm(fake_module)
            fake_optimizer.step()
            fake_lr_scheduler.step()
            fake_optimizer.zero_grad()
            student_optimizer.zero_grad()

        wandb_extras = {}
        log_every = self.args.dmd_log_image_every
        if log_every > 0 and self.vae is not None and (self.global_step == 0 or (self.global_step + 1) % log_every == 0):
            wandb_extras = self._build_image_logs(
                generated_latents=dmd_latents.detach(),
                dmd_log_latents=dmd_outputs["log_latents"] if (do_gen_update and dmd_outputs is not None) else None,
                captions=batch.captions,
                context_latents=joint_attn_kwargs.get("context_latents"),
                height=height,
                width=width,
            )

        loss = dmd_loss.detach() + fake_loss.detach()
        log_extras = {
            "grad_step_idx": grad_step_idx,
            "teacher_forced": int(teacher_forced),
            "t2i_teacher": int(real_module is self.t2i_teacher),
        }
        if fake_grad_norm is not None:
            log_extras["grad_norm_fake"] = fake_grad_norm
        if student_grad_norm is not None:
            log_extras["grad_norm_student"] = student_grad_norm
        return {
            "loss": loss,
            "named_losses": named_losses,
            "timings": {
                "text_encoder_time": text_encoder_time,
                "generate_time": generate_time,
                "create_mask_time": mask_time,
                "fetch_batch": fetch_time_sec,
            },
            "log_extras": log_extras,
            "wandb_extras": wandb_extras,
        }

    def load_checkpoint_state(self, full_path: str):
        # student/fake weights are loaded in configure_transformer before the FSDP wrap, so
        # the full-state load is skipped for the multi-module (student/, fake/) layout.
        if os.path.isdir(os.path.join(full_path, "student")):
            logger.info("DMD multi-module checkpoint; student/fake weights loaded in configure_transformer.")
        else:
            self.accelerator.load_state(full_path, map_location="cpu")

    def save_checkpoint(self):
        """Gather student/fake/EMA state dicts (collectives: every rank enters) and
        write them on rank 0."""
        save_path = os.path.join(
            self.args.checkpoint_local_path,
            f"checkpoint_{self.global_step:06d}",
        )
        self.accelerator.wait_for_everyone()
        logger.info(f"Saving full-FT DMD checkpoint to {save_path}")
        student_sd = self.accelerator.get_state_dict(self.transformer, unwrap=False)
        fake_sd = self.accelerator.get_state_dict(self.fake_transformer, unwrap=False)
        if self.accelerator.is_main_process:
            for sub, sd, unwrapped in (
                ("student", student_sd, self.transformer_unwrapped),
                ("fake", fake_sd, self.fake_transformer_unwrapped),
            ):
                sub_dir = os.path.join(save_path, sub)
                os.makedirs(sub_dir, exist_ok=True)
                unwrapped.save_config(sub_dir)
                cleaned = {k.replace("module.", "").replace("._orig_mod", ""): v.contiguous() for k, v in sd.items()}
                save_file(cleaned, os.path.join(sub_dir, "diffusion_pytorch_model.safetensors"))
                del cleaned
            self.copy_data_input_json(save_path)
        del student_sd, fake_sd
        if self.ema_student is not None:
            self.ema_student.save(save_path, self.accelerator, self.transformer, self.transformer_unwrapped)
        self.accelerator.wait_for_everyone()

    def reference_panel(self, references, height: int, device):
        """One record's images to edit, decoded, scaled to ``height`` and side by side; None for T2I."""
        if not references:
            return None
        panel = []
        for ref in references:
            img = (decode_latents(self.vae, ref[None].to(device)) / 2 + 0.5).clamp(0, 1)
            size = (height, round(img.shape[-1] * height / img.shape[-2]))
            panel.append(F.interpolate(img, size=size, mode="bilinear", antialias=True)[0])
        return torch.cat(panel, dim=-1)

    @torch.no_grad()
    def _build_image_logs(
        self,
        *,
        generated_latents: torch.Tensor,
        dmd_log_latents: dict | None,
        captions,
        context_latents: list | None = None,  # per record, the latents of its images to edit
        height: int,
        width: int,
    ) -> dict:
        """Decode and log rank-0's local samples only — no cross-rank collective.

        gather_object of pickled tensor payloads proved unreliable at scale (rank-permuted
        panel pairings, truncated pickles mid-training). Image logs are cosmetic, so they
        stay rank-0-local; no other rank does collective work here.
        """
        if not self.accelerator.is_main_process:
            return {}
        vae_dtype = next(self.vae.parameters()).dtype

        def take_local(packed, i):
            idx = min(i, packed.shape[0] - 1)
            return packed[idx : idx + 1].to(dtype=vae_dtype).cpu()

        gathered = []
        for i in range(min(8, generated_latents.shape[0])):
            latents = {"generated": take_local(generated_latents, i)}
            for key, packed in (dmd_log_latents or {}).items():
                latents[key] = take_local(packed, i)
            caption = captions[min(i, len(captions) - 1)] if len(captions) else ""
            context = [ref.cpu() for ref in context_latents[min(i, len(context_latents) - 1)]] if context_latents else None
            gathered.append({"caption": caption, "hw": (height, width), "latents": latents, "context": context})
        device = generated_latents.device
        img_logs = {}
        for sample in gathered:
            h, w = sample["hw"]
            caption = self.log_caption_for(sample["caption"])
            panel = self.reference_panel(sample["context"], h, device)
            for key, packed in sample["latents"].items():
                img = decode_to_images(
                    self.vae, packed.to(device), h, w,
                    self.vae_scale_factor, self.vae_config, self.args.do_patching,
                )[0]
                if key == "generated" and panel is not None:
                    img = torch.cat([panel, img], dim=-1)  # images to edit | edited
                arr = (img * 255).clamp(0, 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
                img_logs.setdefault(f"dmd/{key}", []).append(
                    wandb.Image(arr, caption=caption, file_type="jpg")
                )
        return img_logs

    @torch.no_grad()
    def generate_validation_latents(self, caption: str, height: int, width: int, device) -> torch.Tensor:
        """Full DMD N-step student rollout for one caption from a fixed-seed noise tensor.

        Implements the base-trainer validation hook. Returns packed x0 latents
        ``(1, seq_len, C)``; mirrors the ``training_step`` rollout
        but runs every step under no_grad — no grad step.
        """
        caption, images, context = self.validation_inputs(caption, height, width)
        eh, te_layers, prompt_attn_mask, _ = self.run_text_encoder([caption], images)
        eh = eh.to(device=device, dtype=torch.float32)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=torch.float32)

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        # Same fixed seed for every prompt and every pass -> comparable samples across steps.
        gen = torch.Generator(device=device).manual_seed(self.args.validation_seed)
        noise = torch.randn(1, self.vae_config["latent_channels"], latent_h, latent_w, device=device, dtype=torch.float32, generator=gen)
        latents, img_ids = pack_t2i(noise, height, width, self.vae_scale_factor, self.args.do_patching, device)
        latents_seq_len = latents.shape[1]

        attn_mask = build_attention_mask(prompt_attn_mask, latents_seq_len, self.weight_dtype, self.args.attn_backend)
        joint_attn_kwargs = {"attention_mask": attn_mask}
        if context is not None:
            joint_attn_kwargs["context_latents"] = context

        gen_kwargs = dict(
            encoder_hidden_states=eh,
            text_encoder_layers=te_layers,
            img_ids=img_ids,
            attn_kwargs=joint_attn_kwargs,
            latents_seq_len=latents_seq_len,
        )
        infer_timesteps = build_infer_timesteps(self.infer_scheduler, self.args.dmd_num_inference_steps, latents_seq_len, device, mu=self.args.dmd_mu)
        for t in infer_timesteps[:-1]:
            pred = predict_flow_step(self.transformer, latents, t, **gen_kwargs)
            latents = self.infer_scheduler.step(pred, t, latents, return_dict=False)[0]
        t_final = infer_timesteps[-1]
        pred = predict_flow_step(self.transformer, latents, t_final, **gen_kwargs)
        sigma = (t_final.float() / 1000.0).to(device=device, dtype=latents.dtype)
        latents = latents - sigma * pred
        return latents
