"""Decoupled DMD trainers."""

from __future__ import annotations

import os

import torch
import shutil
import wandb
from accelerate.utils import broadcast, gather_object
from diffusers.training_utils import cast_training_params
from safetensors.torch import load_file, save_file

from args.base_args import add_base_args
from args.data_args import add_data_args
from args.dmd_args import add_decoupled_dmd_args
from data.batch import T2IBatch, get_caption_short_description
from data.build import build_dataloader_t2i
from distributed import compile_transformer
from lora.adapters import LORA_TARGET_MODULES
from lora.static_multi_lora import (
    inject_multi_lora,
    load_peft_adapter,
    make_scales,
    save_peft_adapter,
)
from train_utils.ema import FullFtEMA, LoraEMA
from train_utils.checkpoint import resolve_resume_checkpoint
from train_utils.lr_scheduler import build_lr_scheduler
from train_utils.packing import pack_t2i
from train_utils.runtime import CudaTimerContext
from losses.dmd import compute_decoupled_dmd_loss, compute_fake_score_loss
from models.text_encoder import create_attention_matrix
from models.transformer import Fibo2Transformer2DModel, build_transformer_config
from models.vae import build_vae, decode_to_images
from schedulers.flow_matching import build_infer_scheduler, build_infer_timesteps, resolve_mu
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
        self.ema_student = None

    def extra_setup(self):
        self.infer_scheduler = build_infer_scheduler()
        if self.args.dmd_log_image_every > 0 or self.args.validate_every > 0:
            self.vae = build_vae(self.args, self.accelerator, self.weight_dtype)

    ADAPTER_NAMES = ("student", "fake")

    def configure_transformer(self):
        if self.args.lora_rank == 0:
            self._configure_full_ft()
        else:
            self._configure_lora()

    def _configure_lora(self):
        args = self.args
        logger.info(f"Using DMD LoRA adapters {self.ADAPTER_NAMES}: rank={args.lora_rank} alpha={args.lora_alpha}")
        self.transformer.requires_grad_(False)
        self.transformer.to(dtype=torch.float32 if args.use_fsdp else self.weight_dtype)
        replaced = inject_multi_lora(
            self.transformer,
            target_modules=LORA_TARGET_MODULES,
            adapter_names=self.ADAPTER_NAMES,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
        )
        logger.info(f"Injected StaticMultiLoraLinear on {len(replaced)} modules")
        for adapter in self.ADAPTER_NAMES:
            p = dmd_init_path(args, adapter)
            if p and os.path.isdir(p):
                load_peft_adapter(self.transformer, p, adapter, in_subdir="")
                logger.info(f"Loaded LoRA '{adapter}' from {p}")
        for name, param in self.transformer.named_parameters():
            param.requires_grad_("lora_" in name)
        cast_training_params([self.transformer], dtype=torch.float32)
        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")

    def _configure_full_ft(self):
        """Build fake (trainable) and real (frozen) clones of the student, then FSDP-prepare
        all three. Under FSDP, only rank 0 holds materialised weights; other ranks stay on
        meta and receive weights via ``sync_module_states=True`` during ``prepare``.
        """
        args = self.args
        logger.info("Using DMD full fine-tune mode (no LoRA): cloning student into fake + real transformers")
        self.transformer.requires_grad_(True)
        assert self.transformer.dtype == torch.float32

        transformer_config = build_transformer_config(args, self.vae_config, logger)
        with torch.device("meta"):
            self.fake_transformer = Fibo2Transformer2DModel.from_config(transformer_config)
            self.real_transformer = Fibo2Transformer2DModel.from_config(transformer_config)

        if (args.use_fsdp and self.accelerator.is_main_process) or (not args.use_fsdp):
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

        if args.gradient_checkpointing:
            self.fake_transformer.enable_gradient_checkpointing()
            self.real_transformer.enable_gradient_checkpointing()

        self.fake_transformer_unwrapped = self.fake_transformer
        self.real_transformer_unwrapped = self.real_transformer

        if args.use_fsdp:
            self.fake_transformer = self.accelerator.prepare(self.fake_transformer)
            self.real_transformer = self.accelerator.prepare(self.real_transformer)

        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Full-FT student trainable params: {n_trainable:,} / {n_total:,}")

    def build_optimizer(self):
        if self.args.lora_rank == 0:
            student_params = list(self.transformer.parameters())
            fake_params = list(self.fake_transformer.parameters())
        else:
            student_params = []
            fake_params = []
            for name, param in self.transformer.named_parameters():
                if not param.requires_grad or "lora_" not in name:
                    continue
                if "student" in name:
                    student_params.append(param)
                elif "fake" in name:
                    fake_params.append(param)

        fake_lr = self.args.fake_lr if self.args.fake_lr is not None else self.args.learning_rate
        student_optimizer = torch.optim.AdamW(
            student_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
            fused=True,
        )
        fake_optimizer = torch.optim.AdamW(
            fake_params,
            lr=fake_lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
            fused=True,
        )
        self.optimizer = [student_optimizer, fake_optimizer]
        self.lr_scheduler = [
            build_lr_scheduler(self.args, student_optimizer, self.accelerator.num_processes),
            build_lr_scheduler(self.args, fake_optimizer, self.accelerator.num_processes),
        ]

    def post_optimizer_setup(self):
        if self.args.lora_rank > 0 and self.args.ema_decay > 0:
            self.ema_student = LoraEMA(self.transformer, decay=self.args.ema_decay, adapter_name="student")
            logger.info(f"LoraEMA (student-only) enabled with decay={self.args.ema_decay}, tracking {len(self.ema_student.named_live)} LoRA params")
        elif self.args.lora_rank == 0 and self.args.ema_decay > 0:
            self.ema_student = FullFtEMA(self.transformer, decay=self.args.ema_decay)
            logger.info(f"Full-FT student EMA enabled with decay={self.args.ema_decay}, tracking {self.ema_student.local_numel:,} local-shard params per rank")

        if self.args.lora_rank == 0 and (self.args.full_torch_compile or self.args.regional_compile):
            compile_transformer(self.fake_transformer, self.fake_transformer_unwrapped, regional=self.args.regional_compile)
            compile_transformer(self.real_transformer, self.real_transformer_unwrapped, regional=self.args.regional_compile)

    def _clip_grad_norm(self, model):
        # In full-FT we prepare three same-arch FSDP modules; accelerator.clip_grad_norm_
        # crashes inside its dispatcher (tensor list `==`), so call FSDP's sharded clip directly.
        if self.args.use_fsdp and self.args.lora_rank == 0:
            return model.clip_grad_norm_(self.args.max_grad_norm)
        return self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

    @staticmethod
    def _stash_optimizer_grads(optimizer):
        stashed = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                stashed.append((param, param.grad))
                param.grad = None
        return stashed

    @staticmethod
    def _restore_optimizer_grads(stashed):
        for param, grad in stashed:
            param.grad = grad

    def training_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        height, width = batch.resolution or self.args.resolution
        device = self.accelerator.device

        eh, te_layers, prompt_attn_mask, text_encoder_time = self.run_text_encoder(batch.captions)
        pixel_values = batch.pixel_values.to(device=device, dtype=torch.float32)
        eh = eh.to(device=device, dtype=torch.float32)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=torch.float32)

        latents = (pixel_values - self.vae_config["shift_factor"]) * self.vae_config["scaling_factor"]

        bsz = pixel_values.shape[0]
        num_text_tokens = eh.shape[1]

        patched_latents, patched_image_ids = pack_t2i(latents, height, width, self.vae_scale_factor, self.args.do_patching, device)
        latents_seq_len = patched_latents.shape[1]
        mask_time = [0.0]
        if bsz > 1:
            latent_attn = torch.ones([bsz, latents_seq_len], dtype=latents.dtype, device=device)
            attn_mask = torch.cat([prompt_attn_mask, latent_attn], dim=1)
            mask_time = []
            with CudaTimerContext(mask_time):
                attn_mask = create_attention_matrix(attn_mask)
            attn_mask = attn_mask.unsqueeze(dim=1)
        else:
            attn_mask = None
        joint_attn_kwargs = {"attention_mask": attn_mask}
        if self.args.lora_rank > 0:
            joint_attn_kwargs["lora_scales"] = make_scales("student", self.transformer, device=device)

        num_inference_steps = self.args.dmd_num_inference_steps
        infer_timesteps = build_infer_timesteps(self.infer_scheduler, num_inference_steps, latents_seq_len, device, mu=self.args.dmd_mu)

        generated_latents = torch.randn(patched_latents.shape, device=device, dtype=patched_latents.dtype, generator=self.generator)

        current_null_embeds, current_null_layers = self.null_conditioning_for(bsz, num_text_tokens)

        do_gen_update = (self.global_step + 1) % self.args.dfake_gen_update_ratio == 0

        gen_kwargs = dict(
            encoder_hidden_states=eh,
            text_encoder_layers=te_layers,
            img_ids=patched_image_ids,
            attn_kwargs=joint_attn_kwargs,
            latents_seq_len=latents_seq_len,
        )

        generate_time = []
        with CudaTimerContext(generate_time):
            selected_step = torch.randint(low=0, high=len(infer_timesteps), size=(1,), device=device, dtype=torch.long)
            selected_step = broadcast(selected_step, from_process=0)
            grad_step_idx = int(selected_step.item())

            with torch.no_grad():
                for t in infer_timesteps[:grad_step_idx]:
                    pred = predict_flow_step(self.transformer, generated_latents, t, **gen_kwargs)
                    generated_latents = self.infer_scheduler.step(pred, t, generated_latents, return_dict=False)[0]

            t_grad = infer_timesteps[grad_step_idx]
            with torch.set_grad_enabled(do_gen_update):
                pred = predict_flow_step(self.transformer, generated_latents, t_grad, **gen_kwargs)
            sigma = (t_grad.float() / 1000.0).to(device=device, dtype=generated_latents.dtype)
            generated_latents = generated_latents - sigma * pred
            dmd_latents = generated_latents

        return {
            "image": [height, width, device],
            "cond": [eh, te_layers, patched_image_ids, joint_attn_kwargs],
            "null": [current_null_embeds, current_null_layers],
            "rollout": [dmd_latents, do_gen_update, grad_step_idx],
            "real_latents": patched_latents,
            "timings": [text_encoder_time, generate_time[0], mask_time[0], fetch_time_sec],
        }

    def optimization_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        setup = self.training_step(batch, fetch_time_sec=fetch_time_sec)
        height, width, device = setup["image"]
        eh, te_layers, patched_image_ids, joint_attn_kwargs = setup["cond"]
        current_null_embeds, current_null_layers = setup["null"]
        dmd_latents, do_gen_update, grad_step_idx = setup["rollout"]
        text_encoder_time, generate_time, mask_time, fetch_time_sec = setup["timings"]
        student_optimizer, fake_optimizer = self.optimizer[0], self.optimizer[1]
        student_lr_scheduler, fake_lr_scheduler = self.lr_scheduler[0], self.lr_scheduler[1]

        named_losses = {}
        dmd_outputs = None
        dmd_loss = torch.tensor(0.0, device=device)
        student_grad_norm = None

        real_module = self.transformer if self.args.lora_rank > 0 else self.real_transformer
        fake_module = self.transformer if self.args.lora_rank > 0 else self.fake_transformer
        # 1) Student update — DMD loss (internal fake DM fwd + real teacher fwd).
        if do_gen_update:
            mu = resolve_mu(self.infer_scheduler, dmd_latents.shape[1], self.args.dmd_mu)
            dmd_outputs = compute_decoupled_dmd_loss(
                generated_latents=dmd_latents,
                grad_step_idx=grad_step_idx,
                mu=mu,
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
            )
            dmd_loss = dmd_outputs["loss"]
            dmd_named_losses = dmd_outputs.get("named_losses") or {}
            named_losses.update(dmd_named_losses)
            self.record_loss_components({"loss": dmd_loss, "named_losses": dmd_named_losses})

            student_loss = dmd_loss

            self.accelerator.backward(student_loss)
            fake_grad_stash = None
            if self.accelerator.sync_gradients:
                if self.args.lora_rank > 0:
                    fake_grad_stash = self._stash_optimizer_grads(fake_optimizer)
                student_grad_norm = self._clip_grad_norm(self.transformer)
            student_optimizer.step()
            student_optimizer.zero_grad()
            if self.accelerator.sync_gradients and self.ema_student is not None:
                self.ema_student.update()
            if fake_grad_stash is not None:
                self._restore_optimizer_grads(fake_grad_stash)
            dmd_latents = dmd_latents.detach()

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
            args=self.args,
        )
        fake_loss = fake_outputs["loss"]
        fake_named_losses = fake_outputs.get("named_losses") or {}
        named_losses.update(fake_named_losses)
        self.record_loss_components({"loss": fake_loss, "named_losses": fake_named_losses})

        guidance_loss = fake_loss

        self.accelerator.backward(guidance_loss)

        fake_grad_norm = None
        if self.accelerator.sync_gradients:
            fake_grad_norm = self._clip_grad_norm(fake_module)
        fake_optimizer.step()
        fake_lr_scheduler.step()
        fake_optimizer.zero_grad()
        student_optimizer.zero_grad()

        wandb_extras = {}
        log_every = self.args.dmd_log_image_every
        if log_every > 0 and self.vae is not None and (self.global_step + 1) % log_every == 0:
            wandb_extras = self._build_image_logs(
                generated_latents=dmd_latents.detach(),
                dmd_log_latents=dmd_outputs["log_latents"] if (do_gen_update and dmd_outputs is not None) else None,
                captions=batch.captions,
                height=height,
                width=width,
            )

        loss = dmd_loss.detach() + fake_loss.detach()
        log_extras = {
            "grad_step_idx": grad_step_idx,
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
        save_path = os.path.join(
            self.args.checkpoint_local_path,
            f"checkpoint_{self.global_step:06d}",
        )
        if getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
            self.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        self.accelerator.wait_for_everyone()
        if self.args.lora_rank == 0:
            self._save_full_ft_checkpoint(save_path)
        else:
            self._save_lora_checkpoint(save_path)
        self.accelerator.wait_for_everyone()
        logger.info(f"Saved state to {save_path}")

    def _save_lora_checkpoint(self, save_path: str):
        logger.info(f"Saving static DMD LoRA checkpoint to {save_path}")
        state_dict = self.accelerator.get_state_dict(self.transformer, unwrap=False)
        if self.accelerator.is_main_process:
            for adapter_name in self.ADAPTER_NAMES:
                save_peft_adapter(self.transformer_unwrapped, save_path, adapter_name, state_dict=state_dict)
            try:
                shutil.copyfile(self.args.json_data_input_path, os.path.join(save_path, "data_input.json"))
            except (OSError, FileNotFoundError):
                pass
        del state_dict
        if self.ema_student is not None:
            self.ema_student.save(save_path, self.accelerator, self.transformer, self.transformer_unwrapped)

    def _save_full_ft_checkpoint(self, save_path: str):
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
            try:
                shutil.copyfile(self.args.json_data_input_path, os.path.join(save_path, "data_input.json"))
            except (OSError, FileNotFoundError):
                pass
        del student_sd, fake_sd

        if self.ema_student is not None:
            self.ema_student.save(save_path, self.accelerator, self.transformer, self.transformer_unwrapped)

    @torch.no_grad()
    def _build_image_logs(
        self,
        *,
        generated_latents: torch.Tensor,
        dmd_log_latents: dict | None,
        captions,
        height: int,
        width: int,
    ) -> dict:
        """Gather packed latents + per-rank (H, W) and decode on rank 0.

        Dynamic-resolution buckets let ranks pick different aspect ratios at the same step;
        seq_len is identical across ranks (token count is the bucket key), but the unpacked
        (H, W) differs. We gather the packed latents (same shape) and per-sample (H, W),
        then rank 0 unpacks each with its own resolution. ALL ranks must reach this call.
        """
        world = self.accelerator.num_processes
        is_main = self.accelerator.is_main_process
        target = min(8, generated_latents.shape[0] * world)
        per_rank = (target + world - 1) // world
        vae_dtype = next(self.vae.parameters()).dtype

        def take_local(packed):
            local = packed[:per_rank]
            if local.shape[0] < per_rank:
                pad = local[-1:].expand(per_rank - local.shape[0], *local.shape[1:])
                local = torch.cat([local, pad], dim=0)
            return local.to(dtype=vae_dtype)

        def gather_packed(packed):
            return self.accelerator.gather(take_local(packed))[:target]

        local_caps = list(captions[:per_rank])
        while len(local_caps) < per_rank:
            local_caps.append(local_caps[-1] if local_caps else "")
        gathered_caps = gather_object(local_caps)[:target]
        gathered_hw = gather_object([(height, width)] * per_rank)[:target]

        def decode_per_sample(packed_all):
            if not is_main:
                return None
            imgs = []
            for i in range(packed_all.shape[0]):
                h_i, w_i = gathered_hw[i]
                img = decode_to_images(
                    self.vae, packed_all[i : i + 1], h_i, w_i,
                    self.vae_scale_factor, self.vae_config, self.args.do_patching,
                )
                imgs.append(img[0])
            return imgs

        def to_wandb_list(imgs):
            out = []
            for i, t in enumerate(imgs):
                arr = (t * 255).clamp(0, 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
                out.append(wandb.Image(arr, caption=get_caption_short_description(gathered_caps[i]), file_type="jpg"))
            return out

        img_logs = {}
        gen_imgs = decode_per_sample(gather_packed(generated_latents))
        if is_main:
            img_logs["dmd/generated"] = to_wandb_list(gen_imgs)
        for key, latents in (dmd_log_latents or {}).items():
            dec = decode_per_sample(gather_packed(latents))
            if is_main:
                img_logs[f"dmd/{key}"] = to_wandb_list(dec)
        return img_logs

    @torch.no_grad()
    def generate_validation_latents(self, caption: str, height: int, width: int, device) -> torch.Tensor:
        """Full DMD N-step student rollout for one caption from a fixed-seed noise tensor.

        Implements the base-trainer validation hook. Returns packed x0 latents
        ``(1, seq_len, C)``; mirrors the ``training_step`` rollout
        but runs every step under no_grad — no grad step.
        """
        eh, te_layers, prompt_attn_mask, _ = self.run_text_encoder([caption])
        eh = eh.to(device=device, dtype=torch.float32)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=torch.float32)

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        # Same fixed seed for every prompt and every pass -> comparable samples across steps.
        gen = torch.Generator(device=device).manual_seed(self.args.validation_seed)
        noise = torch.randn(1, self.vae_config["latent_channels"], latent_h, latent_w, device=device, dtype=torch.float32, generator=gen)
        latents, img_ids = pack_t2i(noise, height, width, self.vae_scale_factor, self.args.do_patching, device)
        latents_seq_len = latents.shape[1]

        latent_attn = torch.ones([1, latents_seq_len], dtype=latents.dtype, device=device)
        attn_mask = torch.cat([prompt_attn_mask, latent_attn], dim=1)
        attn_mask = create_attention_matrix(attn_mask).unsqueeze(dim=1).to(self.weight_dtype)
        joint_attn_kwargs = {"attention_mask": attn_mask}
        if self.args.lora_rank > 0:
            joint_attn_kwargs["lora_scales"] = make_scales("student", self.transformer, device=device)

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
