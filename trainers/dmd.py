"""Decoupled DMD trainers."""

from __future__ import annotations

import os
import random

import torch
import wandb
from diffusers.training_utils import cast_training_params
from safetensors.torch import load_file

from args.base_args import add_base_args
from args.data_args import add_data_args
from args.dmd_args import add_decoupled_dmd_args
from data.batch import T2IBatch, get_caption_short_description
from data.build import build_dataloader_t2i
from lora.adapters import add_lora, add_lora_adapter
from train_utils.checkpoint import resolve_resume_checkpoint
from train_utils.packing import pack_t2i
from train_utils.runtime import CudaTimerContext
from losses.dmd import compute_decoupled_dmd_loss, compute_fake_score_loss, set_adapter
from models.text_encoder import create_attention_matrix, pad_embedding
from models.vae import build_vae, decode_to_images
from schedulers.flow_matching import build_infer_scheduler, build_infer_timesteps
from schedulers.inference import predict_flow_step
from trainers.base import BaseTrainer, logger


def maybe_resume_dmd_adapters(peft_model, args):
    """Load student and fake adapter weights from a DMD-format checkpoint, pre-FSDP."""
    ckpt_path = resolve_resume_checkpoint(args)
    if ckpt_path is None:
        return
    if not os.path.isdir(os.path.join(ckpt_path, "student")):
        return
    from peft.utils.save_and_load import set_peft_model_state_dict

    for adapter_name in ("student", "fake"):
        sd_path = os.path.join(ckpt_path, adapter_name, "adapter_model.safetensors")
        if not os.path.exists(sd_path):
            raise FileNotFoundError(f"DMD resume: missing {sd_path}")
        logger.info(f"Loading DMD adapter '{adapter_name}' from {sd_path}")
        state_dict = load_file(sd_path)
        load_result = set_peft_model_state_dict(peft_model, state_dict, adapter_name=adapter_name)
        if load_result.unexpected_keys:
            logger.info(f"  unexpected keys: {load_result.unexpected_keys[:5]}...")


class DecoupledDMDT2ITrainer(BaseTrainer):
    PARSER_MIXINS = [add_base_args, add_data_args, add_decoupled_dmd_args]
    BATCH_CLS = T2IBatch
    DATALOADER_BUILDER = staticmethod(build_dataloader_t2i)
    WANDB_PROJECT = "dmd"

    def __init__(self):
        super().__init__()
        self.infer_scheduler = None

    def extra_setup(self):
        self.infer_scheduler = build_infer_scheduler()
        if self.args.dmd_log_image_every > 0:
            self.vae = build_vae(self.args, self.accelerator, self.weight_dtype)

    def configure_transformer(self):
        """Attach student + fake LoRA adapters; resume their weights pre-FSDP if available."""
        args = self.args
        assert args.lora_rank > 0, "DMD trainer requires lora_rank > 0"
        logger.info(f"Using DMD LoRA adapters: student rank {args.lora_rank}, fake rank {args.lora_rank}")
        self.transformer.requires_grad_(False)
        self.transformer.to(dtype=self.weight_dtype)
        self.transformer = add_lora(self.transformer, args.lora_rank, adapter_name="student")
        self.transformer = add_lora_adapter(self.transformer, args.lora_rank, adapter_name="fake")
        maybe_resume_dmd_adapters(self.transformer, args)
        self.transformer.set_adapter("student")
        for name, param in self.transformer.named_parameters():
            param.requires_grad_(("student" in name or "fake" in name) and "lora_" in name)
        cast_training_params([self.transformer], dtype=self.weight_dtype if args.use_fsdp else torch.float32)
        self.transformer.print_trainable_parameters()

    def training_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        height, width = self.args.resolution
        device = self.accelerator.device

        eh, te_layers, prompt_attn_mask, text_encoder_time = self.run_text_encoder(batch.captions)
        pixel_values = batch.pixel_values.to(device=device, dtype=torch.float32)
        eh = eh.to(device=device, dtype=torch.float32)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=torch.float32)

        latents = (pixel_values - self.vae_config["shift_factor"]) * self.vae_config["scaling_factor"]

        bsz = pixel_values.shape[0]
        num_text_tokens = eh.shape[1]

        patched_latents, patched_image_ids = pack_t2i(
            latents,
            height,
            width,
            self.vae_scale_factor,
            self.args.do_patching,
            device,
        )
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

        num_inference_steps = self.args.dmd_num_inference_steps
        infer_timesteps = build_infer_timesteps(
            self.infer_scheduler,
            num_inference_steps,
            latents_seq_len,
            device,
        )

        generated_latents = torch.randn(
            patched_latents.shape,
            device=device,
            dtype=patched_latents.dtype,
            generator=self.generator,
        )

        current_null_embeds, _ = pad_embedding(self.null_conditioning, max_tokens=num_text_tokens)
        current_null_layers = [pad_embedding(layer, max_tokens=num_text_tokens)[0] for layer in self.null_conditioning_layers]

        do_gen_update = self.global_step % self.args.dfake_gen_update_ratio == 0

        gen_kwargs = dict(
            encoder_hidden_states=eh,
            text_encoder_layers=te_layers,
            img_ids=patched_image_ids,
            attn_kwargs=joint_attn_kwargs,
            latents_seq_len=latents_seq_len,
        )

        set_adapter(self.transformer, "student")
        generate_time = []
        # DMD2 backward simulation: roll out under no_grad up to a randomly sampled step,
        # then run one student forward with grad at that step so the DMD loss backprops
        # via x0_pred = x_t - sigma * v. On non-gen-update steps, roll the full trajectory.
        with CudaTimerContext(generate_time):
            if do_gen_update:
                grad_step_idx = random.randint(0, len(infer_timesteps) - 1)
            else:
                grad_step_idx = len(infer_timesteps)
            with torch.no_grad():
                for t in infer_timesteps[:grad_step_idx]:
                    pred = predict_flow_step(
                        self.transformer,
                        generated_latents,
                        t,
                        **gen_kwargs,
                    )
                    generated_latents = self.infer_scheduler.step(
                        pred,
                        t,
                        generated_latents,
                        return_dict=False,
                    )[0]

            if do_gen_update:
                t_grad = infer_timesteps[grad_step_idx]
                pred = predict_flow_step(
                    self.transformer,
                    generated_latents,
                    t_grad,
                    **gen_kwargs,
                )
                sigma = (t_grad.float() / 1000.0).to(device=device, dtype=generated_latents.dtype)
                generated_latents = generated_latents - sigma * pred
                generator_timesteps = t_grad.long().expand(generated_latents.shape[0]).to(device=device)
            else:
                generator_timesteps = None

        named_losses = {}
        dmd_loss = torch.tensor(0.0, device=device)
        if do_gen_update:
            dmd_outputs = compute_decoupled_dmd_loss(
                generated_latents=generated_latents,
                generator_timesteps=generator_timesteps,
                real_transformer=self.transformer,
                fake_transformer=self.transformer,
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
            named_losses.update(dmd_outputs.get("named_losses") or {})

        fake_outputs = compute_fake_score_loss(
            transformer=self.transformer,
            generated_latents=generated_latents,
            cond_hidden_states=eh,
            cond_layers=te_layers,
            img_ids=patched_image_ids,
            attn_kwargs=joint_attn_kwargs,
            model_dtype=self.weight_dtype,
            args=self.args,
        )
        fake_loss = fake_outputs["loss"]
        named_losses.update(fake_outputs.get("named_losses") or {})

        loss = dmd_loss + fake_loss

        wandb_extras = {}
        log_every = self.args.dmd_log_image_every
        if (
            log_every > 0
            and do_gen_update
            and self.accelerator.is_main_process
            and self.vae is not None
            and (self.global_step + 1) % log_every == 0
        ):
            wandb_extras = self._build_image_logs(
                generated_latents=generated_latents,
                dm_update=dmd_outputs["log_latents"]["dm_update"],
                captions=batch.captions,
                height=height,
                width=width,
            )

        return {
            "loss": loss,
            "named_losses": named_losses,
            "timings": {
                "text_encoder_time": text_encoder_time,
                "generate_time": generate_time[0],
                "create_mask_time": mask_time[0],
                "fetch_batch": fetch_time_sec,
            },
            "log_extras": {
                "gen_update": int(do_gen_update),
                "grad_step_idx": grad_step_idx if do_gen_update else -1,
            },
            "wandb_extras": wandb_extras,
        }

    @torch.no_grad()
    def _build_image_logs(
        self,
        *,
        generated_latents: torch.Tensor,
        dm_update: torch.Tensor,
        captions,
        height: int,
        width: int,
    ) -> dict:
        """Rank-0-only VAE decode for wandb. VAE is local (not FSDP), so no collectives
        and no sync risk from other ranks skipping this path."""
        n = min(4, generated_latents.shape[0])
        vae_dtype = next(self.vae.parameters()).dtype

        def _decode(packed):
            return decode_to_images(
                self.vae,
                packed[:n].to(dtype=vae_dtype),
                height,
                width,
                self.vae_scale_factor,
                self.vae_config,
                self.args.do_patching,
            )

        gen_img = _decode(generated_latents)
        update_img = _decode(dm_update)
        # Center the DM-update visualization at 0.5 for viewability.
        update_img = (update_img * 0.5 + 0.5).clamp(0, 1)

        def _to_wandb_list(t):
            arr = (t * 255).clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            return [
                wandb.Image(
                    arr[i],
                    caption=get_caption_short_description(captions[min(i, len(captions) - 1)]),
                )
                for i in range(arr.shape[0])
            ]

        return {
            "dmd/generated": _to_wandb_list(gen_img),
            "dmd/dm_update": _to_wandb_list(update_img),
        }
