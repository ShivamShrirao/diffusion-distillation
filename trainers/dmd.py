"""Decoupled DMD trainers."""
from __future__ import annotations

import random

import torch

from args.base_args import add_base_args
from args.data_args import add_data_args
from args.dmd_args import add_decoupled_dmd_args
from data.batch import T2IBatch
from data.build import build_dataloader_t2i
from train_utils.packing import apply_cfg_dropout_t2i, pack_t2i, unpack_to_image
from train_utils.runtime import CudaTimerContext
from losses.denoising import compute_denoising_loss
from losses.dmd import compute_decoupled_dmd_loss, compute_fake_score_loss, set_adapter
from models.text_encoder import create_attention_matrix, pad_embedding
from schedulers.flow_matching import build_infer_scheduler, build_infer_timesteps, parse_num_inference_steps_arg, pick_inference_steps
from schedulers.inference import predict_flow_step
from trainers.base import BaseTrainer


class DecoupledDMDT2ITrainer(BaseTrainer):
    PARSER_MIXINS = [add_base_args, add_data_args, add_decoupled_dmd_args]
    BATCH_CLS = T2IBatch
    DATALOADER_BUILDER = staticmethod(build_dataloader_t2i)
    WANDB_PROJECT = "dmd"

    def __init__(self):
        super().__init__()
        self.infer_scheduler = None
        self.num_infer_min = None
        self.num_infer_max = None

    def extra_setup(self):
        self.infer_scheduler = build_infer_scheduler()
        self.num_infer_min, self.num_infer_max = parse_num_inference_steps_arg(
            self.args.dmd_num_inference_steps,
        )

    def training_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        height, width = self.args.resolution
        device = self.accelerator.device

        eh, te_layers, prompt_attn_mask, text_encoder_time = self.run_text_encoder(batch.captions)
        pixel_values = batch.pixel_values.to(device=device, dtype=torch.float32)
        eh = eh.to(device=device, dtype=torch.float32)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=torch.float32)

        latents = (pixel_values - self.vae_config["shift_factor"]) * self.vae_config["scaling_factor"]
        noise = torch.randn_like(latents)

        bsz = pixel_values.shape[0]
        sigmas = torch.rand(bsz, device=device)
        timesteps = sigmas * 1000
        while len(sigmas.shape) < len(noise.shape):
            sigmas = sigmas.unsqueeze(-1)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        target = noise - latents

        num_text_tokens = eh.shape[1]
        eh, te_layers, prompt_attn_mask = apply_cfg_dropout_t2i(
            eh, te_layers, prompt_attn_mask,
            self.null_conditioning, self.null_conditioning_layers,
            num_text_tokens, self.args.drop_rate_cfg, self.generator,
        )

        patched_noisy, patched_image_ids = pack_t2i(
            noisy_latents, height, width, self.vae_scale_factor, self.args.do_patching, device,
        )
        latent_attn = torch.ones(
            [patched_noisy.shape[0], patched_noisy.shape[1]], dtype=latents.dtype, device=device,
        )
        attn_mask = torch.cat([prompt_attn_mask, latent_attn], dim=1)
        mask_time = []
        with CudaTimerContext(mask_time):
            attn_mask = create_attention_matrix(attn_mask)
        attn_mask = attn_mask.unsqueeze(dim=1)
        joint_attn_kwargs = {"attention_mask": attn_mask}
        latents_seq_len = patched_noisy.shape[1]

        current_num_inference_steps = pick_inference_steps(
            self.global_step, self.args.seed, self.num_infer_min, self.num_infer_max,
        )
        infer_timesteps = build_infer_timesteps(
            self.infer_scheduler, current_num_inference_steps, latents_seq_len, device,
        )

        init_noise = torch.randn(
            patched_noisy.shape[0], latents_seq_len, patched_noisy.shape[2],
            device=device, dtype=patched_noisy.dtype, generator=self.generator,
        )
        generated_latents = init_noise.clone()

        current_null_embeds, _ = pad_embedding(self.null_conditioning, max_tokens=num_text_tokens)
        current_null_layers = [
            pad_embedding(layer, max_tokens=num_text_tokens)[0]
            for layer in self.null_conditioning_layers
        ]
        use_generation_cfg = self.args.dmd_generation_guidance_scale != 1.0
        if use_generation_cfg:
            generation_enc_states = torch.cat([current_null_embeds, eh], dim=0)
            generation_enc_layers = [
                torch.cat([nl, cl], dim=0) for nl, cl in zip(current_null_layers, te_layers)
            ]
            generation_attn_mask = torch.cat([attn_mask, attn_mask], dim=0)
            generation_attn_kwargs = {"attention_mask": generation_attn_mask if bsz > 1 else None}
        else:
            generation_enc_states = eh
            generation_enc_layers = te_layers
            generation_attn_kwargs = joint_attn_kwargs

        do_gen_update = (self.global_step % self.args.dfake_gen_update_ratio == 0)

        set_adapter(self.transformer, "student")
        generate_time = []
        # DMD2 backward simulation: generate the trajectory under no_grad, then if this is
        # a generator-update step, keep grad through a single student forward at a randomly
        # sampled timestep so the DMD loss backprops via x0_pred = x_t - sigma * v.
        with CudaTimerContext(generate_time):
            if do_gen_update:
                grad_step_idx = random.randint(0, len(infer_timesteps) - 1)
            else:
                grad_step_idx = len(infer_timesteps)  # all steps in no_grad
            with torch.no_grad():
                for t in infer_timesteps[:grad_step_idx]:
                    pred = predict_flow_step(
                        self.transformer,
                        generated_latents,
                        t,
                        encoder_hidden_states=generation_enc_states,
                        text_encoder_layers=generation_enc_layers,
                        img_ids=patched_image_ids,
                        attn_kwargs=generation_attn_kwargs,
                        use_cfg=use_generation_cfg,
                        guidance_scale=self.args.dmd_generation_guidance_scale,
                        latents_seq_len=latents_seq_len,
                    )
                    generated_latents = self.infer_scheduler.step(
                        pred, t, generated_latents, return_dict=False,
                    )[0]
                # Finish the trajectory under no_grad too when this is not a gen step.
                for t in infer_timesteps[grad_step_idx:]:
                    pred = predict_flow_step(
                        self.transformer,
                        generated_latents,
                        t,
                        encoder_hidden_states=generation_enc_states,
                        text_encoder_layers=generation_enc_layers,
                        img_ids=patched_image_ids,
                        attn_kwargs=generation_attn_kwargs,
                        use_cfg=use_generation_cfg,
                        guidance_scale=self.args.dmd_generation_guidance_scale,
                        latents_seq_len=latents_seq_len,
                    )
                    generated_latents = self.infer_scheduler.step(
                        pred, t, generated_latents, return_dict=False,
                    )[0]

            if do_gen_update:
                t_grad = infer_timesteps[grad_step_idx]
                pred = predict_flow_step(
                    self.transformer,
                    generated_latents,
                    t_grad,
                    encoder_hidden_states=generation_enc_states,
                    text_encoder_layers=generation_enc_layers,
                    img_ids=patched_image_ids,
                    attn_kwargs=generation_attn_kwargs,
                    use_cfg=use_generation_cfg,
                    guidance_scale=self.args.dmd_generation_guidance_scale,
                    latents_seq_len=latents_seq_len,
                )
                # x0 from v-prediction at the sampled student timestep
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
            args=self.args,
        )
        fake_loss = fake_outputs["loss"]
        named_losses.update(fake_outputs.get("named_losses") or {})

        forward_time = [0.0]
        denoising_loss = torch.tensor(0.0, device=device)
        if do_gen_update and self.args.denoising_loss_scale > 0:
            forward_time = []
            with CudaTimerContext(forward_time):
                model_pred, _ = self.transformer(
                    hidden_states=patched_noisy,
                    timestep=timesteps,
                    encoder_hidden_states=eh,
                    text_encoder_layers=te_layers,
                    img_ids=patched_image_ids,
                    return_dict=False,
                    joint_attention_kwargs=joint_attn_kwargs,
                )
                model_pred = unpack_to_image(
                    model_pred, height, width, self.vae_scale_factor, self.args.do_patching,
                )
                denoising_loss = compute_denoising_loss(
                    model_pred, target, self.loss_coeff, self.args.denoising_loss_scale,
                )

        loss = dmd_loss + fake_loss + denoising_loss
        named_losses["denoising"] = denoising_loss

        return {
            "loss": loss,
            "named_losses": named_losses,
            "timings": {
                "text_encoder_time": text_encoder_time,
                "generate_time": generate_time[0],
                "forward_time": forward_time[0],
                "create_mask_time": mask_time[0],
                "fetch_batch": fetch_time_sec,
            },
            "log_extras": {
                "num_inference_steps": current_num_inference_steps,
                "gen_update": int(do_gen_update),
                "grad_step_idx": grad_step_idx if do_gen_update else -1,
            },
        }
