"""Decoupled DMD trainers."""

from __future__ import annotations

import os

import torch
import wandb
from accelerate.utils import broadcast
from diffusers.training_utils import cast_training_params
from safetensors.torch import load_file

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
from lora.ema import LoraEMA
from train_utils.checkpoint import resolve_resume_checkpoint
from train_utils.lr_scheduler import build_lr_scheduler
from train_utils.packing import pack_t2i
from train_utils.runtime import CudaTimerContext
from losses.dmd import compute_decoupled_dmd_loss, compute_fake_score_loss
from models.text_encoder import create_attention_matrix, pad_embedding
from models.transformer import Fibo2Transformer2DModel, build_transformer_config
from models.vae import build_vae, decode_to_images
from schedulers.flow_matching import build_infer_scheduler, build_infer_timesteps
from schedulers.inference import predict_flow_step
from trainers.base import BaseTrainer, logger


def maybe_resume_dmd_adapters(transformer, adapter_names, args):
    """Load student + fake adapter weights from a PEFT-format checkpoint
    directory (one safetensors per adapter).

    Resume path takes precedence (continues an interrupted run with full state).
    Falls back to ``--lora_init`` for cold-init from a saved DMD ckpt
    when starting a fresh run.
    """
    ckpt_path = resolve_resume_checkpoint(args)
    if ckpt_path is None:
        ckpt_path = args.lora_init
        if ckpt_path:
            logger.info(f"Cold-initializing DMD adapters from lora_init={ckpt_path}")
    if not ckpt_path or not os.path.isdir(os.path.join(ckpt_path, adapter_names[0])):
        return
    for adapter_name in adapter_names:
        logger.info(f"Loading DMD adapter '{adapter_name}' from {ckpt_path}")
        load_peft_adapter(transformer, ckpt_path, adapter_name)


def _load_full_ft_weights(ckpt_dir: str) -> dict:
    """Load a diffusers ``save_pretrained``-style weights dir into a state_dict.

    Tries diffusers' default safetensors name first, falls back to other common
    names that ``load_transformer_init_weights`` also accepts.
    """
    for fname in ("diffusion_pytorch_model.safetensors", "model.safetensors", "pytorch_model.bin"):
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            return load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu", mmap=True)
    raise FileNotFoundError(f"No transformer weights found in {ckpt_dir} (looked for diffusion_pytorch_model.safetensors / model.safetensors / pytorch_model.bin)")


def maybe_resume_dmd_full_ft(student_transformer, fake_transformer, args):
    """Load student + fake full-FT weights from a checkpoint dir laid out as
    ``ckpt/student/`` and ``ckpt/fake/`` (the layout written by
    ``DecoupledDMDT2ITrainer._save_full_ft_checkpoint``).

    Returns True if weights were loaded, False otherwise. Falls back to
    ``--lora_init`` (reused as the generic DMD cold-init path — layout
    is disambiguated by ``args.lora_rank``).
    """
    ckpt_path = resolve_resume_checkpoint(args)
    if ckpt_path is None:
        # Reuse lora_init as the generic "cold-init from a saved DMD ckpt" path —
        # the trainer's mode (lora_rank) already disambiguates the layout.
        ckpt_path = args.lora_init
        if ckpt_path:
            logger.info(f"Cold-initializing DMD full-FT student+fake from lora_init={ckpt_path}")
    if not ckpt_path or not os.path.isdir(os.path.join(ckpt_path, "student")) or not os.path.isdir(os.path.join(ckpt_path, "fake")):
        return False
    logger.info(f"Loading DMD full-FT student+fake weights from {ckpt_path}")
    student_sd = _load_full_ft_weights(os.path.join(ckpt_path, "student"))
    student_transformer.load_state_dict(student_sd, assign=True)
    del student_sd
    fake_sd = _load_full_ft_weights(os.path.join(ckpt_path, "fake"))
    fake_transformer.load_state_dict(fake_sd, assign=True)
    del fake_sd
    return True


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

    def extra_setup(self):
        self.infer_scheduler = build_infer_scheduler()
        if self.args.dmd_log_image_every > 0:
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
        maybe_resume_dmd_adapters(self.transformer, self.ADAPTER_NAMES, args)
        for name, param in self.transformer.named_parameters():
            param.requires_grad_("lora_" in name)
        cast_training_params([self.transformer], dtype=torch.float32)
        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%)")

    def _configure_full_ft(self):
        """Build fake (trainable) and real (frozen) clones of the student, then FSDP-prepare
        all three. Under FSDP, only rank 0 holds materialised weights; other ranks stay on
        meta and receive weights via ``sync_module_states=True`` during ``prepare``.

        Real teacher is always the cold pretrained weights from TRANSFORMER_INIT (frozen,
        never updated). Student+fake start from those same weights at cold init, and get
        overridden by the resume helper if a saved DMD full-FT checkpoint is available.
        """
        args = self.args
        logger.info("Using DMD full fine-tune mode (no LoRA): cloning student into fake + real transformers")
        self.transformer.requires_grad_(True)
        assert self.transformer.dtype == torch.float32

        transformer_config = build_transformer_config(args, self.vae_config, logger)
        with torch.device("meta"):
            self.fake_transformer = Fibo2Transformer2DModel.from_config(transformer_config)
            self.real_transformer = Fibo2Transformer2DModel.from_config(transformer_config)

        # Copy student weights into the meta-built clones. Under FSDP this only runs on
        # rank 0 (other ranks materialise empty shells in `prepare` via `param_init_fn`
        # and pull the rank-0 weights through `sync_module_states`).
        is_rank0 = self.accelerator.is_main_process
        if (args.use_fsdp and is_rank0) or (not args.use_fsdp):
            src_sd = self.transformer.state_dict()
            self.real_transformer.load_state_dict({k: v.detach().clone() for k, v in src_sd.items()}, assign=True)
            # Cold-init fake from pretrained student. If a resume checkpoint exists, the
            # next call overwrites both student and fake with the saved weights.
            self.fake_transformer.load_state_dict({k: v.detach().clone() for k, v in src_sd.items()}, assign=True)
            del src_sd
            maybe_resume_dmd_full_ft(self.transformer, self.fake_transformer, args)

        self.fake_transformer.requires_grad_(True)
        self.real_transformer.requires_grad_(False)

        if args.gradient_checkpointing:
            self.fake_transformer.enable_gradient_checkpointing()

        self.fake_transformer_unwrapped = self.fake_transformer
        self.real_transformer_unwrapped = self.real_transformer

        if args.use_fsdp:
            # Prepare fake first (trainable, fp32) so it shards like student; real next
            # (frozen, weight_dtype). Each call wraps independently — the student's own
            # prepare still runs in base.setup after this returns.
            self.fake_transformer = self.accelerator.prepare(self.fake_transformer)
            self.real_transformer = self.accelerator.prepare(self.real_transformer)

        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"Full-FT student trainable params: {n_trainable:,} / {n_total:,}")

    def post_optimizer_setup(self):
        if self.args.lora_rank > 0 and self.args.ema_decay > 0:
            self.ema_lora = LoraEMA(self.transformer, decay=self.args.ema_decay)
            logger.info(
                f"LoraEMA enabled with decay={self.args.ema_decay}, tracking {len(self.ema_lora.named_live)} LoRA params",
            )
        elif self.args.lora_rank == 0 and self.args.ema_decay > 0:
            logger.info("EMA disabled under full-FT mode (LoraEMA only walks LoRA params)")

        # In full-FT mode the student is compiled by base.setup; compile fake/real here.
        if self.args.lora_rank == 0 and (self.args.full_torch_compile or self.args.regional_compile):
            compile_transformer(self.fake_transformer, self.fake_transformer_unwrapped, regional=self.args.regional_compile)
            compile_transformer(self.real_transformer, self.real_transformer_unwrapped, regional=self.args.regional_compile)

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
            lr=self.args.learning_rate,
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
        if self.args.lora_rank > 0:
            joint_attn_kwargs["lora_scales"] = make_scales("student", self.transformer, device=device)

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

        do_gen_update = (self.global_step + 1) % self.args.dfake_gen_update_ratio == 0

        gen_kwargs = dict(
            encoder_hidden_states=eh,
            text_encoder_layers=te_layers,
            img_ids=patched_image_ids,
            attn_kwargs=joint_attn_kwargs,
            latents_seq_len=latents_seq_len,
        )

        generate_time = []
        # DMD2 backward simulation: roll out under no_grad up to a randomly sampled step,
        # then run one student forward with grad at that step so the DMD loss backprops
        # via x0_pred = x_t - sigma * v. On non-gen-update steps, roll the full trajectory.
        with CudaTimerContext(generate_time):
            if do_gen_update:
                selected_step = torch.randint(low=0, high=len(infer_timesteps), size=(1,), device=device, dtype=torch.long)
                selected_step = broadcast(selected_step, from_process=0)
                grad_step_idx = int(selected_step.item())
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

        return {
            "image": [height, width, device],
            "cond": [eh, te_layers, patched_image_ids, joint_attn_kwargs],
            "null": [current_null_embeds, current_null_layers],
            "rollout": [generated_latents, generator_timesteps, do_gen_update, grad_step_idx],
            "timings": [text_encoder_time, generate_time[0], mask_time[0], fetch_time_sec],
        }

    def optimization_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        setup = self.training_step(batch, fetch_time_sec=fetch_time_sec)
        height, width, device = setup["image"]
        eh, te_layers, patched_image_ids, joint_attn_kwargs = setup["cond"]
        current_null_embeds, current_null_layers = setup["null"]
        generated_latents, generator_timesteps, do_gen_update, grad_step_idx = setup["rollout"]
        text_encoder_time, generate_time, mask_time, fetch_time_sec = setup["timings"]
        student_optimizer, fake_optimizer = self.optimizer
        student_lr_scheduler, fake_lr_scheduler = self.lr_scheduler

        named_losses = {}
        dmd_outputs = None
        dmd_loss = torch.tensor(0.0, device=device)
        student_grad_norm = None

        # In LoRA mode, real == fake == self.transformer (different lora_scales select the
        # role). In full-FT, the three roles are three distinct modules.
        real_module = self.transformer if self.args.lora_rank > 0 else self.real_transformer
        fake_module = self.transformer if self.args.lora_rank > 0 else self.fake_transformer

        if do_gen_update:
            dmd_outputs = compute_decoupled_dmd_loss(
                generated_latents=generated_latents,
                generator_timesteps=generator_timesteps,
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
            self.accumulate_step_outputs({"loss": dmd_loss, "named_losses": dmd_named_losses})

            self.accelerator.backward(dmd_loss)
            fake_grad_stash = None
            if self.accelerator.sync_gradients:
                if self.args.lora_rank > 0:
                    # LoRA: student+fake share self.transformer; stash fake grads so the
                    # full-model clip only normalizes student grads, restored after step.
                    fake_grad_stash = self._stash_optimizer_grads(fake_optimizer)
                student_grad_norm = self._clip_grad_norm(self.transformer)
            student_optimizer.step()
            student_optimizer.zero_grad()
            if fake_grad_stash is not None:
                self._restore_optimizer_grads(fake_grad_stash)
            generated_latents = generated_latents.detach()

        student_lr_scheduler.step()

        fake_outputs = compute_fake_score_loss(
            transformer=fake_module,
            generated_latents=generated_latents,
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
        self.accumulate_step_outputs({"loss": fake_loss, "named_losses": fake_named_losses})

        self.accelerator.backward(fake_loss)
        fake_grad_norm = None
        if self.accelerator.sync_gradients:
            fake_grad_norm = self._clip_grad_norm(fake_module)
        fake_optimizer.step()
        fake_lr_scheduler.step()
        fake_optimizer.zero_grad()
        student_optimizer.zero_grad()

        if self.ema_lora is not None and self.accelerator.sync_gradients:
            self.ema_lora.update()

        wandb_extras = {}
        log_every = self.args.dmd_log_image_every
        if (
            log_every > 0
            and self.accelerator.is_main_process
            and self.vae is not None
            and (self.global_step + 1) % log_every == 0
        ):
            wandb_extras = self._build_image_logs(
                generated_latents=generated_latents.detach(),
                dmd_log_latents=dmd_outputs["log_latents"] if do_gen_update else None,
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

    def save_checkpoint(self):
        save_path = os.path.join(
            self.args.checkpoint_local_path,
            f"checkpoint_{self.global_step - 1:06d}",
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
                save_peft_adapter(
                    self.transformer_unwrapped,
                    save_path,
                    adapter_name,
                    state_dict=state_dict,
                )
            try:
                import shutil
                shutil.copyfile(
                    self.args.json_data_input_path,
                    os.path.join(save_path, "data_input.json"),
                )
            except (OSError, FileNotFoundError):
                pass
        del state_dict
        if self.ema_lora is not None:
            saved = self.ema_lora.swap_in()
            try:
                ema_state_dict = self.accelerator.get_state_dict(self.transformer, unwrap=False)
            finally:
                self.ema_lora.swap_out(saved)
            if self.accelerator.is_main_process:
                ema_dir = os.path.join(save_path, "ema")
                for adapter_name in self.ADAPTER_NAMES:
                    save_peft_adapter(
                        self.transformer_unwrapped,
                        ema_dir,
                        adapter_name,
                        state_dict=ema_state_dict,
                    )
            del ema_state_dict

    def _save_full_ft_checkpoint(self, save_path: str):
        """Save student and fake full-FT weights under student/ and fake/ subdirs.

        Uses diffusers' ``save_pretrained`` with a gathered state_dict so the artefact
        loads via ``from_pretrained`` for inference. Only rank 0 writes.
        """
        logger.info(f"Saving full-FT DMD checkpoint to {save_path}")
        student_sd = self.accelerator.get_state_dict(self.transformer, unwrap=False)
        fake_sd = self.accelerator.get_state_dict(self.fake_transformer, unwrap=False)
        if self.accelerator.is_main_process:
            self.transformer_unwrapped.save_pretrained(
                os.path.join(save_path, "student"),
                state_dict={k.replace("module.", "").replace("._orig_mod", ""): v for k, v in student_sd.items()},
                is_main_process=True,
            )
            self.fake_transformer_unwrapped.save_pretrained(
                os.path.join(save_path, "fake"),
                state_dict={k.replace("module.", "").replace("._orig_mod", ""): v for k, v in fake_sd.items()},
                is_main_process=True,
            )
            try:
                import shutil
                shutil.copyfile(
                    self.args.json_data_input_path,
                    os.path.join(save_path, "data_input.json"),
                )
            except (OSError, FileNotFoundError):
                pass
        del student_sd, fake_sd

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

        def _to_wandb_list(t):
            arr = (t * 255).clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            return [
                wandb.Image(
                    arr[i],
                    caption=get_caption_short_description(captions[min(i, len(captions) - 1)]),
                )
                for i in range(arr.shape[0])
            ]

        img_logs = {"dmd/generated": _to_wandb_list(_decode(generated_latents))}
        for key, latents in (dmd_log_latents or {}).items():
            img_logs[f"dmd/{key}"] = _to_wandb_list(_decode(latents))
        return img_logs
