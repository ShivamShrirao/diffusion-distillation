"""BaseTrainer — shared training plumbing.

Owns:
  - argparse + accelerator + dtype + seed
  - VAE / transformer / text-encoder construction (via build_*())
  - LoRA, optimizer, lr_scheduler, dataloader wiring + accelerator.prepare
  - torch.compile, wandb init, resume
  - the outer `for step in range(...)` loop with batch fetching + StopIteration recovery
  - named-loss accumulation across gradient_accumulation_steps + wandb emission
  - checkpoint saving (LoRA-only or full-state) + optional EMA-LoRA shadow save

Subclasses declare (as class attributes):
  - ``PARSER_MIXINS``        : list of ``add_*_args(parser)`` functions
  - ``BATCH_CLS``            : the ``@dataclass`` for the dataloader tuple
  - ``DATALOADER_BUILDER``   : staticmethod returning ``(get_dataloader, total_batch)``
  - ``WANDB_PROJECT``        : str

And override these hooks:
  - ``extra_setup()``        : load extra models (teachers, rewards).
  - ``post_optimizer_setup()``: instantiate ``self.ema_lora`` if applicable
  - ``training_step(batch, *, fetch_time_sec)`` : the explicit step body.
        Must return a dict:
            ``loss``           : scalar tensor used by .backward()
            ``named_losses``   : dict[str, tensor] (gathered + accumulated for wandb)
            ``timings``        : dict[str, float] (replaces previous step's, not accumulated)
            ``wandb_extras``   : dict[str, Any]    (e.g. wandb.Image lists; merged into log)
            ``log_extras``     : dict[str, Any]    (plain scalars to add to wandb log)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone

import argparse
import diffusers
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import DEFAULT_NEGATIVE_PROMPT
import torch
import wandb
from diffusers.training_utils import cast_training_params
from tqdm.auto import tqdm

from args.base_args import add_base_args, read_config_defaults
from args.data_args import add_data_args
from data.batch import T2IBatch
from data.build import build_dataloader_t2i
from distributed import build_accelerator, compile_transformer, set_nccl_env
from lora.adapters import add_lora
from train_utils.checkpoint import load_checkpoint
from train_utils.lr_scheduler import build_lr_scheduler
from train_utils.runtime import (
    CudaTimerContext,
    get_logger,
    init_wandb,
    resolve_dtype,
    set_seed_for_rank,
    setup_root_logger,
)
from losses.denoising import compute_denoising_loss
from models.text_encoder import build_text_encoder, create_attention_matrix
from models.transformer import build_transformer
from models.vae import build_vae_config, vae_scale_factor_from_config
from train_utils.packing import apply_cfg_dropout_t2i, pack_t2i, unpack_to_image


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
logger = get_logger(__name__)

# TF32 matmul on Ampere+. Set at import time so it's effective before the first
# transformer forward (Accelerator may construct things during prepare()).
torch.set_float32_matmul_precision("high")

# Per-device tensor-core peak TFLOPs at bf16/fp16, dense (no sparsity), from vendor specs.
GPU_BF16_PEAK_TFLOPS = {
    "H100": 989.0,
    "H200": 989.0,
    "A100": 312.0,
    "A800": 312.0,
    "B200": 2250.0,
    "B100": 1800.0,
    "L40S": 362.0,
    "L40": 181.0,
    "L4": 121.0,
    "RTX 4090": 165.2,
    "RTX 5090": 419.0,
}


def peak_flops_per_sec(dtype: torch.dtype) -> float | None:
    """Return per-device peak FLOPs/sec for the current CUDA device, or None if unknown."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for key, tflops in GPU_BF16_PEAK_TFLOPS.items():
        if key in name:
            peak = tflops * 1e12
            # fp32 (TF32) tensor cores deliver ~half the bf16/fp16 throughput on these GPUs.
            return peak if dtype in (torch.bfloat16, torch.float16) else peak / 2
    return None


class BaseTrainer:
    """Concrete T2I flow-matching denoising trainer.

    Out of the box this trains a transformer with the standard FM v-pred denoising loss
    on T2I latents (`T2IBatch`) — useful by itself for plain fine-tuning runs. Subclasses
    swap class attrs and/or override `training_step()` for other objectives.
    """

    PARSER_MIXINS: list = [add_base_args, add_data_args]
    BATCH_CLS = T2IBatch
    DATALOADER_BUILDER = staticmethod(build_dataloader_t2i)
    WANDB_PROJECT: str = "image-model"

    def __init__(self):
        # Populated by setup()
        self.args: argparse.Namespace | None = None
        self.accelerator = None
        self.weight_dtype = None
        self.vae = None
        self.vae_config: dict | None = None
        self.transformer = None
        self.transformer_unwrapped = None
        self.total_num_layers: int | None = None
        self.tokenizer = None
        self.text_encoder = None
        self.get_prompt_embedds = None
        self.parameters = None
        self.optimizer = None
        self.lr_scheduler = None
        self.get_dataloader = None
        self.total_batch_no_acc: int | None = None
        self.null_conditioning = None
        self.null_conditioning_layers = None
        self.vae_scale_factor: int | None = None
        self.ema_lora = None
        self.generator = None
        self.transformer_param_count: int | None = None
        self.transformer_inner_dim: int | None = None
        self.flops_per_step: float | None = None
        self.peak_flops_per_sec: float | None = None
        self.loss_coeff: float | None = None
        self.seed: int | None = None
        self.rank: int = 0
        self.global_step: int = 0
        # Accumulators reset at every sync_gradients step.
        self.train_loss = 0.0
        self.named_loss_acc: dict[str, float] = {}

    # ---------------------------------------------------------------
    # Override hooks
    # ---------------------------------------------------------------
    def parse_args(self):
        parser = argparse.ArgumentParser(description=self.__class__.__name__)
        for fn in self.PARSER_MIXINS:
            fn(parser)
        parser.set_defaults(**read_config_defaults())
        return parser.parse_args()

    def extra_setup(self):
        """Load reward / perceptual / ref-transformer models. Default: nothing."""

    def configure_transformer(self):
        """Apply LoRA / adapter setup and dtype-grad masking to ``self.transformer``.

        Runs after ``build_transformer`` returns the bare model (init weights) and before ``accelerator.prepare``. The default handles the standard
        single-LoRA and no-LoRA paths; DMD overrides
        this to wire its student/fake dual-adapter setup. Must mutate ``self.transformer``
        in place — may rewrap it as a ``PeftModel``.
        """
        args = self.args
        if args.lora_rank > 0:
            logger.info(f"Using LoRA with rank {args.lora_rank}")
            self.transformer.requires_grad_(False)
            self.transformer.to(dtype=self.weight_dtype)
            self.transformer = add_lora(self.transformer, args.lora_rank)
            cast_training_params([self.transformer], dtype=self.weight_dtype if args.use_fsdp else torch.float32)
            self.transformer.print_trainable_parameters()
        else:
            self.transformer.requires_grad_(True)
            assert self.transformer.dtype == torch.float32

    def post_optimizer_setup(self):
        """Runs after ``accelerator.prepare`` (so FSDP shards exist) and before
        ``torch.compile``. Use this to instantiate EMA-LoRA shadows or other
        live-parameter trackers."""

    def maybe_adjust_resume_step(self, global_step: int) -> int:
        return global_step + 1

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    def setup(self):
        self.args = self.parse_args()
        self.rank = int(os.environ.get("RANK", 0))
        self.seed = set_seed_for_rank(self.args.seed)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.weight_dtype = resolve_dtype(self.args.mixed_precision)

        self.accelerator = build_accelerator(self.args, self.weight_dtype)
        setup_root_logger()
        logger.info(self.accelerator.state, main_process_only=False)
        logger.info(f"TORCH_VERSION {torch.__version__}")
        logger.info(f"DIFFUSERS_VERSION {diffusers.__version__}")

        self.vae_config = build_vae_config()

        # Reward / perceptual / ref-transformer models, and the VAE itself for trainers
        # that decode pixels (DMD-with-image-logging). Lives BEFORE the main
        # transformer so a subclass can size its reward stack against the args first.
        self.extra_setup()

        self.transformer, self.total_num_layers = build_transformer(
            self.args,
            self.vae_config,
            self.accelerator,
            logger,
        )
        self.transformer_inner_dim = self.transformer.inner_dim
        self.configure_transformer()
        self.transformer_unwrapped = self.transformer
        self.transformer_param_count = sum(p.numel() for p in self.transformer.parameters())
        trainable_param_count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        logger.info(f"Transformer Parameters: {self.transformer_param_count / 10**9}B")
        logger.info(f"Trainable transformer Parameters: {trainable_param_count / 10**9}B")
        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        if self.args.use_fsdp:
            self.transformer = self.accelerator.prepare(self.transformer)
        logger.info(f"Using precision of {self.weight_dtype}")

        self.tokenizer, self.text_encoder, self.get_prompt_embedds = build_text_encoder(
            self.args,
            self.accelerator,
            self.weight_dtype,
        )

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.build_optimizer()
        self.get_dataloader, self.total_batch_no_acc = self.DATALOADER_BUILDER(
            self.args,
            self.vae_config,
            logger,
        )
        self.compute_mfu_constants()
        self.lr_scheduler = build_lr_scheduler(self.args, self.optimizer, self.accelerator.num_processes)

        if self.args.use_fsdp:
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            self.transformer, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.transformer,
                self.optimizer,
                self.lr_scheduler,
            )

        self.post_optimizer_setup()

        if self.args.use_torch_compile:
            compile_transformer(
                self.transformer,
                self.transformer_unwrapped,
                regional=self.args.regional_compile,
            )

        if self.accelerator.is_main_process:
            init_wandb(self.accelerator, self.args, project_name=self.WANDB_PROJECT)

        logger.info("***** Running training *****")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        self.global_step = int(os.environ.get("GLOBAL_STEP", 0))
        if self.args.resume_from_checkpoint != "no":
            self.global_step = load_checkpoint(self.accelerator, self.args)
            self.global_step = self.maybe_adjust_resume_step(self.global_step)

        logger.info(f"Using adam with lr: {self.args.learning_rate}, beta2: {self.args.adam_beta2}")

        # Common derived state — everything `training_step` needs.
        null_emb, null_layers, _ = self.get_prompt_embedds([DEFAULT_NEGATIVE_PROMPT])
        assert null_emb.shape[0] == 1
        self.null_conditioning = null_emb.repeat(self.args.train_batch_size, 1, 1).to(dtype=torch.float32)
        self.null_conditioning_layers = [layer.repeat(self.args.train_batch_size, 1, 1).to(dtype=torch.float32) for layer in null_layers]
        self.vae_scale_factor = vae_scale_factor_from_config(self.vae_config)
        self.generator = torch.Generator(device=self.accelerator.device).manual_seed(self.seed)
        assert self.total_batch_no_acc is not None
        self.loss_coeff = WORLD_SIZE / self.total_batch_no_acc
        self.transformer.train()

    def build_optimizer(self):
        if self.args.lora_rank > 0:
            self.parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        else:
            self.parameters = self.transformer.parameters()
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
            fused=True,
        )

    def training_step(self, batch: T2IBatch, *, fetch_time_sec: float) -> dict:
        """Simple T2I flow-matching denoising step.

        Subclasses override this entirely for other objectives.
        """
        height, width = self.args.resolution
        device = self.accelerator.device

        # ---- text encoder ----
        eh, te_layers, prompt_attn_mask, text_encoder_time = self.run_text_encoder(batch.captions)
        pixel_values = batch.pixel_values.to(device=device, dtype=torch.float32)
        eh = eh.to(device=device, dtype=torch.float32)
        prompt_attn_mask = prompt_attn_mask.to(device=device, dtype=torch.float32)

        # ---- noisy latents ----
        latents = (pixel_values - self.vae_config["shift_factor"]) * self.vae_config["scaling_factor"]
        noise = torch.randn_like(latents)
        bsz = pixel_values.shape[0]
        sigmas = torch.rand(bsz, device=device)
        timesteps = sigmas * 1000
        while len(sigmas.shape) < len(noise.shape):
            sigmas = sigmas.unsqueeze(-1)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

        num_text_tokens = eh.shape[1]

        # ---- CFG dropout (single drop rate over prompt) ----
        eh, te_layers, prompt_attn_mask = apply_cfg_dropout_t2i(
            eh,
            te_layers,
            prompt_attn_mask,
            self.null_conditioning,
            self.null_conditioning_layers,
            num_text_tokens,
            self.args.drop_rate_cfg,
            self.generator,
        )

        target = noise - latents

        # ---- pack + attention mask ----
        patched_noisy, patched_image_ids = pack_t2i(
            noisy_latents,
            height,
            width,
            self.vae_scale_factor,
            self.args.do_patching,
            device,
        )
        mask_time = [0.0]
        if bsz > 1:
            latent_attn = torch.ones(
                [patched_noisy.shape[0], patched_noisy.shape[1]],
                dtype=latents.dtype,
                device=device,
            )
            attn_mask = torch.cat([prompt_attn_mask, latent_attn], dim=1)
            mask_time = []
            with CudaTimerContext(mask_time):
                attn_mask = create_attention_matrix(attn_mask)
            attn_mask = attn_mask.unsqueeze(dim=1)
        else:
            attn_mask = None
        joint_attn_kwargs = {"attention_mask": attn_mask}

        # ---- forward + denoising loss ----
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
            model_pred = unpack_to_image(model_pred, height, width, self.vae_scale_factor, self.args.do_patching)
            denoising_loss = compute_denoising_loss(
                model_pred,
                target,
                self.loss_coeff,
                self.args.denoising_loss_scale,
            )

        return {
            "loss": denoising_loss,
            "named_losses": {"denoising": denoising_loss},
            "timings": {
                "text_encoder_time": text_encoder_time,
                "forward_time": forward_time[0],
                "create_mask_time": mask_time[0],
                "fetch_batch": fetch_time_sec,
            },
        }

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------
    def main_loop(self):
        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.update(self.global_step)
        progress_bar.set_description("Steps")

        now = datetime.now(timezone.utc)
        iter_ = iter(self.get_dataloader())
        epoch = 0

        for step in range(
            self.global_step * self.args.gradient_accumulation_steps,
            self.args.max_train_steps * self.args.gradient_accumulation_steps,
        ):
            # ----- next batch (recycle iterator on StopIteration) -----
            fetch_time_sec = 0.0
            while True:
                try:
                    fetch_start = datetime.now(timezone.utc)
                    raw_batch = next(iter_)
                    fetch_time_sec = (datetime.now(timezone.utc) - fetch_start).total_seconds()
                    break
                except StopIteration:
                    iter_ = iter(self.get_dataloader())
                    logger.info(f"Rank {self.rank} reinit iterator")
                    epoch += 1
            batch = self.BATCH_CLS.from_tuple(raw_batch)

            # ----- step -----
            with self.accelerator.accumulate(self.transformer):
                outputs = self.training_step(batch, fetch_time_sec=fetch_time_sec)
                loss = outputs["loss"]
                self.accumulate_step_outputs(outputs)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if self.ema_lora is not None and self.accelerator.sync_gradients:
                    self.ema_lora.update()

            # ----- per-sync admin (log, save, increment) -----
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                if self.accelerator.is_main_process:
                    self.emit_wandb_log(outputs, batch_time_sec=(datetime.now(timezone.utc) - now).total_seconds())
                    now = datetime.now(timezone.utc)
                self.reset_accumulators()

                if (self.global_step - 1) % self.args.checkpointing_steps == 0 and (self.global_step - 1) > 0:
                    torch.cuda.synchronize()
                    self.accelerator.wait_for_everyone()
                    torch.cuda.empty_cache()
                    self.save_checkpoint()
                    now = datetime.now(timezone.utc)

            progress_bar.set_postfix(
                step_loss=loss.detach().item(),
                global_step=self.global_step,
                epoch=epoch,
            )
            if self.global_step >= self.args.max_train_steps:
                break

    # ---------------------------------------------------------------
    # Driver
    # ---------------------------------------------------------------
    def run(self):
        # NCCL/EFA env vars must be set before the process group is initialised in
        # `build_accelerator`. SSH helper has to start early so a stuck job is still
        # reachable. Both happen here so entrypoints stay one-liners.
        set_nccl_env()
        self.print_cuda_info()
        self.setup()
        self.main_loop()
        logger.info("Waiting for everyone :)")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------
    def accumulate_step_outputs(self, outputs: dict):
        loss = outputs["loss"]
        self.train_loss += self.accelerator.gather(loss.detach()).mean() / self.args.gradient_accumulation_steps
        for k, v in (outputs.get("named_losses") or {}).items():
            if v is None:
                continue
            tensor = v if torch.is_tensor(v) else torch.as_tensor(v)
            gathered = self.accelerator.gather(tensor.detach().mean()).mean()
            self.named_loss_acc[k] = self.named_loss_acc.get(k, 0.0) + gathered / self.args.gradient_accumulation_steps

    def emit_wandb_log(self, outputs: dict, batch_time_sec: float):
        def _item(v):
            return v.item() if hasattr(v, "item") else v

        log_params = {
            "train_loss": _item(self.train_loss),
            "num_processes": self.accelerator.num_processes,
            "batch_time_sec": batch_time_sec,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **(outputs.get("timings") or {}),
            **{k: _item(v) for k, v in (outputs.get("log_extras") or {}).items()},
        }
        if self.flops_per_step is not None and batch_time_sec > 0:
            achieved = self.flops_per_step / batch_time_sec
            log_params["tflops_per_sec"] = achieved / 1e12
            if self.peak_flops_per_sec:
                log_params["mfu"] = achieved / self.peak_flops_per_sec
        for k, v in self.named_loss_acc.items():
            log_params[f"train_{k}_loss"] = _item(v)
        wandb.log({**log_params, **(outputs.get("wandb_extras") or {})}, step=self.global_step)
        logger.info(f"train_loss: {log_params['train_loss']}")

    def reset_accumulators(self):
        self.train_loss = 0.0
        self.named_loss_acc = {}

    def run_text_encoder(self, captions):
        """Forward the text encoder, align layer count to the transformer, and return
        ``(encoder_hidden_states, text_encoder_layers, prompt_attention_mask, encoder_time_sec)``.

        Side-effect: trims ``self.null_conditioning_layers`` once on the first call so it
        matches ``total_num_layers``. Subsequent calls are no-ops on it.
        """
        timer = []
        with CudaTimerContext(timer):
            eh, te_layers, prompt_attn_mask = self.get_prompt_embedds(captions)
        te_layers = list(te_layers)
        n = self.total_num_layers
        if len(te_layers) >= n:
            te_layers = te_layers[len(te_layers) - n :]
            self.null_conditioning_layers = self.null_conditioning_layers[len(self.null_conditioning_layers) - n :]
        else:
            te_layers = te_layers + [te_layers[-1]] * (n - len(te_layers))
            self.null_conditioning_layers = self.null_conditioning_layers + [self.null_conditioning_layers[-1]] * (n - len(self.null_conditioning_layers))
        return eh, te_layers, prompt_attn_mask, timer[0]

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------
    def save_checkpoint(self):
        save_path = os.path.join(
            self.args.checkpoint_local_path,
            f"checkpoint_{self.global_step - 1:06d}",
        )
        if self.args.lora_rank > 0:
            if getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                self.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            self.accelerator.wait_for_everyone()
            logger.info(f"Saving LoRA checkpoint to {save_path}")
            state_dict = self.accelerator.get_state_dict(self.transformer, unwrap=False)
            state_dict = {k.replace("module.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()}
            self.transformer_unwrapped.save_pretrained(
                save_path,
                state_dict=state_dict,
                selected_adapters=list(self.transformer_unwrapped.peft_config.keys()),
                is_main_process=self.accelerator.is_main_process,
            )
            del state_dict
            if self.ema_lora is not None:
                self.ema_lora.save(save_path, self.accelerator, self.transformer)
        else:
            self.accelerator.save_state(save_path)

        if self.accelerator.is_main_process:
            try:
                shutil.copyfile(
                    self.args.json_data_input_path,
                    os.path.join(save_path, "data_input.json"),
                )
            except (OSError, FileNotFoundError):
                pass
        logger.info(f"Saved state to {save_path}")

    def compute_mfu_constants(self):
        """Precompute per-step FLOPs and device peak for MFU logging.

        Closed-form Megatron approximation: 6*N*tokens for dense matmuls (fwd+bwd) plus
        12*L*d_model*seq^2*B for joint attention. Multiplied by 4/3 under gradient
        checkpointing to account for the re-forward in backward. Per-device — under FSDP
        each rank still computes the full unsharded model on its local batch.
        """
        height, width = self.args.resolution
        if self.args.do_patching:
            image_seq = (height // (self.vae_scale_factor * 2)) * (width // (self.vae_scale_factor * 2))
        else:
            image_seq = (height // self.vae_scale_factor) * (width // self.vae_scale_factor)
        seq_total = image_seq + self.args.max_sequence_length

        tokens_per_step = self.args.train_batch_size * seq_total * self.args.gradient_accumulation_steps
        dense = 6 * self.transformer_param_count * tokens_per_step
        attn = 12 * self.total_num_layers * self.transformer_inner_dim * (seq_total**2) * self.args.train_batch_size * self.args.gradient_accumulation_steps
        flops = dense + attn
        if self.args.gradient_checkpointing:
            flops *= 4.0 / 3.0

        self.flops_per_step = float(flops)
        self.peak_flops_per_sec = peak_flops_per_sec(self.weight_dtype)
        logger.info(
            f"MFU: {self.flops_per_step / 1e12:.2f} TFLOPs/step/device "
            f"(seq={seq_total}, B={self.args.train_batch_size}, grad_acc={self.args.gradient_accumulation_steps}); "
            f"peak={self.peak_flops_per_sec / 1e12 if self.peak_flops_per_sec else 'unknown'} TFLOPs/s "
            f"on '{torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}'"
        )

    @staticmethod
    def print_cuda_info():
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        try:
            out = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(
                "CUDA Toolkit version (from nvcc):",
                out.stdout.split("release ")[1].split(",")[0],
            )
        except Exception:
            print("No nvcc found.")
