"""Transformer builder.

Builds Fibo 2.0's transformer (diffusers ``BriaFibo2Transformer2DModel``), optionally loads
``--transformer_init`` weights over the pretrained ones, applies optional LoRA, gradient
checkpointing, and ``accelerator.prepare`` under FSDP.
"""
import os

import torch
from diffusers import BriaFibo2Transformer2DModel
from diffusers.training_utils import cast_training_params
from safetensors.torch import load_file

from train_utils.checkpoint import resolve_resume_checkpoint
from lora.adapters import add_lora, add_lora_adapter


class Fibo2Transformer2DModel(BriaFibo2Transformer2DModel):
    """Fibo 2.0 called with the trainers' packed-token convention.

    ``hidden_states`` are 2x2-patch tokens ``(B, N, C * 4)`` laid out on the ``img_ids`` grid,
    ``timestep`` is in ``[0, 1000]``, ``encoder_hidden_states`` is the Perceiver bundle and
    ``text_encoder_layers`` the injection bundles. The text padding mask is read off the
    joint attention mask (text tokens first, then image tokens). Returns ``(v_tokens, None)``.
    """

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        text_encoder_layers,
        img_ids=None,
        return_dict=False,
        joint_attention_kwargs=None,
    ):
        batch_size, _, channels = hidden_states.shape
        patch_size = self.config.patch_size
        height = int(img_ids[:, 1].max()) + 1
        width = int(img_ids[:, 2].max()) + 1
        latents = hidden_states.view(batch_size, height, width, channels // patch_size**2, patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, -1, height * patch_size, width * patch_size)

        text_mask = None
        attention_mask = (joint_attention_kwargs or {}).get("attention_mask")
        if torch.is_tensor(attention_mask):
            text_mask = attention_mask[:, 0, -1, : encoder_hidden_states.shape[1]] == 0
            if text_mask.all():
                text_mask = None
        bundles = torch.stack([encoder_hidden_states, *text_encoder_layers], dim=1)

        out = super().forward(
            hidden_states=latents,
            timestep=timestep / 1000.0,
            encoder_hidden_states=bundles,
            encoder_attention_mask=text_mask,
            return_dict=False,
        )[0]
        return self._patchify(out), None


def maybe_resume_dmd_adapters(peft_model, args, logger):
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


def load_transformer_init_weights(transformer, transformer_init: str, logger):
    if os.path.exists(f"{transformer_init}/pytorch_model.bin"):
        ckpt_path = f"{transformer_init}/pytorch_model.bin"
    elif os.path.exists(f"{transformer_init}/pytorch_model_fsdp.bin"):
        ckpt_path = f"{transformer_init}/pytorch_model_fsdp.bin"
    elif os.path.exists(f"{transformer_init}/model.safetensors"):
        ckpt_path = f"{transformer_init}/model.safetensors"
    else:
        raise Exception(f"{transformer_init} not found")

    logger.info(f"\n--------Loading transformer weights from {ckpt_path}--------\n")
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
    transformer.load_state_dict(state_dict, assign=True)


def build_transformer(args, vae_config: dict, accelerator, weight_dtype, logger):
    """Returns `(transformer, transformer_unwrapped, total_num_layers)`.

    ``transformer_unwrapped`` is kept for `save_pretrained` (FSDP-aware path uses the
    accelerator-prepared module, but `unwrap_model` has a known bug for our setup).
    ``total_num_layers`` is the number of injection bundles the transformer reads, used by
    the entrypoint to align text-encoder hidden-state layers.
    """
    transformer = Fibo2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.float32,
    )
    assert vae_config["latent_channels"] == transformer.config.in_channels
    total_num_layers = len(transformer.config.injection_layer_ids)

    if args.transformer_init:
        if (args.use_fsdp and accelerator.is_main_process) or not args.use_fsdp:
            load_transformer_init_weights(transformer, args.transformer_init, logger)

    is_dmd = hasattr(args, "dmd_num_inference_steps")
    if is_dmd and args.lora_rank > 0:
        logger.info(
            f"Using DMD LoRA adapters: student rank {args.lora_rank}, fake rank {args.lora_rank}"
        )
        transformer.requires_grad_(False)
        transformer.to(dtype=weight_dtype)
        transformer = add_lora(transformer, args.lora_rank, adapter_name="student")
        transformer = add_lora_adapter(transformer, args.lora_rank, adapter_name="fake")
        maybe_resume_dmd_adapters(transformer, args, logger)
        transformer.set_adapter("student")
        for name, param in transformer.named_parameters():
            param.requires_grad_(("student" in name or "fake" in name) and "lora_" in name)
        cast_training_params([transformer], dtype=weight_dtype if args.use_fsdp else torch.float32)
        transformer.print_trainable_parameters()
    elif args.lora_rank > 0:
        logger.info(f"Using LoRA with rank {args.lora_rank}")
        transformer.requires_grad_(False)
        transformer.to(dtype=weight_dtype)
        transformer = add_lora(transformer, args.lora_rank)
        cast_training_params([transformer], dtype=weight_dtype if args.use_fsdp else torch.float32)
        transformer.print_trainable_parameters()
    else:
        transformer.requires_grad_(True)
        assert transformer.dtype == torch.float32

    param_count = sum(p.numel() for p in transformer.parameters()) / 10**9
    trainable_param_count = sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 10**9
    logger.info(f"Transformer Parameters: {param_count}B")
    logger.info(f"Trainable transformer Parameters: {trainable_param_count}B")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_unwrapped = transformer
    if args.use_fsdp:
        transformer = accelerator.prepare(transformer)
    return transformer, transformer_unwrapped, total_num_layers
