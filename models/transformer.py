"""Transformer builder.

Builds the bare transformer: Fibo 2.0's ``BriaFibo2Transformer2DModel`` and an optional
``--transformer_init`` weight load over the pretrained weights.
LoRA / adapter configuration, gradient checkpointing, and ``accelerator.prepare``
are owned by the trainer (see ``BaseTrainer.configure_transformer``).
"""
import os
from contextlib import contextmanager

import torch
from diffusers import BriaFibo2Transformer2DModel
from safetensors.torch import load_file

from lora.static_multi_lora import StaticMultiLoraLinear


class Fibo2Transformer2DModel(BriaFibo2Transformer2DModel):
    """Fibo 2.0 called with the trainers' packed-token convention.

    ``hidden_states`` are 2x2-patch tokens ``(B, N, C * 4)`` laid out on the ``img_ids`` grid,
    ``timestep`` is in ``[0, 1000]``, ``encoder_hidden_states`` is the Perceiver bundle and
    ``text_encoder_layers`` the injection bundles. The text padding mask is read off the
    joint attention mask (text tokens first, then image tokens). ``lora_scales`` in
    ``joint_attention_kwargs`` picks the static multi-LoRA adapter. ``context_latents`` in ``joint_attention_kwargs``
    (per record, the (C, h, w) latents of its images to edit) join the forward as clean context
    tokens. Returns ``(v_tokens, None)``.
    """

    @property
    def inner_dim(self) -> int:
        return self.config.dim

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
        context = (joint_attention_kwargs or {}).get("context_latents")
        if context is not None:  # per record, its images to edit
            if len({tuple(tuple(x.shape) for x in images) for images in context}) > 1:
                raise ValueError("the padded forward needs the same count and sizes of images to edit in every record")
            context = [torch.stack(images).to(latents.dtype) for images in zip(*context)]

        with self.lora_scales((joint_attention_kwargs or {}).get("lora_scales")):
            out = super().forward(
                hidden_states=latents,
                timestep=timestep / 1000.0,
                encoder_hidden_states=bundles,
                encoder_attention_mask=text_mask,
                context_latents=context,
                return_dict=False,
            )[0]
        return self._patchify(out), None

    @contextmanager
    def lora_scales(self, scales):
        """Set the static multi-LoRA adapter scales for one forward. Under gradient
        checkpointing each block sets them again on recompute, so a later forward with
        other scales (student vs fake) does not leak into this forward's backward."""
        if scales is None:
            yield
            return
        set_lora_scales(self, scales)
        checkpoint_func = getattr(self, "_gradient_checkpointing_func", None)
        if self.gradient_checkpointing and checkpoint_func is not None:

            def scaled_checkpoint(block, *args):
                def run(*inner):
                    set_lora_scales(block, scales)
                    return block(*inner)

                return checkpoint_func(run, *args)

            self._gradient_checkpointing_func = scaled_checkpoint
        try:
            yield
        finally:
            if checkpoint_func is not None:
                self._gradient_checkpointing_func = checkpoint_func


def set_lora_scales(module, scales):
    for sub in module.modules():
        if isinstance(sub, StaticMultiLoraLinear):
            sub._lora_scales = scales


def build_transformer_config(args, vae_config: dict, logger) -> dict:
    transformer_config = dict(
        Fibo2Transformer2DModel.load_config(args.pretrained_model_name_or_path, subfolder="transformer")
    )
    assert vae_config["latent_channels"] == transformer_config["in_channels"]

    logger.info("------- Transformer Config ---------")
    logger.info(transformer_config)
    return transformer_config


def load_transformer_init_weights(transformer, transformer_init: str, logger):
    if os.path.exists(f"{transformer_init}/pytorch_model.bin"):
        ckpt_path = f"{transformer_init}/pytorch_model.bin"
    elif os.path.exists(f"{transformer_init}/pytorch_model_fsdp.bin"):
        ckpt_path = f"{transformer_init}/pytorch_model_fsdp.bin"
    elif os.path.exists(f"{transformer_init}/transformer_ema.bin"):
        ckpt_path = f"{transformer_init}/transformer_ema.bin"
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


def build_transformer(args, vae_config: dict, accelerator, logger):
    """Returns ``(transformer, total_num_layers)``.

    Builds the pretrained model and loads ``--transformer_init`` weights over it. The
    returned transformer is in fp32 with ``requires_grad=True`` everywhere; the trainer's
    ``configure_transformer`` hook is responsible for LoRA setup and dtype/grad masking
    before ``accelerator.prepare``.

    ``total_num_layers`` is the number of injection bundles the transformer reads, used by
    the trainer to align text-encoder hidden-state layers.
    """
    transformer = Fibo2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.float32,
    )
    assert vae_config["latent_channels"] == transformer.config.in_channels

    if args.transformer_init:
        # Under FSDP only rank 0 loads; fsdp2_shard_model scatters the weights to all ranks.
        if not args.use_fsdp or accelerator.is_main_process:
            load_transformer_init_weights(transformer, args.transformer_init, logger)

    total_num_layers = len(transformer.config.injection_layer_ids)
    return transformer, total_num_layers
