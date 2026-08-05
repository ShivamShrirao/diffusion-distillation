"""Transformer builder.

Builds the bare transformer: Fibo 2.0's ``BriaFibo2Transformer2DModel`` and an optional
``--transformer_init`` weight load over the pretrained weights.
Gradient checkpointing and ``accelerator.prepare`` are owned by the trainer (see ``BaseTrainer.configure_transformer``).

Packed path (``packed_stream``): diffusers' forward re-expressed per token on one flat stream.
adaLN runs once per record (plus one clean row for the gist tokens) and is gathered per token,
and attention is varlen (``train_utils/attention.varlen_attention``). The block forward is
patched to take a ``PackedAttention`` in the mask slot, so FSDP2's hooks, gradient checkpointing
and regional compile wrap the same modules on both paths.
"""
import os
from dataclasses import dataclass

import torch
from diffusers import BriaFibo2Transformer2DModel
from diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2TransformerBlock
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence

from train_utils.attention import varlen_attention
from train_utils.packing import PackedStream, Segments


@dataclass(frozen=True)
class PackedAttention:
    """A block's packed layout: token segments, each token's adaLN row and the text segments."""

    seg: Segments
    rows: torch.Tensor
    text_seg: Segments | None = None


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotary embedding in fp32; x [tokens, heads, head_dim], freqs [tokens, head_dim / 2] complex."""
    out = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs.unsqueeze(1)
    return torch.view_as_real(out).flatten(-2).type_as(x)


def attend(attn, x, seg, freqs=None, context=None, context_seg=None):
    """A BriaFibo2Attention on a flat stream: self-attention within each record (with RoPE), or
    cross-attention from each record's tokens to its own ``context`` tokens."""
    source = x if context is None else context
    q = attn.norm_q(attn.to_q(x).unflatten(-1, (attn.heads, -1)))
    k = attn.norm_k(attn.to_k(source).unflatten(-1, (attn.heads, -1)))
    v = attn.to_v(source).unflatten(-1, (attn.heads, -1))
    if freqs is not None:
        q, k = rope(q, freqs), rope(k, freqs)
    out = varlen_attention(q, k, v, seg, seg if context_seg is None else context_seg)
    return attn.to_out[0](out.flatten(-2))


def packed_block_forward(self, x, packed: PackedAttention, freqs, temb, text=None):
    """BriaFibo2TransformerBlock.forward on a flat stream; ``temb[rows]`` is each token's adaLN input."""
    scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(temb)[packed.rows].chunk(4, dim=-1)
    gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
    x = x + gate_msa * self.attention_norm2(attend(self.attention, self.attention_norm1(x) * (1 + scale_msa), packed.seg, freqs))
    if self.cross_attention is not None:  # injection block: gated cross-attention to the text
        h = attend(self.cross_attention, self.cross_attn_norm1(x), packed.seg, context=text, context_seg=packed.text_seg)
        x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(h)
    return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * (1 + scale_mlp)))


_upstream_block_forward = BriaFibo2TransformerBlock.forward


def block_forward(self, x, attn_mask, freqs_cis, *args, **kwargs):
    if isinstance(attn_mask, PackedAttention):
        return packed_block_forward(self, x, attn_mask, freqs_cis, *args, **kwargs)
    return _upstream_block_forward(self, x, attn_mask, freqs_cis, *args, **kwargs)


BriaFibo2TransformerBlock.forward = block_forward


class Fibo2Transformer2DModel(BriaFibo2Transformer2DModel):
    """Fibo 2.0 called with the trainers' packed-token convention.

    ``hidden_states`` are 2x2-patch tokens ``(B, N, C * 4)`` laid out on the ``img_ids`` grid,
    ``timestep`` is in ``[0, 1000]``, ``encoder_hidden_states`` is the Perceiver bundle and
    ``text_encoder_layers`` the injection bundles. The text padding mask is read off the
    joint attention mask (text tokens first, then image tokens). ``context_latents`` in ``joint_attention_kwargs``
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
        packed_stream: PackedStream | None = None,
    ):
        if packed_stream is not None:
            return self.packed_forward(hidden_states, timestep, encoder_hidden_states, text_encoder_layers, packed_stream), None
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
                raise ValueError("the padded forward needs the same count and sizes of images to edit in every record; --packed_varlen 1 takes any")
            context = [torch.stack(images).to(latents.dtype) for images in zip(*context)]

        out = super().forward(
            hidden_states=latents,
            timestep=timestep / 1000.0,
            encoder_hidden_states=bundles,
            encoder_attention_mask=text_mask,
            context_latents=context,
            return_dict=False,
        )[0]
        return self._patchify(out), None

    def run_block(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(block, *args)
        return block(*args)

    def gist_tokens(self, text, meta: PackedStream):
        """The Perceiver's gist tokens per record (the count follows the caption length), run padded."""
        device = text.device
        states = pad_sequence(text.split(meta.text_lengths), batch_first=True)
        text_mask = torch.arange(states.shape[1], device=device) < torch.tensor(meta.text_lengths, device=device)[:, None]
        n = max(meta.gist_counts)
        gist_mask = torch.arange(n, device=device) < torch.tensor(meta.gist_counts, device=device)[:, None]
        gist = self.perceiver(states, n, text_mask, None if gist_mask.all() else gist_mask)
        return [g[:count] for g, count in zip(gist, meta.gist_counts)]

    def packed_forward(self, tokens, timestep, text, text_layers, meta: PackedStream):
        """tokens (B, N, C * 4) on the ``meta.grid``; timestep (B,) in [0, 1000]; text and
        text_layers flat (sum L_i, D). Returns the velocity tokens (B, N, C * 4)."""
        cfg, device = self.config, tokens.device
        b, n = tokens.shape[:2]
        h, w = meta.grid
        # adaLN inputs: row i is record i's noise level (its image tokens), row b is t = 0 (gist and context tokens).
        temb = self.t_embedder(torch.cat([timestep.float() / 1000.0, timestep.new_zeros(1, dtype=torch.float32)]) * cfg.t_scale)
        plane = cfg.max_num_gist_tokens + 1  # the image's RoPE plane, whatever the gist count
        image_freqs = self.rope_embedder(self.create_coordinate_grid((1, h, w), (plane, 0, 0), device).flatten(0, 2))
        image_rows = torch.arange(b, device=device).repeat_interleave(n)

        def context_freqs(latents, k):
            # Context image k sits on the k-th RoPE plane after the image's, with its own rows and columns.
            size = (1, latents.shape[-2] // cfg.patch_size, latents.shape[-1] // cfg.patch_size)
            return self.rope_embedder(self.create_coordinate_grid(size, (plane + k + 1, 0, 0), device).flatten(0, 2))

        # Refiner segments, record by record: its image tokens, then the tokens of its images to edit.
        pieces, piece_freqs, piece_rows, lengths = [], [], [], []
        for i, refs in enumerate(meta.context or [[]] * b):
            context = [self._patchify(ref[None].to(tokens.dtype))[0] for ref in refs]
            m = sum(c.shape[0] for c in context)
            pieces += [tokens[i], *context]
            piece_freqs += [image_freqs, *(context_freqs(ref, k) for k, ref in enumerate(refs))]
            piece_rows += [torch.full((n,), i, device=device), torch.full((m,), b, device=device)]
            lengths.append(n + m)
        image = self.x_embedder(torch.cat(pieces))
        refiner_freqs, refiner_rows = torch.cat(piece_freqs), torch.cat(piece_rows)
        refiner = PackedAttention(Segments.from_lengths(lengths, device), refiner_rows)
        for block in self.noise_refiner:
            image = self.run_block(block, image, refiner, refiner_freqs, temb)

        # One stream, record by record: gist tokens first, then the image tokens, then the context tokens (Fibo 2.0's order).
        gist = self.gist_tokens(text, meta)
        gist_freqs = [
            self.rope_embedder(self.create_coordinate_grid((g, 1, 1), (1, 0, 0), device).flatten(0, 2))
            for g in meta.gist_counts
        ]
        stream = torch.cat([t for g, x in zip(gist, image.split(lengths)) for t in (g, x)])
        freqs = torch.cat([t for gf, f in zip(gist_freqs, refiner_freqs.split(lengths)) for t in (gf, f)])
        rows = torch.cat([
            t for g, r in zip(meta.gist_counts, refiner_rows.split(lengths)) for t in (torch.full((g,), b, device=device), r)
        ])
        image_index = torch.cat([
            torch.arange(start + g, start + g + n, device=device)
            for start, g in zip(meta.joint.cu_seqlens[:-1].tolist(), meta.gist_counts)
        ])
        main = PackedAttention(meta.joint, rows, meta.text)
        bundles = [text, *text_layers]
        bundle_index = {layer_id: k + 1 for k, layer_id in enumerate(cfg.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            injected = bundles[bundle_index[layer_id]] if layer_id in bundle_index else None
            stream = self.run_block(block, stream, main, freqs, temb, injected)

        final = self.final_layer
        out = final.linear(final.norm_final(stream[image_index]) * (1 + final.adaLN_modulation(temb)[image_rows]))
        return out.view(b, n, -1)


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
    elif os.path.exists(f"{transformer_init}/diffusion_pytorch_model.safetensors"):
        ckpt_path = f"{transformer_init}/diffusion_pytorch_model.safetensors"
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
    ``configure_transformer`` hook is responsible for dtype/grad masking
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
