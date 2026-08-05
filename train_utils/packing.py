"""Pure-function pack/unpack helpers, packed varlen stream layout, CFG dropout helpers.

The pack/unpack functions lay latents out as 2x2-patch tokens, features ordered
(channel, dy, dx) as the transformer patchifies them. Kept as free functions so
training code doesn't depend on a pipeline instance.
"""
from dataclasses import dataclass
from typing import List

import torch
from diffusers.models.transformers.transformer_bria_fibo2 import compute_gist_count

from models.text_encoder import pad_embedding


@dataclass(frozen=True)
class Segments:
    """Offsets of variable-length records on one flat token axis (the varlen kernels' layout)."""

    lengths: list[int]
    cu_seqlens: torch.Tensor  # [B + 1] int32
    max_len: int

    @classmethod
    def from_lengths(cls, lengths: list[int], device) -> "Segments":
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(lengths, device=device).cumsum(0)
        return cls(list(lengths), cu, max(lengths))


@dataclass(frozen=True)
class PackedStream:
    """One batch packed into a flat stream, record by record ``[gist_i | image_i | context_i]``
    (Fibo 2.0's order; context: the tokens of an edit record's images to edit). Captions go unpadded
    and each record attends only within its own segment, so records with different caption lengths
    (and gist counts) share one forward without padding."""

    text_lengths: list[int]
    gist_counts: list[int]
    grid: tuple[int, int]  # image token grid (h, w), the same for every record of the bucket
    text: Segments
    joint: Segments
    context: list | None = None  # per record: the (C, h, w) latents of its images to edit


def build_packed_layout(text_mask_after_dropout: torch.Tensor, img_ids: torch.Tensor, config, context=None) -> PackedStream:
    """Layout from the post-CFG-dropout (B, L) text mask, the image ids, the transformer config and,
    for edit records, each record's context latents (its images to edit)."""
    device = text_mask_after_dropout.device
    text_lengths = text_mask_after_dropout.bool().sum(dim=1).tolist()
    gist_counts = compute_gist_count(
        text_lengths, config.min_num_gist_tokens, config.max_num_gist_tokens, config.gist_step,
        config.gist_min_text_len, config.gist_max_text_len,
    )
    grid = (int(img_ids[:, 1].max()) + 1, int(img_ids[:, 2].max()) + 1)
    num_image_tokens = grid[0] * grid[1]
    p = config.patch_size
    context_tokens = [sum((x.shape[-2] // p) * (x.shape[-1] // p) for x in refs) for refs in context or [[]] * len(gist_counts)]
    return PackedStream(
        text_lengths=text_lengths,
        gist_counts=gist_counts,
        grid=grid,
        text=Segments.from_lengths(text_lengths, device),
        joint=Segments.from_lengths([g + num_image_tokens + c for g, c in zip(gist_counts, context_tokens)], device),
        context=context,
    )


def repeat_context(context, n: int):
    """Per-record context latents of a batch stacked ``n`` times (CFG-style stacked forwards)."""
    return None if context is None else context * n


def packed_forward(
    transformer,
    meta: PackedStream,
    noisy_target: torch.Tensor,
    timestep: torch.Tensor,
    *,
    text_emb: torch.Tensor,
    text_layers,
    text_mask_after_dropout: torch.Tensor,
    attn_kwargs: dict | None = None,
):
    """Packed forward of (B, N, C) target tokens at per-record ``timestep`` (B,) in [0, 1000].
    The padded captions are flattened to their real tokens; returns ``((B, N, C), None)``."""
    keep = text_mask_after_dropout.bool()
    return transformer(
        hidden_states=noisy_target,
        timestep=timestep.to(device=noisy_target.device, dtype=torch.float32),
        encoder_hidden_states=text_emb[keep],
        text_encoder_layers=[layer[keep] for layer in text_layers],
        return_dict=False,
        joint_attention_kwargs=attn_kwargs,
        packed_stream=meta,
    )


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def pack_latents_no_patch(latents, batch_size, num_channels_latents, height, width):
    latents = latents.permute(0, 2, 3, 1)
    latents = latents.reshape(batch_size, height * width, num_channels_latents)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


def unpack_latents_no_patch(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height, width, channels)
    latents = latents.permute(0, 3, 1, 2)
    return latents


LATENT_IMAGE_IDS_CACHE: dict = {}


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Position ids for one image, built on-device and cached per (h, w, device, dtype). Ids are
    constant per resolution bucket; the stable tensor identity also lets the transformer cache
    its RoPE tables. Callers must not mutate the returned tensor."""
    key = (height, width, str(device), dtype)
    cached = LATENT_IMAGE_IDS_CACHE.get(key)
    if cached is None:
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] += torch.arange(height, device=device)[:, None]
        latent_image_ids[..., 2] += torch.arange(width, device=device)[None, :]
        cached = LATENT_IMAGE_IDS_CACHE[key] = latent_image_ids.reshape(height * width, 3)
    return cached


# ---------------------------------------------------------------------------
# Combined pack helpers used inline by the entrypoint loops.
# ---------------------------------------------------------------------------


def pack_t2i(
    noisy_latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int,
    do_patching: int,
    device,
):
    """Pack noisy latents for the T2I branch. Returns (packed_latents, image_ids)."""
    num_channels_latents = noisy_latents.shape[1]
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    if do_patching == 0:
        packed = pack_latents_no_patch(noisy_latents, noisy_latents.shape[0], num_channels_latents, latent_h, latent_w)
        ids = prepare_latent_image_ids(noisy_latents.shape[0], latent_h, latent_w, device, noisy_latents.dtype)
    else:
        packed = pack_latents(noisy_latents, noisy_latents.shape[0], num_channels_latents, latent_h, latent_w)
        ids = prepare_latent_image_ids(noisy_latents.shape[0], latent_h // 2, latent_w // 2, device, noisy_latents.dtype)
    return packed, ids


def unpack_to_image(packed: torch.Tensor, height: int, width: int, vae_scale_factor: int, do_patching: int):
    if do_patching == 0:
        return unpack_latents_no_patch(packed, height, width, vae_scale_factor)
    return unpack_latents(packed, height, width, vae_scale_factor)


# ---------------------------------------------------------------------------
# CFG dropout helpers
# ---------------------------------------------------------------------------


def apply_cfg_dropout_t2i(
    encoder_hidden_states: torch.Tensor,
    text_encoder_layers: List[torch.Tensor],
    prompt_attention_mask: torch.Tensor,
    null_conditioning: torch.Tensor,
    null_conditioning_layers: List[torch.Tensor],
    num_text_tokens: int,
    drop_rate: float,
    generator: torch.Generator,
):
    """T2I CFG: replace prompt tokens with the null embedding at rate `drop_rate`."""
    if drop_rate <= 0:
        return encoder_hidden_states, text_encoder_layers, prompt_attention_mask

    bsz = encoder_hidden_states.shape[0]
    device = encoder_hidden_states.device
    null_embedding, null_attention_mask = pad_embedding(null_conditioning, max_tokens=num_text_tokens)
    random_p = torch.rand(bsz, device=device, generator=generator)
    prompt_mask = (random_p < drop_rate).reshape(bsz, 1, 1)

    encoder_hidden_states = torch.where(prompt_mask, null_embedding, encoder_hidden_states)
    text_encoder_layers = [
        torch.where(
            prompt_mask,
            pad_embedding(null_conditioning_layers[i], max_tokens=num_text_tokens)[0],
            text_encoder_layers[i],
        )
        for i in range(len(text_encoder_layers))
    ]
    prompt_attention_mask = torch.where(prompt_mask.reshape(bsz, 1), null_attention_mask, prompt_attention_mask)
    return encoder_hidden_states, text_encoder_layers, prompt_attention_mask
