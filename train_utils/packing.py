"""Pure-function pack/unpack helpers + CFG dropout helpers.

The pack/unpack functions lay latents out as 2x2-patch tokens, features ordered
(channel, dy, dx) as the transformer patchifies them. Kept as free functions so
training code doesn't depend on a pipeline instance.
"""
from typing import List

import torch

from models.text_encoder import pad_embedding


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


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    h, w, c = latent_image_ids.shape
    return latent_image_ids.reshape(h * w, c).to(device=device, dtype=dtype)


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
        ids = prepare_latent_image_ids(
            noisy_latents.shape[0], latent_h // 2, latent_w // 2, device, noisy_latents.dtype
        )
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
