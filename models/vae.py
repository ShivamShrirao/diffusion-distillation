"""Fibo 2.0's VAE (diffusers ``AutoencoderKLFlux2``).

The dataloader yields latents that are already normalized the way the Fibo 2.0 pipeline
normalizes them (2x2 patchify, BatchNorm running statistics, unpatchify), so training
uses ``shift_factor = 0`` and ``scaling_factor = 1``.
"""
import torch
from diffusers import AutoencoderKLFlux2

from train_utils.packing import unpack_to_image


def build_vae_config() -> dict:
    return {
        "latent_channels": 32,
        "shift_factor": 0,
        "scaling_factor": 1,
        "compression_rate": 8,
    }


def build_vae(args, accelerator, weight_dtype):
    """Instantiate the VAE model. Trainers that don't need to encode/decode pixels (the
    standard denoising path) should skip this and read scaling factors from
    `build_vae_config` instead.
    """
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype,
    )
    vae.to(device=accelerator.device).eval().requires_grad_(False)
    if args.vae_gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    return vae


def vae_scale_factor_from_config(vae_config: dict) -> int:
    return vae_config["compression_rate"]


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Normalized latents (B, C, H, W) -> images (B, 3, 8H, 8W) in [-1, 1]: undo the
    pipeline's BatchNorm normalization on 2x2 patches, then decode."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, torch.float32)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, torch.float32) + vae.config.batch_norm_eps)
    raw = _unpatchify(_patchify(latents.float()) * std + mean)
    return vae.decode(raw.to(vae.dtype), return_dict=False)[0].float()


def decode_to_images(
    vae,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int,
    vae_config: dict,
    do_patching: int,
) -> torch.Tensor:
    """Unpack → unscale → VAE decode → [0, 1].

    `packed_latents` has shape `(B, seq_len, C)`; the returned tensor has shape
    `(B, 3, H, W)` in `[0, 1]`. Caller is responsible for `with torch.no_grad()` if
    they want to skip building a graph here.
    """
    unpacked = unpack_to_image(packed_latents, height, width, vae_scale_factor, do_patching)
    unpacked = unpacked / vae_config["scaling_factor"] + vae_config["shift_factor"]
    decoded = decode_latents(vae, unpacked)
    return (decoded / 2 + 0.5).clamp(0, 1)
