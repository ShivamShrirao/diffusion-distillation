"""Fibo 2.0's VAE (diffusers ``AutoencoderKLFlux2``).

The dataloader yields latents that are already normalized the way the Fibo 2.0 pipeline
normalizes them (2x2 patchify, BatchNorm running statistics, unpatchify), so training
uses ``shift_factor = 0`` and ``scaling_factor = 1``.
"""
from diffusers import AutoencoderKLFlux2


def build_vae_config() -> dict:
    return {
        "latent_channels": 32,
        "shift_factor": 0,
        "scaling_factor": 1,
        "compression_rate": 8,
    }


def build_vae(args, accelerator, weight_dtype):
    """Returns `(vae, vae_config)`.

    `vae_config` carries `shift_factor`, `scaling_factor`, `latent_channels`, and
    `compression_rate` for downstream packing/scaling.
    """
    vae_config = build_vae_config()
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype,
    )
    vae.to(device=accelerator.device).eval().requires_grad_(False)
    if args.vae_gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    return vae, vae_config


def vae_scale_factor_from_config(vae_config: dict) -> int:
    return vae_config["compression_rate"]
