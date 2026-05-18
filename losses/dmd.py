"""Decoupled DMD loss for flow-matching transformers.

DMD/DMD2 formulation and work in x0 space:

    z_sigma = (1 - sigma) * x0 + sigma * eps
    v       = eps - x0
    x0_hat  = z_sigma - sigma * v_pred

The DM gradient direction is `x0_real - x0_fake` and the per-sample magnitude
normalization is `|x0_real - generator_output|.mean()` (DMD2 weighting trick).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lora.adapters import disable_adapter, set_adapter


def add_flow_noise(x0: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    sigmas = (timesteps.float() / 1000.0).view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    return sigmas * torch.randn_like(x0) + (1.0 - sigmas) * x0


def x0_from_v(noisy: torch.Tensor, pred_v: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    sigmas = (timesteps.float() / 1000.0).view(-1, 1, 1).to(device=noisy.device, dtype=noisy.dtype)
    return noisy - sigmas * pred_v


def replicate_attn_mask(attn_kwargs: dict, n: int) -> dict:
    out = dict(attn_kwargs or {})
    am = out.get("attention_mask")
    if torch.is_tensor(am):
        out["attention_mask"] = torch.cat([am] * n, dim=0)
    return out


def compute_decoupled_dmd_loss(
    *,
    generated_latents: torch.Tensor,
    generator_timesteps: torch.Tensor,
    real_transformer,
    fake_transformer,
    cond_hidden_states: torch.Tensor,
    cond_layers,
    uncond_hidden_states: torch.Tensor,
    uncond_layers,
    img_ids: torch.Tensor,
    attn_kwargs: dict,
    model_dtype: torch.dtype,
    args,
) -> dict:
    batch_size = generated_latents.shape[0]
    device = generated_latents.device

    with torch.no_grad():
        # Per-sample uniform in [dmd_ca_min_step, generator_timestep).
        ca_upper = generator_timesteps.long()
        ca_timesteps = (torch.rand(ca_upper.shape, device=ca_upper.device) * (ca_upper - args.dmd_ca_min_step).float()).long() + args.dmd_ca_min_step
        dm_timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)

        ca_noisy = add_flow_noise(generated_latents, ca_timesteps)
        dm_noisy = add_flow_noise(generated_latents, dm_timesteps)

        cond_h = cond_hidden_states.to(dtype=model_dtype)
        uncond_h = uncond_hidden_states.to(dtype=model_dtype)
        cond_l = [layer.to(dtype=model_dtype) for layer in cond_layers]
        uncond_l = [layer.to(dtype=model_dtype) for layer in uncond_layers]

        # Single batched real-teacher forward: (ca_cond, ca_uncond, dm_cond).
        real_latents = torch.cat([ca_noisy, ca_noisy, dm_noisy], dim=0).to(dtype=model_dtype)
        real_timesteps = torch.cat([ca_timesteps, ca_timesteps, dm_timesteps], dim=0)
        real_enc = torch.cat([cond_h, uncond_h, cond_h], dim=0)
        real_layers = [torch.cat([cl, ul, cl], dim=0) for cl, ul in zip(cond_l, uncond_l)]
        real_attn = replicate_attn_mask(attn_kwargs, 3)

        with disable_adapter(real_transformer):
            real_pred, _ = real_transformer(
                hidden_states=real_latents,
                timestep=real_timesteps.to(device=real_latents.device, dtype=torch.float32),
                encoder_hidden_states=real_enc,
                text_encoder_layers=real_layers,
                img_ids=img_ids,
                return_dict=False,
                joint_attention_kwargs=real_attn,
            )
        real_ca_cond_v, real_ca_uncond_v, real_dm_cond_v = real_pred.chunk(3, dim=0)

        set_adapter(fake_transformer, "fake")
        fake_dm_cond_v, _ = fake_transformer(
            hidden_states=dm_noisy.to(dtype=model_dtype),
            timestep=dm_timesteps.to(device=dm_noisy.device, dtype=torch.float32),
            encoder_hidden_states=cond_h,
            text_encoder_layers=cond_l,
            img_ids=img_ids,
            return_dict=False,
            joint_attention_kwargs=attn_kwargs,
        )
        set_adapter(fake_transformer, "student")

        real_ca_cond_x0 = x0_from_v(ca_noisy, real_ca_cond_v, ca_timesteps)
        real_ca_uncond_x0 = x0_from_v(ca_noisy, real_ca_uncond_v, ca_timesteps)
        real_dm_cond_x0 = x0_from_v(dm_noisy, real_dm_cond_v, dm_timesteps)
        fake_dm_cond_x0 = x0_from_v(dm_noisy, fake_dm_cond_v, dm_timesteps)

        # DMD2 normalization: divide each gradient direction by its own
        # per-sample magnitude, so the post-norm update has unit L1 magnitude and
        # the loss scale is constant across timesteps.
        ca_update = (args.dmd_real_guidance_scale - 1.0) * (real_ca_cond_x0 - real_ca_uncond_x0)
        ca_norm = ca_update.abs().mean(dim=(1, 2), keepdim=True)
        ca_update = ca_update / (ca_norm + 1e-8)

        dm_update = real_dm_cond_x0 - fake_dm_cond_x0
        dm_norm = dm_update.abs().mean(dim=(1, 2), keepdim=True)
        dm_update = dm_update / (dm_norm + 1e-8)

    loss_ca = F.mse_loss(
        generated_latents.float(),
        (generated_latents + ca_update).detach().float(),
        reduction="mean",
    )
    loss_dm = F.mse_loss(
        generated_latents.float(),
        (generated_latents + dm_update).detach().float(),
        reduction="mean",
    )
    loss = args.dmd_loss_scale * (args.dmd_ca_weight * loss_ca + args.dmd_dm_weight * loss_dm)

    return {
        "loss": loss,
        "named_losses": {
            "dmd_loss": loss,
            "dmd_ca_loss": loss_ca,
            "dmd_dm_loss": loss_dm,
            "dmd_ca_norm_mean": ca_norm.mean(),
        },
        "log_latents": {
            "dm_update": dm_update.detach(),
            "real_ca_cfg_x0": (real_ca_uncond_x0 + args.dmd_real_guidance_scale * (real_ca_cond_x0 - real_ca_uncond_x0)).detach(),
            "real_dm_cond_x0": real_dm_cond_x0.detach(),
            "fake_dm_cond_x0": fake_dm_cond_x0.detach(),
        },
    }


def compute_fake_score_loss(
    *,
    transformer,
    generated_latents: torch.Tensor,
    cond_hidden_states: torch.Tensor,
    cond_layers,
    img_ids: torch.Tensor,
    attn_kwargs: dict,
    model_dtype: torch.dtype,
    args,
) -> dict:
    batch_size = generated_latents.shape[0]
    device = generated_latents.device

    timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
    clean_latents = generated_latents.detach()
    noise = torch.randn_like(clean_latents)
    sigmas = (timesteps.float() / 1000.0).view(-1, 1, 1).to(device=device, dtype=clean_latents.dtype)
    noisy_latents = sigmas * noise + (1.0 - sigmas) * clean_latents
    target_v = noise - clean_latents

    set_adapter(transformer, "fake")
    noisy_latents = noisy_latents.to(dtype=model_dtype)
    fake_pred, _ = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps.to(device=noisy_latents.device, dtype=torch.float32),
        encoder_hidden_states=cond_hidden_states.to(dtype=model_dtype),
        text_encoder_layers=[layer.to(dtype=model_dtype) for layer in cond_layers],
        img_ids=img_ids,
        return_dict=False,
        joint_attention_kwargs=attn_kwargs,
    )
    set_adapter(transformer, "student")

    fake_loss = F.mse_loss(fake_pred.float(), target_v.float(), reduction="mean") * args.dmd_fake_loss_scale

    return {
        "loss": fake_loss,
        "named_losses": {
            "dmd_fake_loss": fake_loss,
        },
    }
