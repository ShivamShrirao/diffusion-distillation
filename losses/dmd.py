"""Decoupled DMD loss for flow-matching transformers.

The referenced SD implementation predicts epsilon and converts to x0 with DDPM
alphas. This package trains flow-matching v-prediction models, where

    z_sigma = (1 - sigma) * x0 + sigma * eps
    v       = eps - x0
    x0      = z_sigma - sigma * v

So the same decoupled CA/DM update is applied in packed latent x0 space.
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


def sample_ca_timesteps(generator_timesteps, max_timestep: int) -> torch.Tensor:
    generator_timesteps = generator_timesteps.long()
    ca_timesteps = torch.empty_like(generator_timesteps)
    for i, t in enumerate(generator_timesteps):
        upper = int(min(t.item(), max_timestep + 1))
        ca_timesteps[i] = torch.randint(
            0, upper, (1,), device=generator_timesteps.device, dtype=torch.long,
        )
    return ca_timesteps


def sample_dm_timesteps(batch_size: int, max_timestep: int, device) -> torch.Tensor:
    return torch.randint(0, max_timestep + 1, (batch_size,), device=device, dtype=torch.long)


def transformer_v_prediction(
    transformer,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    encoder_hidden_states: torch.Tensor,
    text_encoder_layers,
    img_ids: torch.Tensor,
    attn_kwargs: dict,
    latents_seq_len: int,
) -> torch.Tensor:
    try:
        model_dtype = next(transformer.parameters()).dtype
    except StopIteration:
        model_dtype = latents.dtype
    latents = latents.to(dtype=model_dtype)
    encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)
    text_encoder_layers = [layer.to(dtype=model_dtype) for layer in text_encoder_layers]
    pred, _ = transformer(
        hidden_states=latents,
        timestep=timesteps.to(device=latents.device, dtype=torch.float32),
        encoder_hidden_states=encoder_hidden_states,
        text_encoder_layers=text_encoder_layers,
        img_ids=img_ids,
        return_dict=False,
        joint_attention_kwargs=attn_kwargs,
    )
    return pred[:, :latents_seq_len]


def guided_v_prediction(
    transformer,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    cond_hidden_states: torch.Tensor,
    cond_layers,
    uncond_hidden_states: torch.Tensor,
    uncond_layers,
    img_ids: torch.Tensor,
    attn_kwargs: dict,
    guidance_scale: float,
    latents_seq_len: int,
) -> torch.Tensor:
    model_input = torch.cat([latents, latents], dim=0)
    timestep = torch.cat([timesteps, timesteps], dim=0)
    guided_attn_kwargs = dict(attn_kwargs or {})
    attention_mask = guided_attn_kwargs.get("attention_mask")
    if torch.is_tensor(attention_mask):
        guided_attn_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask], dim=0)
    pred = transformer_v_prediction(
        transformer,
        model_input,
        timestep,
        encoder_hidden_states=torch.cat([uncond_hidden_states, cond_hidden_states], dim=0),
        text_encoder_layers=[
            torch.cat([uncond_layer, cond_layer], dim=0)
            for uncond_layer, cond_layer in zip(uncond_layers, cond_layers)
        ],
        img_ids=img_ids,
        attn_kwargs=guided_attn_kwargs,
        latents_seq_len=latents_seq_len,
    )
    pred_uncond, pred_cond = pred.chunk(2)
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


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
    args,
) -> dict:
    batch_size, latents_seq_len, _ = generated_latents.shape
    max_timestep = int(args.dmd_max_timestep)

    with torch.no_grad():
        ca_timesteps = sample_ca_timesteps(generator_timesteps, max_timestep)
        dm_timesteps = sample_dm_timesteps(batch_size, max_timestep, generated_latents.device)

        ca_noisy = add_flow_noise(generated_latents, ca_timesteps)
        dm_noisy = add_flow_noise(generated_latents, dm_timesteps)

        set_adapter(fake_transformer, "student")
        with disable_adapter(real_transformer):
            real_ca_cond_v = transformer_v_prediction(
                real_transformer,
                ca_noisy,
                ca_timesteps,
                encoder_hidden_states=cond_hidden_states,
                text_encoder_layers=cond_layers,
                img_ids=img_ids,
                attn_kwargs=attn_kwargs,
                latents_seq_len=latents_seq_len,
            )
            real_ca_uncond_v = transformer_v_prediction(
                real_transformer,
                ca_noisy,
                ca_timesteps,
                encoder_hidden_states=uncond_hidden_states,
                text_encoder_layers=uncond_layers,
                img_ids=img_ids,
                attn_kwargs=attn_kwargs,
                latents_seq_len=latents_seq_len,
            )
            real_dm_cond_v = transformer_v_prediction(
                real_transformer,
                dm_noisy,
                dm_timesteps,
                encoder_hidden_states=cond_hidden_states,
                text_encoder_layers=cond_layers,
                img_ids=img_ids,
                attn_kwargs=attn_kwargs,
                latents_seq_len=latents_seq_len,
            )
        set_adapter(fake_transformer, "fake")
        fake_dm_cond_v = guided_v_prediction(
            fake_transformer,
            dm_noisy,
            dm_timesteps,
            cond_hidden_states=cond_hidden_states,
            cond_layers=cond_layers,
            uncond_hidden_states=uncond_hidden_states,
            uncond_layers=uncond_layers,
            img_ids=img_ids,
            attn_kwargs=attn_kwargs,
            guidance_scale=args.dmd_fake_guidance_scale,
            latents_seq_len=latents_seq_len,
        )
        set_adapter(fake_transformer, "student")

        real_ca_cond_x0 = x0_from_v(ca_noisy, real_ca_cond_v, ca_timesteps)
        real_ca_uncond_x0 = x0_from_v(ca_noisy, real_ca_uncond_v, ca_timesteps)
        real_dm_cond_x0 = x0_from_v(dm_noisy, real_dm_cond_v, dm_timesteps)
        fake_dm_cond_x0 = x0_from_v(dm_noisy, fake_dm_cond_v, dm_timesteps)

        ca_update = (args.dmd_real_guidance_scale - 1.0) * (
            real_ca_cond_x0 - real_ca_uncond_x0
        )
        ca_norm = (real_ca_cond_x0 - generated_latents).abs().mean(dim=(1, 2), keepdim=True)
        ca_update = ca_update / (ca_norm + 1e-8)

        dm_update = real_dm_cond_x0 - fake_dm_cond_x0
        dm_norm = (real_dm_cond_x0 - generated_latents).abs().mean(dim=(1, 2), keepdim=True)
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
            "dmd": loss,
            "dmd_ca": loss_ca,
            "dmd_dm": loss_dm,
            "dmd_ca_norm": ca_norm.detach().mean(),
            "dmd_dm_norm": dm_norm.detach().mean(),
            "dmd_ca_timestep": ca_timesteps.float().mean(),
            "dmd_dm_timestep": dm_timesteps.float().mean(),
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
    args,
) -> dict:
    batch_size, latents_seq_len, _ = generated_latents.shape
    timesteps = sample_dm_timesteps(
        batch_size, int(args.dmd_max_timestep), generated_latents.device,
    )
    clean_latents = generated_latents.detach()
    noise = torch.randn_like(clean_latents)
    sigmas = (timesteps.float() / 1000.0).view(-1, 1, 1).to(device=clean_latents.device, dtype=clean_latents.dtype)
    noisy_latents = sigmas * noise + (1.0 - sigmas) * clean_latents
    target_v = noise - clean_latents

    set_adapter(transformer, "fake")
    fake_pred = transformer_v_prediction(
        transformer,
        noisy_latents,
        timesteps,
        encoder_hidden_states=cond_hidden_states,
        text_encoder_layers=cond_layers,
        img_ids=img_ids,
        attn_kwargs=attn_kwargs,
        latents_seq_len=latents_seq_len,
    )
    set_adapter(transformer, "student")

    fake_loss = F.mse_loss(fake_pred.float(), target_v.float(), reduction="mean") * args.dmd_fake_loss_scale
    return {
        "loss": fake_loss,
        "named_losses": {
            "dmd_fake": fake_loss,
            "dmd_fake_timestep": timesteps.float().mean(),
        },
    }
