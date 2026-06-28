"""Decoupled DMD loss for flow-matching transformers.

DMD/DMD2 formulation in x0 space:

    z_sigma = (1 - sigma) * x0 + sigma * eps
    v       = eps - x0
    x0_hat  = z_sigma - sigma * v_pred

The DM gradient direction is `x0_real - x0_fake` normalised to unit L1
magnitude (DMD2 weighting trick); the CA gradient direction is
`(rgs - 1) * (x0_real_cond - x0_real_uncond)` normalised the same way.

Active LoRA adapter per forward is selected via an explicit
`lora_scales` tensor passed through `joint_attention_kwargs`. The
transformer's checkpoint wrapper threads this tensor through the
gradient-checkpoint boundary so backward recompute uses the same scales as
the original forward (mutating module state between forward and backward
would otherwise corrupt the recomputed activations).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lora.static_multi_lora import StaticMultiLoraLinear, make_scales


def add_flow_noise(x0: torch.Tensor, timesteps: torch.Tensor, noise_max_t: int) -> torch.Tensor:
    sigmas = (timesteps.float() / noise_max_t).view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    return sigmas * torch.randn_like(x0) + (1.0 - sigmas) * x0


def x0_from_v(noisy: torch.Tensor, pred_v: torch.Tensor, timesteps: torch.Tensor, noise_max_t: int) -> torch.Tensor:
    sigmas = (timesteps.float() / noise_max_t).view(-1, 1, 1).to(device=noisy.device, dtype=noisy.dtype)
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
    seg_lo: float | None = None,
    seg_hi: float | None = None,
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
    use_lora_scales = real_transformer is fake_transformer
    fake_scales = make_scales("fake", fake_transformer, device=device) if use_lora_scales else None
    disabled_scales = make_scales(None, real_transformer, device=device) if use_lora_scales else None

    with torch.no_grad():
        # seg_hi == t_grad == infer_timesteps[grad_step_idx]; the old linspace+logit+sigmoid
        # recomputed exactly that flow-match time-shift, so use the rollout timestep directly.
        if getattr(args, "tdm_segment_renoise", 0) and seg_hi is not None:
            # TDM: CA (CFG anchor) and DM (distribution matching) both renoise within the
            # selected step's segment [seg_lo, seg_hi], so each step specializes on its segment.
            u_ca = torch.rand(batch_size, device=device)
            ca_timesteps = (seg_lo + u_ca * (seg_hi - seg_lo)).long().clamp(min=0, max=args.noise_max_t - 1)
            u_dm = torch.rand(batch_size, device=device)
            dm_timesteps = (seg_lo + u_dm * (seg_hi - seg_lo)).long().clamp(min=0, max=args.noise_max_t - 1)
        else:
            # Non-TDM: CA = uniform in (0, gen_sigma]; DM = global uniform.
            gen_sigma = torch.tensor(seg_hi / args.noise_max_t, device=device)
            ca_upper = int(seg_hi)
            ca_sigmas = torch.rand(batch_size, device=device) * gen_sigma
            ca_timesteps = (ca_sigmas * args.noise_max_t).long().clamp(max=ca_upper)
            dm_timesteps = (torch.rand(batch_size, device=device) * args.noise_max_t).long().clamp(max=args.noise_max_t - 1)

        ca_noisy = add_flow_noise(generated_latents, ca_timesteps, args.noise_max_t)
        dm_noisy = add_flow_noise(generated_latents, dm_timesteps, args.noise_max_t)

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
        if disabled_scales is not None:
            real_attn["lora_scales"] = disabled_scales

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

        fake_attn_kwargs = {**(attn_kwargs or {})}
        if fake_scales is not None:
            fake_attn_kwargs["lora_scales"] = fake_scales
        fake_dm_cond_v, _ = fake_transformer(
            hidden_states=dm_noisy.to(dtype=model_dtype),
            timestep=dm_timesteps.to(device=dm_noisy.device, dtype=torch.float32),
            encoder_hidden_states=cond_h,
            text_encoder_layers=cond_l,
            img_ids=img_ids,
            return_dict=False,
            joint_attention_kwargs=fake_attn_kwargs,
        )

        real_ca_cond_x0 = x0_from_v(ca_noisy, real_ca_cond_v, ca_timesteps, args.noise_max_t)
        real_ca_uncond_x0 = x0_from_v(ca_noisy, real_ca_uncond_v, ca_timesteps, args.noise_max_t)
        real_dm_cond_x0 = x0_from_v(dm_noisy, real_dm_cond_v, dm_timesteps, args.noise_max_t)
        fake_dm_cond_x0 = x0_from_v(dm_noisy, fake_dm_cond_v, dm_timesteps, args.noise_max_t)

        ca_update = (args.dmd_real_guidance_scale - 1.0) * (real_ca_cond_x0 - real_ca_uncond_x0)
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
            "dmd_loss": loss,
            "dmd_ca_loss": loss_ca,
            "dmd_dm_loss": loss_dm,
            "dmd_ca_norm_mean": ca_norm.mean(),
            "dmd_dm_norm_mean": dm_norm.mean(),
        },
        "log_latents": {
            "dm_update": dm_update.detach(),
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
    seg_lo: float | None = None,
    seg_hi: float | None = None,
    args=None,
) -> dict:
    batch_size = generated_latents.shape[0]
    device = generated_latents.device
    use_lora_scales = any(isinstance(sub, StaticMultiLoraLinear) for sub in transformer.modules())
    fake_scales = make_scales("fake", transformer, device=device) if use_lora_scales else None

    # --- fake-score renoise timestep ---
    if getattr(args, "tdm_segment_renoise", 0) and seg_hi is not None:
        # TDM segment matching: train the fake score within the selected step's segment.
        u = torch.rand(batch_size, device=device)
        timesteps = (seg_lo + u * (seg_hi - seg_lo)).long().clamp(min=0, max=args.noise_max_t - 1)
    else:
        # uniform (global):
        timesteps = (torch.rand(batch_size, device=device) * args.noise_max_t).long().clamp(max=args.noise_max_t - 1)
    clean_latents = generated_latents.detach()
    noise = torch.randn_like(clean_latents)
    sigmas = (timesteps.float() / args.noise_max_t).view(-1, 1, 1).to(device=device, dtype=clean_latents.dtype)
    noisy_latents = sigmas * noise + (1.0 - sigmas) * clean_latents
    target_v = noise - clean_latents

    noisy_latents = noisy_latents.to(dtype=model_dtype)
    fake_attn_kwargs = {**(attn_kwargs or {})}
    if fake_scales is not None:
        fake_attn_kwargs["lora_scales"] = fake_scales
    fake_pred, _ = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps.to(device=noisy_latents.device, dtype=torch.float32),
        encoder_hidden_states=cond_hidden_states.to(dtype=model_dtype),
        text_encoder_layers=[layer.to(dtype=model_dtype) for layer in cond_layers],
        img_ids=img_ids,
        return_dict=False,
        joint_attention_kwargs=fake_attn_kwargs,
    )

    fake_loss = F.mse_loss(fake_pred.float(), target_v.float(), reduction="mean") * args.dmd_fake_loss_scale

    return {
        "loss": fake_loss,
        "named_losses": {
            "dmd_fake_loss": fake_loss,
        },
    }
