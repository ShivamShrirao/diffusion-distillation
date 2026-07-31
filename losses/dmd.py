"""Decoupled DMD loss for flow-matching transformers.

DMD/DMD2 formulation in x0 space:

    z_sigma = (1 - sigma) * x0 + sigma * eps
    v       = eps - x0
    x0_hat  = z_sigma - sigma * v_pred

The DM gradient direction is `x0_real - x0_fake` normalised to unit L1
magnitude (DMD2 weighting trick); the CA gradient direction is
`(rgs - 1) * (x0_real_cond - x0_real_uncond)` normalised the same way.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.pseudo_huber import normalized_pseudo_huber_loss
from train_utils.attention import replicate_attention_mask


def add_flow_noise(x0: torch.Tensor, timesteps: torch.Tensor, noise_max_t: int) -> torch.Tensor:
    sigmas = (timesteps.float() / noise_max_t).view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    return sigmas * torch.randn_like(x0) + (1.0 - sigmas) * x0


def x0_from_v(noisy: torch.Tensor, pred_v: torch.Tensor, timesteps: torch.Tensor, noise_max_t: int) -> torch.Tensor:
    sigmas = (timesteps.float() / noise_max_t).view(-1, 1, 1).to(device=noisy.device, dtype=noisy.dtype)
    return noisy - sigmas * pred_v


def add_segment_bridge_noise(
    x0: torch.Tensor, v: torch.Tensor, seg_lo: torch.Tensor, timesteps: torch.Tensor, noise_max_t: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TDM boundary-anchored renoise: deterministic to the segment's lower boundary (ODE point
    z_lo = x0 + sigma_lo * v), then forward-diffused to `timesteps`. Returns (noisy, target_v,
    fresh_bridge_noise); seg_lo = 0 reduces to plain fresh-noise renoise of x0."""
    sigma_lo = (seg_lo.float() / noise_max_t).view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    sigma_t = (timesteps.float() / noise_max_t).view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
    fresh = torch.randn_like(x0)
    z_lo = x0 + sigma_lo * v
    a = (1.0 - sigma_t) / (1.0 - sigma_lo).clamp(min=1e-6)
    beta = (sigma_t.square() - (a * sigma_lo).square()).clamp(min=0.0).sqrt()
    noisy = a * z_lo + beta * fresh
    eps_mix = torch.where(
        sigma_t > 0, (a * sigma_lo * (v + x0) + beta * fresh) / sigma_t.clamp(min=1e-8), fresh
    )
    return noisy, eps_mix - x0, fresh


def replicate_attn_mask(attn_kwargs: dict, n: int) -> dict:
    out = dict(attn_kwargs or {})
    am = out.get("attention_mask")
    if am is not None:
        out["attention_mask"] = replicate_attention_mask(am, n)
    if out.get("context_latents") is not None:
        out["context_latents"] = out["context_latents"] * n
    return out


def sample_decoupled_dmd_timesteps(
    *, batch_size: int, device: torch.device, seg_lo, seg_hi, args
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
    """Draw CA and DM renoise timesteps for either the TDM segment or global schedule."""
    tdm_renoise = getattr(args, "tdm_segment_renoise", 0) and seg_hi is not None
    if tdm_renoise:
        lo_sigma = torch.as_tensor(seg_lo, device=device, dtype=torch.float32) / args.noise_max_t
        hi_sigma = torch.as_tensor(seg_hi, device=device, dtype=torch.float32) / args.noise_max_t

        return (
            (
                (lo_sigma + torch.rand(batch_size, device=device) * (hi_sigma - lo_sigma))
                * args.noise_max_t
            ).long().clamp(min=0, max=args.noise_max_t - 1),
            (
                (lo_sigma + torch.rand(batch_size, device=device) * (hi_sigma - lo_sigma))
                * args.noise_max_t
            ).long().clamp(min=0, max=args.noise_max_t - 1),
            lo_sigma,
            True,
        )

    gen_sigma = torch.tensor(seg_hi / args.noise_max_t, device=device)
    ca_sigmas = torch.rand(batch_size, device=device) * gen_sigma
    ca_timesteps = (ca_sigmas * args.noise_max_t).long().clamp(max=int(seg_hi))
    dm_timesteps = (torch.rand(batch_size, device=device) * args.noise_max_t).long().clamp(max=args.noise_max_t - 1)
    return ca_timesteps, dm_timesteps, None, False


def renoise_decoupled_dmd_latents(
    *,
    generated_latents: torch.Tensor,
    generated_v: torch.Tensor | None,
    lower_sigma: torch.Tensor | None,
    ca_timesteps: torch.Tensor,
    dm_timesteps: torch.Tensor,
    tdm_renoise: bool,
    noise_max_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare the CA and DM teacher inputs using TDM bridge or standard flow noising."""
    if tdm_renoise and generated_v is not None:
        seg_lo = lower_sigma * noise_max_t
        ca_noisy, _, _ = add_segment_bridge_noise(
            generated_latents, generated_v, seg_lo, ca_timesteps, noise_max_t
        )
        dm_noisy, _, _ = add_segment_bridge_noise(
            generated_latents, generated_v, seg_lo, dm_timesteps, noise_max_t
        )
        return ca_noisy, dm_noisy
    return (
        add_flow_noise(generated_latents, ca_timesteps, noise_max_t),
        add_flow_noise(generated_latents, dm_timesteps, noise_max_t),
    )


def dmd_component_loss(
    generated_latents: torch.Tensor,
    update_raw: torch.Tensor,
    update: torch.Tensor,
    norm: torch.Tensor,
    huber_c: float,
) -> torch.Tensor:
    if huber_c > 0:
        return normalized_pseudo_huber_loss(generated_latents, update_raw, norm, huber_c)
    return F.mse_loss(
        generated_latents.float(),
        (generated_latents + update).detach().float(),
        reduction="mean",
    )


def compute_decoupled_dmd_loss(
    *,
    generated_latents: torch.Tensor,
    generated_v: torch.Tensor | None = None,  # detached student velocity; enables TDM bridge renoise
    seg_lo: float | torch.Tensor | None = None,
    seg_hi: float | torch.Tensor | None = None,
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
    real_guidance_scale: float | torch.Tensor | None = None,  # scalar or per-record (B, 1, 1)
) -> dict:
    batch_size = generated_latents.shape[0]
    device = generated_latents.device

    with torch.no_grad():
        # seg_hi is the selected rollout timestep, so no schedule reconstruction is needed.
        ca_timesteps, dm_timesteps, lower_sigma, tdm_renoise = sample_decoupled_dmd_timesteps(
            batch_size=batch_size,
            device=device,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            args=args,
        )
        ca_noisy, dm_noisy = renoise_decoupled_dmd_latents(
            generated_latents=generated_latents,
            generated_v=generated_v,
            lower_sigma=lower_sigma,
            ca_timesteps=ca_timesteps,
            dm_timesteps=dm_timesteps,
            tdm_renoise=tdm_renoise,
            noise_max_t=args.noise_max_t,
        )

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

        guidance_scale = args.dmd_real_guidance_scale if real_guidance_scale is None else real_guidance_scale
        ca_update_raw = (guidance_scale - 1.0) * (real_ca_cond_x0 - real_ca_uncond_x0)
        ca_norm = (real_ca_cond_x0 - generated_latents).abs().mean(dim=(1, 2), keepdim=True)
        ca_update = ca_update_raw / (ca_norm + 1e-8)

        dm_update_raw = real_dm_cond_x0 - fake_dm_cond_x0
        dm_norm = (real_dm_cond_x0 - generated_latents).abs().mean(dim=(1, 2), keepdim=True)
        dm_update = dm_update_raw / (dm_norm + 1e-8)

    huber_c = getattr(args, "dmd_pseudo_huber_c", 0.0)

    loss_ca = dmd_component_loss(generated_latents, ca_update_raw, ca_update, ca_norm, huber_c)
    loss_dm = dmd_component_loss(generated_latents, dm_update_raw, dm_update, dm_norm, huber_c)
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
    seg_lo: float | torch.Tensor | None = None,
    seg_hi: float | torch.Tensor | None = None,
    generated_v: torch.Tensor | None = None,  # detached student velocity; enables TDM bridge renoise
    args=None,
) -> dict:
    batch_size = generated_latents.shape[0]
    device = generated_latents.device
    clean_latents = generated_latents.detach()

    # --- fake-score renoise timestep ---
    if getattr(args, "tdm_segment_renoise", 0) and seg_hi is not None:
        # TDM segment matching: train the fake score within the selected step's segment.
        lo = torch.as_tensor(seg_lo, device=device, dtype=torch.float32)
        hi = torch.as_tensor(seg_hi, device=device, dtype=torch.float32)
        u = torch.rand(batch_size, device=device)
        timesteps = (lo + u * (hi - lo)).long().clamp(min=0, max=args.noise_max_t - 1)
    else:
        # uniform (global):
        timesteps = (torch.rand(batch_size, device=device) * args.noise_max_t).long().clamp(max=args.noise_max_t - 1)
    if getattr(args, "tdm_segment_renoise", 0) and seg_hi is not None and generated_v is not None:
        noisy_latents, target_v, fresh_noise = add_segment_bridge_noise(clean_latents, generated_v.detach(), lo, timesteps, args.noise_max_t)
    else:
        noise = torch.randn_like(clean_latents)
        sigmas = (timesteps.float() / args.noise_max_t).view(-1, 1, 1).to(device=device, dtype=clean_latents.dtype)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * clean_latents
        target_v = noise - clean_latents
        fresh_noise = noise

    noisy_latents = noisy_latents.to(dtype=model_dtype)
    fake_attn_kwargs = {**(attn_kwargs or {})}
    fake_pred, _ = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps.to(device=noisy_latents.device, dtype=torch.float32),
        encoder_hidden_states=cond_hidden_states.to(dtype=model_dtype),
        text_encoder_layers=[layer.to(dtype=model_dtype) for layer in cond_layers],
        img_ids=img_ids,
        return_dict=False,
        joint_attention_kwargs=fake_attn_kwargs,
    )

    per_sample = F.mse_loss(fake_pred.float(), target_v.float(), reduction="none").mean(dim=(1, 2))
    gamma = getattr(args, "dmd_fake_min_snr_gamma", 0.0)
    if gamma > 0:
        # Official TDM critic weighting is min(SNR, gamma) on the x0-space MSE. In v-param,
        # mse_x0 = sigma^2 * mse_v and flow-matching SNR = ((1-sigma)/sigma)^2, so the combined
        # v-space weight is min((1-sigma)^2, gamma*sigma^2).
        sigmas_f = timesteps.float() / args.noise_max_t
        per_sample = per_sample * torch.minimum((1.0 - sigmas_f).square(), gamma * sigmas_f.square())
    if getattr(args, "dmd_fake_is_weight", 0):
        # Official TDM importance weight: Gaussian density ratio of the mixed effective noise
        # vs the fresh bridge noise, correcting the bridge-renoised sample back toward the
        # forward-diffusion marginal the critic loss is an expectation over.
        eps_mix = target_v.float() + clean_latents.float()
        per_sample = per_sample * torch.exp(0.5 * (fresh_noise.float().square().mean(dim=(1, 2)) - eps_mix.square().mean(dim=(1, 2))))
    fake_loss = per_sample.mean() * args.dmd_fake_loss_scale

    return {
        "loss": fake_loss,
        "named_losses": {
            "dmd_fake_loss": fake_loss,
        },
    }
