"""SMD: segment matching distillation for flow-matching transformers (arXiv 2610.18472).

No fake critic. The student's one step from ``z`` at sigma_hi (its velocity ``v``) matches the
teacher's CFG velocities at ``K`` stratified query sigmas inside its TDM segment [sigma_lo, sigma_hi],
minus a repulsion from the weak-CFG teacher. The query points lie on a path from ``z``:

    none     the student's straight line z + (sigma - sigma_hi) v
    alpha    a pivot walk: Euler steps along v pulled toward the last teacher target by alpha
    learned  x(tau) = z + (sigma_lo - sigma_hi) [tau v + sum_m a_m B_m(tau)], tau = (sigma_hi - sigma) /
             (sigma_hi - sigma_lo), with per-token coefficients a_m from the Pivot head
             (``models/pivot.py``). Every B_m is 0 at both ends, so the bend keeps the endpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.dmd import padded_attn_kwargs
from train_utils.packing import build_packed_layout, packed_forward, repeat_context


def smd_points(lo: torch.Tensor, hi: torch.Tensor, k: int, u: torch.Tensor, sigma_floor: float = 1e-3) -> torch.Tensor:
    """(B, k) descending stratified sigmas in each record's [lo, hi]: one per equal stratum, sharing
    the uniform offset ``u`` (B,)."""
    strata = torch.arange(k, device=hi.device, dtype=torch.float32)
    width = (hi.float() - lo.float())[:, None] / k
    return (hi.float()[:, None] - (strata + u.float()[:, None]) * width).clamp_min(sigma_floor)


@torch.no_grad()
def smd_pivot_step(z: torch.Tensor, v: torch.Tensor, target: torch.Tensor, delta: torch.Tensor, alpha: float) -> torch.Tensor:
    """Euler step of ``delta`` in sigma along the student velocity pulled toward the teacher's
    target by ``alpha`` (0 = the student's straight line)."""
    return z.float() + delta.float() * (v.float() + alpha * (target.float() - v.float()))


def pivot_basis(tau: torch.Tensor, terms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """B_m(tau) = tau (1 - tau) (2 tau - 1)^(m - 1) and dB_m/dtau, m = 1..terms, shaped [..., terms].
    The path velocity offset r(tau) = sum_m a_m B_m'(tau) integrates to 0 over the segment."""
    x, chord = 2 * tau - 1, tau * (1 - tau)
    basis, deriv = [], []
    for m in range(1, terms + 1):
        basis.append(chord * x ** (m - 1))
        deriv.append((1 - 2 * tau) * x ** (m - 1) + (2 * (m - 1) * chord * x ** (m - 2) if m > 1 else 0))
    return torch.stack(basis, -1), torch.stack(deriv, -1)


def smd_loss(v: torch.Tensor, targets: torch.Tensor, weak_targets: torch.Tensor, lam: float, r: torch.Tensor | None = None):
    """v (B, N, C): the student velocity; targets, weak_targets (K, B, N, C): the teacher at the K
    query points with the training and the weak CFG; r (K, B, N, C): the learned path's velocity
    offsets (None = no bend).

    The path velocity at query k is v + r_k. Matching splits into the mean (student, with r's mean
    detached) and the deviations from it (Pivot head):
    |v + mean r - mean t|^2 + |r - mean r - (t - mean t)|^2, which is mean_k |v - t_k|^2 for r = 0.
    Repulsion: - lam * mean_k |v + sg(r_k) - w_k|^2."""
    v, t, w = v.float(), targets.detach().float(), weak_targets.detach().float()
    r = torch.zeros_like(t) if r is None else r.float()
    mean_r, mean_t = r.mean(0), t.mean(0)
    match = (v + mean_r.detach() - mean_t).square().mean() + (r - mean_r - (t - mean_t)).square().mean()
    repel = (v + r.detach() - w).square().mean()  # repulsion trains the student only
    loss = match - lam * repel
    flat_v, flat_t = v.detach().flatten(1), mean_t.flatten(1)
    return loss, {
        "smd_loss": loss.detach(),
        "smd_match": match.detach(),
        "smd_repel": repel.detach(),
        "smd_cos": F.cosine_similarity(flat_v, flat_t, dim=1).mean(),
        "smd_gap": ((flat_t - flat_v).norm(dim=1) / flat_v.norm(dim=1).clamp_min(1e-12)).mean(),
    }


def compute_smd_loss(
    *,
    student_input: torch.Tensor,
    generated_latents: torch.Tensor,
    seg_lo: float | torch.Tensor,
    seg_hi: float | torch.Tensor,
    real_transformer,
    cond_hidden_states: torch.Tensor,
    cond_layers,
    uncond_hidden_states: torch.Tensor,
    uncond_layers,
    img_ids: torch.Tensor,
    model_dtype: torch.dtype,
    args,
    real_guidance_scale: float | torch.Tensor | None = None,  # scalar or per-record (B, 1, 1)
    coefficients: torch.Tensor | None = None,  # (B, terms, N, C) from the Pivot head
    generator: torch.Generator | None = None,
    packed_stream=None,
    text_mask_after_dropout: torch.Tensor | None = None,
    uncond_text_mask: torch.Tensor | None = None,
    context: list | None = None,  # per record: the latents of its images to edit (padded path; the packed stream holds them)
) -> dict:
    """SMD loss of the student's step from ``student_input`` at seg_hi to its x0 ``generated_latents``."""
    batch_size, device = generated_latents.shape[0], generated_latents.device

    def per_record(values):
        return torch.as_tensor(values, device=device, dtype=torch.float32).expand(batch_size) / args.noise_max_t

    lo, hi = per_record(seg_lo), per_record(seg_hi)
    z = student_input.detach().float()
    # The student velocity; its gradient reaches the student through x0.
    v = (z - generated_latents.float()) / hi.view(-1, 1, 1)
    vs = v.detach()
    guidance = args.dmd_real_guidance_scale if real_guidance_scale is None else real_guidance_scale
    guidance = torch.as_tensor(guidance, device=device, dtype=torch.float32).reshape(-1, 1, 1)

    with torch.no_grad():
        points = smd_points(lo, hi, args.smd_points, torch.rand(batch_size, device=device, generator=generator), args.smd_sigma_floor)
        enc = torch.cat([cond_hidden_states, uncond_hidden_states]).to(dtype=model_dtype)
        layers = [torch.cat([c, u]).to(dtype=model_dtype) for c, u in zip(cond_layers, uncond_layers)]
        mask = torch.cat([text_mask_after_dropout, uncond_text_mask])
        if packed_stream is not None:
            layout = build_packed_layout(mask, img_ids, real_transformer.config, repeat_context(packed_stream.context, 2))
        else:
            padded_kwargs = padded_attn_kwargs(mask, generated_latents.shape[1], model_dtype, args, repeat_context(context, 2))

        def teacher(query, sigma):
            """(CFG, weak-CFG) teacher velocities at ``query`` (B, N, C), sigma (B,)."""
            latents = torch.cat([query, query]).to(dtype=model_dtype)
            timesteps = torch.cat([sigma, sigma]) * args.noise_max_t
            if packed_stream is not None:
                pred = packed_forward(real_transformer, layout, latents, timesteps, text_emb=enc, text_layers=layers, text_mask_after_dropout=mask)[0]
            else:
                pred = real_transformer(
                    hidden_states=latents,
                    timestep=timesteps.to(dtype=torch.float32),
                    encoder_hidden_states=enc,
                    text_encoder_layers=layers,
                    img_ids=img_ids,
                    return_dict=False,
                    joint_attention_kwargs=padded_kwargs,
                )[0]
            cond, uncond = pred.float().chunk(2)
            return uncond + guidance * (cond - uncond), uncond + args.smd_weak_guidance * (cond - uncond)

        def column(x):
            return x.view(-1, 1, 1)

        alpha = args.smd_pivot_alpha if args.smd_pivot == "alpha" else 0.0
        if coefficients is not None:
            taus = (hi[:, None] - points) / (hi - lo)[:, None]
            basis, deriv = pivot_basis(taus, args.pivot_terms)  # (B, K, terms)
            a = coefficients.detach().float()
        query = z + column(points[:, 0] - hi) * vs
        targets, weak = [], []
        for k in range(args.smd_points):
            if coefficients is not None:
                query = z + column(lo - hi) * (column(taus[:, k]) * vs + torch.einsum("bm,bmnc->bnc", basis[:, k], a))
            target, weak_target = teacher(query, points[:, k])
            targets.append(target)
            weak.append(weak_target)
            if coefficients is None and k + 1 < args.smd_points:
                query = smd_pivot_step(query, vs, target, column(points[:, k + 1] - points[:, k]), alpha)

    r = None if coefficients is None else torch.einsum("bkm,bmnc->kbnc", deriv, coefficients.float())
    loss, logs = smd_loss(v, torch.stack(targets), torch.stack(weak), args.smd_repulsion_lambda, r)
    if alpha > 0:
        straight = (z + column(points[:, -1] - hi) * vs).flatten(1)
        logs["smd_pivot_disp"] = ((query.flatten(1) - straight).norm(dim=1) / straight.norm(dim=1).clamp_min(1e-12)).mean()
    logs["teacher_guidance"] = guidance.mean()
    return {"loss": loss * args.dmd_loss_scale, "named_losses": logs}
