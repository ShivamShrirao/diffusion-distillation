"""Shared flow-matching inference-step helpers."""

from __future__ import annotations

import torch

from train_utils.packing import packed_forward


def predict_flow_step(
    transformer,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    encoder_hidden_states: torch.Tensor,
    text_encoder_layers,
    packed_stream,
    text_mask_after_dropout,
):
    # Rollout passes a 0-dim scheduler t (or per-sample (B,)); the packed forward takes (B,).
    packed_t = timestep.reshape(-1).expand(latents.shape[0])
    model_pred, _ = packed_forward(
        transformer, packed_stream, latents, packed_t,
        text_emb=encoder_hidden_states, text_layers=text_encoder_layers,
        text_mask_after_dropout=text_mask_after_dropout,
    )
    return model_pred
