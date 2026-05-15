"""Shared flow-matching inference-step helpers."""

from __future__ import annotations

import torch


def predict_flow_step(
    transformer,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    encoder_hidden_states: torch.Tensor,
    text_encoder_layers,
    img_ids: torch.Tensor,
    attn_kwargs: dict,
    latents_seq_len: int,
):
    model_input = latents

    expanded_timestep = timestep.expand(model_input.shape[0]).to(
        model_input.device,
        dtype=model_input.dtype,
    )
    model_pred, _ = transformer(
        hidden_states=model_input,
        timestep=expanded_timestep,
        encoder_hidden_states=encoder_hidden_states,
        text_encoder_layers=text_encoder_layers,
        img_ids=img_ids,
        return_dict=False,
        joint_attention_kwargs=attn_kwargs,
    )

    return model_pred[:, :latents_seq_len]
