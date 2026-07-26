"""Pseudo-Huber generator loss from official TDM (train_tdm_demo.py, --use_huber, c=1e-3)."""

import torch


def normalized_pseudo_huber_loss(x: torch.Tensor, raw_update: torch.Tensor, norm: torch.Tensor, c: float) -> torch.Tensor:
    """Official-TDM generator loss: elementwise pseudo-Huber on the RAW update, divided by the
    DMD2 normaliser — mean((sqrt(diff^2 + c^2) - c) / norm). For MSE, pre-normalising the update
    gives the same gradient as dividing the loss, but Huber saturates elementwise at |diff| >> c,
    so the normaliser must divide the loss: grad ~= sign(diff) / norm (an L1-style update)."""
    diff = x.float() - (x + raw_update).detach().float()
    return (((diff.square() + c * c).sqrt() - c) / (norm.float() + 1e-8)).mean()
