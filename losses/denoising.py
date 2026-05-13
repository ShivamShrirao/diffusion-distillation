import torch


def compute_denoising_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_coeff: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Standard flow-matching v-pred MSE, scaled by `loss_coeff * scale`.

    `loss_coeff = WORLD_SIZE / TOTAL_BATCH_NO_ACC` matches the convention used by both
    train scripts: Accelerate divides by gradient_accumulation_steps and `.backward()`
    divides by WORLD_SIZE, so we multiply by WORLD_SIZE and divide by the total batch.
    """
    if scale <= 0:
        return torch.tensor(0.0, device=model_pred.device)
    mse = (model_pred.float() - target.float()) ** 2
    per_sample = torch.mean(mse.reshape(target.shape[0], -1), dim=1)
    return per_sample.sum() * loss_coeff * scale
