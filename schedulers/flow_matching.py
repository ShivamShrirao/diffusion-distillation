"""Flow-matching inference scheduler builders.

`build_infer_scheduler()` returns a `FlowMatchEulerDiscreteScheduler` with dynamic shifting.
`build_infer_timesteps()` wraps the linspace-sigmas + calculate_shift +
retrieve_timesteps combo used per-step inside the student rollout.
"""

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps


_DEFAULT_INFER_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


def build_infer_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_config(_DEFAULT_INFER_SCHEDULER_CONFIG)


def build_infer_timesteps(infer_scheduler, current_num_inference_steps: int, latents_seq_len: int, device):
    """Compute the per-step timesteps for one inference pass."""
    sigmas = np.linspace(1.0, 1 / current_num_inference_steps, current_num_inference_steps)
    mu = calculate_shift(
        latents_seq_len,
        infer_scheduler.config.base_image_seq_len,
        infer_scheduler.config.max_image_seq_len,
        infer_scheduler.config.base_shift,
        infer_scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        infer_scheduler,
        num_inference_steps=current_num_inference_steps,
        device=device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps
