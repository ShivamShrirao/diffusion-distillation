import logging
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch
from accelerate.logging import get_logger as _accel_get_logger


def setup_root_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def get_logger(name: str = __name__):
    return _accel_get_logger(name, log_level="INFO")


def set_seed_for_rank(base_seed: int) -> int:
    rank = int(os.environ.get("RANK", 0))
    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


def resolve_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


class CudaTiming:
    """Deferred GPU-interval reading: resolves to seconds only when consumed (at wandb-log
    time), so the hot loop never calls torch.cuda.synchronize(). float() waits on the end
    event only — a bounded, single-stream sync instead of a device-wide one."""

    def __init__(self, before_event, after_event):
        self.before_event = before_event
        self.after_event = after_event

    def __float__(self):
        if not self.after_event.query():
            self.after_event.synchronize()
        return self.before_event.elapsed_time(self.after_event) / 1000


class CudaTimerContext:
    """Appends one elapsed time to ``times_arr``; with ``enabled=False`` appends 0.0 and records nothing."""

    def __init__(self, times_arr, enabled=True):
        self.times_arr = times_arr
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return
        self.before_event = torch.cuda.Event(enable_timing=True)
        self.after_event = torch.cuda.Event(enable_timing=True)
        self.before_event.record()

    def __exit__(self, type, value, traceback):
        if not self.enabled:
            self.times_arr.append(0.0)
            return
        self.after_event.record()
        self.times_arr.append(CudaTiming(self.before_event, self.after_event))


def init_wandb(accelerator, args, project_name: str = "image-model", extend_name: bool = False):
    import wandb

    if os.environ.get("WANDB_TOKEN"):
        wandb.login(key=os.environ["WANDB_TOKEN"])
    conf = {
        **vars(args),
        "accelerator_use_distributed": accelerator.use_distributed,
        "accelerator_mixed_precision": accelerator.mixed_precision,
        "accelerator_distributed_type": str(accelerator.distributed_type),
    }
    if extend_name:
        name = (
            f"{args.wandb_name}_acc_{args.gradient_accumulation_steps}"
            f"_proc_{accelerator.num_processes}"
        )
    else:
        name = args.wandb_name
    name += "_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")

    init_kwargs = dict(
        project=project_name,
        config=conf,
        mode="online" if project_name else "disabled",
        name=name,
    )
    resume_id = getattr(args, "wandb_resume_from", "") or None
    if resume_id:
        os.environ["WANDB_RUN_ID"] = resume_id
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "must"

    wandb.init(**init_kwargs)
    accelerator.init_trackers(project_name or "distillation")
    wandb.config.update(vars(args), allow_val_change=True)
