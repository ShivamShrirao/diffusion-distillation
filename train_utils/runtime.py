import logging
import os
import random
from datetime import datetime, timezone

import torch
from accelerate.logging import get_logger as _accel_get_logger
from accelerate.utils import set_seed as _accelerate_set_seed


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
    _accelerate_set_seed(seed)
    random.seed(seed)
    return seed


def resolve_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


class CudaTimerContext:
    def __init__(self, times_arr):
        self.times_arr = times_arr

    def __enter__(self):
        self.before_event = torch.cuda.Event(enable_timing=True)
        self.after_event = torch.cuda.Event(enable_timing=True)
        self.before_event.record()

    def __exit__(self, type, value, traceback):
        self.after_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.before_event.elapsed_time(self.after_event) / 1000
        self.times_arr.append(elapsed_time)


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

    wandb.init(
        project=project_name,
        config=conf,
        mode="online" if project_name else "disabled",
        name=name,
    )
    accelerator.init_trackers(project_name or "distillation")
    wandb.config.update(vars(args), allow_val_change=True)
