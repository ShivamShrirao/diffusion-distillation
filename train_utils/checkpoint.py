import os

from accelerate.logging import get_logger

_logger = get_logger(__name__, log_level="INFO")


def resolve_resume_checkpoint(args):
    """Returns the absolute path of the resume checkpoint dir or None."""
    if args.resume_from_checkpoint == "no":
        return None
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        if not os.path.isdir(args.checkpoint_local_path):
            return None
        dirs = [d for d in os.listdir(args.checkpoint_local_path) if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = dirs[-1] if dirs else None
    if path is None:
        return None
    full = os.path.join(args.checkpoint_local_path, path)
    if not os.path.exists(full):
        return None
    return full


def load_checkpoint(accelerator, args) -> int:
    """Resume `accelerator.state` from a checkpoint dir under `args.checkpoint_local_path`.

    Returns the global_step number to continue from (the saved step + 2, matching the
    legacy behaviour the trainers rely on).
    """
    full_path = resolve_resume_checkpoint(args)
    if full_path is None:
        accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        args.resume_from_checkpoint = None
        return 0

    accelerator.print(f"Resuming from checkpoint {full_path}")
    # DMD checkpoints save each adapter under its own subdir (student/, fake/); their
    # weights are loaded by DMDTrainer.configure_transformer before FSDP wrap, so skip
    # the full-state load here.
    if os.path.isdir(os.path.join(full_path, "student")):
        accelerator.print("Detected multi-adapter LoRA checkpoint; adapter weights were loaded by DMDTrainer.configure_transformer.")
    else:
        accelerator.load_state(full_path, map_location="cpu")
    global_step = int(os.path.basename(full_path).split("_")[-1])
    global_step += 2
    return global_step
