import os


def resolve_resume_checkpoint(args):
    """Returns the absolute path of the resume checkpoint dir or None."""
    if args.resume_from_checkpoint == "no":
        return None
    if args.resume_from_checkpoint != "latest":
        # Absolute paths resolve verbatim, so a run can resume a checkpoint that lives
        # outside its own checkpoint_local_path.
        if os.path.isabs(args.resume_from_checkpoint):
            full = args.resume_from_checkpoint.rstrip("/")
            return full if os.path.exists(full) else None
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
