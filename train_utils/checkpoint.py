import logging
import os
import shutil
import threading

# stdlib logger, not accelerate's: these messages come from a rank-0-only background
# thread, and the failure path must never itself raise (accelerate's logger asserts on
# process state).
logger = logging.getLogger(__name__)

pending_checkpoint_thread = None


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


def finish_pending_checkpoint():
    """Block until the previous async checkpoint write (if any) has fully landed."""
    global pending_checkpoint_thread
    t = pending_checkpoint_thread
    pending_checkpoint_thread = None
    if t is not None:
        t.join()


def write_checkpoint_async(final_path: str, writes: list, is_main_process: bool):
    """Run `writes` (closures taking a staging dir) in a background thread, then publish
    the staging dir at `final_path` with an atomic rename.

    The training loop only pays for the (collective) state-dict gathers; serializing
    50-100GB of safetensors to disk overlaps with the next steps. The staging dir is
    hidden (`.writing_<name>`) so `resolve_resume_checkpoint` never picks up a
    half-written checkpoint; a crash mid-write leaves only the hidden dir behind.

    Joins the previous pending write first, so at most one checkpoint's gathered state
    is held in RAM at a time. Gathers must happen BEFORE this call — the closures run
    off-thread and must only touch CPU tensors and disk. Non-main ranks pass through
    (their `writes` list is empty). The thread is non-daemon: process exit waits for
    the write to finish rather than corrupting it.
    """
    finish_pending_checkpoint()
    if not is_main_process:
        return

    def run():
        staging = os.path.join(os.path.dirname(final_path), f".writing_{os.path.basename(final_path)}")
        try:
            if os.path.exists(staging):
                shutil.rmtree(staging)
            os.makedirs(staging)
            for write in writes:
                write(staging)
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.replace(staging, final_path)
            logger.info(f"Async checkpoint write complete: {final_path}")
        except Exception:
            logger.error(f"Async checkpoint write FAILED for {final_path}", exc_info=True)
            shutil.rmtree(staging, ignore_errors=True)

    global pending_checkpoint_thread
    pending_checkpoint_thread = threading.Thread(target=run, daemon=False, name="checkpoint-writer")
    pending_checkpoint_thread.start()
