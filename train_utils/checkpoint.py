import os
import shutil
import threading
from pathlib import Path

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


class AsyncCheckpointSaver:
    """Stage checkpoint dir on /dev/shm, then async-move it to the final destination.

    Usage:
        with AsyncCheckpointSaver(final_save_path, is_main_process) as temp_path:
            accelerator.save_state(temp_path)
        # context exit kicks off the move
    """

    _lock = threading.Lock()
    _last_move_thread = None

    def __init__(self, final_save_path: str, is_main_process: bool):
        self.final_save_path = Path(final_save_path)
        self.temp_dir = None
        self.is_main_process = is_main_process

    def __enter__(self):
        checkpoint_name = self.final_save_path.name
        self.temp_dir = Path("/dev/shm/") / checkpoint_name
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        return str(self.temp_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_main_process:
            return False
        t = threading.Thread(
            target=self._move_checkpoint,
            args=(self.temp_dir, self.final_save_path),
            daemon=False,
        )
        with AsyncCheckpointSaver._lock:
            if AsyncCheckpointSaver._last_move_thread and AsyncCheckpointSaver._last_move_thread.is_alive():
                AsyncCheckpointSaver._last_move_thread.join()
            AsyncCheckpointSaver._last_move_thread = t
        t.start()
        return False

    @staticmethod
    def _move_checkpoint(temp_dir: Path, final_path: Path):
        try:
            _logger.info(f"Moving checkpoint from {temp_dir} to {final_path}")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                shutil.rmtree(final_path)
            shutil.move(str(temp_dir), str(final_path))
            _logger.info(f"Successfully moved checkpoint from {temp_dir} to {final_path}")
        except Exception as e:
            _logger.error(f"Error moving checkpoint from {temp_dir} to {final_path}: {e}")
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
