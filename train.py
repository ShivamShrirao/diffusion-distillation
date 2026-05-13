"""Single training entrypoint. Dispatches on ``--trainer``.

The value of ``--trainer`` is an importable ``module.path.ClassName`` (e.g.
``trainers.dmd.DecoupledDMDT2ITrainer``).
"""
import argparse
import importlib

from args.base_args import read_config_defaults


def resolve_trainer(path: str):
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def parse_trainer_path() -> str:
    """Peek at sys.argv for --trainer without consuming other args; trainer's own
    parse_args runs later and re-reads sys.argv in full, where --trainer is registered
    via add_base_args so it doesn't error."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trainer", default="trainers.base.BaseTrainer")
    parser.set_defaults(**{k: v for k, v in read_config_defaults().items() if k == "trainer"})
    args, _ = parser.parse_known_args()
    return args.trainer


if __name__ == "__main__":
    resolve_trainer(parse_trainer_path())().run()
