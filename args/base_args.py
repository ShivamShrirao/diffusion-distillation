"""Shared base/training/optim/checkpointing args."""
import argparse

import yaml


def read_config_defaults() -> dict:
    """Peek at sys.argv for --config: a YAML whose keys become the parser defaults, so
    flags on the command line still win."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    if args.config is None:
        return {}
    with open(args.config) as f:
        return yaml.safe_load(f) or {}


def add_base_args(parser):
    parser.add_argument("--config", type=str, default=None, help="YAML file of argument defaults.")
    parser.add_argument(
        "--checkpoint_local_path",
        type=str,
        default="checkpoints",
        help="The local path to save checkpoints to.",
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="briaai/Fibo-2.0-dev",
        help="Fibo 2.0 diffusers repo (transformer, text_encoder, processor, vae).",
    )
    parser.add_argument(
        "--transformer_init",
        type=str,
        default=None,
        help="Optional dir with transformer weights that replace the pretrained ones.",
    )
    parser.add_argument("--seed", type=int, default=10, help="Seed for reproducible training.")
    parser.add_argument("--max_sequence_length", type=int, default=4096, help="Max sequence length for text encoder.")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=10_000_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help="One of {linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, constant_with_warmup_cosine_decay}.",
    )
    parser.add_argument("--constant_steps", type=int, default=-1)
    parser.add_argument("--lr_warmup_steps", type=int, default=10_000)
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-15)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=10_000)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="no",
        help='Path under `--checkpoint_local_path` or "latest".',
    )
    parser.add_argument("--use_fsdp", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=int, default=0)
    parser.add_argument("--vae_gradient_checkpointing", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--wandb_name", type=str, default="run")
    parser.add_argument("--wandb_project", type=str, default="", help="wandb project; empty = no wandb.")
    parser.add_argument("--do_patching", type=int, default=1)
    parser.add_argument("--use_torch_compile", type=int, default=0)
    parser.add_argument("--regional_compile", type=int, default=0)
    parser.add_argument(
        "--denoising_loss_scale",
        type=float,
        default=1.0,
        help="Scale factor for the flow-matching denoising loss. 0 disables.",
    )
    parser.add_argument(
        "--drop_rate_cfg",
        type=float,
        default=0.1,
        help="Probability of dropping prompt to null at training time (T2I CFG).",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="trainers.base.BaseTrainer",
        help="Importable `module.path.ClassName` of the trainer to run.",
    )
