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
        "--student_init",
        type=str,
        default=None,
        help="Optional DMD checkpoint `student/` dir to cold-start the student from.",
    )
    parser.add_argument(
        "--fake_init",
        type=str,
        default=None,
        help="Optional DMD checkpoint `fake/` dir to cold-start the fake critic from.",
    )
    parser.add_argument(
        "--ema_init",
        type=str,
        default=None,
        help="Optional DMD checkpoint `ema/` dir to seed the full-FT student EMA shadow from.",
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
    parser.add_argument("--global_step", type=int, default=0, help="Initial optimizer step for logs, loop offset, and checkpoint numbering.")
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
    parser.add_argument("--lora_alpha", type=int, default=-1, help="LoRA alpha. -1 means use 2 * lora_rank (legacy default).")
    parser.add_argument("--wandb_name", type=str, default="run")
    parser.add_argument("--wandb_project", type=str, default="", help="wandb project; empty = no wandb.")
    parser.add_argument(
        "--wandb_resume_from",
        type=str,
        default="",
        help="Resume (append) into an existing wandb run id. Empty = fresh run.",
    )
    parser.add_argument("--do_patching", type=int, default=1)
    parser.add_argument("--full_torch_compile", type=int, default=0)
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
    # --- Validation (periodic full-inference sampling on a fixed prompt set) ---
    parser.add_argument(
        "--validate_every",
        type=int,
        default=0,
        help="Run full inference on a fixed prompt set every N global steps and log the samples to wandb under val/. 0 disables. Requires the trainer to implement generate_validation_latents().",
    )
    parser.add_argument(
        "--validation_prompts_path",
        type=str,
        default="",
        help="Validation prompts source: a dir of per-prompt JSON files (globbed) or a single JSON file holding a list of prompts. Required for --validate_every.",
    )
    parser.add_argument(
        "--validation_num_prompts",
        type=int,
        default=8,
        help="Cap on how many validation prompts to run each validation pass.",
    )
    parser.add_argument(
        "--validation_resolution",
        type=int,
        default=1024,
        help="Square resolution (H=W) for the validation generations. Fixed across all ranks so the per-rank training aspect buckets don't make the gathered images shape-mismatch. Set to match the run's data resolution (e.g. 512 for the 512 datasets).",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=100050,
        help="Fixed RNG seed for the validation noise. Same seed for every prompt and every pass so samples stay comparable across steps.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="trainers.base.BaseTrainer",
        help="Importable `module.path.ClassName` of the trainer to run.",
    )
