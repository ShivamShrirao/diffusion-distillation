"""Decoupled DMD args."""


def add_decoupled_dmd_args(parser):
    parser.add_argument(
        "--dmd_loss_scale",
        type=float,
        default=1.0,
        help="Scale for decoupled DMD loss.",
    )
    parser.add_argument(
        "--dmd_num_inference_steps",
        type=int,
        default=4,
        help="Number of student generation steps per training iteration.",
    )
    parser.add_argument(
        "--dmd_fake_loss_scale",
        type=float,
        default=1.0,
        help="Scale for training the fake LoRA score model on generated samples.",
    )
    parser.add_argument(
        "--dmd_real_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale used for CA real teacher update.",
    )
    parser.add_argument(
        "--dmd_ca_weight",
        type=float,
        default=1.0,
        help="Weight for classifier-alignment component.",
    )
    parser.add_argument(
        "--dmd_ca_min_step",
        type=int,
        default=20,
        help="Minimum timestep sampled for the classifier-alignment component.",
    )
    parser.add_argument(
        "--dmd_dm_weight",
        type=float,
        default=1.0,
        help="Weight for distribution-matching component.",
    )
    parser.add_argument(
        "--dfake_gen_update_ratio",
        type=int,
        default=5,
        help="Run the generator update every Nth step; the fake/guidance score updates every step.",
    )
    parser.add_argument(
        "--dmd_log_image_every",
        type=int,
        default=20,
        help="Decode and log generated / teacher x0 / fake x0 / DM-update images every N global steps. 0 disables (also skips loading the VAE weights).",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay for the student/fake LoRA shadow. Saved as a side ema/ subdir per checkpoint. 0 disables.",
    )
