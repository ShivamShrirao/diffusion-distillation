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
        type=str,
        default="4",
        help="Student generation steps as a single value ('4') or inclusive range ('1-4').",
    )
    parser.add_argument(
        "--dmd_generation_guidance_scale",
        type=float,
        default=5.0,
        help="CFG scale used while generating student samples for DMD.",
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
        "--dmd_fake_guidance_scale",
        type=float,
        default=1.0,
        help="CFG scale used for the fake/model score in the DM update.",
    )
    parser.add_argument(
        "--dmd_ca_weight",
        type=float,
        default=1.0,
        help="Weight for classifier-alignment component.",
    )
    parser.add_argument(
        "--dmd_dm_weight",
        type=float,
        default=1.0,
        help="Weight for distribution-matching component.",
    )
    parser.add_argument(
        "--dmd_max_timestep",
        type=int,
        default=980,
        help="Maximum DMD timestep in [0, 1000].",
    )
    parser.add_argument(
        "--dfake_gen_update_ratio",
        type=int,
        default=5,
        help="Run the generator update every Nth step; the fake/guidance score updates every step.",
    )
