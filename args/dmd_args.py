"""Decoupled DMD args."""


def add_decoupled_dmd_args(parser):
    parser.add_argument(
        "--noise_max_t",
        type=int,
        default=1000,
        help="Max timestep for flow-matching noise (sigma = t / noise_max_t). Used by the DMD DM-update / fake-score noise (capped at this value), and the add_flow_noise / x0_from_v conversions. Default 1000 matches the standard convention.",
    )
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
        "--teacher_force_prob",
        type=float,
        default=0.0,
        help="Per-step probability (shared-RNG, rank-uniform) of teacher-forcing the student "
        "input PiD-style: skip the no-grad rollout and grad-forward from the GT target latents "
        "flow-noised to the selected step's timestep, instead of the student's own trajectory "
        "state. 0 = always roll out from noise (default).",
    )
    parser.add_argument(
        "--dmd_mu",
        type=float,
        default=None,
        help="Flow-matching time shift mu for the student schedule (sigma' = e^mu * sigma / (1 + (e^mu - 1) * sigma)). None = resolution-dependent calculate_shift (default; ~1.15 at 1024). Lower mu biases intermediate timesteps toward cleaner sigmas.",
    )
    parser.add_argument(
        "--dmd_fake_loss_scale",
        type=float,
        default=1.0,
        help="Scale for training the fake score model on generated samples.",
    )
    parser.add_argument(
        "--dmd_real_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale used for CA real teacher update.",
    )
    parser.add_argument(
        "--dmd_real_guidance_min",
        type=float,
        default=None,
        help="With --dmd_real_guidance_max, the CA CFG scale is drawn per record in [min, max] instead of the fixed scale.",
    )
    parser.add_argument("--dmd_real_guidance_max", type=float, default=None)
    parser.add_argument(
        "--dmd_pdmd",
        type=str,
        default="off",
        choices=["off", "dm", "all"],
        help="PDMD: project the fake critic's error x0_fake - x out of the DM direction (dm) or out of both directions (all).",
    )
    # A second frozen teacher with the main CFG, on a rank-shared per-step coin.
    parser.add_argument("--t2i_teacher_path", type=str, default=None, help="Second frozen teacher weights (local checkpoint dir).")
    parser.add_argument("--t2i_teacher_prob", type=float, default=0.0, help="Share of steps whose real score comes from t2i_teacher; 0 = off.")
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
        "--dmd_pseudo_huber_c",
        type=float,
        default=0.0,
        help="Pseudo-Huber constant c for the generator CA/DM losses (official TDM: 1e-3 via "
        "--use_huber): loss = mean((sqrt(diff^2 + c^2) - c) / normalizer) on the RAW update, "
        "replacing MSE on the normalized update. Gives an ~L1 gradient (sign(diff)/norm) once "
        "|diff| >> c. 0 = MSE (default, unchanged behaviour).",
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
        help="EMA decay for the student EMA shadow. Saved as a side ema/ subdir per checkpoint. 0 disables.",
    )
    parser.add_argument(
        "--ema_start_step",
        type=int,
        default=0,
        help="global_step at which EMA starts tracking; the shadow is re-seeded from the live student then.",
    )
    parser.add_argument(
        "--tdm_step_list",
        type=str,
        default="",
        help="TDM steps-aware training: comma-separated few-step counts to sample K from each iteration (e.g. '2,4,8'), broadcast across FSDP ranks. Empty = fixed --dmd_num_inference_steps (TDM off). Validation always uses --dmd_num_inference_steps.",
    )
    parser.add_argument(
        "--tdm_segment_renoise",
        type=int,
        default=0,
        help="TDM segment matching: draw the CA/DM and fake-score renoise timesteps uniformly WITHIN the selected step's segment [t_{i+1}, t_i] (non-overlapping intervals), instead of uniformly over the full range. 0 = global renoise.",
    )
    parser.add_argument(
        "--dmd_fake_min_snr_gamma",
        type=float,
        default=0.0,
        help="Official-TDM critic weighting: min(SNR, gamma) on the x0-space fake-score MSE "
        "(applied as min((1-sigma)^2, gamma*sigma^2) on our v-space MSE). Official uses 5. "
        "0 disables (default, unweighted v-MSE).",
    )
    parser.add_argument(
        "--dmd_fake_is_weight",
        type=int,
        default=0,
        help="Official-TDM importance weight on the critic loss: per-sample Gaussian density "
        "ratio exp(0.5*(mean(fresh^2) - mean(eps_mix^2))) correcting the bridge-renoised sample "
        "toward the forward-diffusion marginal. 0 disables.",
    )
    parser.add_argument(
        "--fake_lr",
        type=float,
        default=None,
        help="Learning rate for the fake/guidance optimizer. Defaults to --learning_rate.",
    )
