def add_smd_args(parser):
    """SMD: the student's one step matches the teacher's CFG velocities inside its TDM segment."""
    parser.add_argument("--smd_points", type=int, default=3, help="Stratified teacher query sigmas per segment.")
    parser.add_argument(
        "--smd_pivot",
        type=str,
        default="alpha",
        choices=["none", "alpha", "learned"],
        help="Query path: none = the student's straight line, alpha = pivot walk toward the teacher, "
        "learned = Pivot head bend (needs --packed_varlen 1).",
    )
    parser.add_argument("--smd_pivot_alpha", type=float, default=0.2, help="Pull of the pivot walk toward the teacher.")
    parser.add_argument("--smd_repulsion_lambda", type=float, default=0.1, help="Weight of the repulsion from the weak-CFG teacher.")
    parser.add_argument("--smd_weak_guidance", type=float, default=1.0, help="Teacher CFG of the repulsion target.")
    parser.add_argument("--smd_sigma_floor", type=float, default=1e-3, help="Lowest query sigma.")
    parser.add_argument("--pivot_terms", type=int, default=4, help="Bend basis terms B_m, m = 1..terms.")
    parser.add_argument("--pivot_blocks", type=int, default=2, help="Pivot head transformer blocks.")
    parser.add_argument("--pivot_dim", type=int, default=1024, help="Pivot head width.")
    parser.add_argument(
        "--pivot_feature_block_from_end",
        type=int,
        default=4,
        help="The head reads the student's tokens at the input of main block n_layers - K.",
    )
    parser.add_argument("--pivot_lr", type=float, default=1e-4, help="Pivot head learning rate (constant).")
