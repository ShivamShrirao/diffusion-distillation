def add_dmdr_args(parser):
    """DMDR: a frozen Qwen3-VL judge's pairwise preference (student x0 vs the real image)."""
    parser.add_argument(
        "--dmdr_scale",
        type=float,
        default=0.0,
        help="Weight of the judge loss (1 - P(student better than real)) on the student update. 0 disables.",
    )
    parser.add_argument(
        "--dmdr_prob",
        type=float,
        default=0.5,
        help="Share of student updates that run the judge (rank-shared coin).",
    )
    parser.add_argument(
        "--dmdr_min_segment",
        type=int,
        default=2,
        help="Only records whose TDM segment index is >= this (lower noise) are judged.",
    )
    parser.add_argument(
        "--dmdr_judge",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL judge (HF id or local path).",
    )
    parser.add_argument(
        "--dmdr_prompt_ver",
        type=str,
        default="aesthetic_with_caption",
        choices=["aesthetic", "aesthetic_with_caption", "caption"],
        help="Judge prompt variant.",
    )
