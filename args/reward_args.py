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
    parser.add_argument(
        "--dmdr_both_orders",
        type=int,
        default=0,
        help="1 = ask the pair in both orders in one pass and average the student's wins (cancels the judge's position bias).",
    )
    parser.add_argument(
        "--dmdr_text_scale",
        type=float,
        default=0.0,
        help="Weight of the one-image check that the student renders the caption's text_render exactly; 0 = off.",
    )
    parser.add_argument(
        "--dmdr_text_prob",
        type=float,
        default=1.0,
        help="Share of text records that take the text check instead of the pair (per-rank coin).",
    )
    parser.add_argument(
        "--dmdr_groups",
        type=str,
        default="T2I,EDIT",
        help="Step kinds the judge scores: T2I, EDIT or T2I,EDIT.",
    )
    parser.add_argument(
        "--dmdr_edit_prompt",
        type=str,
        default="aesthetic_with_caption",
        choices=["aesthetic_with_caption", "edit"],
        help="Edit steps: 'aesthetic_with_caption' = the T2I pair prompt (dmdr_prompt_ver) on the caption of the result; "
        "'edit' = the pairwise edit prompt with the record's images to edit.",
    )
