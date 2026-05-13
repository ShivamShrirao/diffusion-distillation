"""Data-side args."""


def add_data_args(parser):
    parser.add_argument(
        "--json_data_input_path",
        type=str,
        default="data_input.json.txt",
        help="Path to the JSON describing per-rank data allocation.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Training resolution (overridden by per-rank JSON allocation).",
    )
    parser.add_argument(
        "--random_latents",
        type=int,
        default=0,
        help="Use a synthetic random-tensor dataset (no data reads).",
    )
