"""Data-side args."""


def add_data_args(parser):
    parser.add_argument(
        "--json_data_input_path",
        type=str,
        default="data_input.json.txt",
        help="Path to the dataset JSON.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution for the synthetic --random_latents dataset; real data uses per-bucket resolution.",
    )
    parser.add_argument(
        "--random_latents",
        type=int,
        default=0,
        help="Use a synthetic random-tensor dataset (no data reads).",
    )
