import json


def json_to_data(rank: int, json_file: str):
    """Resolve `(training_dir, ratio, width, height, batch_size)` for this rank.

    The JSON describes how to allocate ranks across datasets/resolutions. Each top-level
    entry holds a `gpu_allocation_map` summing to the slice of ranks it owns. The path
    is built from the entry's `path`.
    """
    with open(json_file, "r") as f:
        json_data = json.load(f)
    gpu_cumsum = 0
    curr_gpu_config = None
    curr_gpu_name = None
    for key in sorted(json_data.keys()):
        value = json_data[key]
        if rank < gpu_cumsum + sum(value["gpu_allocation_map"]):
            curr_gpu_config = value
            curr_gpu_name = key
            break
        gpu_cumsum += sum(value["gpu_allocation_map"])

    if curr_gpu_config is None or curr_gpu_name is None:
        raise Exception(f"Rank {rank} not found in {json_file}")

    resolutions_wh = curr_gpu_config["resolutions_wh"]
    ratios = curr_gpu_config["ratios"]
    gpu_allocation_map = curr_gpu_config["gpu_allocation_map"]
    batch_size = curr_gpu_config["batch_size"]
    shift = gpu_cumsum
    current_sum = 0
    for i in range(len(gpu_allocation_map)):
        current_sum += gpu_allocation_map[i]
        if rank - shift < current_sum:
            width_str, height_str = resolutions_wh[i].split(",")
            width = int(width_str)
            height = int(height_str)
            ratio = ratios[i]
            training_dir = f"{curr_gpu_config['path'].rstrip('/')}/{width}x{height}"
            return training_dir, ratio, width, height, batch_size


def total_batch_no_acc(json_file: str) -> int:
    """Sum batch contributions across all entries in the data input JSON."""
    with open(json_file, "r") as f:
        data_config = json.load(f)
    return sum(v["batch_size"] * sum(v["gpu_allocation_map"]) for v in data_config.values())
