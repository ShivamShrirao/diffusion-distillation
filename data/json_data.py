import json
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ResolutionData:
    training_dir: str
    width: int
    height: int
    batch_size: int


def dataset_active(value: dict) -> bool:
    """Whether a dataset entry participates in training (its ``active`` flag)."""
    return bool(value["active"])


def list_resolution_data(json_file: str) -> List[ResolutionData]:
    """Expand the data JSON into per-resolution entries for every active dataset.

    All listed resolutions of an active dataset participate; the dataloader weights
    them by tar count.
    """
    with open(json_file, "r") as f:
        json_data = json.load(f)

    out = []
    for key in sorted(json_data.keys()):
        value = json_data[key]
        if not dataset_active(value):
            continue
        base_dir = value["path"].rstrip("/")
        for res in value["resolutions_wh"]:
            width_str, height_str = res.split(",")
            width = int(width_str)
            height = int(height_str)
            out.append(
                ResolutionData(
                    training_dir=f"{base_dir}/{width}x{height}",
                    width=width,
                    height=height,
                    batch_size=value["batch_size"],
                )
            )
    return out
