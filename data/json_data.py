import json
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ResolutionData:
    training_dir: str
    width: int
    height: int
    batch_size: int
    sample_percentage: float = 1.0


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
        resolutions = value["resolutions_wh"]
        sample_percentages = value.get("sample_percentages", [1.0] * len(resolutions))
        if len(sample_percentages) != len(resolutions):
            raise ValueError(
                f"{key} sample_percentages must have one value per resolution "
                f"({len(sample_percentages)} != {len(resolutions)})"
            )
        for res, sample_percentage in zip(resolutions, sample_percentages):
            sample_percentage = float(sample_percentage)
            if not 0.0 <= sample_percentage <= 1.0:
                raise ValueError(f"{key} sample percentage for {res} must be in [0, 1], got {sample_percentage}")
            width_str, height_str = res.split(",")
            width = int(width_str)
            height = int(height_str)
            out.append(
                ResolutionData(
                    training_dir=f"{base_dir}/{width}x{height}",
                    width=width,
                    height=height,
                    batch_size=value["batch_size"],
                    sample_percentage=sample_percentage,
                )
            )
    return out
