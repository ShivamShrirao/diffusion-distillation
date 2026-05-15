"""Dataloader builder for the T2I entrypoints.

Each builder returns:
    `(get_dataloader: Callable[[], DataLoader], total_batch_no_acc: int)`
and also mutates `args.train_batch_size` and `args.resolution` to the values resolved
from the data input JSON for this rank.
"""

import glob
import os
import random
from typing import Callable, List, Tuple

import numpy as np
import torch
import webdataset as wds
from datasets import IterableDataset

from data.json_data import json_to_data, total_batch_no_acc


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def load_dataset_from_tars(
    training_dirs: List[str],
    rank: int,
    world_size: int,
    seed: int = 0,
    slice: bool = False,
):
    tar_files = []
    for train_data_dir in training_dirs:
        files = glob.glob(f"{train_data_dir}/*.tar")
        print(f"We have {len(files)} data files on {train_data_dir}")
        tar_files += files

    print(f"We have {len(tar_files)} data files in total")

    if slice:
        print("Using slicing")
        tar_files = tar_files[rank::world_size]
        print(f"Process {rank} will use data files {len(tar_files)} files")

    np.random.seed(seed)
    np.random.shuffle(tar_files)

    def duplicate_name_handler(e):
        if type(e) is ValueError:
            return True  # duplicate vhashes
        raise e

    return wds.WebDataset(
        tar_files,
        nodesplitter=wds.split_by_worker,
        handler=duplicate_name_handler,
        shardshuffle=True,
    ).decode("torch")


def setup(args, default_dir, default_size, default_bs, logger):
    """Resolve rank/seed/resolution/training_dirs/total_batch shared by every builder.

    Mutates `args.train_batch_size` and `args.resolution`. `default_size` is `(width, height)`.
    Returns `(rank, seed, training_dirs, height, width, total_batch)`; `training_dirs` is None
    when `args.random_latents` is set.
    """
    rank = int(os.environ.get("RANK", 0))
    seed = args.seed + rank
    logger.info("Overriding Batch size with dynamic batch")
    logger.info(f"Hello, Im Rank {rank}, from world size {WORLD_SIZE}")

    if not args.random_latents:
        if WORLD_SIZE > 1:
            td, _, width, height, batch_size = json_to_data(rank, json_file=args.json_data_input_path)
            args.train_batch_size = batch_size
            training_dirs = [td]
        else:
            training_dirs = [default_dir]
            width, height = default_size
            args.train_batch_size = default_bs
        assert len(training_dirs) == 1
    else:
        training_dirs = None
        height, width = args.resolution if isinstance(args.resolution, tuple) else (args.resolution, args.resolution)

    args.resolution = (height, width)

    if not args.random_latents:
        print(f"Rank {rank}: Chosen ratio is {training_dirs}, batch size is {args.train_batch_size} for resolution {width},{height}")
    else:
        print(f"Rank {rank}: Converting batch size to {args.train_batch_size} for resolution {width},{height}")

    total_batch = total_batch_no_acc(args.json_data_input_path)
    print(f"Total batch: {total_batch * args.gradient_accumulation_steps}")

    return rank, seed, training_dirs, height, width, total_batch


def tar_dataset_fn(training_dirs, rank):
    def get_dataset(current_seed):
        ds = load_dataset_from_tars(
            training_dirs=training_dirs,
            rank=rank,
            world_size=WORLD_SIZE,
            seed=current_seed,
            slice=False,
        )
        return ds.shuffle(10000, rng=random.Random(current_seed))

    return get_dataset


def dummy_dataset_fn(build_sample, logger):
    """Wrap a zero-arg `build_sample()` (returning a dict) into the IterableDataset pattern."""
    logger.info("Using DummyPrecompute")

    def get_dataset(current_seed):
        def dataset_gen(current_seed):
            data = build_sample()
            while True:
                yield data

        return IterableDataset.from_generator(dataset_gen, gen_kwargs={"current_seed": current_seed})

    return get_dataset


def dataloader_fn(get_dataset, preprocess, collate_fn, args, seed, pin_memory=True):
    rng = random.Random(seed)

    def get_dataloader():
        current_seed = rng.randint(0, 10**6)
        train_dataset = get_dataset(current_seed).map(preprocess)
        return torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
            prefetch_factor=None if args.debug else 4,
            pin_memory=pin_memory,
        )

    return get_dataloader


# ---------------------------------------------------------------------------
# T2I
# ---------------------------------------------------------------------------


def build_dataloader_t2i(args, vae_config, logger) -> Tuple[Callable, int]:
    rank, seed, training_dirs, height, width, total_batch = setup(
        args,
        default_dir="data/256x256",
        default_size=(256, 256),
        default_bs=2,
        logger=logger,
    )
    h, w = args.resolution

    def preprocess(record):
        if "pickle" in record:
            record = record["pickle"]
        example = {"caption": record["caption"]}
        if f"latents_{width}_{height}" in record:
            example["pixel_values"] = record[f"latents_{width}_{height}"]
        elif f"good_latents_{width}_{height}" in record:
            example["pixel_values"] = record[f"good_latents_{width}_{height}"]
        else:
            raise Exception("latents not found in record")
        return example

    def collate_fn(examples):
        captions = [e["caption"] for e in examples]
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return pixel_values, captions

    if args.random_latents:

        def build_sample():
            return {
                f"latents_{width}_{height}": torch.randn([vae_config["latent_channels"], int(h / 8), int(w / 8)], dtype=torch.float32),
                f"images_{width}_{height}": torch.randn([3, h, w], dtype=torch.float32),
                "caption": "_".join(["hello"] * args.max_sequence_length),
            }

        get_dataset = dummy_dataset_fn(build_sample, logger)
    else:
        get_dataset = tar_dataset_fn(training_dirs, rank)

    return dataloader_fn(get_dataset, preprocess, collate_fn, args, seed), total_batch
