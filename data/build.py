"""Dataloader builders for the T2I and edit trainers.

Each builder returns:
    `(get_dataloader: Callable[[], DataLoader], total_batch_no_acc: int)`
and mutates `args.train_batch_size` / `args.resolution` to representative values
resolved from the data input.

Resolution is always bucketed: every rank uses the same batch token count on a given
step, while ranks may choose different same-token-count resolutions. The builders
differ only in which keys they pull out of each record and what the collate produces —
the scheduler / tar-loading / batching wiring is shared.

`--random_latents` (T2I only) swaps the real tar source for a synthetic single-bucket stream and
reuses the same preprocess / collate, so no data reads happen.
"""

import glob
import io
import os
import queue
import random
import threading
from dataclasses import dataclass, is_dataclass
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch
import webdataset as wds
from diffusers.pipelines.bria_fibo2.image_processor import BriaFibo2ImageProcessor
from PIL import Image

from data.batch import edit_prompt
from data.json_data import ResolutionData, list_resolution_data
from models.vae import vae_scale_factor_from_config
from train_utils.runtime import get_logger


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DYNAMIC_RESOLUTION_SCHEDULE_STEPS = 128

logger = get_logger(__name__)


def load_dataset_from_tars(
    training_dirs: List[str],
    rank: int,
    world_size: int,
    seed: int = 0,
    slice_by_rank: bool = False,
    tar_files: List[str] | None = None,
):
    if tar_files is None:
        tar_files = []
        for train_data_dir in training_dirs:
            files = glob.glob(f"{train_data_dir}/*.tar")
            logger.info(f"We have {len(files)} data files on {train_data_dir}")
            tar_files += files
        logger.info(f"We have {len(tar_files)} data files in total")

    tar_files = list(tar_files)

    if slice_by_rank:
        all_tar_files = tar_files
        tar_files = all_tar_files[rank::world_size]
        if not tar_files and all_tar_files:
            # If a tiny bucket has fewer shards than participating ranks, shard-level
            # splitting cannot give every rank a unique tar. Fall back to a deterministic
            # repeated shard instead of returning an empty stream.
            tar_files = [all_tar_files[rank % len(all_tar_files)]]
            logger.info(
                f"Process {rank} had no sliced files for world_size={world_size}; "
                f"reusing one shard from {len(all_tar_files)} available files"
            )

    if not tar_files:
        raise RuntimeError(f"No tar files found for training_dirs={training_dirs}")

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
        shardshuffle=100,
    ).decode("torch")


def resolve_resolution(args) -> Tuple[int, int]:
    """Return `(height, width)` from `args.resolution` (a scalar or `(h, w)` tuple)."""
    if isinstance(args.resolution, (tuple, list)):
        return tuple(args.resolution)
    return (args.resolution, args.resolution)


@dataclass(frozen=True)
class BucketSpec:
    index: int
    training_dir: str
    tar_files: Tuple[str, ...]
    width: int
    height: int
    batch_size: int
    tar_count: int
    token_count: int

    @property
    def batch_token_count(self) -> int:
        return self.token_count * self.batch_size


class DynamicResolutionScheduler:
    """Deterministic weighted scheduler shared by every distributed rank."""

    def __init__(
        self,
        buckets: List[BucketSpec],
        world_size: int,
        seed: int,
        schedule_steps: int,
    ):
        if not buckets:
            raise ValueError("DynamicResolutionScheduler requires at least one bucket")
        self.buckets = buckets
        self.world_size = world_size
        self.seed = seed
        self.groups = self._build_groups(buckets)
        min_steps = len(self.groups)
        self.schedule_steps = max(1, schedule_steps, min_steps)
        self.group_schedule = self._build_group_schedule()

    @staticmethod
    def _build_groups(buckets: List[BucketSpec]) -> List[List[BucketSpec]]:
        # Key on (token_count, batch_size), not their product: ranks draw different buckets
        # from the same group each step, so every bucket in a group must share seq_len and
        # batch size or cross-rank collectives see mismatched shapes.
        groups: Dict[tuple, List[BucketSpec]] = {}
        for bucket in buckets:
            groups.setdefault((bucket.token_count, bucket.batch_size), []).append(bucket)
        return [groups[k] for k in sorted(groups.keys())]

    @staticmethod
    def _allocate_counts(weights: List[int], total_slots: int) -> List[int]:
        if len(weights) > total_slots:
            raise ValueError(f"Cannot allocate {len(weights)} groups into {total_slots} schedule slots")
        weight_sum = sum(weights)
        raw = [w * total_slots / weight_sum for w in weights]
        counts = [max(1, int(v)) for v in raw]
        while sum(counts) > total_slots:
            candidates = [(raw[i] - counts[i], i) for i in range(len(counts)) if counts[i] > 1]
            _, idx = min(candidates)
            counts[idx] -= 1
        needed = total_slots - sum(counts)
        if needed > 0:
            remainders = sorted([(raw[i] - int(raw[i]), i) for i in range(len(counts))], reverse=True)
            for _, idx in remainders[:needed]:
                counts[idx] += 1
        return counts

    def _build_group_schedule(self) -> List[int]:
        total_slots = self.schedule_steps
        group_weights = [sum(b.tar_count for b in group) for group in self.groups]
        counts = self._allocate_counts(group_weights, total_slots)
        schedule = []
        for group_idx, count in enumerate(counts):
            schedule.extend([group_idx] * count)
        rng = random.Random(self.seed)
        rng.shuffle(schedule)
        return schedule

    def _choose_bucket(self, group: List[BucketSpec], step: int, rank: int) -> BucketSpec:
        if len(group) == 1:
            return group[0]
        rng = random.Random(self.seed + 1_000_003 * step + 9_176 * rank)
        total = sum(bucket.tar_count for bucket in group)
        pick = rng.randrange(total)
        running = 0
        for bucket in group:
            running += bucket.tar_count
            if pick < running:
                return bucket
        return group[-1]

    def bucket_for_step(self, step: int, rank: int) -> BucketSpec:
        group_idx = self.group_schedule[step % len(self.group_schedule)]
        return self._choose_bucket(self.groups[group_idx], step, rank)


class DynamicResolutionBatchIterator:
    def __init__(self, scheduler, rank, seed, preprocess_fn, collate_fn, start_step=0):
        self.scheduler = scheduler
        self.rank = rank
        self.seed = seed
        self.preprocess_fn = preprocess_fn
        self.collate_fn = collate_fn
        self.start_step = start_step
        self.step = start_step
        self.iterators = {}
        self.epochs = {}

    def __iter__(self):
        return self

    def _iterator_key(self, bucket: BucketSpec):
        return (bucket.index, self.rank, self.scheduler.world_size)

    def _make_iterator(self, bucket: BucketSpec):
        key = self._iterator_key(bucket)
        epoch = self.epochs.get(key, 0)
        current_seed = (self.seed + 1_000_003 * self.start_step + 10_007 * bucket.index + epoch) % (2**32)
        ds = load_dataset_from_tars(
            training_dirs=[bucket.training_dir],
            rank=self.rank,
            world_size=self.scheduler.world_size,
            seed=current_seed,
            slice_by_rank=True,
            tar_files=list(bucket.tar_files),
        )
        ds = ds.shuffle(10000, rng=random.Random(current_seed))
        return iter(ds.map(lambda record: self.preprocess_fn(record, bucket)))

    def _next_example(self, bucket: BucketSpec):
        key = self._iterator_key(bucket)
        if key not in self.iterators:
            self.iterators[key] = self._make_iterator(bucket)
        while True:
            try:
                return next(self.iterators[key])
            except StopIteration:
                self.epochs[key] = self.epochs.get(key, 0) + 1
                self.iterators[key] = self._make_iterator(bucket)

    def __next__(self):
        bucket = self.scheduler.bucket_for_step(self.step, self.rank)
        examples = [self._next_example(bucket) for _ in range(bucket.batch_size)]
        self.step += 1
        return self.collate_fn(examples, bucket)


def _latent_token_count(width: int, height: int, vae_scale_factor: int, do_patching: int) -> int:
    if do_patching:
        return (height // (vae_scale_factor * 2)) * (width // (vae_scale_factor * 2))
    return (height // vae_scale_factor) * (width // vae_scale_factor)


def _sample_tar_files(tar_files: Tuple[str, ...], sample_percentage: float, seed: int, bucket_index: int) -> Tuple[str, ...]:
    if sample_percentage >= 1.0:
        return tar_files
    if sample_percentage <= 0.0:
        return ()
    sample_count = max(1, int(len(tar_files) * sample_percentage))
    rng = random.Random(seed + 1_000_003 * bucket_index)
    return tuple(sorted(rng.sample(list(tar_files), sample_count)))


def build_dynamic_scheduler(args, vae_config, logger):
    vae_scale_factor = vae_scale_factor_from_config(vae_config)
    entries: List[ResolutionData] = list_resolution_data(args.json_data_input_path)
    buckets = []
    for entry_index, entry in enumerate(entries):
        tar_files = tuple(sorted(glob.glob(f"{entry.training_dir}/*.tar")))
        total_tar_count = len(tar_files)
        if total_tar_count <= 0:
            logger.warning(f"Skipping dynamic bucket {entry.width}x{entry.height}: no tars under {entry.training_dir}")
            continue
        tar_files = _sample_tar_files(tar_files, entry.sample_percentage, args.seed, entry_index)
        tar_count = len(tar_files)
        if tar_count <= 0:
            logger.warning(
                f"Skipping dynamic bucket {entry.width}x{entry.height}: "
                f"sample_percentage={entry.sample_percentage} selected no tars from {total_tar_count}"
            )
            continue
        if tar_count != total_tar_count:
            logger.info(
                f"Using {tar_count}/{total_tar_count} tars for dynamic bucket "
                f"{entry.width}x{entry.height} (sample_percentage={entry.sample_percentage})"
            )
        buckets.append(
            BucketSpec(
                index=len(buckets),
                training_dir=entry.training_dir,
                tar_files=tar_files,
                width=entry.width,
                height=entry.height,
                batch_size=entry.batch_size,
                tar_count=tar_count,
                token_count=_latent_token_count(entry.width, entry.height, vae_scale_factor, args.do_patching),
            )
        )
    if not buckets:
        raise RuntimeError(f"No non-empty dynamic resolution buckets found in {args.json_data_input_path}")

    scheduler = DynamicResolutionScheduler(
        buckets=buckets,
        world_size=WORLD_SIZE,
        seed=args.seed,
        schedule_steps=DYNAMIC_RESOLUTION_SCHEDULE_STEPS,
    )
    group_lines = []
    for i, group in enumerate(scheduler.groups):
        resolutions = ", ".join(f"{b.width}x{b.height}@bs{b.batch_size}" for b in group)
        tar_total = sum(b.tar_count for b in group)
        group_lines.append(f"    [{i}] batch_tokens={group[0].batch_token_count}, tars={tar_total}: {resolutions}")
    logger.info(
        "Dynamic resolution buckets:\n"
        f"  batch_sizes={sorted({b.batch_size for b in buckets})}\n"
        f"  schedule_steps={scheduler.schedule_steps}\n"
        f"  all ranks use the same batch token count per step\n"
        f"  groups ({len(scheduler.groups)}):\n"
        + "\n".join(group_lines)
    )
    return scheduler, buckets[0].batch_size


class SyntheticBatchIterator:
    """Infinite synthetic batches for `--random_latents` (no data reads).

    Reuses a builder's `preprocess` / `collate` against a single fixed bucket, so the
    synthetic path produces the exact same batch shape as the real dynamic loader.
    """

    def __init__(self, bucket: BucketSpec, build_sample, preprocess_fn, collate_fn):
        self.bucket = bucket
        self.build_sample = build_sample
        self.preprocess_fn = preprocess_fn
        self.collate_fn = collate_fn

    def __iter__(self):
        return self

    def __next__(self):
        examples = [self.preprocess_fn(self.build_sample(), self.bucket) for _ in range(self.bucket.batch_size)]
        return self.collate_fn(examples, self.bucket)


def pin_batch(batch):
    """Recursively `pin_memory()` CPU tensors inside a collate output (tuple/list/None).

    Runs on the prefetch worker thread; must NOT touch CUDA (pinning only, no `.to`).
    Non-tensor / non-container fields (captions, resolution tuples, product names) pass
    through unchanged, and the container type is preserved so `*Batch.from_tuple` still
    unpacks the result identically.
    """
    if torch.is_tensor(batch):
        if batch.device.type == "cpu" and not batch.is_pinned():
            return batch.pin_memory()
        return batch
    if isinstance(batch, tuple):
        return type(batch)(pin_batch(x) for x in batch) if not hasattr(batch, "_fields") else batch
    if isinstance(batch, list):
        return [pin_batch(x) for x in batch]
    if is_dataclass(batch) and not isinstance(batch, type):
        for field_name in batch.__dataclass_fields__:
            setattr(batch, field_name, pin_batch(getattr(batch, field_name)))
        return batch
    return batch


class PrefetchIterator:
    """Overlap batch production with the training step via a background daemon thread.

    Pulls batches from `inner` into a bounded queue and pins their CPU tensors, so the
    H2D copy the consumer issues (`.to(device, non_blocking=True)`) can overlap the
    compute of the previous step. The worker never touches CUDA. Batch ordering is
    preserved exactly, so a `DynamicResolutionBatchIterator` yields the identical
    sequence with prefetch on or off. Exceptions raised by `inner` are re-raised in the
    consumer with the original traceback; `StopIteration` terminates the consumer.
    """

    STOP = object()

    def __init__(self, inner: Iterator, queue_depth: int = 2):
        self.inner = inner
        self.queue = queue.Queue(maxsize=max(1, queue_depth))
        self.error = None
        self.thread = threading.Thread(target=self.worker, name="prefetch", daemon=True)
        self.thread.start()

    def worker(self):
        try:
            for batch in self.inner:
                self.queue.put(pin_batch(batch))
        except StopIteration:
            pass
        except BaseException as exc:  # re-raised in the consumer with original traceback
            self.error = exc
        finally:
            self.queue.put(self.STOP)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is self.STOP:
            if self.error is not None:
                error = self.error
                self.error = None
                raise error.with_traceback(error.__traceback__)
            raise StopIteration
        return item


def maybe_prefetch(iterator: Iterator, args) -> Iterator:
    """Wrap `iterator` in a `PrefetchIterator` when `dataloader_num_workers >= 1`.

    `dataloader_num_workers == 0` keeps today's synchronous behavior. Otherwise a single
    background prefetch thread runs with queue depth `dataloader_num_workers + 1`.
    """
    num_workers = int(getattr(args, "dataloader_num_workers", 0))
    if num_workers < 1:
        return iterator
    return PrefetchIterator(iterator, queue_depth=num_workers + 1)


def build_dynamic_dataloader(args, vae_config, logger, preprocess, collate) -> Tuple[Callable, int]:
    """Wire the tar-count-weighted resolution scheduler to a builder's preprocess/collate."""
    rank = int(os.environ.get("RANK", 0))
    seed = args.seed + rank
    scheduler, batch_size = build_dynamic_scheduler(args, vae_config, logger)
    args.train_batch_size = batch_size
    first_bucket = scheduler.buckets[0]
    args.resolution = (first_bucket.height, first_bucket.width)
    total_batch = WORLD_SIZE * batch_size
    logger.info(f"Rank {rank}: dynamic resolution buckets, batch size {batch_size}, total batch {total_batch}")

    def get_dataloader(start_step: int | None = None):
        if start_step is None:
            start_step = int(getattr(args, "global_step", 0)) * int(getattr(args, "gradient_accumulation_steps", 1))
        iterator = DynamicResolutionBatchIterator(
            scheduler=scheduler,
            rank=rank,
            seed=seed,
            preprocess_fn=preprocess,
            collate_fn=collate,
            start_step=start_step,
        )
        return maybe_prefetch(iterator, args)

    return get_dataloader, total_batch


def build_random_dataloader(args, build_sample, preprocess, collate, logger) -> Tuple[Callable, int]:
    """Single fixed-resolution synthetic stream sharing the dynamic preprocess/collate."""
    logger.info("Using synthetic random-latents dataset (no data reads)")
    height, width = resolve_resolution(args)
    args.resolution = (height, width)
    bucket = BucketSpec(
        index=0,
        training_dir="",
        tar_files=(),
        width=width,
        height=height,
        batch_size=args.train_batch_size,
        tar_count=0,
        token_count=0,
    )
    total_batch = WORLD_SIZE * bucket.batch_size
    logger.info(f"Rank {int(os.environ.get('RANK', 0))}: random latents, batch size {bucket.batch_size} for resolution {width}x{height}")

    def get_dataloader(start_step: int | None = None):
        return maybe_prefetch(SyntheticBatchIterator(bucket, build_sample, preprocess, collate), args)

    return get_dataloader, total_batch


# ---------------------------------------------------------------------------
# T2I
# ---------------------------------------------------------------------------


def record_latents(record, width: int, height: int) -> torch.Tensor:
    """A record's VAE latents at the bucket resolution."""
    for key in (f"latents_{width}_{height}", f"good_latents_{width}_{height}"):
        if key in record:
            return record[key]
    raise Exception(f"latents not found in record for resolution {width}x{height}")


def build_dataloader_t2i(args, vae_config, logger) -> Tuple[Callable, int]:
    def preprocess(record, bucket: BucketSpec):
        if "pickle" in record:
            record = record["pickle"]
        return {"caption": record["caption"], "pixel_values": record_latents(record, bucket.width, bucket.height)}

    def collate(examples, bucket: BucketSpec):
        captions = [e["caption"] for e in examples]
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        return pixel_values, captions, (bucket.height, bucket.width)

    if args.random_latents:

        def build_sample():
            height, width = resolve_resolution(args)
            return {
                f"latents_{width}_{height}": torch.randn([vae_config["latent_channels"], height // 8, width // 8], dtype=torch.float32),
                "caption": "_".join(["hello"] * args.max_sequence_length),
            }

        return build_random_dataloader(args, build_sample, preprocess, collate, logger)

    return build_dynamic_dataloader(args, vae_config, logger, preprocess, collate)


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def preprocess_references(images, height: int, width: int):
    """The images to edit as Fibo 2.0 was trained on them (its image processor): the VAE's crops, (1, 3, H, W)
    in [-1, 1] (one image at the target size, several each at its own aspect ratio and about the target
    area), and Qwen3-VL's copies."""
    processor = BriaFibo2ImageProcessor(vae_scale_factor=16)
    return processor.preprocess(images, height=height, width=width, area=height * width)


def build_dataloader_edit(args, vae_config, logger) -> Tuple[Callable, int]:
    """Edit records. An edit record also holds ``reference_images`` (the encoded images to edit) and ``instruction``;
    its ``caption``, of the result, is optional."""

    def preprocess(record, bucket: BucketSpec):
        if "pickle" in record:
            record = record["pickle"]
        example = {"caption": record.get("caption"), "references": None}
        example["pixel_values"] = record_latents(record, bucket.width, bucket.height)
        if record.get("reference_images"):
            images = [x if isinstance(x, Image.Image) else Image.open(io.BytesIO(x)) for x in record["reference_images"]]
            example["references"] = preprocess_references(images, bucket.height, bucket.width)
            example["caption"] = edit_prompt(record.get("caption"), record["instruction"])
        return example

    def collate(examples, bucket: BucketSpec):
        captions = [e["caption"] for e in examples]
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        return pixel_values, captions, (bucket.height, bucket.width), [e["references"] for e in examples]

    if args.random_latents:
        raise ValueError("--random_latents has no images to edit")
    return build_dynamic_dataloader(args, vae_config, logger, preprocess, collate)
