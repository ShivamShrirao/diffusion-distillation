"""Distributed-training setup: NCCL env, Accelerator, transformer compile.

`set_nccl_env()` must be called BEFORE the process group is initialised —
`BaseTrainer.run()` calls it first thing, so entrypoints don't need to do anything.
"""

import os
from datetime import timedelta

import diffusers
import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    disable_fsdp_ram_efficient_loading,
)
from huggingface_hub import login
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2TransformerBlock


def set_nccl_env() -> None:
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_MIN_NCHANNELS"] = "8"
    os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_CROSS_NIC"] = "1"


def build_accelerator(args, weight_dtype):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    fsdp_plugin = None
    if args.use_fsdp:
        # cuDNN SDPA backward returns NaN at head_dim=128/bf16 on FSDP2's all-gathered
        # tensors (FSDP1 didn't select it); fall back to flash/mem-efficient.
        torch.backends.cuda.enable_cudnn_sdp(False)
        # FSDP2 (per-parameter sharding). Trainable models are built on meta,
        # loaded on rank 0, then scattered into shards before optimizer construction.
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy=transformer_auto_wrap_policy,
            transformer_cls_names_to_wrap=[BriaFibo2TransformerBlock.__name__],
            reshard_after_forward=True,
            mixed_precision_policy=MixedPrecisionPolicy(param_dtype=weight_dtype, reduce_dtype=torch.float32),
            cpu_ram_efficient_loading=True,
            state_dict_type="FULL_STATE_DICT",
        )
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(1000))]
        print("Using FSDP2")
    else:
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    if fsdp_plugin is not None:
        # The plugin exports FSDP_CPU_RAM_EFFICIENT_LOADING=True, which makes every later
        # transformers.from_pretrained load rank-0-only placeholders — wrong for models that
        # are never FSDP-sharded (text encoder, rewards). Turn the env flag back off; the
        # plugin attribute stays True so fsdp2_shard_model still scatters rank-0 weights.
        disable_fsdp_ram_efficient_loading()

    with accelerator.main_process_first():
        if accelerator.is_local_main_process and os.environ.get("HF_API_TOKEN"):
            login(token=os.environ["HF_API_TOKEN"], add_to_git_credential=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return accelerator


def fsdp2_shard_frozen(model, accelerator, wrap_classes, *, param_dtype=torch.bfloat16,
                       reshard_after_forward=True, ignored_modules=None):
    """FSDP2-shard a frozen, inference-only module in place (no optimizer, unlike
    ``accelerator.prepare``). Weights must already be on every rank; ``fully_shard`` moves each
    rank's slice to GPU as it wraps. Shards ``wrap_classes`` (types or names) bottom-up then the
    root; ``ignored_modules`` keeps their params unsharded in the original dtype (e.g. fp32 head).
    reduce_dtype == param_dtype: a frozen model never all-reduces param grads."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=param_dtype),
    }
    # Shard on the same mesh / fsdp sub-mesh as the student (HSDP-aware).
    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is not None:
        pc = getattr(accelerator, "parallelism_config", None)
        dim_names = getattr(pc, "fsdp_dim_names", None) if pc is not None else None
        fsdp_kwargs["mesh"] = mesh[tuple(dim_names)] if dim_names else mesh

    wrap_names = {c if isinstance(c, str) else c.__name__ for c in wrap_classes}
    blocks = [m for m in model.modules() if type(m).__name__ in wrap_names]
    if not blocks:
        raise ValueError(
            f"fsdp2_shard_frozen: no wrap targets {wrap_names} found in {type(model).__name__}; "
            "root-only fully_shard would all-gather the whole model onto one GPU."
        )
    ignored_params = {p for m in (ignored_modules or []) for p in m.parameters()}
    for block in blocks:
        fully_shard(block, **fsdp_kwargs)
    root_kwargs = {**fsdp_kwargs, **({"ignored_params": ignored_params} if ignored_params else {})}
    fully_shard(model, **root_kwargs)

    # Move ignored params + buffers (untouched by fully_shard) onto the compute device.
    device = accelerator.device
    for p in ignored_params:
        if p.device.type != device.type:
            p.data = p.data.to(device)
    for name, buf in list(model.named_buffers()):
        if buf is None or isinstance(buf, DTensor) or buf.device.type == device.type:
            continue
        *parents, leaf = name.split(".")
        parent = model
        for attr in parents:
            parent = getattr(parent, attr)
        parent._buffers[leaf] = buf.to(device)
    return model


def fsdp2_set_reshard_after_forward(model, value: bool, *, reshard_now: bool = False):
    """Toggle FSDP2 post-forward resharding on every FSDP submodule; no-op for
    unsharded models. Turning it off around a multi-forward no-grad loop (the DMD
    rollout) keeps params gathered instead of re-all-gathering the whole model on every
    forward. Turn it back on before any weight update or grad forward —
    ``reshard_now=True`` frees the gathered params immediately. While resharding is off,
    weight updates would NOT propagate to the cached unsharded params, so only use this
    around loops that don't touch weights."""
    from torch.distributed.fsdp import FSDPModule

    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.set_reshard_after_forward(value)
            if reshard_now:
                m.reshard()


def fsdp2_shard_model(model, accelerator, *, transformer_cls_names_to_wrap=None):
    """FSDP2-shard a model in place with the plugin's policy (rank-0 weights are scattered
    to all ranks via ``cpu_ram_efficient_loading``); no-op without FSDP2.

    Used for every same-arch transformer (student/fake/real), instead of
    ``accelerator.prepare``: prepare() only supports one model+optimizer per call and remaps
    optimizer params afterwards, which doesn't fit DMD's multi-model / frozen-teacher setup.
    Trainable models must be sharded BEFORE their optimizer is built so the optimizer holds
    the live DTensor params. ``transformer_cls_names_to_wrap`` overrides the plugin's block
    classes for this call ([] = shard the root only)."""
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None or getattr(plugin, "fsdp_version", None) != 2:
        return model

    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

    original_auto_wrap_policy = plugin.auto_wrap_policy
    original_transformer_cls_names_to_wrap = plugin.transformer_cls_names_to_wrap
    try:
        if transformer_cls_names_to_wrap is not None:
            plugin.transformer_cls_names_to_wrap = transformer_cls_names_to_wrap
        return fsdp2_prepare_model(accelerator, model)
    finally:
        plugin.auto_wrap_policy = original_auto_wrap_policy
        plugin.transformer_cls_names_to_wrap = original_transformer_cls_names_to_wrap


def fsdp2_load_full_state(model, full_sd: dict):
    """Load a full (unsharded) state dict into a live FSDP2-sharded model. Every rank
    passes the full sd it loaded and DCP takes the rank's own shard of each tensor.

    strict=False is required for DCP's full-state scatter, so guard against silently
    resuming from a mismatched checkpoint by checking key coverage ourselves."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    missing = [k for k in model.state_dict() if k not in full_sd]
    if missing:
        raise ValueError(
            f"FSDP2 full-state load: checkpoint is missing {len(missing)} model keys "
            f"(e.g. {missing[:5]}) — refusing to silently resume from a mismatched checkpoint"
        )
    set_model_state_dict(
        model,
        full_sd,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False, strict=False),
    )


def compile_transformer(transformer, transformer_unwrapped, regional: bool = False, **compile_kwargs):
    """Compile the transformer in-place.

    `regional=True` compiles each block runner individually (cheaper recompile on shape
    changes, checkpoint kept inside the compiled region — see build_block_runners);
    otherwise the full module is compiled.
    """
    torch._dynamo.config.recompile_limit = 64
    if regional:
        for i in range(len(transformer_unwrapped.noise_refiner)):
            transformer_unwrapped.noise_refiner[i] = torch.compile(
                transformer_unwrapped.noise_refiner[i], **compile_kwargs
            )
        for i in range(len(transformer_unwrapped.layers)):
            transformer_unwrapped.layers[i] = torch.compile(
                transformer_unwrapped.layers[i], **compile_kwargs
            )
    else:
        transformer.compile(**compile_kwargs)
