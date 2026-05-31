"""Distributed-training setup: NCCL env, Accelerator, transformer compile.

`set_nccl_env()` must be called BEFORE the process group is initialised —
`BaseTrainer.run()` calls it first thing, so entrypoints don't need to do anything.
"""

import functools
import os
from datetime import timedelta

import diffusers
import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration
from huggingface_hub import login
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
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
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            reshard_after_forward="FULL_SHARD",
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={BriaFibo2TransformerBlock},
            ),
            mixed_precision_policy=MixedPrecision(
                param_dtype=weight_dtype,
                reduce_dtype=torch.float32,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sync_module_states=True,
            use_orig_params=args.lora_rank > 0,
            limit_all_gathers=True,
            # Materialise meta-device params on the local GPU before sharding.
            param_init_fn=lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False),
        )
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(1000))]
        print("Using fsdp")
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


def compile_transformer(transformer, transformer_unwrapped, regional: bool = False, **compile_kwargs):
    """Compile the transformer in-place.

    `regional=True` compiles each transformer block individually (cheaper recompile on
    shape changes); otherwise the full module is compiled.
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
