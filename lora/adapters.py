from peft import LoraConfig, PeftModel, get_peft_model


_TARGET_MODULES = [
    # Attention (self and cross) and SwiGLU feed-forward of every block
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
]


def has_lora(transformer, adapter_name: str = "default") -> bool:
    return isinstance(transformer, PeftModel) and adapter_name in getattr(transformer, "peft_config", {})


def add_lora(transformer, lora_rank: int, adapter_name: str = "default"):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        init_lora_weights="gaussian",
        target_modules=_TARGET_MODULES,
    )
    return get_peft_model(transformer, lora_config, adapter_name=adapter_name)


def add_lora_adapter(transformer, lora_rank: int, adapter_name: str):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        init_lora_weights="gaussian",
        target_modules=_TARGET_MODULES,
    )
    transformer.add_adapter(adapter_name, lora_config)
    return transformer


def load_lora(transformer, input_dir: str, is_trainable: bool = False, adapter_name: str = "default"):
    return PeftModel.from_pretrained(
        transformer,
        input_dir,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        low_cpu_mem_usage=True,
    )


def set_adapter(transformer, adapter_name: str):
    """Set the active LoRA adapter, unwrapping FSDP/DDP if needed."""
    module = transformer.module if hasattr(transformer, "module") else transformer
    if hasattr(module, "set_adapter"):
        module.set_adapter(adapter_name)


def disable_adapter(transformer):
    """Context manager: run with all LoRA adapters disabled (skips lora_A/lora_B matmul).

    Use this instead of ``joint_attention_kwargs["scale"] = 0.0`` — it produces a leaner
    compiled graph (no zero-multiplied lora compute) and is compile-cache-friendly
    because Dynamo guards on the bool ``disable_adapters`` rather than a float scale.
    """
    module = transformer.module if hasattr(transformer, "module") else transformer
    return module.disable_adapter()
