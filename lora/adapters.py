from contextlib import contextmanager

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer


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
    """Set the active LoRA adapter without touching ``requires_grad``.

    PEFT's ``PeftModel.set_adapter`` flips ``requires_grad=False`` on every adapter
    that's not the new active one — and PyTorch's ``AccumulateGrad`` checks
    ``requires_grad`` at backward time, so a mid-step switch can silently drop the
    inactive adapter's gradients (DMD's student/fake setup hit this). We mutate
    ``_active_adapter`` directly on each ``LoraLayer`` instead: same forward routing,
    no grad mutation.
    """
    module = transformer.module if hasattr(transformer, "module") else transformer
    target = [adapter_name]
    for sub in module.modules():
        if isinstance(sub, LoraLayer):
            sub._active_adapter = target


@contextmanager
def disable_adapter(transformer):
    """Context manager: run with all LoRA adapters disabled (skips lora_A/lora_B matmul).

    PEFT's ``PeftModel.disable_adapter`` also flips ``requires_grad=False`` on all
    adapter params on entry and only restores grads on the *active* adapter on exit —
    that's the same silent-grad-drop hazard as ``set_adapter``. We just toggle
    ``LoraLayer._disable_adapters`` directly.
    """
    module = transformer.module if hasattr(transformer, "module") else transformer
    layers = [m for m in module.modules() if isinstance(m, LoraLayer)]
    for layer in layers:
        layer._disable_adapters = True
    try:
        yield
    finally:
        for layer in layers:
            layer._disable_adapters = False
