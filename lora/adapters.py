"""PEFT-based LoRA helpers for non-DMD trainers.

DMD training uses the static multi-adapter wrapper in
``lora/static_multi_lora.py`` — that path does NOT go through PEFT and does
not use these helpers. Everything here is the stock PEFT API for trainers
(DPO, draft, kl_lora, etc.) that need a single adapter.
"""

from contextlib import contextmanager

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer


LORA_TARGET_MODULES = (
    # Attention (self and cross) and SwiGLU feed-forward of every block
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)


def _resolve_lora_alpha(lora_rank: int, lora_alpha: int | None) -> int:
    if lora_alpha is None or lora_alpha < 0:
        return 2 * lora_rank
    return lora_alpha


def add_lora(transformer, lora_rank: int, adapter_name: str = "default", lora_alpha: int | None = None):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=_resolve_lora_alpha(lora_rank, lora_alpha),
        init_lora_weights="gaussian",
        target_modules=list(LORA_TARGET_MODULES),
    )
    return get_peft_model(transformer, lora_config, adapter_name=adapter_name)


def add_lora_adapter(transformer, lora_rank: int, adapter_name: str, lora_alpha: int | None = None):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=_resolve_lora_alpha(lora_rank, lora_alpha),
        init_lora_weights="gaussian",
        target_modules=list(LORA_TARGET_MODULES),
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


def _root_module(transformer):
    return transformer.module if hasattr(transformer, "module") else transformer


@contextmanager
def disable_adapter(transformer):
    """Disable all LoRA adapters for the duration of the context (no-grad
    paths only — used by ref-model evaluations in kl_lora / DPO). Flips
    ``_disable_adapters`` on every PEFT LoraLayer and restores on exit.
    """
    root = _root_module(transformer)
    layers = [m for m in root.modules() if isinstance(m, LoraLayer)]
    for layer in layers:
        layer._disable_adapters = True
    try:
        yield
    finally:
        for layer in layers:
            layer._disable_adapters = False
