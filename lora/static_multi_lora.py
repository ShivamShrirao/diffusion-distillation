"""Static multi-adapter LoRA used by DMD.

Adapter choice is a tuple of Python floats (`lora_scales`) threaded through
block forwards, so checkpoint recompute gets the same adapter scales as the
original forward. Zero-scaled adapters are skipped entirely in the forward:
the branch is on a Python float, so torch.compile just specializes per
scale-set (scales only flip between a few fixed configurations, so the
recompile count is bounded).
"""

from __future__ import annotations

import os
import re
from typing import Iterable

import torch
import torch.nn as nn
from peft import LoraConfig
from safetensors.torch import load_file, save_file


class StaticMultiLoraLinear(nn.Module):
    """Frozen linear + N PEFT-shaped LoRA adapters gated by `_lora_scales`."""

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_names: Iterable[str],
        rank: int,
        alpha: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.adapter_names = tuple(adapter_names)
        eff_alpha = alpha if (alpha is not None and alpha > 0) else 2 * rank
        in_dim, out_dim = base_layer.in_features, base_layer.out_features

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        scaling_vals = []
        for name in self.adapter_names:
            a = nn.Linear(in_dim, rank, bias=False)
            b = nn.Linear(rank, out_dim, bias=False)
            nn.init.normal_(a.weight, std=1.0 / rank)
            nn.init.zeros_(b.weight)
            self.lora_A[name] = a
            self.lora_B[name] = b
            self.lora_dropout[name] = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            scaling_vals.append(float(eff_alpha) / float(rank))
        self.scaling = tuple(scaling_vals)
        self._lora_scales: tuple[float, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        result_dtype = result.dtype
        scales = self._lora_scales
        if scales is None:
            return result
        for i, name in enumerate(self.adapter_names):
            # Python-float branch: zero-scaled adapters cost nothing, and no
            # zero grads are written to their params (grad stays None).
            coeff = scales[i] * self.scaling[i]
            if coeff == 0.0:
                continue
            xi = x.to(self.lora_A[name].weight.dtype)
            xi = self.lora_dropout[name](xi)
            result = result + self.lora_B[name](self.lora_A[name](xi)) * coeff
        return result.to(result_dtype)


def _root(transformer):
    return transformer.module if hasattr(transformer, "module") else transformer


def inject_multi_lora(
    transformer: nn.Module,
    target_modules: Iterable[str],
    adapter_names: Iterable[str],
    rank: int,
    alpha: int | None = None,
    dropout: float = 0.0,
) -> list[str]:
    """Wrap target `nn.Linear`s with `StaticMultiLoraLinear`; run pre-FSDP."""
    root = _root(transformer)
    target_suffixes = tuple(target_modules)
    candidates = [
        (name, mod)
        for name, mod in root.named_modules()
        if isinstance(mod, nn.Linear)
        and not isinstance(mod, StaticMultiLoraLinear)
        and any(name == suffix or name.endswith("." + suffix) for suffix in target_suffixes)
    ]
    replaced: list[str] = []
    for path, mod in candidates:
        root.set_submodule(path, StaticMultiLoraLinear(mod, adapter_names, rank, alpha=alpha, dropout=dropout))
        replaced.append(path)
    return replaced


def make_scales(active_adapter: str | None, transformer: nn.Module) -> tuple[float, ...]:
    names = None
    for sub in _root(transformer).modules():
        if isinstance(sub, StaticMultiLoraLinear):
            names = sub.adapter_names
            break
    if names is None:
        raise RuntimeError("No StaticMultiLoraLinear found; call inject_multi_lora first.")
    if active_adapter is not None and active_adapter not in names:
        raise ValueError(f"Unknown adapter '{active_adapter}'; expected one of {names}")
    return tuple(1.0 if name == active_adapter else 0.0 for name in names)


def lora_scales_pre_hook(module, args, kwargs):
    """Pre-hook: applies kwargs["lora_scales"] to every StaticMultiLoraLinear
    inside `module`, then strips the kwarg so the block's real forward runs
    unchanged. This is THE only way to set scales — `lora_scales` flows in
    as a forward input, so autograd's checkpoint replay re-applies the same
    scales on recompute (no mutable-state corruption across multiple grad
    forwards in one step)."""
    if "lora_scales" not in kwargs:
        return None
    scales = kwargs["lora_scales"]
    for sub in module.modules():
        if isinstance(sub, StaticMultiLoraLinear):
            sub._lora_scales = scales
    return args, {k: v for k, v in kwargs.items() if k != "lora_scales"}


# PEFT bridge: PEFT stores one adapter per file without adapter name in keys;
# this module stores all adapters together as lora_A/B.{adapter}.weight.

_PEFT_PREFIX = "base_model.model."


_FSDP_WRAPPED_RE = re.compile(r"(^|\.)_fsdp_wrapped_")


def _clean_key(key: str) -> str:
    key = key.replace("module.", "").replace("._orig_mod", "")
    return _FSDP_WRAPPED_RE.sub(r"\1", key)


def _peft_key_to_own(peft_key: str, adapter_name: str) -> str | None:
    if not peft_key.startswith(_PEFT_PREFIX):
        return None
    path = peft_key[len(_PEFT_PREFIX):]
    for kind in (".lora_A.weight", ".lora_B.weight"):
        if path.endswith(kind):
            letter = kind[6]  # 'A' or 'B'
            return f"{path[:-len(kind)]}.lora_{letter}.{adapter_name}.weight"
    return None


def _own_key_to_peft(own_key: str, adapter_name: str) -> str | None:
    for letter in ("A", "B"):
        suffix = f".lora_{letter}.{adapter_name}.weight"
        if own_key.endswith(suffix):
            return f"{_PEFT_PREFIX}{own_key[:-len(suffix)]}.lora_{letter}.weight"
    return None


def _adapter_state_keys(transformer: nn.Module, adapter_name: str) -> set[str]:
    return {
        f"{f'{_clean_key(prefix)}.' if prefix else ''}lora_{letter}.{adapter_name}.weight"
        for prefix, sub in _root(transformer).named_modules()
        if isinstance(sub, StaticMultiLoraLinear)
        for letter in ("A", "B")
    }


def _normalise_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_clean_key(k): v for k, v in state_dict.items()}


def load_peft_adapter(
    transformer: nn.Module,
    ckpt_dir: str,
    adapter_name: str,
    in_subdir: str | None = None,
) -> None:
    sub = adapter_name if in_subdir is None else in_subdir
    sd_path = os.path.join(ckpt_dir, sub, "adapter_model.safetensors")
    if not os.path.exists(sd_path):
        raise FileNotFoundError(sd_path)
    peft_sd = load_file(sd_path)
    own_sd = {new_k: v for k, v in peft_sd.items() if (new_k := _peft_key_to_own(k, adapter_name))}
    if not own_sd:
        raise RuntimeError(f"No LoRA keys recognised in {sd_path}")
    if missing := sorted(_adapter_state_keys(transformer, adapter_name) - set(own_sd)):
        raise RuntimeError(f"Missing keys when loading adapter '{adapter_name}': {', '.join(missing[:5])}")
    _, unexpected = _root(transformer).load_state_dict(own_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading adapter '{adapter_name}': {unexpected[:5]}")


def build_peft_lora_config(transformer: nn.Module, adapter_name: str) -> LoraConfig:
    """Reconstruct a PEFT LoraConfig from the injected StaticMultiLoraLinear modules.

    target_modules uses the dotted suffix from LORA_TARGET_MODULES (the same list
    inject_multi_lora used). The last path segment alone (e.g. ``ff.net.2`` -> ``"2"``)
    collides with too many unrelated modules and breaks PEFT's load_lora.
    """
    from lora.adapters import LORA_TARGET_MODULES

    rank = alpha = None
    dropout = 0.0
    targets: set[str] = set()
    for prefix, sub in _root(transformer).named_modules():
        if not isinstance(sub, StaticMultiLoraLinear):
            continue
        idx = sub.adapter_names.index(adapter_name)
        r = sub.lora_A[adapter_name].out_features
        a = int(round(sub.scaling[idx] * r))
        d_mod = sub.lora_dropout[adapter_name]
        d = float(getattr(d_mod, "p", 0.0)) if not isinstance(d_mod, nn.Identity) else 0.0
        if rank is None:
            rank, alpha, dropout = r, a, d
        cleaned = _clean_key(prefix)
        match = next(
            (suffix for suffix in LORA_TARGET_MODULES if cleaned == suffix or cleaned.endswith("." + suffix)),
            None,
        )
        if match is None:
            raise RuntimeError(f"StaticMultiLoraLinear at '{cleaned}' doesn't match any LORA_TARGET_MODULES suffix")
        targets.add(match)
    if rank is None:
        raise RuntimeError(f"No StaticMultiLoraLinear modules found for adapter '{adapter_name}'")
    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=sorted(targets))


def save_peft_adapter(
    transformer: nn.Module,
    ckpt_dir: str,
    adapter_name: str,
    state_dict: dict[str, torch.Tensor] | None = None,
    out_subdir: str | None = None,
) -> None:
    own_sd = _normalise_state_dict_keys(state_dict if state_dict is not None else _root(transformer).state_dict())
    if missing := sorted(_adapter_state_keys(transformer, adapter_name) - set(own_sd)):
        raise RuntimeError(f"Missing keys when saving adapter '{adapter_name}': {', '.join(missing[:5])}")
    peft_sd = {
        new_k: v.detach().cpu().contiguous()
        for k, v in own_sd.items()
        if (new_k := _own_key_to_peft(k, adapter_name))
    }
    if not peft_sd:
        raise RuntimeError(f"No LoRA keys found for adapter '{adapter_name}'")
    sub = adapter_name if out_subdir is None else out_subdir
    out_dir = os.path.join(ckpt_dir, sub)
    os.makedirs(out_dir, exist_ok=True)
    save_file(peft_sd, os.path.join(out_dir, "adapter_model.safetensors"))
    build_peft_lora_config(transformer, adapter_name).save_pretrained(out_dir)
