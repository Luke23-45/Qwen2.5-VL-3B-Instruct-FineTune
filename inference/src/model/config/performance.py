from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from .settings import cfg_get


def resolve_batch_size(cfg: DictConfig, runtime_plan: Any | None = None) -> int:
    configured = cfg_get(cfg.runtime, "batch_size", 1)
    if isinstance(configured, int):
        return max(configured, 1)

    configured_str = str(configured).strip().lower()
    if configured_str not in {"auto", "default"}:
        return max(int(configured), 1)

    gpu = getattr(runtime_plan, "gpu", None)
    total_memory_gb = getattr(gpu, "total_memory_gb", None)
    quantized = bool(cfg_get(cfg.quantization, "enabled", True))

    if total_memory_gb is None:
        return 1
    if not quantized:
        if total_memory_gb >= 40:
            return 4
        if total_memory_gb >= 24:
            return 2
        return 1

    if total_memory_gb >= 40:
        return 8
    if total_memory_gb >= 20:
        return 4
    if total_memory_gb >= 12:
        return 2
    return 1
