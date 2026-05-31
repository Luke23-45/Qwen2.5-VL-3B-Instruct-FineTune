from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..core.types import ModelPaths


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def optional_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def torch_dtype(torch_module: Any, name: str | None) -> Any:
    if not name or str(name).lower() in {"auto", "none", "null"}:
        return "auto"
    return getattr(torch_module, str(name))


def resolve_model_paths(cfg: DictConfig) -> ModelPaths:
    model_cfg = cfg.model
    artifact_dir = optional_path(cfg_get(model_cfg, "artifact_dir"))
    adapter_path = optional_path(cfg_get(model_cfg, "adapter_path")) or artifact_dir
    processor_path = optional_path(cfg_get(model_cfg, "processor_path")) or artifact_dir
    return ModelPaths(
        base_model=str(cfg_get(model_cfg, "name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct")),
        artifact_dir=artifact_dir,
        adapter_path=adapter_path,
        processor_path=processor_path,
    )


def validate_model_paths(paths: ModelPaths) -> None:
    if paths.artifact_dir and not paths.artifact_dir.exists():
        raise FileNotFoundError(f"Inference artifact_dir does not exist: {paths.artifact_dir}")
    if paths.adapter_path and not (paths.adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter path is missing adapter_config.json: {paths.adapter_path}")
    if paths.processor_path and not (paths.processor_path / "processor_config.json").exists():
        raise FileNotFoundError(f"Processor path is missing processor_config.json: {paths.processor_path}")


def quantization_config(cfg: DictConfig, torch_module: Any) -> Any | None:
    quant_cfg = cfg_get(cfg, "quantization")
    if not bool(cfg_get(quant_cfg, "enabled", False)):
        return None

    from transformers import BitsAndBytesConfig

    bits = int(cfg_get(quant_cfg, "bits", 4))
    if bits != 4:
        raise ValueError(f"Only 4-bit inference quantization is supported by this config; got bits={bits}.")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg_get(quant_cfg, "quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg_get(quant_cfg, "double_quant", True)),
        bnb_4bit_compute_dtype=torch_dtype(torch_module, cfg_get(quant_cfg, "compute_dtype", "bfloat16")),
    )
