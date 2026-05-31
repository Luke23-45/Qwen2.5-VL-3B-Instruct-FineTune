from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .settings import cfg_get


logger = logging.getLogger(__name__)


def configure_torch_runtime(cfg: DictConfig, torch_module: Any):
    runtime_cfg = cfg_get(cfg, "runtime")
    backend = cfg_get(runtime_cfg, "backend", "transformers_peft")
    if backend != "transformers_peft":
        raise ValueError(f"Unsupported inference backend '{backend}'. Use backend=transformers_peft for this runner.")

    plan = None
    if bool(cfg_get(runtime_cfg, "auto_prepare", True)):
        plan = _prepare_inference_runtime(cfg, report_path=cfg_get(runtime_cfg, "report_path"))

    if bool(cfg_get(runtime_cfg, "tf32", True)) and torch_module.cuda.is_available():
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True

    _patch_transformers_flash_attn_mapping()
    return plan


def _prepare_inference_runtime(cfg: DictConfig, report_path: str | Path | None = None):
    from vlm_training.runtime.accelerator import prepare_runtime

    shim_cfg = _runtime_shim_config(cfg)
    plan = prepare_runtime(shim_cfg, report_path=report_path)

    cfg.model.attn_implementation = shim_cfg.model.attn_implementation
    cfg.model.torch_dtype = shim_cfg.model.torch_dtype
    cfg.quantization.compute_dtype = shim_cfg.quantization.compute_dtype
    cfg.runtime.tf32 = bool(shim_cfg.training.tf32)
    cfg.runtime.num_workers = int(shim_cfg.training.dataloader_num_workers)

    logger.info(
        "Inference runtime plan: state=%s, attention=%s, tf32=%s, dtype=%s",
        plan.state,
        plan.attention_backend,
        plan.tf32_enabled,
        plan.torch_dtype,
    )
    for warning in plan.warnings:
        logger.warning("Inference runtime fallback: %s", warning)
    return plan


def _runtime_shim_config(cfg: DictConfig) -> DictConfig:
    runtime_cfg = cfg.runtime
    quant_cfg = cfg.quantization
    model_cfg = cfg.model
    return OmegaConf.create(
        {
            "model": {
                "attn_implementation": cfg_get(model_cfg, "attn_implementation", "auto"),
                "torch_dtype": cfg_get(model_cfg, "torch_dtype", "bfloat16"),
            },
            "quantization": {
                "enabled": bool(cfg_get(quant_cfg, "enabled", True)),
                "compute_dtype": cfg_get(quant_cfg, "compute_dtype", "bfloat16"),
            },
            "training": {
                "bf16": str(cfg_get(model_cfg, "torch_dtype", "bfloat16")) == "bfloat16",
                "fp16": str(cfg_get(model_cfg, "torch_dtype", "bfloat16")) == "float16",
                "tf32": bool(cfg_get(runtime_cfg, "tf32", True)),
                "use_liger_kernel": False,
                "dataloader_num_workers": int(cfg_get(runtime_cfg, "num_workers", 0) or 0),
                "dataloader_persistent_workers": False,
                "matmul_precision": str(cfg_get(runtime_cfg, "matmul_precision", "high")),
                "torch_compile": False,
            },
            "runtime": {
                "auto_install": bool(cfg_get(runtime_cfg, "auto_install", True)),
                "prefer_hub_kernels": bool(cfg_get(runtime_cfg, "prefer_hub_kernels", True)),
                "install_liger_kernel": str(cfg_get(runtime_cfg, "install_liger_kernel", "auto")),
                "liger_mode": str(cfg_get(runtime_cfg, "liger_mode", "auto")),
                "torch_compile_mode": str(cfg_get(runtime_cfg, "torch_compile_mode", "auto")),
                "pip_timeout_sec": int(cfg_get(runtime_cfg, "pip_timeout_sec", 1800)),
            },
        }
    )


def _patch_transformers_flash_attn_mapping() -> None:
    try:
        from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    except Exception:
        return

    if "flash_attn" in PACKAGE_DISTRIBUTION_MAPPING:
        return

    candidates = []
    for dist_name in ("flash-attn", "flash_attn"):
        try:
            importlib_metadata.version(dist_name)
            candidates.append(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue

    if candidates:
        PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] = tuple(candidates)
        logger.info("Patched transformers flash_attn package mapping: %s", candidates)
