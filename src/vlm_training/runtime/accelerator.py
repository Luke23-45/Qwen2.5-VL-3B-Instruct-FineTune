from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


FLASH_KERNEL_BACKEND = "kernels-community/flash-attn2@>=2.0,<3.0"
SDPA_BACKEND = "sdpa"
EAGER_BACKEND = "eager"


@dataclass(slots=True)
class GPUInfo:
    available: bool
    device_name: str | None
    major: int | None
    minor: int | None
    total_memory_gb: float | None
    cuda_version: str | None
    platform_system: str
    platform_machine: str
    bf16_supported: bool
    tf32_supported: bool


@dataclass(slots=True)
class RuntimePlan:
    state: str
    attention_backend: str
    attention_reason: str
    use_liger_kernel: bool
    liger_reason: str
    torch_compile_enabled: bool
    torch_compile_reason: str
    tf32_enabled: bool
    bf16_enabled: bool
    fp16_enabled: bool
    torch_dtype: str
    quant_compute_dtype: str
    dataloader_num_workers: int
    dataloader_persistent_workers: bool
    install_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gpu: GPUInfo | None = None


def _get(cfg: DictConfig, *keys: str, default=None):
    current = cfg
    for key in keys:
        if not isinstance(current, DictConfig) or key not in current:
            return default
        current = current[key]
    return current


def module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def detect_gpu_info() -> GPUInfo:
    available = torch.cuda.is_available()
    system = platform.system().lower()
    machine = platform.machine().lower()
    cuda_version = torch.version.cuda

    if not available:
        return GPUInfo(
            available=False,
            device_name=None,
            major=None,
            minor=None,
            total_memory_gb=None,
            cuda_version=cuda_version,
            platform_system=system,
            platform_machine=machine,
            bf16_supported=False,
            tf32_supported=False,
        )

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    major = int(props.major)
    minor = int(props.minor)
    total_memory_gb = round(props.total_memory / (1024**3), 2)
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    tf32_supported = major >= 8 and cuda_version is not None
    return GPUInfo(
        available=True,
        device_name=props.name,
        major=major,
        minor=minor,
        total_memory_gb=total_memory_gb,
        cuda_version=cuda_version,
        platform_system=system,
        platform_machine=machine,
        bf16_supported=bf16_supported,
        tf32_supported=tf32_supported,
    )


def _can_use_flash_attention(gpu: GPUInfo) -> bool:
    if not gpu.available or gpu.major is None:
        return False
    # FlashAttention-2 officially supports Ampere, Ada, and Hopper.
    return gpu.major >= 8 and gpu.platform_system == "linux" and gpu.platform_machine in {"x86_64", "amd64"}


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 2
    # Image decoding workers can consume a lot of host RAM. Keep enough CPU
    # parallelism to feed the GPU without letting prefetch queues dominate RAM.
    return max(1, min(8, cpu_count))


def _run_pip_install(packages: list[str], timeout_sec: int) -> None:
    work_root = Path(os.environ.get("KV_RUNTIME_PIP_ROOT", Path(tempfile.gettempdir()) / "krishivaidya-runtime-pip")).resolve()
    tmp_root = work_root / "tmp"
    cache_root = work_root / "cache"
    build_root = work_root / "build"
    for path in (tmp_root, cache_root, build_root):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_root)
    env["TEMP"] = str(tmp_root)
    env["TMP"] = str(tmp_root)
    env["PIP_CACHE_DIR"] = str(cache_root)
    command = [sys.executable, "-m", "pip", "install", "-U", *packages]
    logger.info("Installing runtime package(s): %s", " ".join(packages))
    subprocess.run(command, check=True, timeout=timeout_sec, cwd=build_root, env=env)


def _ensure_package(module_name: str, packages: list[str], install_actions: list[str], timeout_sec: int) -> bool:
    if module_available(module_name):
        return True
    try:
        _run_pip_install(packages, timeout_sec)
    except Exception as exc:
        install_actions.append(f"failed:{' '.join(packages)}:{exc}")
        return False
    install_actions.append(f"installed:{' '.join(packages)}")
    return module_available(module_name)


def build_runtime_plan(cfg: DictConfig, gpu: GPUInfo | None = None) -> RuntimePlan:
    runtime_cfg = _get(cfg, "runtime", default=None)
    gpu = gpu or detect_gpu_info()
    warnings: list[str] = []
    install_actions: list[str] = []

    auto_install = bool(_get(runtime_cfg, "auto_install", default=True))
    install_liger_mode = str(_get(runtime_cfg, "install_liger_kernel", default="auto"))
    prefer_hub_kernels = bool(_get(runtime_cfg, "prefer_hub_kernels", default=True))
    torch_compile_mode = str(_get(runtime_cfg, "torch_compile_mode", default="auto"))
    pip_timeout_sec = int(_get(runtime_cfg, "pip_timeout_sec", default=1800))

    bf16_enabled = bool(getattr(cfg.training, "bf16", False))
    fp16_enabled = bool(getattr(cfg.training, "fp16", False))
    torch_dtype = str(cfg.model.torch_dtype)
    quant_compute_dtype = str(cfg.quantization.compute_dtype)

    if bf16_enabled and not gpu.bf16_supported:
        bf16_enabled = False
        fp16_enabled = True
        torch_dtype = "float16"
        quant_compute_dtype = "float16"
        warnings.append("GPU does not support bf16. Falling back to fp16/float16.")

    tf32_enabled = bool(_get(cfg.training, "tf32", default=True)) and gpu.tf32_supported
    if bool(_get(cfg.training, "tf32", default=True)) and not gpu.tf32_supported:
        warnings.append("TF32 requested but not supported on this GPU. Disabling TF32.")

    attention_backend = SDPA_BACKEND if gpu.available else EAGER_BACKEND
    attention_reason = "No CUDA GPU detected; using eager attention." if not gpu.available else "Using PyTorch SDPA as the safe CUDA baseline."
    state = "fallback"

    if _can_use_flash_attention(gpu):
        state = "flash_candidate"
        if prefer_hub_kernels:
            kernels_ok = module_available("kernels")
            if kernels_ok:
                attention_backend = FLASH_KERNEL_BACKEND
                attention_reason = "Ampere-or-newer Linux GPU detected; using Hugging Face Hub FlashAttention kernel."
                state = "flash_kernels"
            else:
                warnings.append("Kernels package is not already installed. Skipping auto-install for kernels to avoid core Hugging Face dependency conflicts.")

        if attention_backend == SDPA_BACKEND:
            flash_attn_ok = module_available("flash_attn")
            if not flash_attn_ok and auto_install:
                flash_attn_ok = _ensure_package(
                    "flash_attn",
                    ["flash-attn>=2.7.4.post1", "--no-build-isolation"],
                    install_actions,
                    pip_timeout_sec,
                )
            if flash_attn_ok:
                attention_backend = "flash_attention_2"
                attention_reason = "Ampere-or-newer Linux GPU detected; using flash-attn Python package."
                state = "flash_attn_package"
            else:
                warnings.append("flash-attn unavailable after installation attempt. Falling back to SDPA.")
    elif gpu.available:
        warnings.append("GPU architecture is below Ampere or platform is unsupported for FlashAttention. Using SDPA fallback.")

    use_liger_kernel = bool(_get(cfg.training, "use_liger_kernel", default=False))
    liger_reason = "Disabled in config."
    if str(_get(runtime_cfg, "liger_mode", default="auto")) == "auto":
        use_liger_kernel = attention_backend != EAGER_BACKEND and gpu.available
        liger_reason = "Enabled automatically for CUDA training." if use_liger_kernel else "Disabled because CUDA acceleration is unavailable."

    if use_liger_kernel and not module_available("liger_kernel"):
        if auto_install and install_liger_mode in {"auto", "always"}:
            if gpu.platform_system != "linux":
                liger_ok = False
                warnings.append("Skipping liger-kernel auto-install because this platform is not Linux.")
            else:
                liger_ok = _ensure_package("liger_kernel", ["liger-kernel"], install_actions, pip_timeout_sec)
            use_liger_kernel = liger_ok
            liger_reason = "Installed liger-kernel automatically." if liger_ok else "liger-kernel installation failed; disabling."
            if not liger_ok:
                warnings.append("Could not install liger-kernel automatically. Continuing without it.")
        else:
            use_liger_kernel = False
            liger_reason = "liger-kernel is not installed."

    worker_count = int(_get(cfg.training, "dataloader_num_workers", default=_default_num_workers()) or 0)
    if worker_count <= 0:
        worker_count = _default_num_workers()

    torch_compile_enabled = bool(_get(cfg.training, "torch_compile", default=False))
    torch_compile_reason = "Disabled in config."
    if torch_compile_mode == "always":
        torch_compile_enabled = bool(gpu.available)
        torch_compile_reason = "Forced on by runtime.torch_compile_mode=always." if gpu.available else "Requested, but no CUDA GPU is available."
    elif torch_compile_mode == "off":
        torch_compile_enabled = False
        torch_compile_reason = "Disabled by runtime.torch_compile_mode=off."
    elif torch_compile_mode == "auto":
        quantized = bool(_get(cfg.quantization, "enabled", default=False))
        if gpu.available and gpu.major is not None and gpu.major >= 8 and not quantized:
            torch_compile_enabled = True
            torch_compile_reason = "Enabled automatically on Ampere-or-newer GPU without quantization."
        elif quantized:
            torch_compile_enabled = False
            torch_compile_reason = "Disabled automatically because 4-bit quantization is enabled and torch.compile is less reliable in this stack."
        else:
            torch_compile_enabled = False
            torch_compile_reason = "Disabled automatically because the GPU/runtime is not a strong compile candidate."

    return RuntimePlan(
        state=state,
        attention_backend=attention_backend,
        attention_reason=attention_reason,
        use_liger_kernel=use_liger_kernel,
        liger_reason=liger_reason,
        torch_compile_enabled=torch_compile_enabled,
        torch_compile_reason=torch_compile_reason,
        tf32_enabled=tf32_enabled,
        bf16_enabled=bf16_enabled,
        fp16_enabled=fp16_enabled,
        torch_dtype=torch_dtype,
        quant_compute_dtype=quant_compute_dtype,
        dataloader_num_workers=worker_count,
        dataloader_persistent_workers=worker_count > 0,
        install_actions=install_actions,
        warnings=warnings,
        gpu=gpu,
    )


def apply_runtime_plan(cfg: DictConfig, plan: RuntimePlan) -> None:
    cfg.model.attn_implementation = plan.attention_backend
    cfg.model.torch_dtype = plan.torch_dtype
    cfg.quantization.compute_dtype = plan.quant_compute_dtype
    cfg.training.bf16 = plan.bf16_enabled
    cfg.training.fp16 = plan.fp16_enabled
    cfg.training.tf32 = plan.tf32_enabled
    cfg.training.use_liger_kernel = plan.use_liger_kernel
    cfg.training.torch_compile = plan.torch_compile_enabled
    cfg.training.dataloader_num_workers = plan.dataloader_num_workers
    cfg.training.dataloader_persistent_workers = plan.dataloader_persistent_workers

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = plan.tf32_enabled
        torch.backends.cudnn.allow_tf32 = plan.tf32_enabled
        matmul_precision = str(_get(cfg.training, "matmul_precision", default="high"))
        if plan.tf32_enabled and matmul_precision:
            torch.set_float32_matmul_precision(matmul_precision)


def write_runtime_report(plan: RuntimePlan, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    return path


def prepare_runtime(cfg: DictConfig, report_path: str | Path | None = None) -> RuntimePlan:
    plan = build_runtime_plan(cfg)
    apply_runtime_plan(cfg, plan)
    if report_path is not None:
        write_runtime_report(plan, report_path)
    return plan
