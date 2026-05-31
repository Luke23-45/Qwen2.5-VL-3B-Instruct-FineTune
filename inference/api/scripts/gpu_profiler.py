#!/usr/bin/env python3
"""GPU profiler for vLLM inference — auto-configures engine arguments
based on detected GPU hardware.

Produces an :class:`InferencePlan` that maps hardware capabilities to optimal
``vllm serve`` arguments.  Follows the same pattern as the training-side
``vlm_training.runtime.accelerator`` module.

Auto-detects
-----------
- GPU compute capability (SM version)
- Total VRAM
- bf16 / fp8 / TF32 support
- Platform OS and architecture

Usage
-----
    from inference.api.scripts.gpu_profiler import GPUProfiler

    profiler = GPUProfiler()
    plan = profiler.build_plan()
    cli_args = plan.to_cli_args()
    print(plan.report())
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field

import torch


logger = logging.getLogger("krishivaidya.gpu_profiler")


# ── Helpers ────────────────────────────────────────────────────


def _gb(bytes_: int) -> float:
    return round(bytes_ / (1024 ** 3), 2)


def _cuda_version_ge(version_str: str | None, target_str: str) -> bool:
    if version_str is None:
        return False
    try:
        parts = [int(x) for x in version_str.split(".")]
        target = [int(x) for x in target_str.split(".")]
        while len(parts) < len(target):
            parts.append(0)
        return tuple(parts) >= tuple(target)
    except (ValueError, IndexError):
        return False


# ── GPUInfo ────────────────────────────────────────────────────


@dataclass(slots=True)
class GPUInfo:
    """Normalised description of the available GPU hardware.

    Mirrors the ``GPUInfo`` dataclass from
    ``vlm_training.runtime.accelerator`` but adds ``fp8_supported`` for
    KV-cache dtype selection.
    """
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
    fp8_supported: bool


# ── InferencePlan ──────────────────────────────────────────────


@dataclass(slots=True)
class InferencePlan:
    """GPU-tuned vLLM inference configuration.

    Every decision carries a ``*_reason`` string that explains *why* the
    value was chosen, enabling full transparency at startup.
    """

    # Attention
    attention_backend: str | None
    attention_reason: str

    # Data type
    dtype: str
    dtype_reason: str

    # KV cache
    kv_cache_dtype: str
    kv_cache_reason: str
    block_size: int
    block_size_reason: str

    # Memory utilisation
    gpu_memory_utilization: float
    gpu_memory_reason: str
    max_model_len: int
    max_model_len_reason: str

    # Multi-modal pixel budget
    min_pixels: int
    max_pixels: int
    pixel_reason: str
    mm_processor_cache_gb: float
    mm_cache_reason: str

    # Performance knobs
    enforce_eager: bool
    enforce_eager_reason: str
    optimization_level: int
    optimization_reason: str

    # Scheduling
    max_num_seqs: int
    max_num_seqs_reason: str
    max_num_batched_tokens: int
    batched_tokens_reason: str

    # Triton JIT warmup (environment variable)
    determinism_warmup: int
    warmup_reason: str

    # Diagnostics
    warnings: list[str] = field(default_factory=list)
    gpu: GPUInfo | None = field(default=None, compare=False)

    # ── Public API ────────────────────────────────────────────

    def to_cli_args(self) -> list[str]:
        """Render the plan as an argument list for ``vllm serve``.

        Returns a flat list of strings suitable for ``subprocess.run``.
        """
        args: list[str] = []

        args.extend(["--dtype", self.dtype])
        args.extend(["--kv-cache-dtype", self.kv_cache_dtype])
        args.extend(["--block-size", str(self.block_size)])
        args.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        args.extend(["--max-model-len", str(self.max_model_len)])
        args.extend(["--max-num-seqs", str(self.max_num_seqs)])
        args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])

        if self.attention_backend is not None:
            args.extend(["--attention-backend", self.attention_backend])

        if self.enforce_eager:
            args.append("--enforce-eager")
        elif self.optimization_level > 0:
            args.extend(["--optimization-level", str(self.optimization_level)])

        mm_kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "processor_cache_gb": self.mm_processor_cache_gb}
        args.extend(["--mm-processor-kwargs", json.dumps(mm_kwargs, separators=(",", ":"))])

        return args

    def env_vars(self) -> dict[str, str]:
        """Environment variables to propagate to the vLLM subprocess."""
        env: dict[str, str] = {}
        if self.determinism_warmup > 0:
            env["VLLM_DETERMINISM_WARMUP_ITERATIONS"] = str(self.determinism_warmup)
        return env

    def report(self) -> str:
        """Human-readable summary of the plan."""
        lines = [
            "=" * 72,
            " KrishiVaidya — vLLM GPU Profile Report",
            "=" * 72,
        ]
        if self.gpu:
            g = self.gpu
            lines.append(f"  GPU:            {g.device_name or 'N/A'}")
            lines.append(f"  Compute cap:    {g.major}.{g.minor}")
            lines.append(f"  VRAM:           {g.total_memory_gb} GiB" if g.total_memory_gb else "  VRAM:           N/A")
            lines.append(f"  CUDA toolkit:   {g.cuda_version or 'N/A'}")
            lines.append(f"  Platform:       {g.platform_system} / {g.platform_machine}")
            lines.append(f"  bf16:           {'yes' if g.bf16_supported else 'no'}")
            lines.append(f"  fp8:            {'yes' if g.fp8_supported else 'no'}")
            lines.append(f"  TF32:           {'yes' if g.tf32_supported else 'no'}")

        lines.extend([
            "",
            " Decisions:",
            f"  attention backend:    {self.attention_backend or 'vLLM-auto':20s}  ({self.attention_reason})",
            f"  dtype:                {self.dtype:20s}  ({self.dtype_reason})",
            f"  kv-cache-dtype:       {self.kv_cache_dtype:20s}  ({self.kv_cache_reason})",
            f"  block-size:           {self.block_size:<20d}  ({self.block_size_reason})",
            f"  gpu-memory-util:      {self.gpu_memory_utilization:<20.2f}  ({self.gpu_memory_reason})",
            f"  max-model-len:        {self.max_model_len:<20d}  ({self.max_model_len_reason})",
            f"  enforce-eager:        {str(self.enforce_eager):20s}  ({self.enforce_eager_reason})",
            f"  optimization-level:   {self.optimization_level:<20d}  ({self.optimization_reason})",
            f"  max-num-seqs:         {self.max_num_seqs:<20d}  ({self.max_num_seqs_reason})",
            f"  max-batched-tokens:   {self.max_num_batched_tokens:<20d}  ({self.batched_tokens_reason})",
            f"  pixels (min/max):     {self.min_pixels}/{self.max_pixels}  ({self.pixel_reason})",
            f"  mm-cache (GiB):       {self.mm_processor_cache_gb:<20.1f}  ({self.mm_cache_reason})",
            f"  warmup iterations:    {self.determinism_warmup:<20d}  ({self.warmup_reason})",
        ])

        if self.warnings:
            lines.extend(["", " Warnings:"])
            for w in self.warnings:
                lines.append(f"  !  {w}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def log(self) -> None:
        """Log the plan decisions via ``krishivaidya.gpu_profiler``."""
        if self.gpu:
            g = self.gpu
            logger.info(
                "GPU: %s | SM %s.%s | VRAM %s GiB | bf16=%s fp8=%s",
                g.device_name or "N/A",
                g.major, g.minor,
                g.total_memory_gb or "?",
                g.bf16_supported, g.fp8_supported,
            )

        _loggable = [
            ("attention-backend", self.attention_backend or "vLLM-auto"),
            ("dtype", self.dtype),
            ("kv-cache-dtype", self.kv_cache_dtype),
            ("block-size", str(self.block_size)),
            ("gpu-memory-utilization", str(self.gpu_memory_utilization)),
            ("max-model-len", str(self.max_model_len)),
            ("enforce-eager", str(self.enforce_eager)),
            ("optimization-level", str(self.optimization_level)),
            ("max-num-seqs", str(self.max_num_seqs)),
            ("max-num-batched-tokens", str(self.max_num_batched_tokens)),
            ("min-pixels", str(self.min_pixels)),
            ("max-pixels", str(self.max_pixels)),
            ("mm-processor-cache-gb", str(self.mm_processor_cache_gb)),
            ("warmup-iterations", str(self.determinism_warmup)),
        ]
        for name, val in _loggable:
            reason = getattr(self, name.replace("-", "_") + "_reason", None)
            if reason:
                logger.info("  %s: %s  (%s)", name, val, reason)

        if self.warnings:
            for w in self.warnings:
                logger.warning("%s", w)


# ── GPUProfiler ────────────────────────────────────────────────


class GPUProfiler:
    """Build a GPU-aware :class:`InferencePlan` for vLLM.

    Usage::

        profiler = GPUProfiler()
        plan = profiler.build_plan()
        cli_args = plan.to_cli_args()
    """

    @staticmethod
    def detect_gpu() -> GPUInfo:
        """Probe the system and return a :class:`GPUInfo`."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        cuda_version = torch.version.cuda

        if not torch.cuda.is_available():
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
                fp8_supported=False,
            )

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        major = int(props.major)
        minor = int(props.minor)
        total_memory_gb = _gb(props.total_memory)

        bf16_supported = bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        tf32_supported = major >= 8 and cuda_version is not None
        fp8_supported = major > 8 or (major == 8 and minor >= 9)

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
            fp8_supported=fp8_supported,
        )

    @staticmethod
    def build_plan(gpu: GPUInfo | None = None) -> InferencePlan:
        """Produce an :class:`InferencePlan` tuned for the detected GPU.

        Parameters
        ----------
        gpu : GPUInfo or None
            Pre-detected GPU info.  If ``None``, auto-detect.

        Returns
        -------
        InferencePlan
        """
        gpu = gpu or GPUProfiler.detect_gpu()
        warnings: list[str] = []

        # ── 1. Attention backend ──────────────────────────────────────────
        # FlashInfer paged attention crashes on SM < 8.0 (Turing, T4, GTX 16xx).
        # TRITON_ATTN is portable and works across all compute capabilities.
        # On SM >= 8.0 we let vLLM auto-select (FLASH_ATTN > FLASHINFER > TRITON_ATTN).
        if gpu.available and gpu.major is not None and gpu.major < 8:
            attention_backend = "TRITON_ATTN"
            attention_reason = (
                f"GPU SM {gpu.major}.{gpu.minor} < 8.0: FlashInfer paged attention "
                f"is unstable on this architecture; using TRITON_ATTN."
            )
        else:
            attention_backend = None
            attention_reason = (
                "GPU SM >= 8.0: letting vLLM auto-select optimal backend "
                "(FLASH_ATTN > FLASHINFER > TRITON_ATTN)."
            )

        # ── 2. Data type ─────────────────────────────────────────────────
        # bf16 requires SM >= 8.0 (Ampere). Older GPUs must use float16.
        if gpu.bf16_supported:
            dtype = "bfloat16"
            dtype_reason = "GPU natively supports bfloat16 (SM >= 8.0)."
        else:
            dtype = "float16"
            dtype_reason = (
                "GPU lacks bfloat16 support (SM < 8.0); "
                "using float16 to avoid runtime casting penalty."
            )

        # ── 3. KV cache dtype ────────────────────────────────────────────
        # FP8 KV cache halves memory but requires SM >= 8.9 (Ada Lovelace+)
        # for native compute.  On older GPUs use 'auto' (= model dtype).
        if gpu.fp8_supported and gpu.total_memory_gb is not None and gpu.total_memory_gb >= 24:
            kv_cache_dtype = "fp8"
            kv_cache_reason = (
                "GPU supports FP8 compute (SM >= 8.9) with VRAM >= 24 GiB; "
                "FP8 KV cache doubles effective capacity."
            )
        else:
            kv_cache_dtype = "auto"
            kv_cache_reason = (
                "Using model dtype for KV cache "
                "(FP8 not supported or VRAM < 24 GiB)."
            )

        # ── 4. Block size ────────────────────────────────────────────────
        # PagedAttention block size: larger = higher throughput for long
        # sequences, smaller = less fragmentation for short sequences.
        # Per Leeroopedia & PagedAttention tuning guides.
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 24:
                block_size = 32
                block_size_reason = "VRAM >= 24 GiB: larger blocks maximise throughput."
            elif gpu.total_memory_gb >= 8:
                block_size = 16
                block_size_reason = "Standard block size (16) for balanced throughput."
            else:
                block_size = 8
                block_size_reason = "VRAM < 8 GiB: smaller blocks reduce memory waste."
        else:
            block_size = 16
            block_size_reason = "Default block size (VRAM unknown)."

        # ── 5. GPU memory utilisation ─────────────────────────────────────
        # Controls how much VRAM vLLM reserves for model weights + KV cache.
        # Conservative on low-VRAM GPUs to avoid OOM.
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 40:
                gpu_memory = 0.95
                gpu_memory_reason = "VRAM >= 40 GiB: high utilisation."
            elif gpu.total_memory_gb >= 24:
                gpu_memory = 0.92
                gpu_memory_reason = "VRAM >= 24 GiB: generous KV allocation."
            elif gpu.total_memory_gb >= 16:
                gpu_memory = 0.88
                gpu_memory_reason = "VRAM >= 16 GiB: balanced allocation."
            elif gpu.total_memory_gb >= 12:
                gpu_memory = 0.85
                gpu_memory_reason = "VRAM >= 12 GiB: conservative."
            elif gpu.total_memory_gb >= 8:
                gpu_memory = 0.80
                gpu_memory_reason = "VRAM >= 8 GiB: tight budget."
            else:
                gpu_memory = 0.75
                gpu_memory_reason = "VRAM < 8 GiB: minimal allocation; OOM risk."
                warnings.append(
                    f"Only {gpu.total_memory_gb} GiB VRAM detected. "
                    f"The model may not fit. Consider a smaller model or "
                    f"increasing --gpu-memory-utilization."
                )
        else:
            gpu_memory = 0.90
            gpu_memory_reason = "VRAM unknown: using default 0.90."

        # ── 6. Max model length ──────────────────────────────────────────
        # Trades KV cache capacity for context length.
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 24:
                max_model_len = 8192
                max_model_len_reason = "VRAM >= 24 GiB: longer context supported."
            elif gpu.total_memory_gb >= 12:
                max_model_len = 4096
                max_model_len_reason = "VRAM >= 12 GiB: standard context length."
            else:
                max_model_len = 2048
                max_model_len_reason = "VRAM < 12 GiB: reduced context to fit KV cache."
        else:
            max_model_len = 4096
            max_model_len_reason = "Default (VRAM unknown)."

        # ── 7. Multi-modal pixel budget ──────────────────────────────────
        # Controls image resolution → image token count → encoder VRAM.
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 24:
                min_px, max_px = 200704, 451584
                pixel_reason = "High quality (up to ~672x672 pixels)."
            elif gpu.total_memory_gb >= 12:
                min_px, max_px = 100352, 301056
                pixel_reason = "Medium quality (up to ~448x672)."
            elif gpu.total_memory_gb >= 8:
                min_px, max_px = 50176, 200704
                pixel_reason = "Reduced quality (up to ~448x448) to save encoder VRAM."
            else:
                min_px, max_px = 25088, 100352
                pixel_reason = "Minimal quality (up to ~224x448) critical VRAM constraint."
        else:
            min_px, max_px = 200704, 451584
            pixel_reason = "Default (VRAM unknown)."

        # ── 8. Multi-modal processor cache ───────────────────────────────
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 24:
                mm_cache = 4.0
                mm_cache_reason = "Full multimodal cache (4 GiB)."
            elif gpu.total_memory_gb >= 16:
                mm_cache = 3.0
                mm_cache_reason = "Reduced multimodal cache (3 GiB)."
            elif gpu.total_memory_gb >= 8:
                mm_cache = 2.0
                mm_cache_reason = "Conservative multimodal cache (2 GiB)."
            else:
                mm_cache = 1.0
                mm_cache_reason = "Minimal multimodal cache (1 GiB)."
        else:
            mm_cache = 4.0
            mm_cache_reason = "Default multimodal cache (4 GiB)."

        # ── 9. Enforce eager / CUDA graphs ───────────────────────────────
        # CUDA graphs consume 2-4 GiB extra GPU memory.  On low-VRAM / older
        # GPUs the memory is better spent on KV cache.
        # Threshold: SM >= 8.0 AND VRAM >= 16 GiB → allow CUDA graphs.
        _has_cuda_graph_headroom = (
            gpu.available
            and gpu.major is not None
            and gpu.major >= 8
            and gpu.total_memory_gb is not None
            and gpu.total_memory_gb >= 16
        )
        if not gpu.available:
            enforce_eager = True
            enforce_eager_reason = "No CUDA GPU: eager mode required."
        elif not _has_cuda_graph_headroom:
            if gpu.total_memory_gb is not None and gpu.total_memory_gb < 16:
                enforce_eager = True
                enforce_eager_reason = (
                    f"VRAM {gpu.total_memory_gb} GiB < 16 GiB: CUDA graphs "
                    f"consume 2-4 GiB better used for KV cache."
                )
            else:
                enforce_eager = True
                enforce_eager_reason = (
                    f"SM {gpu.major}.{gpu.minor} < 8.0: CUDA graphs provide "
                    f"limited benefit on this architecture."
                )
        else:
            enforce_eager = False
            if gpu.total_memory_gb is not None and gpu.total_memory_gb >= 24:
                enforce_eager_reason = (
                    "VRAM >= 24 GiB: CUDA graphs enabled for optimal latency."
                )
            else:
                enforce_eager_reason = (
                    "SM >= 8.0 and VRAM >= 16 GiB: CUDA graphs enabled."
                )

        # ── 10. Optimisation level ───────────────────────────────────────
        # -O0 = no compile, no cudagraphs (equivalent to enforce_eager).
        # -O1 = compile + PIECEWISE cudagraphs (lighter memory).
        # -O2 = compile + FULL_AND_PIECEWISE cudagraphs (max perf, default).
        if _has_cuda_graph_headroom:
            if gpu.total_memory_gb is not None and gpu.total_memory_gb >= 40:
                opt_level = 2
                opt_reason = "VRAM >= 40 GiB: full -O2 optimisation."
            elif gpu.total_memory_gb is not None and gpu.total_memory_gb >= 24:
                opt_level = 1
                opt_reason = "VRAM >= 24 GiB: -O1 (compile + piecewise cudagraphs)."
            else:
                opt_level = 0
                opt_reason = "VRAM 16-24 GiB: no explicit -O flag (vLLM defaults to -O2)."
        else:
            opt_level = 0
            opt_reason = "No CUDA or GPU < SM 8.0 or VRAM < 16 GiB: -O0 no optimisation."

        # ── 11. Max number of sequences ──────────────────────────────────
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 40:
                max_seqs = 512
                max_seqs_reason = "High concurrency: up to 512 sequences."
            elif gpu.total_memory_gb >= 24:
                max_seqs = 256
                max_seqs_reason = "Generous concurrency: up to 256 sequences."
            elif gpu.total_memory_gb >= 12:
                max_seqs = 128
                max_seqs_reason = "Moderate concurrency: up to 128 sequences."
            elif gpu.total_memory_gb >= 8:
                max_seqs = 64
                max_seqs_reason = "Limited concurrency: up to 64 sequences."
            else:
                max_seqs = 32
                max_seqs_reason = "Minimal concurrency: up to 32 sequences."
        else:
            max_seqs = 256
            max_seqs_reason = "Default (VRAM unknown)."

        # ── 12. Max batched tokens ───────────────────────────────────────
        # Controls chunked prefill granularity.
        if gpu.total_memory_gb is not None:
            if gpu.total_memory_gb >= 24:
                batched_tokens = 8192
                batched_tokens_reason = "VRAM >= 24 GiB: large batches for throughput."
            elif gpu.total_memory_gb >= 12:
                batched_tokens = 4096
                batched_tokens_reason = "VRAM >= 12 GiB: standard batches."
            else:
                batched_tokens = 2048
                batched_tokens_reason = "VRAM < 12 GiB: smaller batches bound memory."
        else:
            batched_tokens = 8192
            batched_tokens_reason = "Default (VRAM unknown)."

        # ── 13. Triton JIT determinism warmup ───────────────────────────
        # When TRITON_ATTN is used (SM < 8.0), Triton kernels may JIT-compile
        # during the first inference request, causing latency spikes.
        # Pre-warming with dummy iterations avoids this (vLLM PR #33537).
        if attention_backend == "TRITON_ATTN":
            warmup = 3
            warmup_reason = (
                "TRITON_ATTN backend: pre-warming Triton JIT kernels "
                "to avoid first-request latency spikes (PR #33537)."
            )
        else:
            warmup = 0
            warmup_reason = "No Triton JIT warmup needed."

        # ── GPU-specific warnings ────────────────────────────────────────
        if gpu.available and gpu.major is not None:
            if gpu.major < 7:
                warnings.append(
                    f"GPU SM {gpu.major}.{gpu.minor} < 7.0: vLLM requires "
                    f"compute capability 7.0+. Server may fail to start."
                )

            if gpu.major < 8:
                warnings.append(
                    f"GPU SM {gpu.major}.{gpu.minor} < 8.0: FlashAttention-2 "
                    f"for vision encoder is unavailable; falling back to "
                    f"PyTorch SDPA. FlashInfer paged attention disabled."
                )

            if gpu.major == 8 and gpu.minor < 9 and gpu.total_memory_gb is not None and gpu.total_memory_gb >= 24:
                warnings.append(
                    f"GPU SM {gpu.major}.{gpu.minor} does not natively support "
                    f"FP8 compute. KV cache quantisation set to 'auto'."
                )

        # Check for FlashInfer GDN fallback needed for CUDA < 12.6 on SM90
        # (vLLM PR #37507).
        if gpu.available and gpu.major == 9 and gpu.cuda_version is not None:
            if not _cuda_version_ge(gpu.cuda_version, "12.6"):
                warnings.append(
                    f"CUDA toolkit {gpu.cuda_version} < 12.6 detected on SM90 "
                    f"GPU. FlashInfer GDN prefill kernel may fail; vLLM will "
                    f"fall back to Triton/FLA automatically (PR #37507)."
                )

        if not gpu.available:
            warnings.append(
                "No CUDA GPU detected. Falling back to CPU-safe defaults. "
                "Inference will be extremely slow or may not work."
            )

        if gpu.available and gpu.total_memory_gb is not None and gpu.total_memory_gb < 6:
            warnings.append(
                f"Only {gpu.total_memory_gb} GiB VRAM. A 3B model in float16 "
                f"requires ~6 GiB just for weights. The model may not load."
            )

        # ── Assemble ────────────────────────────────────────────────────
        return InferencePlan(
            attention_backend=attention_backend,
            attention_reason=attention_reason,
            dtype=dtype,
            dtype_reason=dtype_reason,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_reason=kv_cache_reason,
            block_size=block_size,
            block_size_reason=block_size_reason,
            gpu_memory_utilization=gpu_memory,
            gpu_memory_reason=gpu_memory_reason,
            max_model_len=max_model_len,
            max_model_len_reason=max_model_len_reason,
            min_pixels=min_px,
            max_pixels=max_px,
            pixel_reason=pixel_reason,
            mm_processor_cache_gb=mm_cache,
            mm_cache_reason=mm_cache_reason,
            enforce_eager=enforce_eager,
            enforce_eager_reason=enforce_eager_reason,
            optimization_level=opt_level,
            optimization_reason=opt_reason,
            max_num_seqs=max_seqs,
            max_num_seqs_reason=max_seqs_reason,
            max_num_batched_tokens=batched_tokens,
            batched_tokens_reason=batched_tokens_reason,
            determinism_warmup=warmup,
            warmup_reason=warmup_reason,
            warnings=warnings,
            gpu=gpu,
        )


# ── Standalone CLI entry point ──────────────────────────────────


def main() -> None:
    """Print the GPU profile report to stderr and exit."""
    import sys
    profiler = GPUProfiler()
    plan = profiler.build_plan()
    print(plan.report(), file=sys.stderr)
    plan.log()
    sys.exit(1 if plan.warnings else 0)


if __name__ == "__main__":
    main()
