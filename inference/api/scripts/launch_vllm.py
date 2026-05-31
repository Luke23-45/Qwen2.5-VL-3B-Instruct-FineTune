#!/usr/bin/env python3
"""vLLM server launcher.

Forwards arguments to ``vllm serve``.

On GPUs with compute capability < 8.0 (e.g. T4) FlashInfer's paged
attention kernel crashes.  The script auto-detects this and explicitly
sets ``--attention-backend TRITON_ATTN`` which works reliably on
older GPUs while V1 engine stays active.

Usage
-----
    python -m inference.api.scripts.launch_vllm
    python -m inference.api.scripts.launch_vllm --port 8090 --max-model-len 4096
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys


logger = logging.getLogger("krishivaidya.launch_vllm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def _compute_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability, or None if CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return (torch.cuda.get_device_capability(0))
    except Exception:
        return None


def _attention_backend() -> str | None:
    """Return attention backend override, or None if FlashInfer should be used.

    FlashInfer's paged attention crashes on compute < 8.0 (T4, etc.).
    TRITON_ATTN is a reliable fallback that works on older GPUs.
    """
    cc = _compute_capability()
    if cc is None:
        return None
    major, minor = cc
    if major < 8:
        logger.warning(
            "GPU compute capability %d.%d < 8.0 — FlashInfer paged attention "
            "crashes on this GPU. Using --attention-backend TRITON_ATTN.",
            major, minor,
        )
        return "TRITON_ATTN"
    return None


def build_cmd(args: argparse.Namespace, attention_backend: str | None) -> list[str]:
    """Build the vLLM serve command list."""
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--enforce-eager",
        "--kv-cache-dtype", args.kv_cache_dtype,
    ]

    if attention_backend is not None:
        cmd.extend(["--attention-backend", attention_backend])

    if args.limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", args.limit_mm_per_prompt])
    if args.mm_processor_kwargs:
        cmd.extend(["--mm-processor-kwargs", args.mm_processor_kwargs])

    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM server launcher.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--limit-mm-per-prompt", default='{"image":1}')
    parser.add_argument(
        "--mm-processor-kwargs",
        default='{"min_pixels":200704,"max_pixels":451584}',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    attention_backend = _attention_backend()
    cmd = build_cmd(args, attention_backend)

    logger.info("Launching vLLM: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error(
            "The ``vllm`` command was not found on PATH.  "
            "Ensure vLLM is installed:  pip install vllm"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        logger.error("vLLM exited with code %d", exc.returncode)
        sys.exit(exc.returncode)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
