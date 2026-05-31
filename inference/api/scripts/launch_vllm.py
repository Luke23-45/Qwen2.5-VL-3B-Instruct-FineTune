#!/usr/bin/env python3
"""vLLM server launcher.

Forwards arguments to ``vllm serve``.  Attention backend is intentionally
not set — vLLM 0.22+ auto-selects via its built-in fallback chain
(FLASH_ATTN → FLASHINFER → TRITON_ATTN → …) which correctly handles
older GPUs (T4, etc.) without explicit configuration.

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


def build_cmd(args: argparse.Namespace) -> list[str]:
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
    cmd = build_cmd(args)

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
