#!/usr/bin/env python3
"""vLLM server launcher with GPU-aware auto-configuration.

Auto-detects GPU hardware via :class:`GPUProfiler` and tunes every
``vllm serve`` argument — attention backend, dtype, KV-cache, block
size, memory utilisation, context length, pixel budget, scheduling,
CUDA graphs, and Triton JIT warmup — for optimal performance on the
detected GPU.

Usage
-----
    python -m inference.api.scripts.launch_vllm
    python -m inference.api.scripts.launch_vllm --port 8090 --max-model-len 4096

Any GPU-tuned CLI argument passed explicitly overrides the profiler's
recommendation.  Pass ``--extra-args`` for flags not listed here::

    python -m inference.api.scripts.launch_vllm --extra-args "--cpu-offload-gb 8"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys

from inference.api.scripts.gpu_profiler import GPUProfiler


logger = logging.getLogger("krishivaidya.launch_vllm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ── Helpers ────────────────────────────────────────────────────


def _override_arg(cmd: list[str], flag: str, value: str) -> None:
    """Remove any previous occurrence of *flag* from *cmd* and re-add at end.

    This lets user-provided CLI args take priority over plan defaults.
    """
    for i in range(len(cmd) - 1, -1, -1):
        if cmd[i] == flag:
            del cmd[i:i + 2]
            break
    cmd.extend([flag, value])


# ── Command builder ────────────────────────────────────────────


def build_cmd(args: argparse.Namespace, plan) -> list[str]:
    """Build the ``vllm serve`` command list.

    Priority (high → low): user CLI override → plan (GPU-auto) → vLLM default.
    """
    cmd: list[str] = [
        "vllm",
        "serve",
        args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--trust-remote-code",
    ]

    # 1. GPU-tuned defaults from the profiler plan.
    cmd.extend(plan.to_cli_args())

    # 2. User CLI overrides — re-add at end to win.
    if args.dtype is not None:
        _override_arg(cmd, "--dtype", args.dtype)
    if args.max_model_len is not None:
        _override_arg(cmd, "--max-model-len", str(args.max_model_len))
    if args.gpu_memory_utilization is not None:
        _override_arg(cmd, "--gpu-memory-utilization", str(args.gpu_memory_utilization))
    if args.kv_cache_dtype is not None:
        _override_arg(cmd, "--kv-cache-dtype", args.kv_cache_dtype)
    if args.mm_processor_kwargs is not None:
        _override_arg(cmd, "--mm-processor-kwargs", args.mm_processor_kwargs)

    # 3. Model-specific (not GPU-dependent).
    if args.limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", args.limit_mm_per_prompt])

    # 4. Escape hatch for any other vLLM flags.
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    return cmd


# ── CLI parser ─────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM server launcher with GPU-aware auto-config.",
    )

    # -- Non-GPU parameters (always required / use provided default).
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--limit-mm-per-prompt", default='{"image":1}')

    # -- GPU-tuned parameters (default None → plan value is used).
    parser.add_argument("--dtype", default=None,
                        help="Override auto-detected dtype (plan: bfloat16 or float16).")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override auto-detected max context length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Override auto-detected GPU memory ratio (0-1).")
    parser.add_argument("--kv-cache-dtype", default=None,
                        help="Override auto-detected KV cache dtype.")
    parser.add_argument("--mm-processor-kwargs", default=None,
                        help="Override auto-detected processor kwargs JSON.")

    # -- Escape hatch.
    parser.add_argument("--extra-args", default=None,
                        help="Additional vLLM CLI args passed verbatim "
                             "(e.g. '--cpu-offload-gb 8').")

    return parser.parse_args(argv)


# ── Entry point ────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Build GPU-aware configuration plan.
    profiler = GPUProfiler()
    plan = profiler.build_plan()
    plan.log()

    if plan.warnings:
        for w in plan.warnings:
            logger.warning("%s", w)

    cmd = build_cmd(args, plan)

    logger.info("Launching vLLM: %s", " ".join(cmd))

    # Propagate environment variables required by the plan.
    vllm_env = os.environ.copy()
    vllm_env.update(plan.env_vars())

    try:
        proc = subprocess.run(cmd, check=True, env=vllm_env)
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
