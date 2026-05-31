#!/usr/bin/env python3
"""Cross-platform launcher for the KrishiVaidya Inference API.

Optionally starts vLLM as a background subprocess (if
``KRISHI_VLLM_AUTOSTART=True`` in the ``.env`` file), waits for
it to become healthy, then launches the FastAPI gateway.  On
shutdown the vLLM process is cleaned up automatically.

This allows the entire pipeline to be started with a single
command — essential for cloud notebooks where only one process
can run at a time.

Usage
-----
    python run_api.py

To run without auto-starting vLLM (e.g. when vLLM is already
running elsewhere):
    KRISHI_VLLM_AUTOSTART=False python run_api.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import nest_asyncio

# Ensure the project root is on ``sys.path`` so that ``inference``
# can be imported regardless of the working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn

from inference.api.config import get_settings
from inference.api.scripts.gpu_profiler import GPUProfiler

logger = logging.getLogger("krishivaidya.runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def _build_vllm_cmd(settings) -> tuple[list[str], dict[str, str]]:
    """Build the vllm serve command using GPU-profiled configuration.

    Returns
    -------
    (command_list, env_vars)
        The vLLM command arguments and a dict of environment variables
        that must be propagated to the subprocess (e.g. Triton JIT warmup).
    """
    profiler = GPUProfiler()
    plan = profiler.build_plan()
    plan.log()

    if plan.warnings:
        for w in plan.warnings:
            logger.warning("%s", w)

    cmd = [
        "vllm",
        "serve",
        settings.vllm_model_name,
        "--host", settings.vllm_host,
        "--port", str(settings.vllm_port),
        "--trust-remote-code",
    ]
    cmd.extend(plan.to_cli_args())
    cmd.extend(["--limit-mm-per-prompt", '{"image":1}'])
    return cmd, plan.env_vars()


def _wait_for_vllm(
    *,
    base_url: str = "http://127.0.0.1:8091",
    proc: subprocess.Popen | None = None,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
) -> None:
    """Poll ``/health`` until vLLM responds ``200`` or *timeout* expires.

    If *proc* is provided the process-alive status is checked on
    every iteration so we fail fast when the subprocess dies.
    """
    health_url = f"{base_url.rstrip('/')}/health"
    started = time.monotonic()
    last_log = 0.0

    with httpx.Client() as client:
        while time.monotonic() - started < timeout:
            elapsed = time.monotonic() - started

            # Short-circuit if the subprocess has already exited.
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM process (PID {proc.pid}) exited unexpectedly "
                    f"with code {proc.returncode} before becoming ready."
                )

            try:
                resp = client.get(health_url, timeout=10.0)
                if resp.status_code == 200:
                    logger.info(
                        "vLLM ready after %.0fs  url=%s", elapsed, base_url
                    )
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            # Log progress every 30 s so the user knows we are still
            # waiting (model loading can take several minutes).
            if elapsed - last_log >= 30.0:
                logger.info("Waiting for vLLM … (%.0fs elapsed)", elapsed)
                last_log = elapsed

            time.sleep(poll_interval)

        raise RuntimeError(
            f"vLLM did not become ready within {timeout:.0f}s "
            f"(polled {health_url}).  Check the vLLM logs above."
        )


def _pipe_vllm_logs(stream) -> None:
    """Read lines from the vLLM subprocess stdout and forward them to our
    logger with a ``[vllm]`` prefix so they are interleaved with the
    gateway's own log output."""
    assert stream is not None
    for line in iter(stream.readline, b""):
        logger.info("[vllm] %s", line.decode("utf-8", errors="replace").rstrip())
    stream.close()


def _start_vllm(settings) -> subprocess.Popen | None:
    """Start vLLM as a background subprocess.

    Returns ``None`` if ``vllm_autostart`` is ``False``.
    ``_stop_vllm`` is called automatically if the startup fails.
    """
    if not settings.vllm_autostart:
        logger.info("vLLM autostart disabled (KRISHI_VLLM_AUTOSTART=False) — skipping.")
        return None

    cmd, env_vars = _build_vllm_cmd(settings)
    vllm_env = {**os.environ, **env_vars}
    logger.info("Starting vLLM subprocess: %s", " ".join(cmd))
    if env_vars:
        for k, v in env_vars.items():
            logger.info("  env: %s=%s", k, v)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=vllm_env,
        )
    except FileNotFoundError:
        logger.error(
            "The ``vllm`` command was not found on PATH.  "
            "Ensure vLLM is installed:  pip install vllm"
        )
        raise

    # Forward vLLM logs in a daemon thread so the main thread can
    # poll for readiness without blocking.
    t = threading.Thread(
        target=_pipe_vllm_logs,
        args=(proc.stdout,),
        daemon=True,
    )
    t.start()

    # Block until vLLM becomes healthy.
    # If it fails, clean up the subprocess before propagating the error.
    try:
        _wait_for_vllm(
            base_url=f"http://{settings.vllm_host}:{settings.vllm_port}",
            proc=proc,
            timeout=600.0,
        )
    except Exception:
        _stop_vllm(proc)
        raise

    return proc


def _stop_vllm(proc: subprocess.Popen | None) -> None:
    """Gracefully terminate the vLLM subprocess.

    Sends ``SIGTERM`` (or ``terminate()`` on Windows) and waits up
    to 30 s for a clean exit.  Falls back to ``SIGKILL`` if the
    process does not respond in time.
    """
    if proc is None:
        return
    if proc.poll() is not None:
        logger.debug("vLLM (PID %d) already exited.", proc.pid)
        return

    logger.info("Stopping vLLM (PID %d) …", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=30.0)
        logger.info("vLLM (PID %d) stopped.", proc.pid)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM did not exit in 30 s — killing.")
        proc.kill()
        proc.wait()
        logger.info("vLLM (PID %d) killed.", proc.pid)


# ── Entry-point ──────────────────────────────────────────────


def main() -> None:
    """Orchestrate the full pipeline lifecycle."""
    # Change CWD to ``inference/`` so relative paths in ``.env``
    # (e.g. ``./keys/public_key.pem``) resolve correctly regardless
    # of how the script was invoked.
    os.chdir(Path(__file__).parent)

    settings = get_settings()

    nest_asyncio.apply()

    vllm_proc = _start_vllm(settings)
    try:
        uvicorn.run(
            "inference.api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level="debug" if settings.debug else "info",
        )
    finally:
        _stop_vllm(vllm_proc)


if __name__ == "__main__":
    main()
