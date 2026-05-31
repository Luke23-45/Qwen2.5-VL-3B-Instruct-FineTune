#!/usr/bin/env bash
# Launch the vLLM server for Qwen2.5-VL-3B-Instruct.
#
# For GPU-aware attention backend auto-detection, use:
#   python -m inference.api.scripts.launch_vllm
#
# Requirements:
#   - Linux with NVIDIA GPU
#   - pip install vllm
#
# Override defaults via environment variables:
#   VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
#   VLLM_HOST=127.0.0.1
#   VLLM_PORT=8091
#   VLLM_DTYPE=half
#   VLLM_MAX_MODEL_LEN=2048

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8091}"
DTYPE="${VLLM_DTYPE:-half}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
MIN_PIXELS="${VLLM_MIN_PIXELS:-200704}"
MAX_PIXELS="${VLLM_MAX_PIXELS:-451584}"

echo "============================================================"
echo " KrishiVaidya - vLLM Server"
echo "============================================================"
echo " Model:          ${MODEL}"
echo " Host:           ${HOST}:${PORT}"
echo " Dtype:          ${DTYPE}"
echo " Max model len:  ${MAX_MODEL_LEN}"
echo " Pixel budget:   ${MIN_PIXELS} - ${MAX_PIXELS}"
echo "============================================================"

exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "{\"image\":1}" \
    --mm-processor-kwargs "{\"min_pixels\":${MIN_PIXELS},\"max_pixels\":${MAX_PIXELS}}" \
    --enforce-eager \
    --trust-remote-code
