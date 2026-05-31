#!/usr/bin/env bash
# Launch the vLLM server for Qwen2.5-VL-3B-Instruct.
#
# Requirements:
#   - Linux with NVIDIA GPU
#   - pip install vllm
#
# Override defaults via environment variables:
#   VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
#   VLLM_HOST=127.0.0.1
#   VLLM_PORT=8080
#   VLLM_DTYPE=bfloat16
#   VLLM_MAX_MODEL_LEN=4096
#   VLLM_MM_CACHE_GB=4

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8080}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MM_CACHE_GB="${VLLM_MM_CACHE_GB:-4}"
MIN_PIXELS="${VLLM_MIN_PIXELS:-200704}"
MAX_PIXELS="${VLLM_MAX_PIXELS:-451584}"

echo "============================================================"
echo " KrishiVaidya - vLLM Server"
echo "============================================================"
echo " Model:          ${MODEL}"
echo " Host:           ${HOST}:${PORT}"
echo " Dtype:          ${DTYPE}"
echo " Max model len:  ${MAX_MODEL_LEN}"
echo " MM cache:       ${MM_CACHE_GB} GB"
echo " Pixel budget:   ${MIN_PIXELS} - ${MAX_PIXELS}"
echo "============================================================"

exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "{\"image\":1}" \
    --mm-processor-kwargs "{\"min_pixels\":${MIN_PIXELS},\"max_pixels\":${MAX_PIXELS}}" \
    --mm-processor-cache-gb "${MM_CACHE_GB}" \
    --trust-remote-code
