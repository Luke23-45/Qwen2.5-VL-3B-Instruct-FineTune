# Cloud Deployment

This API is a cloud gateway for a separate vLLM runtime. Do not load model weights inside the FastAPI process.

## Architecture

- `gateway`: `uvicorn inference.api.main:app --host 0.0.0.0 --port 8000`
- `vllm`: `vllm serve Qwen/Qwen2.5-VL-3B-Instruct ...`

Recommended network posture:

- expose the gateway publicly on port `8000`
- keep vLLM private on `127.0.0.1:8091` or an internal VPC address
- set `KRISHI_VLLM_BASE_URL` to the private vLLM URL

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Environment

Copy `inference/api/.env.example` to `.env` or export the same `KRISHI_*` variables in your service manager.

Minimum required variables:

```bash
KRISHI_VLLM_BASE_URL=http://127.0.0.1:8080
KRISHI_JWT_PUBLIC_KEY_PATH=/opt/krishivaidya/secrets/public_key.pem
```

## Start vLLM (with GPU-aware auto-configuration)

```bash
python -m inference.api.scripts.launch_vllm --port 8091
```

## Start gateway

```bash
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
```

Or use the installed console script:

```bash
krishivaidya-inference-api
```

## JWT utilities

Generate keys:

```bash
python -m inference.api.scripts.generate_keys
```

Generate a test token:

```bash
python -m inference.api.scripts.generate_token
```

## Smoke test

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Then send a signed `POST /v1/inference` request with a valid JWT and either a base64 image or image URL.
