"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the FastAPI gateway process is alive.",
    response_class=JSONResponse,
)
async def health() -> dict:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 if the gateway is alive and the vLLM backend is "
        "reachable with the expected model loaded."
    ),
    response_class=JSONResponse,
)
async def ready(request: Request) -> JSONResponse:
    """Check vLLM reachability."""
    vllm_client = getattr(request.app.state, "vllm_client", None)
    if vllm_client is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "vLLM client not initialised."},
        )

    if await vllm_client.is_healthy():
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "vllm": "connected"},
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "reason": "Cannot reach vLLM backend or model not loaded.",
        },
    )
