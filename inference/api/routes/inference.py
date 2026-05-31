"""Main inference endpoint for ``POST /v1/inference``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ..auth.jwt_verifier import JWTVerifier
from ..auth.token_models import TokenPayload
from ..config import get_settings
from ..middleware.rate_limiter import limiter
from ..services.vllm_client import VLLMClient
from ..validation.image_validator import (
    ValidatedImage,
    validate_from_base64,
    validate_from_url,
)
from ..validation.request_schemas import InferenceRequestBody, InferenceResponseBody

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["inference"])


def _get_vllm_client(request: Request) -> VLLMClient:
    """Retrieve the shared vLLM client from app state."""
    client: VLLMClient | None = getattr(request.app.state, "vllm_client", None)
    if client is None:
        raise RuntimeError("vLLM client not initialised. Is the app started correctly?")
    return client


def _get_auth_dependency(request: Request) -> JWTVerifier:
    """Retrieve the shared JWT verifier from app state."""
    verifier: JWTVerifier | None = getattr(request.app.state, "jwt_verifier", None)
    if verifier is None:
        raise RuntimeError("JWT verifier not initialised. Is the app started correctly?")
    return verifier


@router.post(
    "/inference",
    response_model=InferenceResponseBody,
    summary="Run crop disease inference",
    description=(
        "Send an image (base64 or URL) and receive a crop disease prediction "
        "from the Qwen2.5-VL model served by vLLM."
    ),
    responses={
        401: {"description": "Missing or invalid JWT token."},
        403: {"description": "Token lacks required scope."},
        413: {"description": "Image exceeds size limit."},
        422: {"description": "Invalid request body or image."},
        429: {"description": "Rate limit exceeded."},
        502: {"description": "vLLM backend error."},
        503: {"description": "vLLM backend unreachable."},
        504: {"description": "vLLM inference timed out."},
    },
)
@limiter.limit(lambda: get_settings().rate_limit)
async def inference(
    request: Request,
    body: InferenceRequestBody,
) -> InferenceResponseBody:
    """Execute the full inference pipeline."""
    verifier = _get_auth_dependency(request)
    token: TokenPayload = await verifier(request)
    request.state.token_payload = token

    logger.info(
        "Inference request  request_id=%s  client=%s  has_base64=%s  has_url=%s",
        body.request_id,
        token.sub,
        body.image.base64 is not None,
        body.image.url is not None,
    )

    settings = request.app.state.settings
    validation_kwargs = dict(
        max_size_bytes=settings.max_image_size_bytes,
        max_dimension=settings.max_image_dimension,
        min_dimension=settings.min_image_dimension,
        allowed_formats=settings.allowed_image_formats,
        max_pixel_budget=settings.max_pixel_budget,
    )

    validated: ValidatedImage
    if body.image.base64 is not None:
        validated = await validate_from_base64(body.image.base64, **validation_kwargs)
    else:
        validated = await validate_from_url(
            str(body.image.url),
            fetch_timeout=settings.image_url_fetch_timeout_seconds,
            **validation_kwargs,
        )

    logger.info(
        "Image validated  request_id=%s  fmt=%s  size=%d  dims=%dx%d  tokens~=%d",
        body.request_id,
        validated.format,
        validated.size_bytes,
        validated.width,
        validated.height,
        validated.estimated_visual_tokens,
    )

    vllm = _get_vllm_client(request)
    prompt = body.prompt or settings.default_prompt
    result = await vllm.infer(
        image_bytes=validated.raw_bytes,
        image_format=validated.format,
        prompt=prompt,
        max_tokens=body.max_new_tokens,
        temperature=body.temperature,
        do_sample=body.do_sample,
    )

    logger.info(
        "Inference complete  request_id=%s  latency_ms=%.1f  in_tok=%d  out_tok=%d",
        body.request_id,
        result.latency_ms,
        result.input_tokens,
        result.output_tokens,
    )

    return InferenceResponseBody(
        request_id=body.request_id or "",
        prediction=result.text,
        latency_ms=round(result.latency_ms, 1),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
    )
