"""Global exception handlers that always return structured JSON."""

from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from ..services.vllm_client import VLLMError
from ..validation.image_validator import ImageValidationError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI app."""

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = " -> ".join(str(item) for item in err.get("loc", []))
            messages.append(f"{loc}: {err.get('msg', 'invalid')}")
        combined = "; ".join(messages)
        logger.warning("Request validation failed: %s", combined)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": combined,
                    "details": errors,
                }
            },
        )

    @app.exception_handler(ImageValidationError)
    async def _image_validation_error(
        request: Request, exc: ImageValidationError
    ) -> JSONResponse:
        logger.warning("Image validation failed: [%s] %s", exc.code, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "field": "image",
                }
            },
        )

    @app.exception_handler(VLLMError)
    async def _vllm_error(request: Request, exc: VLLMError) -> JSONResponse:
        logger.error("vLLM error: [%d] %s", exc.status_code, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "inference_backend_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            content = {"error": detail}
        else:
            content = {
                "error": {
                    "code": "http_error",
                    "message": str(detail),
                }
            }
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.critical(
            "Unhandled exception on %s %s:\n%s",
            request.method,
            request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected error occurred. Please try again later.",
                }
            },
        )
