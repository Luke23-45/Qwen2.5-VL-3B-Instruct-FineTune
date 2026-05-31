"""Structured request / response logging middleware.

Logs every request with:
- request_id, client_id, method, path, status, latency_ms, image_size_bytes

Security: **never** logs image content, JWT tokens, or other secrets.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("krishivaidya.api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that emits one structured log line per request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or reuse a request-level correlation ID
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id

        started = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            status_code = response.status_code if response else 500

            # Extract client ID from JWT payload if available
            client_id = "anonymous"
            token_payload = getattr(request.state, "token_payload", None)
            if token_payload is not None:
                client_id = getattr(token_payload, "sub", "anonymous")

            logger.info(
                "request_id=%s client=%s method=%s path=%s status=%d latency_ms=%.1f",
                request_id,
                client_id,
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )

            # Propagate the request ID in the response headers
            if response is not None:
                response.headers["X-Request-ID"] = request_id
