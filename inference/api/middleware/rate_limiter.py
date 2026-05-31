"""In-memory rate limiting using slowapi.

Limits are applied **per client** identified by the ``sub`` claim in the
verified JWT.  If no token is present (e.g. health endpoints) the limit
is applied per remote IP.

Default: 30 requests / minute per client.
"""

from __future__ import annotations

import logging

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def _key_func(request: Request) -> str:
    """Extract the rate-limit key from the request.

    Prefers the JWT ``sub`` claim (set by the auth dependency) over the
    remote IP address so that rate limits follow the authenticated client
    rather than the network origin.
    """
    token_payload = getattr(request.state, "token_payload", None)
    if token_payload is not None:
        return getattr(token_payload, "sub", get_remote_address(request))
    return get_remote_address(request)


# Shared limiter instance — imported by routes
limiter = Limiter(key_func=_key_func)


def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a structured 429 response with Retry-After header."""
    retry_after = getattr(exc, "retry_after", 60)
    logger.warning(
        "Rate limit exceeded  client=%s  path=%s  limit=%s",
        _key_func(request),
        request.url.path,
        str(exc.detail),
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "rate_limit_exceeded",
                "message": f"Too many requests. Retry after {retry_after} seconds.",
            }
        },
        headers={"Retry-After": str(retry_after)},
    )
