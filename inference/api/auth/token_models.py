"""Pydantic models representing JWT token payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenPayload(BaseModel):
    """Validated claims extracted from a verified JWT.

    Standard registered claims (RFC 7519) plus application-specific fields.
    """

    # Registered claims
    sub: str = Field(..., description="Subject — the client / user identifier.")
    iss: str = Field(..., description="Issuer — who signed this token.")
    aud: str = Field(..., description="Audience — intended recipient service.")
    exp: int = Field(..., description="Expiration time (Unix timestamp, UTC).")
    iat: int = Field(..., description="Issued-at time (Unix timestamp, UTC).")

    # Optional registered claim
    jti: str | None = Field(
        default=None,
        description="Unique token ID — useful for revocation lists.",
    )

    # Application claims
    scopes: list[str] = Field(
        default_factory=lambda: ["inference"],
        description="Permission scopes granted to this token.",
    )
