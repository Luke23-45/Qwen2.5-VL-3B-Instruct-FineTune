"""RS256 JWT verification — public-key only, no signing on this server.

Flow
----
1. Load the **public key** once at startup from disk (PEM file).
2. On every request, extract the ``Authorization: Bearer <token>`` header.
3. Decode + verify signature, expiry, issuer, and audience.
4. Return a :class:`TokenPayload` on success or raise ``HTTPException(401)``.

Security notes
--------------
* The **private key** is never loaded here.  It lives with the issuer only.
* ``algorithms`` is locked to the configured algorithm (default RS256) to
  prevent algorithm-confusion attacks (e.g. ``alg: none`` or ``HS256`` with
  the public key as secret).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .token_models import TokenPayload

logger = logging.getLogger(__name__)

# FastAPI security scheme — auto-populates the Swagger "Authorize" button
_bearer_scheme = HTTPBearer(auto_error=False)


class JWTVerifier:
    """Stateful verifier initialised once with public key + expected claims."""

    def __init__(
        self,
        public_key_path: Path | None = None,
        public_key_content: str | None = None,
        algorithm: str = "RS256",
        issuer: str = "krishivaidya",
        audience: str = "krishivaidya-inference",
    ) -> None:
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience
        self._public_key = self._load_public_key(public_key_path, public_key_content)
        logger.info(
            "JWT verifier initialised  alg=%s  iss=%s  aud=%s",
            algorithm,
            issuer,
            audience,
        )

    # ── Public interface ────────────────────────────────────────

    async def __call__(self, request: Request) -> TokenPayload:
        """FastAPI dependency — verify the bearer token on every request."""
        credentials: HTTPAuthorizationCredentials | None = await _bearer_scheme(
            request
        )
        if credentials is None:
            raise self._unauthorized("Missing Authorization header.")

        token = credentials.credentials
        payload = self._decode(token)
        self._check_scope(payload, required_scope="inference")
        return payload

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _load_public_key(path: Path | None, content: str | None) -> str:
        if content is not None:
            key_text = content
            logger.info("Loaded JWT public key from raw environment variable.")
        elif path is not None:
            resolved = path.expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(
                    f"JWT public key not found at {resolved}. "
                    "Generate one with:  python -m inference.api.scripts.generate_keys"
                )
            key_text = resolved.read_text(encoding="utf-8")
            logger.info("Loaded JWT public key from %s", resolved)
        else:
            raise ValueError("Either public_key_path or public_key_content must be provided.")

        if "BEGIN PUBLIC KEY" not in key_text and "BEGIN RSA PUBLIC KEY" not in key_text:
            raise ValueError("Provided key does not look like a PEM public key.")
            
        return key_text

    def _decode(self, token: str) -> TokenPayload:
        """Decode, verify signature, and validate standard claims."""
        try:
            raw: dict[str, Any] = jwt.decode(
                token,
                self._public_key,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["exp", "iss", "aud", "sub", "iat"],
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except jwt.ExpiredSignatureError:
            raise self._unauthorized("Token has expired.")
        except jwt.InvalidIssuerError:
            raise self._unauthorized("Invalid token issuer.")
        except jwt.InvalidAudienceError:
            raise self._unauthorized("Invalid token audience.")
        except jwt.InvalidSignatureError:
            raise self._unauthorized("Invalid token signature.")
        except jwt.DecodeError as exc:
            raise self._unauthorized(f"Token decode failed: {exc}")
        except jwt.InvalidTokenError as exc:
            raise self._unauthorized(f"Invalid token: {exc}")

        return TokenPayload(**raw)

    @staticmethod
    def _check_scope(payload: TokenPayload, required_scope: str) -> None:
        if required_scope not in payload.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "insufficient_scope",
                    "message": (
                        f"Token lacks required scope '{required_scope}'. "
                        f"Present scopes: {payload.scopes}"
                    ),
                },
            )

    @staticmethod
    def _unauthorized(message: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "authentication_failed", "message": message},
            headers={"WWW-Authenticate": "Bearer"},
        )
