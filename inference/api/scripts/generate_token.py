"""Mint a signed JWT for a client.

Usage::

    python -m inference.api.scripts.generate_token \
        --private-key ./keys/private_key.pem \
        --sub my-client-id \
        --expires-in 24h

The generated token should be sent in the ``Authorization: Bearer <token>``
header of every API request.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from pathlib import Path

import jwt


_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d)$")

_UNIT_TO_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def _parse_duration(value: str) -> int:
    """Parse a human-friendly duration string to seconds.

    Examples: ``30s``, ``15m``, ``24h``, ``7d``.
    """
    match = _DURATION_RE.match(value.strip())
    if not match:
        raise ValueError(
            f"Invalid duration '{value}'. Use format like 30s, 15m, 24h, 7d."
        )
    amount, unit = int(match.group(1)), match.group(2)
    return amount * _UNIT_TO_SECONDS[unit]


def generate_token(
    private_key_path: Path,
    sub: str,
    expires_in_seconds: int,
    issuer: str = "krishivaidya",
    audience: str = "krishivaidya-inference",
    scopes: list[str] | None = None,
) -> str:
    """Create and sign a JWT with the given claims."""
    private_key_pem = private_key_path.expanduser().resolve().read_text(encoding="utf-8")

    now = int(time.time())
    payload = {
        "sub": sub,
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + expires_in_seconds,
        "jti": uuid.uuid4().hex,
        "scopes": scopes or ["inference"],
    }

    token: str = jwt.encode(payload, private_key_pem, algorithm="RS256")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a signed JWT for API authentication."
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        required=True,
        help="Path to the PEM private key file.",
    )
    parser.add_argument(
        "--sub",
        required=True,
        help="Subject / client identifier.",
    )
    parser.add_argument(
        "--expires-in",
        default="24h",
        help="Token lifetime (e.g. 30s, 15m, 24h, 7d).  Default: 24h.",
    )
    parser.add_argument(
        "--issuer",
        default="krishivaidya",
        help="JWT issuer claim.  Default: krishivaidya.",
    )
    parser.add_argument(
        "--audience",
        default="krishivaidya-inference",
        help="JWT audience claim.  Default: krishivaidya-inference.",
    )
    parser.add_argument(
        "--scopes",
        nargs="*",
        default=["inference"],
        help="Permission scopes.  Default: inference.",
    )
    args = parser.parse_args()

    try:
        ttl = _parse_duration(args.expires_in)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    token = generate_token(
        private_key_path=args.private_key,
        sub=args.sub,
        expires_in_seconds=ttl,
        issuer=args.issuer,
        audience=args.audience,
        scopes=args.scopes,
    )

    print("=" * 60)
    print("JWT generated successfully!")
    print("=" * 60)
    print()
    print(f"  Subject:     {args.sub}")
    print(f"  Issuer:      {args.issuer}")
    print(f"  Audience:    {args.audience}")
    print(f"  Expires in:  {args.expires_in}")
    print(f"  Scopes:      {args.scopes}")
    print()
    print("Token:")
    print(token)
    print()
    print("Usage:")
    print("  curl -X POST http://localhost:8000/v1/inference \\")
    print(f'    -H "Authorization: Bearer {token[:40]}..." \\')
    print('    -H "Content-Type: application/json" \\')
    print("    -d '{...}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
