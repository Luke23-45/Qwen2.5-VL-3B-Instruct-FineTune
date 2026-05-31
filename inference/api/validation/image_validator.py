"""Comprehensive image validation for Qwen2.5-VL inference.

Every image goes through this pipeline *before* it reaches vLLM:

1. **Source resolution** — decode base64 or fetch from URL.
2. **Size check** — raw bytes must not exceed the configured limit.
3. **Format check** — must be one of the allowed formats (JPEG, PNG, …).
4. **Corruption check** — PIL ``verify()`` + re-open + ``load()``.
5. **Dimension checks** — min 28×28, max 4096×4096.
6. **Aspect ratio** — between 1:10 and 10:1.
7. **Colour mode normalisation** — convert to RGB if needed.
8. **Pixel-budget warning** — log if estimated visual tokens exceed config.

All failures raise :class:`ImageValidationError` with a structured message.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ── Qwen VL constants ──────────────────────────────────────────
# The vision encoder tokenises images into 28×28 pixel patches.
_PATCH_SIZE = 28


class ImageValidationError(Exception):
    """Raised when an image fails any pre-inference check."""

    def __init__(self, message: str, *, code: str = "invalid_image", status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    """Container returned after successful validation."""

    image: Image.Image
    raw_bytes: bytes
    width: int
    height: int
    format: str  # e.g. "JPEG"
    size_bytes: int
    estimated_visual_tokens: int


# ── Public API ──────────────────────────────────────────────────


async def validate_from_base64(
    data: str,
    *,
    max_size_bytes: int,
    max_dimension: int,
    min_dimension: int,
    allowed_formats: frozenset[str],
    max_pixel_budget: int,
) -> ValidatedImage:
    """Decode base64 string and run the full validation pipeline."""
    raw = _decode_base64(data)
    _check_byte_size(raw, max_size_bytes)
    return _validate_raw(
        raw,
        max_dimension=max_dimension,
        min_dimension=min_dimension,
        allowed_formats=allowed_formats,
        max_pixel_budget=max_pixel_budget,
    )


async def validate_from_url(
    url: str,
    *,
    max_size_bytes: int,
    max_dimension: int,
    min_dimension: int,
    allowed_formats: frozenset[str],
    max_pixel_budget: int,
    fetch_timeout: int = 10,
) -> ValidatedImage:
    """Fetch image from URL and run the full validation pipeline."""
    raw = await _fetch_url(url, max_size_bytes=max_size_bytes, timeout=fetch_timeout)
    return _validate_raw(
        raw,
        max_dimension=max_dimension,
        min_dimension=min_dimension,
        allowed_formats=allowed_formats,
        max_pixel_budget=max_pixel_budget,
    )


# ── Source resolution ───────────────────────────────────────────


def _decode_base64(data: str) -> bytes:
    """Decode a base64 string, handling optional data-URI prefix and padding."""
    # Strip data URI prefix: "data:image/jpeg;base64,..."
    if data.startswith("data:"):
        try:
            _, data = data.split(",", 1)
        except ValueError:
            raise ImageValidationError(
                "Invalid data URI format. Expected 'data:<media>;base64,<data>'.",
                code="invalid_base64",
            )
    # Fix padding
    padding_needed = len(data) % 4
    if padding_needed:
        data += "=" * (4 - padding_needed)
    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ImageValidationError(
            f"Invalid base64 encoding: {exc}",
            code="invalid_base64",
        )


async def _fetch_url(url: str, *, max_size_bytes: int, timeout: int) -> bytes:
    """Download image bytes from a public URL with size and type guards."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        raise ImageValidationError(
            f"Timed out fetching image from URL after {timeout}s.",
            code="image_fetch_timeout",
            status_code=422,
        )
    except httpx.HTTPStatusError as exc:
        raise ImageValidationError(
            f"Failed to fetch image: HTTP {exc.response.status_code}.",
            code="image_fetch_failed",
            status_code=422,
        )
    except httpx.RequestError as exc:
        raise ImageValidationError(
            f"Failed to fetch image from URL: {exc}",
            code="image_fetch_failed",
            status_code=422,
        )

    content_type = response.headers.get("content-type", "")
    if content_type and not content_type.startswith("image/"):
        raise ImageValidationError(
            f"URL did not return an image. Content-Type: {content_type}",
            code="invalid_content_type",
        )

    raw = response.content
    _check_byte_size(raw, max_size_bytes)
    return raw


# ── Core validation pipeline ───────────────────────────────────


def _check_byte_size(raw: bytes, max_size_bytes: int) -> None:
    if len(raw) > max_size_bytes:
        mb = len(raw) / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"Image size {mb:.1f} MB exceeds the {limit_mb:.0f} MB limit.",
            code="image_too_large",
            status_code=413,
        )


def _validate_raw(
    raw: bytes,
    *,
    max_dimension: int,
    min_dimension: int,
    allowed_formats: frozenset[str],
    max_pixel_budget: int,
) -> ValidatedImage:
    """Run format, corruption, dimension, aspect-ratio, and mode checks."""

    # ── 1. Open + format detection ──────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise ImageValidationError(
            "Cannot identify image format. The file may be corrupt or not an image.",
            code="unrecognised_format",
        )
    except Exception as exc:
        raise ImageValidationError(
            f"Failed to open image: {exc}",
            code="image_open_failed",
        )

    fmt = (img.format or "UNKNOWN").upper()
    if fmt not in allowed_formats:
        raise ImageValidationError(
            f"Unsupported image format: {fmt}. Allowed: {sorted(allowed_formats)}.",
            code="unsupported_format",
        )

    # ── 2. Corruption check (verify + reload) ──────────────────
    try:
        img.verify()  # checks headers / structure; image is unusable after this
    except Exception as exc:
        raise ImageValidationError(
            f"Image file is corrupt or unreadable: {exc}",
            code="corrupt_image",
        )

    # Re-open because verify() closes the image
    img = Image.open(io.BytesIO(raw))
    try:
        img.load()  # force full decode of pixel data
    except Exception as exc:
        raise ImageValidationError(
            f"Image pixel data is corrupt: {exc}",
            code="corrupt_image",
        )

    width, height = img.size

    # ── 3. Dimension checks ─────────────────────────────────────
    if width < min_dimension or height < min_dimension:
        raise ImageValidationError(
            f"Image too small: {width}×{height} px. Minimum is {min_dimension}×{min_dimension}.",
            code="image_too_small",
        )
    if width > max_dimension or height > max_dimension:
        raise ImageValidationError(
            f"Image too large: {width}×{height} px. Maximum is {max_dimension}×{max_dimension}.",
            code="image_too_large",
        )

    # ── 4. Aspect ratio ─────────────────────────────────────────
    ratio = max(width, height) / max(min(width, height), 1)
    if ratio > 10.0:
        raise ImageValidationError(
            f"Extreme aspect ratio ({ratio:.1f}:1). Must be between 1:10 and 10:1.",
            code="extreme_aspect_ratio",
        )

    # ── 5. Colour mode normalisation ────────────────────────────
    if img.mode != "RGB":
        logger.info("Converting image from %s to RGB.", img.mode)
        img = img.convert("RGB")

    # ── 6. Visual-token budget warning ──────────────────────────
    total_pixels = width * height
    estimated_tokens = _estimate_visual_tokens(width, height)
    if total_pixels > max_pixel_budget:
        logger.warning(
            "Image %d×%d (%d pixels) exceeds pixel budget %d. "
            "vLLM will resize, but quality may degrade.",
            width,
            height,
            total_pixels,
            max_pixel_budget,
        )

    return ValidatedImage(
        image=img,
        raw_bytes=raw,
        width=width,
        height=height,
        format=fmt,
        size_bytes=len(raw),
        estimated_visual_tokens=estimated_tokens,
    )


def _estimate_visual_tokens(width: int, height: int) -> int:
    """Estimate how many visual tokens Qwen VL will produce.

    Qwen2.5-VL divides the image into 28×28 patches, then each patch
    maps to one visual token.
    """
    patches_w = math.ceil(width / _PATCH_SIZE)
    patches_h = math.ceil(height / _PATCH_SIZE)
    return patches_w * patches_h
