"""Shared FastAPI dependencies (singletons, lifespan resources)."""

from __future__ import annotations

import functools

from .config import APISettings, get_settings


@functools.lru_cache(maxsize=1)
def settings() -> APISettings:
    """Return the cached application settings singleton."""
    return get_settings()
