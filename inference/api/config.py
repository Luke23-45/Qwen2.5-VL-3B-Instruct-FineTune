"""Centralized configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Application settings loaded from environment or .env files."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_prefix="KRISHI_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    vllm_base_url: str = "http://127.0.0.1:8091"
    vllm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vllm_timeout_seconds: int = 60
    vllm_autostart: bool = Field(
        default=False,
        description="Automatically start vLLM as a subprocess when using run_api.py.",
    )
    vllm_host: str = Field(
        default="127.0.0.1",
        description="Host address for the vLLM server subprocess.",
    )
    vllm_port: int = Field(
        default=8091,
        description="Port for the vLLM server subprocess.",
    )
    vllm_dtype: str = Field(
        default="half",
        description="Data type for vLLM inference (half, bfloat16, auto).",
    )
    vllm_max_model_len: int = Field(
        default=4096,
        description="Maximum model length for vLLM.",
    )
    vllm_gpu_memory_utilization: float = Field(
        default=0.80,
        description="GPU memory utilization for vLLM (0.0 to 1.0).",
    )

    jwt_public_key_content: str | None = Field(
        default=None,
        description="Raw PEM string of the public key.",
    )
    jwt_public_key_path: Path | None = Field(
        default=None,
        description="Path to the PEM-encoded RSA public key used to verify JWTs.",
    )
    jwt_private_key_content: str | None = Field(
        default=None,
        description="Raw PEM string of the private key.",
    )
    jwt_private_key_path: Path | None = Field(
        default=None,
        description="Path to the PEM-encoded RSA private key.",
    )
    jwt_algorithm: str = "RS256"
    jwt_issuer: str = "krishivaidya"
    jwt_audience: str = "krishivaidya-inference"

    ngrok_enabled: bool = Field(
        default=False,
        description="Set to True to start an ngrok tunnel on startup.",
    )
    ngrok_authtoken: str | None = Field(
        default=None,
        description="Ngrok auth token for persistent URLs.",
    )
    ngrok_domain: str | None = Field(
        default=None,
        description="Custom domain for ngrok.",
    )

    max_image_size_bytes: int = 10 * 1024 * 1024
    max_image_dimension: int = 4096
    min_image_dimension: int = 28
    allowed_image_formats: frozenset[str] = frozenset(
        {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}
    )
    image_url_fetch_timeout_seconds: int = 10

    rate_limit: str = "30/minute"

    default_prompt: str = (
        "Identify the crop disease visible in this image. Return only compact "
        "JSON with keys crop, disease, canonical_label, and confidence_label."
    )
    default_max_new_tokens: int = 160
    max_pixel_budget: int = 451584
    min_pixel_budget: int = 200704
    """Pixel budget defaults — overridden at startup by GPUProfiler when available."""


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """Return the cached application settings singleton."""
    return APISettings()  # type: ignore[call-arg]
