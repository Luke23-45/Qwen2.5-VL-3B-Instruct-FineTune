"""Pydantic request and response schemas for the inference API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, HttpUrl, model_validator
from typing_extensions import Self


class ImageSource(BaseModel):
    """Exactly one of ``base64`` or ``url`` must be provided."""

    base64: str | None = Field(
        default=None,
        description="Base64-encoded image bytes with an optional data URI prefix.",
        min_length=1,
    )
    url: HttpUrl | None = Field(
        default=None,
        description="Public URL to fetch the image from.",
    )
    media_type: str | None = Field(
        default=None,
        pattern=r"^image/(jpeg|png|webp|tiff|bmp)$",
        description=(
            "Optional MIME type hint. If provided it must match the actual image."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        has_b64 = self.base64 is not None
        has_url = self.url is not None
        if has_b64 == has_url:
            if has_b64:
                raise ValueError(
                    "Provide exactly one of 'base64' or 'url' in the image source, not both."
                )
            raise ValueError(
                "Provide exactly one of 'base64' or 'url' in the image source; neither was provided."
            )
        return self


class InferenceRequestBody(BaseModel):
    """Top-level inference request payload."""

    image: ImageSource = Field(
        ...,
        description="The source image for disease classification.",
    )
    prompt: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "Custom prompt override. When omitted the default crop-disease "
            "classification prompt is used."
        ),
    )
    max_new_tokens: int = Field(
        default=160,
        ge=1,
        le=512,
        description="Maximum number of tokens the model may generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Zero means deterministic greedy decode.",
    )
    do_sample: bool = Field(
        default=False,
        description="Enable stochastic sampling. Ignored when temperature is zero.",
    )
    request_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description=(
            "Client-provided idempotency or correlation ID. Auto-generated if omitted."
        ),
    )

    @model_validator(mode="after")
    def _ensure_request_id(self) -> Self:
        if self.request_id is None:
            object.__setattr__(self, "request_id", uuid.uuid4().hex)
        return self

    @model_validator(mode="after")
    def _temperature_sample_coherence(self) -> Self:
        if self.temperature == 0.0 and self.do_sample:
            object.__setattr__(self, "do_sample", False)
        return self


class InferenceResponseBody(BaseModel):
    """Successful inference response."""

    request_id: str
    prediction: str = Field(..., description="Model output text, usually JSON.")
    latency_ms: float = Field(..., description="End-to-end inference latency in ms.")
    input_tokens: int
    output_tokens: int
    model: str = Field(..., description="Model identifier served by vLLM.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorDetail(BaseModel):
    """Structured error payload."""

    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    field: str | None = Field(
        default=None,
        description="The request field that caused the error, if applicable.",
    )


class ErrorResponse(BaseModel):
    """Top-level error envelope."""

    error: ErrorDetail
