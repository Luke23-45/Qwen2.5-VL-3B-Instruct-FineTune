"""Async client that proxies validated requests to the vLLM server.

The vLLM server exposes an **OpenAI-compatible** ``/v1/chat/completions``
endpoint.  This client builds the exact request shape that vLLM expects
for Qwen2.5-VL multimodal inference (inline base64 image in a chat
message) and maps vLLM's responses / errors back to our API's types.

Connection pooling
------------------
A single :class:`httpx.AsyncClient` is reused for the lifetime of the
FastAPI process.  Call :meth:`VLLMClient.close` (or use the async context
manager) on shutdown to release connections cleanly.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class VLLMError(Exception):
    """Raised when the vLLM backend returns an error or is unreachable."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class VLLMInferenceResult:
    """Parsed result from a vLLM chat completion."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


class VLLMClient:
    """Async HTTP client for the vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
        logger.info(
            "vLLM client initialised  url=%s  model=%s  timeout=%ds",
            self._base_url,
            self._model_name,
            timeout_seconds,
        )

    # ── Lifecycle ───────────────────────────────────────────────

    async def close(self) -> None:
        await self._client.aclose()
        logger.info("vLLM client connection pool closed.")

    async def __aenter__(self) -> VLLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── Health ──────────────────────────────────────────────────

    async def is_healthy(self) -> bool:
        """Check if the vLLM server is reachable and serving the model."""
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            body = resp.json()
            model_ids = [m.get("id", "") for m in body.get("data", [])]
            return self._model_name in model_ids
        except Exception:
            return False

    # ── Inference ───────────────────────────────────────────────

    async def infer(
        self,
        image_bytes: bytes,
        image_format: str,
        prompt: str,
        *,
        max_tokens: int = 160,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> VLLMInferenceResult:
        """Send a single image+prompt to vLLM and return the result.

        Parameters
        ----------
        image_bytes:
            Raw image bytes (already validated).
        image_format:
            PIL format string (``JPEG``, ``PNG``, …).  Used for the data-URI
            media type.
        prompt:
            The text prompt for the model.
        max_tokens:
            Max tokens for generation.
        temperature:
            Sampling temperature (0 = greedy).
        do_sample:
            Whether to enable stochastic sampling.
        """
        media_type = _format_to_media_type(image_format)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{media_type};base64,{b64}"

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature if do_sample else 0.0,
            "stream": False,
        }

        started = time.perf_counter()
        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise VLLMError(
                f"vLLM inference timed out after {self._client.timeout.read}s. "
                "The model may be overloaded.",
                status_code=504,
            )
        except httpx.ConnectError:
            raise VLLMError(
                f"Cannot connect to vLLM at {self._base_url}. Is the server running?",
                status_code=503,
            )
        except httpx.RequestError as exc:
            raise VLLMError(
                f"vLLM request failed: {exc}",
                status_code=502,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code != 200:
            _handle_vllm_error(resp)

        body = resp.json()
        return _parse_completion(body, elapsed_ms)


# ── Helpers ─────────────────────────────────────────────────────


_FORMAT_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
}


def _format_to_media_type(fmt: str) -> str:
    mime = _FORMAT_TO_MIME.get(fmt.upper())
    if mime is None:
        raise VLLMError(f"Unsupported image format for vLLM data URI: {fmt}")
    return mime


def _handle_vllm_error(resp: httpx.Response) -> None:
    """Map vLLM HTTP errors to VLLMError with appropriate status codes."""
    try:
        detail = resp.json()
        msg = detail.get("message") or detail.get("error", {}).get("message", resp.text)
    except Exception:
        msg = resp.text or f"HTTP {resp.status_code}"

    if resp.status_code == 429:
        raise VLLMError(f"vLLM rate limited: {msg}", status_code=429)
    if resp.status_code >= 500:
        raise VLLMError(
            f"vLLM internal error (HTTP {resp.status_code}): {msg}",
            status_code=502,
        )
    raise VLLMError(
        f"vLLM rejected the request (HTTP {resp.status_code}): {msg}",
        status_code=502,
    )


def _parse_completion(body: dict[str, Any], latency_ms: float) -> VLLMInferenceResult:
    """Extract text, token counts, and model name from an OpenAI chat response."""
    try:
        choices = body["choices"]
        if not choices:
            raise VLLMError("vLLM returned no choices in the completion response.")
        text = choices[0]["message"]["content"]
        usage = body.get("usage", {})
        return VLLMInferenceResult(
            text=text.strip(),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=body.get("model", "unknown"),
            latency_ms=latency_ms,
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMError(
            f"Unexpected vLLM response structure: {exc}. Body: {body}",
            status_code=502,
        )
