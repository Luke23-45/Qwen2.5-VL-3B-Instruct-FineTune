from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    image: Path
    prompt: str
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelPaths:
    base_model: str
    artifact_dir: Path | None
    adapter_path: Path | None
    processor_path: Path | None
