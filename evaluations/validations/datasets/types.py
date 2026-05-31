from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetRoot:
    name: str
    path: Path
    kind: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ImageRecord:
    root_name: str
    root_kind: str
    path: Path
    relative_path: str
    category: str | None
    split: str | None
    file_size_bytes: int | None
    width: int | None
    height: int | None
    mode: str | None
    file_format: str | None
    status: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JsonDatasetRecord:
    split: str
    json_path: Path
    sample_id: str | None
    image: Path | None
    category: str | None
    source_name: str | None
    exists: bool
    metadata: dict[str, Any] = field(default_factory=dict)
