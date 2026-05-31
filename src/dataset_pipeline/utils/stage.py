from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .files import read_json, write_json


def stage_metadata_path(stage_dir: Path, stage_name: str) -> Path:
    return stage_dir / f"{stage_name}_metadata.json"


def is_stage_reusable(stage_dir: Path, stage_name: str, config_hash: str) -> bool:
    metadata_path = stage_metadata_path(stage_dir, stage_name)
    if not metadata_path.exists():
        return False
    metadata = read_json(metadata_path)
    return metadata.get("config_hash") == config_hash


def write_stage_metadata(stage_dir: Path, stage_name: str, payload: dict[str, Any]) -> None:
    metadata = {
        "stage_name": stage_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    write_json(stage_metadata_path(stage_dir, stage_name), metadata)
