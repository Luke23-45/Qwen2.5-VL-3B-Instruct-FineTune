from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluations.validations.data import canonical_label_from_record

from .types import JsonDatasetRecord


def iter_qwen_json_records(export_root: Path) -> list[JsonDatasetRecord]:
    records: list[JsonDatasetRecord] = []
    for json_path in sorted(export_root.glob("qwen25_vl_*.json")):
        split = json_path.stem.replace("qwen25_vl_", "")
        if split == "sft_summary":
            continue
        records.extend(_read_split_records(json_path, split))
    return records


def _read_split_records(json_path: Path, split: str) -> list[JsonDatasetRecord]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return []

    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        image = _resolve_image(json_path.parent, item.get("image"))
        metadata: dict[str, Any] = item.get("metadata") or {}
        records.append(
            JsonDatasetRecord(
                split=split,
                json_path=json_path,
                sample_id=item.get("id"),
                image=image,
                category=canonical_label_from_record(item),
                source_name=metadata.get("source_name"),
                exists=bool(image and image.exists()),
                metadata=metadata,
            )
        )
    return records


def _resolve_image(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
