from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from vlm_training.data.resolver import prepare_training_dataset


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def resolve_split_json(cfg: DictConfig, split: str) -> Path:
    prepared = prepare_training_dataset(cfg.data)
    dataset_dir = prepared.final_root / cfg.data.get("final_subdir", "qwen25_vl_sft")
    explicit_key = f"{split}_json_name"
    json_name = cfg.data.get(explicit_key) or f"qwen25_vl_{split}.json"
    path = dataset_dir / str(json_name)
    if not path.exists():
        if split == "val" and prepared.val_json and prepared.val_json.exists():
            return prepared.val_json
        if split == "train" and prepared.train_json.exists():
            return prepared.train_json
        raise FileNotFoundError(f"Validation split file does not exist: {path}")
    return path


def load_qwen_records(json_path: str | Path) -> list[dict[str, Any]]:
    path = Path(json_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")

    resolved: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {path} is not an object.")
        item = dict(record)
        image = item.get("image")
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            item["image"] = str(image_path)
        resolved.append(item)
    return resolved


def user_prompt(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    if not conversations:
        raise ValueError(f"Record {record.get('id')} has no conversations.")
    value = str(conversations[0].get("value", ""))
    return value.replace("<image>", "").strip()


def target_answer(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError(f"Record {record.get('id')} has no assistant target.")
    return str(conversations[1].get("value", "")).strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def canonical_label_from_record(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") or {}
    if metadata.get("canonical_label"):
        return str(metadata["canonical_label"])
    parsed = parse_json_object(target_answer(record))
    if parsed and parsed.get("canonical_label"):
        return str(parsed["canonical_label"])
    return None
