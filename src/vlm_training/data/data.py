from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, Image, load_dataset


def load_qwen_json_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path).resolve()
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset_root = dataset_path.parent

    def _resolve_image_path(example: dict[str, Any]) -> dict[str, Any]:
        image_value = example.get("image")
        if isinstance(image_value, str):
            image_path = Path(image_value)
            if not image_path.is_absolute():
                example["image"] = str((dataset_root / image_path).resolve())
        return example

    dataset = dataset.map(_resolve_image_path)
    return dataset.cast_column("image", Image())


def to_trl_messages(example: dict[str, Any]) -> dict[str, Any]:
    conversations = example["conversations"]
    user_text = conversations[0]["value"].replace("<image>", "").strip()
    assistant_text = conversations[1]["value"].strip()
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
    }
