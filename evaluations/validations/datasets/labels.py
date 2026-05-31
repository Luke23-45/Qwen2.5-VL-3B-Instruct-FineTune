from __future__ import annotations

from pathlib import Path


KNOWN_SPLITS = {"train", "val", "validation", "test"}


def infer_split(root: Path, image_path: Path) -> str | None:
    parts = image_path.relative_to(root).parts
    for part in parts:
        normalized = part.lower()
        if normalized in KNOWN_SPLITS:
            return "val" if normalized == "validation" else normalized
    return None


def infer_category(root: Path, image_path: Path) -> str | None:
    relative_parts = image_path.relative_to(root).parts
    split = infer_split(root, image_path)
    if split and split in relative_parts:
        index = relative_parts.index(split)
        if index + 1 < len(relative_parts) - 1:
            return relative_parts[index + 1]
    if len(relative_parts) > 1:
        return relative_parts[-2]
    return None
