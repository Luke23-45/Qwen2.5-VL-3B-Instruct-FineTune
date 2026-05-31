from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .labels import infer_category, infer_split
from .qwen_export import iter_qwen_json_records
from .types import DatasetRoot, ImageRecord, JsonDatasetRecord
from .walk import iter_image_paths


@dataclass(slots=True)
class DatasetScanResult:
    image_records: list[ImageRecord]
    json_records: list[JsonDatasetRecord]
    issues: list[dict[str, Any]]


class DatasetScanner:
    def __init__(self, check_decode: bool = True):
        self.check_decode = check_decode

    def scan(self, roots: list[DatasetRoot]) -> DatasetScanResult:
        image_records: list[ImageRecord] = []
        json_records: list[JsonDatasetRecord] = []
        issues: list[dict[str, Any]] = []

        for root in roots:
            if not root.path.exists():
                issues.append({"severity": "warning", "root": root.name, "issue": "root_missing", "path": str(root.path)})
                continue
            image_records.extend(self._scan_images(root, issues))
            if root.kind == "qwen_export":
                json_records.extend(iter_qwen_json_records(root.path))

        missing_json_images = [
            {
                "severity": "error",
                "issue": "qwen_record_image_missing",
                "split": record.split,
                "sample_id": record.sample_id,
                "image": str(record.image) if record.image else None,
            }
            for record in json_records
            if not record.exists
        ]
        issues.extend(missing_json_images)
        return DatasetScanResult(image_records=image_records, json_records=json_records, issues=issues)

    def _scan_images(self, root: DatasetRoot, issues: list[dict[str, Any]]) -> list[ImageRecord]:
        records = []
        for image_path in iter_image_paths(root.path):
            records.append(self._image_record(root, image_path, issues))
        return records

    def _image_record(self, root: DatasetRoot, image_path: Path, issues: list[dict[str, Any]]) -> ImageRecord:
        stat = image_path.stat()
        width = height = None
        mode = file_format = None
        status = "ok"
        error = None

        if self.check_decode:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    mode = image.mode
                    file_format = image.format
                    image.verify()
            except Exception as exc:  # noqa: BLE001 - validation records exact sample failures.
                status = "decode_failed"
                error = str(exc)
                issues.append({"severity": "error", "issue": "image_decode_failed", "image": str(image_path), "error": error})

        return ImageRecord(
            root_name=root.name,
            root_kind=root.kind,
            path=image_path,
            relative_path=str(image_path.relative_to(root.path)),
            category=infer_category(root.path, image_path),
            split=infer_split(root.path, image_path),
            file_size_bytes=stat.st_size,
            width=width,
            height=height,
            mode=mode,
            file_format=file_format,
            status=status,
            error=error,
        )


def image_records_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "root_name": record.root_name,
                "root_kind": record.root_kind,
                "path": str(record.path),
                "relative_path": record.relative_path,
                "category": record.category,
                "split": record.split,
                "file_size_bytes": record.file_size_bytes,
                "width": record.width,
                "height": record.height,
                "mode": record.mode,
                "file_format": record.file_format,
                "status": record.status,
                "error": record.error,
            }
            for record in records
        ]
    )


def json_records_dataframe(records: list[JsonDatasetRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": record.split,
                "json_path": str(record.json_path),
                "sample_id": record.sample_id,
                "image": str(record.image) if record.image else None,
                "category": record.category,
                "source_name": record.source_name,
                "exists": record.exists,
            }
            for record in records
        ]
    )


def category_summary(records: list[ImageRecord], json_records: list[JsonDatasetRecord]) -> dict[str, Any]:
    image_categories = Counter(record.category or "__unknown__" for record in records)
    json_categories = Counter(record.category or "__unknown__" for record in json_records)
    split_counts = Counter(record.split or "__unknown__" for record in records)
    json_split_counts = Counter(record.split for record in json_records)
    root_counts = Counter(record.root_name for record in records)
    return {
        "total_images_found": len(records),
        "total_qwen_records": len(json_records),
        "images_per_category": dict(sorted(image_categories.items())),
        "qwen_records_per_category": dict(sorted(json_categories.items())),
        "images_per_split": dict(sorted(split_counts.items())),
        "qwen_records_per_split": dict(sorted(json_split_counts.items())),
        "images_per_root": dict(sorted(root_counts.items())),
        "decode_failed_images": sum(1 for record in records if record.status != "ok"),
        "missing_qwen_images": sum(1 for record in json_records if not record.exists),
    }
