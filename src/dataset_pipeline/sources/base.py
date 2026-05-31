from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlretrieve

from ..core.models import SourceConfig, SourceRecord
from ..utils.files import IMAGE_EXTENSIONS, METADATA_EXTENSIONS


class BaseSourceAdapter:
    def __init__(self, config: SourceConfig):
        self.config = config

    @property
    def data_root(self) -> Path:
        if self.config.data_subdir:
            return self.config.path / self.config.data_subdir
        return self.config.path

    def download_or_link(self) -> dict[str, str]:
        path = self.config.local_mirror or self.config.path
        if path.exists():
            return {"status": "available", "path": str(path)}

        download_mode = self.config.extra.get("download_mode", "manual")
        if not self.config.url or download_mode == "manual":
            return {
                "status": "manual_required",
                "path": str(path),
                "message": "Source requires manual placement or mirror setup before execution.",
            }

        if download_mode == "kaggle":
            if not shutil.which("kaggle"):
                return {
                    "status": "manual_required",
                    "path": str(path),
                    "message": "Kaggle CLI not found. Install and authenticate the Kaggle API first.",
                }
            dataset_slug = self.config.extra.get("dataset_slug")
            if not dataset_slug:
                parsed = urlparse(self.config.url)
                slug = parsed.path.strip("/")
                dataset_slug = "/".join(slug.split("/")[-2:]) if slug else None
            if not dataset_slug:
                return {"status": "manual_required", "path": str(path), "message": "Kaggle dataset slug could not be derived."}
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["kaggle", "datasets", "download", dataset_slug, "--path", str(path), "--unzip", "--force"],
                check=True,
            )
            return {"status": "downloaded", "path": str(path), "dataset_slug": dataset_slug}

        if download_mode == "direct":
            path.mkdir(parents=True, exist_ok=True)
            filename = self.config.extra.get("download_filename") or Path(urlparse(self.config.url).path).name or "downloaded_source"
            destination = path / filename
            urlretrieve(self.config.url, destination)
            return {"status": "downloaded", "path": str(destination)}

        return {"status": "manual_required", "path": str(path), "message": f"Unsupported download mode: {download_mode}"}

    def discover(self) -> list[Path]:
        if not self.data_root.exists():
            return []
        return [path for path in self.data_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

    def _metadata_candidates(self, image_path: Path) -> list[Path]:
        return [image_path.with_suffix(ext) for ext in METADATA_EXTENSIONS if image_path.with_suffix(ext).exists()]

    def _read_sidecar_metadata(self, image_path: Path) -> dict:
        for candidate in self._metadata_candidates(image_path):
            if candidate.suffix.lower() == ".json":
                with candidate.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
        return {}

    def _annotation_lookup(self) -> dict[str, dict]:
        annotation_file = self.config.annotation_file
        if not annotation_file:
            return {}
        path = self.config.path / annotation_file
        if not path.exists():
            return {}
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            return {str(item["relative_path"]): item for item in records}
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                return {str(row["relative_path"]): row for row in reader}
        return {}

    def extract_label(self, image_path: Path) -> str | None:
        strategy = self.config.label_strategy.get("type", "parent_dir")
        if strategy == "parent_dir":
            return image_path.parent.name
        if strategy == "metadata_or_parent_dir":
            metadata = self._read_sidecar_metadata(image_path)
            return metadata.get("label") or metadata.get("class") or image_path.parent.name
        if strategy == "annotation_file":
            relative_path = str(image_path.relative_to(self.data_root)).replace("\\", "/")
            record = self._annotation_lookup().get(relative_path)
            if not record:
                return image_path.parent.name if image_path.parent != self.data_root else None
            return record.get("label") or record.get("class") or record.get("canonical_label")
        return image_path.parent.name

    def extract_group_key(self, image_path: Path) -> str | None:
        metadata = self._read_sidecar_metadata(image_path)
        keys: list[str] = []
        for rule in self.config.group_keys:
            if rule == "parent_dir":
                keys.append(image_path.parent.name)
            elif rule == "stem_prefix":
                keys.append(image_path.stem.split("_")[0])
            elif rule.startswith("metadata:"):
                key = rule.split(":", 1)[1]
                if metadata.get(key):
                    keys.append(str(metadata[key]))
        values = [value for value in keys if value]
        return "::".join(values) if values else None

    def extract_annotation_refs(self, image_path: Path) -> dict[str, str]:
        refs: dict[str, str] = {}
        metadata = self._read_sidecar_metadata(image_path)
        for key in ("mask_path", "boxes_path", "annotation_path"):
            if metadata.get(key):
                refs[key] = str((image_path.parent / metadata[key]).resolve())
        return refs

    def scan_records(self) -> Iterable[SourceRecord]:
        label_counts: Counter[str] = Counter()
        class_cap = self.config.max_samples_per_class
        for image_path in self.discover():
            relative_path = str(image_path.relative_to(self.data_root)).replace("\\", "/")
            source_label = self.extract_label(image_path)
            label_key = source_label or "__unlabeled__"
            if class_cap is not None and label_counts[label_key] >= class_cap:
                continue
            label_counts[label_key] += 1
            yield SourceRecord(
                source_name=self.config.source_name or self.config.name,
                source_id=f"{self.config.name}:{relative_path}",
                original_path=str(image_path.resolve()),
                relative_path=relative_path,
                source_label=source_label,
                metadata_path=None,
                annotation_refs=self.extract_annotation_refs(image_path),
                group_hint=self.extract_group_key(image_path),
                source_split_original=None,
                source_metadata={
                    "field_realistic": self.config.field_realistic,
                    "holdout_eligible": self.config.holdout_eligible,
                    "source_priority": self.config.source_priority,
                    "source_trust": self.config.source_trust,
                    "max_samples_per_class": self.config.max_samples_per_class,
                    "allowed_crops": self.config.allowed_crops,
                },
            )
