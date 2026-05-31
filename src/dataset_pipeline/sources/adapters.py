from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import BaseSourceAdapter


class LocalFolderAdapter(BaseSourceAdapter):
    pass


class ManualMirrorAdapter(BaseSourceAdapter):
    pass


class URLArchiveAdapter(BaseSourceAdapter):
    pass


class MetadataFolderAdapter(BaseSourceAdapter):
    def _metadata_rows(self) -> dict[str, dict]:
        metadata_file = self.config.annotation_file or self.config.extra.get("metadata_file")
        if not metadata_file:
            return {}
        path = self.config.path / metadata_file
        if not path.exists():
            return {}
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                key_column = self.config.extra.get("annotation_path_column", "relative_path")
                return {str(row.get(key_column, "")).replace("\\", "/"): row for row in reader if row.get(key_column)}
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            key_column = self.config.extra.get("annotation_path_column", "relative_path")
            return {str(row.get(key_column, "")).replace("\\", "/"): row for row in payload if row.get(key_column)}
        return {}

    def _metadata_for(self, image_path: Path) -> dict:
        relative_path = str(image_path.relative_to(self.data_root)).replace("\\", "/")
        rows = self._metadata_rows()
        if relative_path in rows:
            return rows[relative_path]
        filename = image_path.name
        for row in rows.values():
            if row.get("filename") == filename or row.get("image_name") == filename:
                return row
        return self._read_sidecar_metadata(image_path)

    def extract_label(self, image_path: Path) -> str | None:
        metadata = self._metadata_for(image_path)
        label_fields = self.config.extra.get("label_fields", ["label", "class", "disease", "disease_name", "canonical_label"])
        crop_fields = self.config.extra.get("crop_fields", ["crop", "crop_name", "plant", "host"])
        crop_value = next((metadata.get(field) for field in crop_fields if metadata.get(field)), None)
        label_value = next((metadata.get(field) for field in label_fields if metadata.get(field)), None)
        if crop_value and label_value:
            return f"{crop_value}_{label_value}"
        if label_value:
            return str(label_value)
        return super().extract_label(image_path)

    def extract_group_key(self, image_path: Path) -> str | None:
        metadata = self._metadata_for(image_path)
        for field in self.config.extra.get("group_fields", ["group_id", "session_id", "plant_id"]):
            if metadata.get(field):
                return str(metadata[field])
        return super().extract_group_key(image_path)

    def extract_annotation_refs(self, image_path: Path) -> dict[str, str]:
        refs = super().extract_annotation_refs(image_path)
        metadata = self._metadata_for(image_path)
        for field in self.config.extra.get("annotation_ref_fields", ["mask_path", "annotation_path", "boxes_path"]):
            if metadata.get(field):
                refs[field] = str((self.config.path / metadata[field]).resolve())
        return refs


class AnnotationFileAdapter(BaseSourceAdapter):
    def _annotation_rows(self) -> dict[str, dict]:
        annotation_file = self.config.annotation_file or self.config.extra.get("metadata_file")
        if not annotation_file:
            return {}
        path = self.config.path / annotation_file
        if not path.exists():
            return {}
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                key_columns = self.config.extra.get("annotation_path_columns", ["relative_path", "image_path", "filepath", "file_path", "filename"])
                rows = {}
                for row in reader:
                    for key in key_columns:
                        if row.get(key):
                            rows[str(row[key]).replace("\\", "/")] = row
                            break
                return rows
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            key_columns = self.config.extra.get("annotation_path_columns", ["relative_path", "image_path", "filepath", "file_path", "filename"])
            rows = {}
            for row in payload:
                for key in key_columns:
                    if row.get(key):
                        rows[str(row[key]).replace("\\", "/")] = row
                        break
            return rows
        return {}

    def _annotation_for(self, image_path: Path) -> dict:
        relative_path = str(image_path.relative_to(self.data_root)).replace("\\", "/")
        rows = self._annotation_rows()
        if relative_path in rows:
            return rows[relative_path]
        filename = image_path.name
        if filename in rows:
            return rows[filename]
        return {}

    def extract_label(self, image_path: Path) -> str | None:
        row = self._annotation_for(image_path)
        if not row:
            return super().extract_label(image_path)
        label_fields = self.config.extra.get("label_fields", ["label", "class", "canonical_label", "disease", "disease_name"])
        crop_fields = self.config.extra.get("crop_fields", ["crop", "crop_name", "plant", "host"])
        crop_value = next((row.get(field) for field in crop_fields if row.get(field)), None)
        label_value = next((row.get(field) for field in label_fields if row.get(field)), None)
        if crop_value and label_value:
            return f"{crop_value}_{label_value}"
        return str(label_value) if label_value else super().extract_label(image_path)

    def extract_group_key(self, image_path: Path) -> str | None:
        row = self._annotation_for(image_path)
        for field in self.config.extra.get("group_fields", ["group_id", "session_id", "plant_id"]):
            if row.get(field):
                return str(row[field])
        return super().extract_group_key(image_path)

    def extract_annotation_refs(self, image_path: Path) -> dict[str, str]:
        refs = super().extract_annotation_refs(image_path)
        row = self._annotation_for(image_path)
        for field in self.config.extra.get("annotation_ref_fields", ["mask_path", "annotation_path", "boxes_path"]):
            if row.get(field):
                refs[field] = str((self.config.path / row[field]).resolve())
        return refs
