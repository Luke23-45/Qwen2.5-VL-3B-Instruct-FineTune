from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourceConfig:
    name: str
    adapter: str
    path: Path
    data_subdir: str | None = None
    enabled: bool = True
    source_name: str | None = None
    expected_format: str = "folder_per_class"
    source_trust: str = "medium"
    field_realistic: bool = False
    holdout_eligible: bool = False
    source_priority: int = 50
    local_mirror: Path | None = None
    url: str | None = None
    annotation_file: str | None = None
    checksum_manifest: str | None = None
    max_samples_per_class: int | None = None
    allowed_crops: list[str] = field(default_factory=list)
    group_keys: list[str] = field(default_factory=list)
    label_strategy: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    source_cache_dir: Path
    dataset_root_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    final_dir: Path
    reports_dir: Path
    manifests_dir: Path
    splits_dir: Path
    artifacts_dir: Path


@dataclass(slots=True)
class ProjectConfig:
    project_name: str
    dataset_name: str
    dataset_version: str
    seed: int
    paths: ProjectPaths
    logging: dict[str, Any]
    stages: dict[str, bool]
    split: dict[str, Any]
    curation: dict[str, Any]
    dedupe: dict[str, Any]
    export: dict[str, Any]
    taxonomy: dict[str, Any]
    sources: dict[str, SourceConfig]
    config_hash: str
    pipeline_config_path: Path
    taxonomy_config_path: Path
    sources_config_path: Path


@dataclass(slots=True)
class SourceRecord:
    source_name: str
    source_id: str
    original_path: str
    relative_path: str
    source_label: str | None
    metadata_path: str | None = None
    annotation_refs: dict[str, str] = field(default_factory=dict)
    group_hint: str | None = None
    source_split_original: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScanRecord(SourceRecord):
    image_width: int | None = None
    image_height: int | None = None
    aspect_ratio: float | None = None
    file_size_bytes: int | None = None
    file_format: str | None = None
    image_mode: str | None = None
    exact_hash: str | None = None
    phash: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    warning_flags: list[str] = field(default_factory=list)
    scan_status: str = "ok"
    scan_error: str | None = None


@dataclass(slots=True)
class ManifestRecord:
    canonical_sample_id: str
    dataset_version: str
    source_name: str
    source_id: str
    source_split_original: str | None
    original_path: str
    relative_path: str
    crop_name: str | None
    disease_name: str | None
    canonical_label: str | None
    task_type: str
    annotation_type: str
    split: str
    is_locked_test: bool
    group_id: str | None
    exact_hash: str | None
    phash: str | None
    duplicate_status: str
    duplicate_of: str | None
    quality_flags: list[str]
    warning_flags: list[str]
    image_width: int | None
    image_height: int | None
    aspect_ratio: float | None
    file_size_bytes: int | None
    file_format: str | None
    label_status: str
    lineage: list[str]
    notes: str | None
    created_at: str
    annotation_refs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_flags"] = sorted(set(self.quality_flags))
        payload["warning_flags"] = sorted(set(self.warning_flags))
        payload["lineage"] = list(dict.fromkeys(self.lineage))
        return payload


@dataclass(slots=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    stage_name: str
    outputs: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
