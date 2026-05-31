from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import ProjectConfig, ProjectPaths, SourceConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _default_dataset_version(dataset_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{dataset_name}-{timestamp}"


def load_project_config(config_path: str | Path) -> ProjectConfig:
    pipeline_path = Path(config_path).resolve()
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")

    root = pipeline_path.parents[2] if pipeline_path.parent.name == "configs" else pipeline_path.parent
    taxonomy_path = pipeline_path.with_name("taxonomy.yaml")
    sources_path = pipeline_path.with_name("sources.yaml")

    pipeline_data = _read_yaml(pipeline_path)
    taxonomy_data = _read_yaml(taxonomy_path)
    sources_data = _read_yaml(sources_path)

    dataset_version = os.environ.get("KV_DATASET_VERSION") or pipeline_data.get("dataset_version") or _default_dataset_version(pipeline_data["dataset_name"])
    paths_cfg = pipeline_data["paths"]
    processed_dir_value = paths_cfg.get("processed_dir") or paths_cfg.get("final_dir")
    if not processed_dir_value:
        raise ValueError("Pipeline config paths must define either 'processed_dir' or 'final_dir'.")
    final_dir_value = paths_cfg.get("final_dir") or processed_dir_value
    paths = ProjectPaths(
        root=root,
        source_cache_dir=_resolve(root, paths_cfg.get("source_cache_dir", "data/source_cache")),
        dataset_root_dir=_resolve(root, paths_cfg.get("dataset_root_dir", "data/datasets")),
        raw_dir=_resolve(root, paths_cfg["raw_dir"]),
        interim_dir=_resolve(root, paths_cfg["interim_dir"]),
        processed_dir=_resolve(root, processed_dir_value),
        final_dir=_resolve(root, final_dir_value),
        reports_dir=_resolve(root, paths_cfg["reports_dir"]),
        manifests_dir=_resolve(root, paths_cfg["manifests_dir"]),
        splits_dir=_resolve(root, paths_cfg["splits_dir"]),
        artifacts_dir=_resolve(root, paths_cfg.get("artifacts_dir", "data/datasets/artifacts")),
    )

    sources: dict[str, SourceConfig] = {}
    for name, raw_source in sources_data.get("sources", {}).items():
        path = _resolve(root, raw_source["path"])
        local_mirror = raw_source.get("local_mirror")
        sources[name] = SourceConfig(
            name=name,
            adapter=raw_source["adapter"],
            path=path,
            data_subdir=raw_source.get("data_subdir"),
            enabled=bool(raw_source.get("enabled", True)),
            source_name=raw_source.get("source_name", name),
            expected_format=raw_source.get("expected_format", "folder_per_class"),
            source_trust=raw_source.get("source_trust", "medium"),
            field_realistic=bool(raw_source.get("field_realistic", False)),
            holdout_eligible=bool(raw_source.get("holdout_eligible", False)),
            source_priority=int(raw_source.get("source_priority", 50)),
            local_mirror=_resolve(root, local_mirror) if local_mirror else None,
            url=raw_source.get("url"),
            annotation_file=raw_source.get("annotation_file"),
            checksum_manifest=raw_source.get("checksum_manifest"),
            max_samples_per_class=raw_source.get("max_samples_per_class"),
            allowed_crops=list(raw_source.get("allowed_crops", [])),
            group_keys=list(raw_source.get("group_keys", [])),
            label_strategy=dict(raw_source.get("label_strategy", {})),
            extra={k: v for k, v in raw_source.items() if k not in {
                "adapter", "path", "enabled", "source_name", "expected_format", "source_trust",
                "field_realistic", "holdout_eligible", "source_priority", "local_mirror", "url",
                "annotation_file", "checksum_manifest", "max_samples_per_class", "allowed_crops",
                "group_keys", "label_strategy", "data_subdir"
            }},
        )

    hash_payload = {
        "pipeline": pipeline_data,
        "taxonomy": taxonomy_data,
        "sources": sources_data,
        "dataset_version": dataset_version,
    }

    return ProjectConfig(
        project_name=pipeline_data["project_name"],
        dataset_name=pipeline_data["dataset_name"],
        dataset_version=dataset_version,
        seed=int(pipeline_data["seed"]),
        paths=paths,
        logging=dict(pipeline_data.get("logging", {})),
        stages=dict(pipeline_data.get("stages", {})),
        split=dict(pipeline_data.get("split", {})),
        curation=dict(pipeline_data.get("curation", {})),
        dedupe=dict(pipeline_data.get("dedupe", {})),
        export=dict(pipeline_data.get("export", {})),
        taxonomy=taxonomy_data,
        sources=sources,
        config_hash=_stable_hash(hash_payload),
        pipeline_config_path=pipeline_path,
        taxonomy_config_path=taxonomy_path,
        sources_config_path=sources_path,
    )
