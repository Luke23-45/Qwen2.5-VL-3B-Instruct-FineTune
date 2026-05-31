from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from dataset_pipeline.core.config import load_project_config
from vlm_training.data.resolver import prepare_training_dataset

from .types import DatasetRoot


def configured_roots(cfg: DictConfig) -> list[DatasetRoot]:
    roots = []
    for item in cfg.dataset_validation.get("dataset_roots", []):
        roots.append(
            DatasetRoot(
                name=str(item["name"]),
                path=Path(str(item["path"])).expanduser().resolve(),
                kind=str(item.get("kind", "generic")),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return roots


def project_stage_roots(cfg: DictConfig) -> list[DatasetRoot]:
    if not bool(cfg.dataset_validation.get("include_project_stage_roots", True)):
        return []
    project_config = load_project_config(cfg.data.pipeline_config)
    stages = {
        "raw": project_config.paths.raw_dir,
        "interim": project_config.paths.interim_dir,
        "final": project_config.paths.final_dir,
        "splits": project_config.paths.splits_dir,
        "manifests": project_config.paths.manifests_dir,
    }
    return [DatasetRoot(name=name, path=path, kind=f"project_{name}") for name, path in stages.items()]


def prepared_export_root(cfg: DictConfig) -> DatasetRoot:
    prepared = prepare_training_dataset(cfg.data)
    dataset_dir = prepared.final_root / cfg.data.get("final_subdir", "qwen25_vl_sft")
    return DatasetRoot(name="prepared_qwen_export", path=dataset_dir, kind="qwen_export")


def resolve_dataset_roots(cfg: DictConfig) -> list[DatasetRoot]:
    roots = configured_roots(cfg)
    if bool(cfg.dataset_validation.get("include_prepared_qwen_export", True)):
        roots.append(prepared_export_root(cfg))
    roots.extend(project_stage_roots(cfg))

    deduped: dict[tuple[str, str], DatasetRoot] = {}
    for root in roots:
        if root.enabled:
            deduped[(root.name, str(root.path))] = root
    return list(deduped.values())


def safe_resolve_dataset_roots(cfg: DictConfig) -> tuple[list[DatasetRoot], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    roots = configured_roots(cfg)

    if bool(cfg.dataset_validation.get("include_prepared_qwen_export", True)):
        try:
            roots.append(prepared_export_root(cfg))
        except Exception as exc:  # noqa: BLE001 - dataset validation should report config/data availability issues.
            issues.append(
                {
                    "severity": "error" if bool(cfg.dataset_validation.get("require_prepared_qwen_export", True)) else "warning",
                    "issue": "prepared_qwen_export_unavailable",
                    "error": str(exc),
                }
            )

    try:
        roots.extend(project_stage_roots(cfg))
    except Exception as exc:  # noqa: BLE001
        issues.append({"severity": "warning", "issue": "project_stage_roots_unavailable", "error": str(exc)})

    deduped: dict[tuple[str, str], DatasetRoot] = {}
    for root in roots:
        if root.enabled:
            deduped[(root.name, str(root.path))] = root
    return list(deduped.values()), issues
