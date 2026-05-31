from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from ..core.models import ProjectConfig
from ..utils.files import ensure_dir, write_dataframe, write_json, write_yaml


def dataset_output_dirs(config: ProjectConfig) -> dict[str, Path]:
    version = config.dataset_version
    final_dir = ensure_dir(config.paths.final_dir / version)
    return {
        "processed": ensure_dir(config.paths.processed_dir / version),
        "final": final_dir,
        "manifests": ensure_dir(config.paths.manifests_dir / version),
        "splits": ensure_dir(config.paths.splits_dir / version),
        "reports": ensure_dir(config.paths.reports_dir / version),
        "interim": ensure_dir(config.paths.interim_dir / version),
        "raw": ensure_dir(config.paths.raw_dir / version),
        "artifacts": ensure_dir(config.paths.artifacts_dir / version),
    }


def export_dataframe(config: ProjectConfig, df: pd.DataFrame, base_path: Path) -> dict[str, str]:
    return write_dataframe(
        df,
        base_path,
        write_csv=bool(config.export.get("write_csv", True)),
        write_parquet=bool(config.export.get("write_parquet", True)),
    )


def save_config_snapshot(config: ProjectConfig, output_dir: Path) -> None:
    write_yaml(output_dir / "pipeline.yaml", {
        "project_name": config.project_name,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "split": config.split,
        "curation": config.curation,
        "dedupe": config.dedupe,
        "export": config.export,
    })
    write_yaml(output_dir / "taxonomy.yaml", config.taxonomy)
    write_yaml(output_dir / "sources.yaml", {
        "sources": {
            name: {
                "adapter": source.adapter,
                "path": str(source.path),
                "data_subdir": source.data_subdir,
                "enabled": source.enabled,
                "source_name": source.source_name,
                "expected_format": source.expected_format,
                "source_trust": source.source_trust,
                "field_realistic": source.field_realistic,
                "holdout_eligible": source.holdout_eligible,
                "source_priority": source.source_priority,
            }
            for name, source in config.sources.items()
        }
    })


def collect_build_metadata(config: ProjectConfig) -> dict:
    git_revision = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.paths.root, check=True, capture_output=True, text=True)
        git_revision = result.stdout.strip()
    except Exception:
        git_revision = None
    return {
        "dataset_version": config.dataset_version,
        "config_hash": config.config_hash,
        "seed": config.seed,
        "git_revision": git_revision,
    }


def write_build_metadata(config: ProjectConfig, output_dir: Path) -> None:
    write_json(output_dir / "build_metadata.json", collect_build_metadata(config))
