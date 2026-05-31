from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .core.models import ProjectConfig
from .utils.files import create_archive, ensure_dir, extract_archive, write_json


LATEST_DATASET_POINTER = "latest_dataset.json"
DEFAULT_INCLUDED_STAGE_DIRS = ("final", "manifests", "splits", "reports")


@dataclass(slots=True)
class DatasetBundleMetadata:
    dataset_name: str
    dataset_version: str
    bundle_root: str
    included_stage_dirs: list[str]
    archives: dict[str, str]


def dataset_registry_path(config: ProjectConfig) -> Path:
    return config.paths.dataset_root_dir / LATEST_DATASET_POINTER


def stage_latest_pointer(stage_root: Path) -> Path:
    return stage_root / "latest"


def write_latest_dataset_pointer(config: ProjectConfig, dirs: dict[str, Path]) -> Path:
    payload = {
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "dataset_root_dir": str(config.paths.dataset_root_dir),
        "stage_paths": {
            stage: str(path)
            for stage, path in dirs.items()
            if stage in {"raw", "interim", "processed", "final", "manifests", "splits", "reports", "artifacts"}
        },
    }
    pointer_path = dataset_registry_path(config)
    write_json(pointer_path, payload)
    return pointer_path


def resolve_dataset_version(config: ProjectConfig, requested_version: str | None) -> str:
    if requested_version and requested_version != "latest":
        return requested_version

    pointer_path = dataset_registry_path(config)
    if pointer_path.exists():
        with pointer_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)["dataset_version"]

    latest_final = stage_latest_pointer(config.paths.final_dir)
    if latest_final.exists():
        try:
            return latest_final.resolve().name
        except OSError:
            return latest_final.name

    raise FileNotFoundError(
        f"Could not resolve dataset version. No pointer found at {pointer_path} and no usable latest directory exists."
    )


def dataset_stage_dir(config: ProjectConfig, stage_name: str, dataset_version: str) -> Path:
    stage_roots = {
        "raw": config.paths.raw_dir,
        "interim": config.paths.interim_dir,
        "processed": config.paths.processed_dir,
        "final": config.paths.final_dir,
        "manifests": config.paths.manifests_dir,
        "splits": config.paths.splits_dir,
        "reports": config.paths.reports_dir,
        "artifacts": config.paths.artifacts_dir,
    }
    if stage_name not in stage_roots:
        raise KeyError(f"Unknown dataset stage: {stage_name}")
    return stage_roots[stage_name] / dataset_version


def update_stage_latest_links(dirs: dict[str, Path], logger) -> None:
    for stage, path in dirs.items():
        if stage not in {"raw", "interim", "processed", "final", "manifests", "splits", "reports", "artifacts"}:
            continue
        latest_path = path.parent / "latest"
        try:
            latest_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            shutil.rmtree(latest_path, ignore_errors=True)
        try:
            latest_path.symlink_to(path.name, target_is_directory=True)
        except OSError as exc:
            logger.warning("Could not create symlink %s: %s", latest_path, exc)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def stage_bundle_tree(
    config: ProjectConfig,
    dirs: dict[str, Path],
    included_stage_dirs: Iterable[str] = DEFAULT_INCLUDED_STAGE_DIRS,
) -> tuple[Path, Path]:
    bundle_root = ensure_dir(dirs["artifacts"] / "bundle")
    bundle_dir = bundle_root / f"{config.dataset_name}-{config.dataset_version}"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for stage_name in included_stage_dirs:
        if stage_name not in dirs:
            raise KeyError(f"Stage '{stage_name}' is not available in runtime dirs.")
        stage_root = bundle_dir / stage_name / config.dataset_version
        _copy_tree(dirs[stage_name], stage_root)

    return bundle_root, bundle_dir


def build_dataset_archives(
    config: ProjectConfig,
    dirs: dict[str, Path],
    archive_formats: Iterable[str],
    included_stage_dirs: Iterable[str] = DEFAULT_INCLUDED_STAGE_DIRS,
) -> dict[str, str]:
    bundle_root, bundle_dir = stage_bundle_tree(config, dirs, included_stage_dirs)
    outputs: dict[str, str] = {}
    archive_stem = f"{config.dataset_name}-{config.dataset_version}"

    for archive_format in archive_formats:
        suffix = ".tar.gz" if archive_format == "tar.gz" else f".{archive_format}"
        archive_path = dirs["artifacts"] / f"{archive_stem}{suffix}"
        create_archive(bundle_dir, archive_path, archive_format)
        outputs[archive_format] = str(archive_path)

    metadata = DatasetBundleMetadata(
        dataset_name=config.dataset_name,
        dataset_version=config.dataset_version,
        bundle_root=bundle_dir.name,
        included_stage_dirs=list(included_stage_dirs),
        archives=outputs,
    )
    write_json(
        dirs["artifacts"] / "archive_manifest.json",
        {
            "dataset_name": metadata.dataset_name,
            "dataset_version": metadata.dataset_version,
            "bundle_root": metadata.bundle_root,
            "included_stage_dirs": metadata.included_stage_dirs,
            "archives": metadata.archives,
        },
    )
    return outputs


def extract_dataset_archive(archive_path: str | Path, dataset_root_dir: str | Path) -> Path:
    archive = Path(archive_path)
    destination = ensure_dir(Path(dataset_root_dir))
    extracted_root = extract_archive(archive, destination)
    if extracted_root.is_file():
        raise ValueError(f"Archive extraction returned a file instead of a directory: {extracted_root}")

    for stage_dir in extracted_root.iterdir():
        target_stage_root = ensure_dir(destination / stage_dir.name)
        for version_dir in stage_dir.iterdir():
            target_version_dir = target_stage_root / version_dir.name
            if target_version_dir.exists():
                shutil.rmtree(target_version_dir)
            shutil.move(str(version_dir), str(target_version_dir))

    shutil.rmtree(extracted_root, ignore_errors=True)
    return destination
