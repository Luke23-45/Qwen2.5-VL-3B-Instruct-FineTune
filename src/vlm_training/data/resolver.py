from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from dataset_pipeline.core.config import load_project_config
from dataset_pipeline.dataset_store import (
    dataset_registry_path,
    dataset_stage_dir,
    extract_dataset_archive,
    resolve_dataset_version,
)
from dataset_pipeline.utils.files import ensure_dir


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedTrainingDataset:
    dataset_version: str
    final_root: Path
    train_json: Path
    val_json: Path | None


def _required_json_paths(final_root: Path, cfg: DictConfig) -> tuple[Path, Path | None]:
    dataset_subdir = cfg.get("final_subdir", "qwen25_vl_sft")
    train_name = cfg.get("train_json_name", "qwen25_vl_train.json")
    val_name = cfg.get("val_json_name", "qwen25_vl_val.json")
    dataset_dir = final_root / dataset_subdir
    train_json = dataset_dir / train_name
    val_json = dataset_dir / val_name if val_name else None
    return train_json, val_json


def _local_dataset_is_ready(final_root: Path, cfg: DictConfig) -> bool:
    train_json, val_json = _required_json_paths(final_root, cfg)
    return train_json.exists() and (val_json is None or val_json.exists())


def _resolve_archive_name(config, hf_cfg: DictConfig, dataset_version: str) -> str:
    archive_template = hf_cfg.get("archive_name_template")
    if archive_template:
        return str(archive_template).format(
            dataset_name=config.dataset_name,
            dataset_version=dataset_version,
        )

    archive_name = hf_cfg.get("archive_name")
    if archive_name:
        archive_name = str(archive_name)
        if dataset_version != "latest" and "latest" in archive_name:
            return f"{config.dataset_name}-{dataset_version}.tar.gz"
        return archive_name

    return f"{config.dataset_name}-{dataset_version}.tar.gz"


def _download_archive(config, hf_cfg: DictConfig, dataset_version: str, force_download: bool) -> Path:
    repo_id = hf_cfg.get("repo_id")
    archive_name = _resolve_archive_name(config, hf_cfg, dataset_version)
    if not repo_id:
        raise FileNotFoundError(
            "Local dataset is missing and Hugging Face download is not fully configured. "
            "Set data.huggingface.repo_id and optionally data.huggingface.archive_name_template."
        )
    repo_id = str(repo_id)
    if "placeholder" in repo_id or repo_id.startswith("your-org/"):
        raise FileNotFoundError(
            "Local dataset is missing and the Hugging Face dataset repo is still a placeholder. "
            f"Requested dataset_version={dataset_version}, expected archive={archive_name}. "
            "Set data.huggingface.repo_id to the real dataset repository and provide HF_TOKEN if needed."
        )

    revision = hf_cfg.get("revision", "main")
    token_env = hf_cfg.get("token_env", "HF_TOKEN")
    token = os.environ.get(token_env) or hf_cfg.get("token")
    download_dir = ensure_dir(config.paths.artifacts_dir / dataset_version / "remote_cache")

    logger.info("Downloading dataset archive %s from %s@%s", archive_name, repo_id, revision)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=archive_name,
        repo_type=hf_cfg.get("repo_type", "dataset"),
        revision=revision,
        token=token,
        local_dir=str(download_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
    )
    return Path(downloaded)


def prepare_training_dataset(cfg: DictConfig) -> PreparedTrainingDataset:
    pipeline_config_path = Path(cfg.get("pipeline_config", "data/configs/pipeline.yaml")).resolve()
    project_config = load_project_config(pipeline_config_path)
    dataset_version = resolve_dataset_version(project_config, cfg.get("dataset_version", "latest"))
    final_root = dataset_stage_dir(project_config, "final", dataset_version)

    if _local_dataset_is_ready(final_root, cfg):
        train_json, val_json = _required_json_paths(final_root, cfg)
        logger.info("Using local dataset version %s from %s", dataset_version, final_root)
        return PreparedTrainingDataset(dataset_version, final_root, train_json, val_json)

    hf_cfg = cfg.get("huggingface")
    if hf_cfg is None:
        raise FileNotFoundError(
            "Prepared dataset was not found locally and no Hugging Face fallback is configured. "
            f"Checked {final_root} and registry {dataset_registry_path(project_config)}."
        )

    archive_path = _download_archive(
        project_config,
        hf_cfg,
        dataset_version=dataset_version,
        force_download=bool(hf_cfg.get("force_download", False)),
    )
    extract_dataset_archive(archive_path, project_config.paths.dataset_root_dir)

    if not _local_dataset_is_ready(final_root, cfg):
        train_json, val_json = _required_json_paths(final_root, cfg)
        raise FileNotFoundError(
            "Dataset archive was downloaded and extracted, but the expected training files are still missing: "
            f"train={train_json}, val={val_json}"
        )

    train_json, val_json = _required_json_paths(final_root, cfg)
    logger.info("Prepared dataset version %s from downloaded archive.", dataset_version)
    return PreparedTrainingDataset(dataset_version, final_root, train_json, val_json)
