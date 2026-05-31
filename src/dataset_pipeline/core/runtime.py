from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import load_project_config
from ..export.export import dataset_output_dirs
from .models import ProjectConfig
from ..utils.logging import configure_logging
from ..utils.stage import is_stage_reusable


@dataclass(slots=True)
class StageRuntime:
    config: ProjectConfig
    dirs: dict[str, Path]
    logger: object


def build_stage_runtime(config_path: str | Path, log_name: str) -> StageRuntime:
    config = load_project_config(config_path)
    dirs = dataset_output_dirs(config)
    logger = configure_logging(
        dirs["reports"] / log_name,
        config.logging.get("level", "INFO"),
        config.logging.get("format"),
    )
    return StageRuntime(config=config, dirs=dirs, logger=logger)


def should_skip_stage(runtime: StageRuntime, stage_dir_key: str, stage_name: str, force: bool) -> bool:
    if force:
        return False
    return is_stage_reusable(runtime.dirs[stage_dir_key], stage_name, runtime.config.config_hash)
