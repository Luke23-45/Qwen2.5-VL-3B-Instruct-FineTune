from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluations.validations.runner import ValidationOrchestrator  # noqa: E402


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="validation")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper()))
    logger.info("Validation config:\n%s", OmegaConf.to_yaml(cfg))
    summary = ValidationOrchestrator(cfg).run()
    logger.info("Validation run finished: %s", summary["status"])


if __name__ == "__main__":
    main()
