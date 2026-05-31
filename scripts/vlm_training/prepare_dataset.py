from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_pipeline.core.config import load_project_config
from vlm_training.data.resolver import prepare_training_dataset


def main() -> None:
    cfg = OmegaConf.load(Path("configs/vlm_training/qwen25_vl_3b.yaml"))
    prepared = prepare_training_dataset(cfg.data)
    project_config = load_project_config(cfg.data.pipeline_config)
    print(f"dataset_version={prepared.dataset_version}")
    print(f"dataset_root={project_config.paths.dataset_root_dir}")
    print(f"final_root={prepared.final_root}")
    print(f"train_json={prepared.train_json}")
    print(f"val_json={prepared.val_json}")


if __name__ == "__main__":
    main()
