from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vlm_training.runtime.accelerator import prepare_runtime


def main() -> None:
    cfg = OmegaConf.load(ROOT / "configs" / "vlm_training" / "qwen25_vl_3b.yaml")
    plan = prepare_runtime(cfg, ROOT / "outputs" / "runtime_plan.json")
    print(OmegaConf.to_yaml(cfg))
    print(f"state={plan.state}")
    print(f"attention_backend={plan.attention_backend}")
    print(f"use_liger_kernel={plan.use_liger_kernel}")
    print(f"tf32_enabled={plan.tf32_enabled}")


if __name__ == "__main__":
    main()
