from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.src.model import DEFAULT_DISEASE_PROMPT, QwenVLInferenceEngine


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper()))
    logger.info("Inference config:\n%s", OmegaConf.to_yaml(cfg))

    if not cfg.input.image:
        raise ValueError("Set input.image=/path/to/image for single-image inference.")

    engine = QwenVLInferenceEngine.from_config(cfg)
    result = engine.predict_one(Path(cfg.input.image).expanduser().resolve(), cfg.input.prompt or DEFAULT_DISEASE_PROMPT)
    payload = {
        "image": str(Path(cfg.input.image).expanduser().resolve()),
        "prediction": result.text,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
