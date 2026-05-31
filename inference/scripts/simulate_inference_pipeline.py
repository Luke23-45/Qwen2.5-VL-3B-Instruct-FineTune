from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


INFERENCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = INFERENCE_ROOT / "config" / "inference.yaml"
IMAGE_PATCH_AREA = 28 * 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run the Qwen2.5-VL inference pipeline without loading model weights."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def cfg_get(config: dict[str, Any], *path: str, default: Any = None) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def visual_token_range(model_cfg: dict[str, Any]) -> dict[str, int]:
    min_pixels = int(model_cfg.get("min_pixels") or 0)
    max_pixels = int(model_cfg.get("max_pixels") or 0)
    return {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "min_visual_tokens": max(min_pixels // IMAGE_PATCH_AREA, 0),
        "max_visual_tokens": max(max_pixels // IMAGE_PATCH_AREA, 0),
    }


def build_trace(config: dict[str, Any], image: Path | None, prompt: str | None) -> dict[str, Any]:
    model_cfg = cfg_get(config, "model", default={})
    quant_cfg = cfg_get(config, "quantization", default={})
    runtime_cfg = cfg_get(config, "runtime", default={})
    generation_cfg = cfg_get(config, "generation", default={})

    return {
        "runner": "transformers_peft",
        "model": {
            "name_or_path": model_cfg.get("name_or_path"),
            "artifact_dir": model_cfg.get("artifact_dir"),
            "adapter_path": model_cfg.get("adapter_path"),
            "processor_path": model_cfg.get("processor_path"),
            "dtype": model_cfg.get("torch_dtype"),
            "attention": model_cfg.get("attn_implementation"),
            "device_map": model_cfg.get("device_map"),
            "use_cache": bool(model_cfg.get("use_cache", True)),
            "use_fast_processor": bool(model_cfg.get("use_fast_processor", True)),
        },
        "vision_budget": visual_token_range(model_cfg),
        "quantization": {
            "enabled": bool(quant_cfg.get("enabled", False)),
            "bits": quant_cfg.get("bits"),
            "type": quant_cfg.get("quant_type"),
            "double_quant": bool(quant_cfg.get("double_quant", False)),
            "compute_dtype": quant_cfg.get("compute_dtype"),
        },
        "runtime": {
            "auto_prepare": bool(runtime_cfg.get("auto_prepare", True)),
            "tf32": bool(runtime_cfg.get("tf32", True)),
            "batch_size": runtime_cfg.get("batch_size"),
            "num_workers": runtime_cfg.get("num_workers"),
        },
        "request": {
            "image": str(image.expanduser().resolve()) if image else cfg_get(config, "input", "image"),
            "prompt_source": "override" if prompt else "config/default",
            "max_new_tokens": int(generation_cfg.get("max_new_tokens", 160)),
            "do_sample": bool(generation_cfg.get("do_sample", False)),
        },
        "stages": [
            "resolve model, adapter, and processor paths",
            "prepare torch runtime and attention backend",
            "load Qwen2.5-VL model and optional PEFT adapter",
            "load AutoProcessor with min/max pixel budget",
            "build Qwen chat messages",
            "load and resize image through qwen_vl_utils",
            "tokenize text and pack image tensors through AutoProcessor",
            "move BatchFeature to model.device",
            "generate output tokens under torch.inference_mode",
            "trim prompt tokens and batch_decode generated ids",
        ],
        "likely_hotspots_without_gpu_profile": [
            "cold model load",
            "image decode and resize",
            "vision encoder prefill",
            "LLM decode loop for max_new_tokens",
            "bitsandbytes 4-bit dequantization if memory is not the limiting factor",
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    trace = build_trace(config, args.image, args.prompt)
    print(json.dumps(trace, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
