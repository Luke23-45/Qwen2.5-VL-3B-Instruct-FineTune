from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from ..config.runtime import configure_torch_runtime
from ..config.settings import cfg_get, quantization_config, resolve_model_paths, torch_dtype, validate_model_paths


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadedQwenVL:
    model: Any
    processor: Any
    runtime_plan: Any | None = None


class QwenVLLoader:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.paths = resolve_model_paths(cfg)

    def load(self) -> LoadedQwenVL:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        validate_model_paths(self.paths)
        runtime_plan = configure_torch_runtime(self.cfg, torch)

        model_cfg = self.cfg.model
        logger.info("Loading base model %s", self.paths.base_model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.paths.base_model,
            torch_dtype=torch_dtype(torch, cfg_get(model_cfg, "torch_dtype", "bfloat16")),
            attn_implementation=cfg_get(model_cfg, "attn_implementation", "flash_attention_2"),
            device_map=cfg_get(model_cfg, "device_map", "auto"),
            quantization_config=quantization_config(self.cfg, torch),
            low_cpu_mem_usage=bool(cfg_get(model_cfg, "low_cpu_mem_usage", True)),
        )
        model = self._apply_adapter(model)
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = bool(cfg_get(model_cfg, "use_cache", True))

        processor = self._load_processor()
        return LoadedQwenVL(model=model, processor=processor, runtime_plan=runtime_plan)

    def _apply_adapter(self, model: Any) -> Any:
        if not self.paths.adapter_path:
            return model

        from peft import PeftModel

        logger.info("Loading PEFT adapter from %s", self.paths.adapter_path)
        model = PeftModel.from_pretrained(model, str(self.paths.adapter_path))
        if bool(cfg_get(self.cfg.model, "merge_adapter", False)):
            logger.info("Merging adapter into base model")
            return model.merge_and_unload()
        return model

    def _load_processor(self) -> Any:
        from transformers import AutoProcessor

        model_cfg = self.cfg.model
        processor_source = self.paths.processor_path or self.paths.base_model
        processor_kwargs = {}
        for key in ("min_pixels", "max_pixels"):
            value = cfg_get(model_cfg, key)
            if value is not None:
                processor_kwargs[key] = value

        logger.info("Loading processor from %s", processor_source)
        processor = AutoProcessor.from_pretrained(
            str(processor_source),
            use_fast=bool(cfg_get(model_cfg, "use_fast_processor", True)),
            **processor_kwargs,
        )
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        return processor
