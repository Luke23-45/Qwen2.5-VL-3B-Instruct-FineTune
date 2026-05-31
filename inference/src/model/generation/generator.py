from __future__ import annotations

import time
from typing import Any

from omegaconf import DictConfig

from ..config.settings import cfg_get
from ..core.types import InferenceRequest, InferenceResult
from .messages import render_chat_texts
from .vision import collect_vision_inputs


class QwenVLGenerator:
    def __init__(self, model: Any, processor: Any, cfg: DictConfig):
        self.model = model
        self.processor = processor
        self.cfg = cfg

    def generate(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        import torch

        if not requests:
            return []

        texts, messages_batch = render_chat_texts(self.processor, requests)
        image_inputs, video_inputs = collect_vision_inputs(messages_batch)
        started = time.perf_counter()
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        per_item_latency = elapsed_ms / max(len(decoded), 1)

        return [
            InferenceResult(
                text=text.strip(),
                latency_ms=per_item_latency,
                input_tokens=int(len(input_ids)),
                output_tokens=int(len(output_ids)),
                sample_id=request.sample_id,
                metadata=request.metadata,
            )
            for request, input_ids, output_ids, text in zip(requests, inputs.input_ids, trimmed, decoded)
        ]

    def _generation_kwargs(self) -> dict[str, Any]:
        gen_cfg = self.cfg.generation
        do_sample = bool(cfg_get(gen_cfg, "do_sample", False))
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(cfg_get(gen_cfg, "max_new_tokens", 160)),
            "do_sample": do_sample,
            "top_p": float(cfg_get(gen_cfg, "top_p", 1.0)),
            "repetition_penalty": float(cfg_get(gen_cfg, "repetition_penalty", 1.0)),
        }
        if do_sample:
            kwargs["temperature"] = float(cfg_get(gen_cfg, "temperature", 0.7))
        return kwargs
