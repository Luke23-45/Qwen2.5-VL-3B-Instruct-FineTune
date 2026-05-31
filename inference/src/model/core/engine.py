from pathlib import Path

from omegaconf import DictConfig

from ..config.defaults import DEFAULT_DISEASE_PROMPT
from ..config.performance import resolve_batch_size
from ..generation.generator import QwenVLGenerator
from ..loading.loader import QwenVLLoader
from ..utils.batches import batched
from .types import InferenceRequest, InferenceResult


class QwenVLInferenceEngine:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.generator: QwenVLGenerator | None = None
        self.runtime_plan = None
        self.batch_size = 1

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "QwenVLInferenceEngine":
        engine = cls(cfg)
        engine.load()
        return engine

    def load(self) -> None:
        loaded = QwenVLLoader(self.cfg).load()
        self.generator = QwenVLGenerator(loaded.model, loaded.processor, self.cfg)
        self.runtime_plan = loaded.runtime_plan
        self.batch_size = resolve_batch_size(self.cfg, loaded.runtime_plan)
        self.cfg.runtime.batch_size = self.batch_size

    def predict_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        if self.generator is None:
            raise RuntimeError("Inference engine is not loaded.")
        return self.generator.generate(requests)

    def predict_many(self, requests: list[InferenceRequest], batch_size: int) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for batch in batched(requests, batch_size):
            results.extend(self.predict_batch(batch))
        return results

    def predict_one(self, image: str | Path, prompt: str | None = None) -> InferenceResult:
        request = InferenceRequest(image=Path(image).expanduser().resolve(), prompt=prompt or DEFAULT_DISEASE_PROMPT)
        return self.predict_batch([request])[0]
