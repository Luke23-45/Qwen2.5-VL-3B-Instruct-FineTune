from __future__ import annotations

from collections.abc import Callable

from omegaconf import DictConfig

from .contracts import ValidationResult


ValidationFn = Callable[[DictConfig, str], ValidationResult]


class ValidationRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ValidationFn] = {}

    def register(self, name: str, fn: ValidationFn) -> None:
        if name in self._items:
            raise ValueError(f"Validation already registered: {name}")
        self._items[name] = fn

    def get(self, name: str) -> ValidationFn:
        try:
            return self._items[name]
        except KeyError as exc:
            raise ValueError(f"Unknown validation '{name}'. Available: {self.names()}") from exc

    def names(self) -> list[str]:
        return sorted(self._items)


def build_registry() -> ValidationRegistry:
    from evaluations.validations.dataset_validation import run as run_dataset_validation
    from evaluations.validations.model.prediction_metrics import run as run_prediction_metrics

    registry = ValidationRegistry()
    registry.register("dataset_validation", run_dataset_validation)
    registry.register("prediction_metrics", run_prediction_metrics)
    return registry
