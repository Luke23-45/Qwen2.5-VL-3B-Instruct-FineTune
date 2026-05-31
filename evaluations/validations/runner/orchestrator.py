from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from evaluations.validations.core.contracts import ValidationResult
from evaluations.validations.core.outputs import ValidationOutputWriter, utc_run_id
from evaluations.validations.core.registry import build_registry


class ValidationOrchestrator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.registry = build_registry()

    def run(self) -> dict[str, Any]:
        run_id = self.cfg.run_id or utc_run_id()
        results = [self._run_one(name, run_id) for name in self._enabled()]
        status = "passed" if all(result.passed for result in results) else "failed"
        summary = {
            "run_id": run_id,
            "status": status,
            "validations": [result.as_dict() for result in results],
        }

        writer = ValidationOutputWriter(self.cfg.output.root_dir, run_id, "runner", self.cfg)
        writer.json("summary.json", summary)
        result = ValidationResult(
            name="runner",
            status=status,
            output_dir=str(writer.output_dir),
            summary=summary,
            outputs=dict(writer.outputs),
        )
        writer.finish(result)
        return summary

    def _enabled(self) -> list[str]:
        names = list(self.cfg.validations.enabled)
        unknown = sorted(set(names) - set(self.registry.names()))
        if unknown:
            raise ValueError(f"Unknown validation(s): {unknown}. Available: {self.registry.names()}")
        return names

    def _run_one(self, name: str, run_id: str) -> ValidationResult:
        return self.registry.get(name)(self.cfg, run_id)
