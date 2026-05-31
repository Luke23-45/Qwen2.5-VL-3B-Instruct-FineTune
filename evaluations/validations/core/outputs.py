from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .contracts import ValidationResult


def utc_run_id(prefix: str = "validation") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


class ValidationOutputWriter:
    def __init__(self, root_dir: str | Path, run_id: str, validation_name: str, cfg: DictConfig):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.run_id = run_id
        self.validation_name = validation_name
        self.cfg = cfg
        self.output_dir = self.root_dir / run_id / validation_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.outputs: dict[str, str] = {}

    def json(self, name: str, payload: Any) -> Path:
        path = self.output_dir / name
        _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        self.outputs[path.stem] = str(path)
        return path

    def jsonl(self, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.output_dir / name
        lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
        _atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))
        self.outputs[path.stem] = str(path)
        return path

    def csv(self, name: str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        self.outputs[path.stem] = str(path)
        return path

    def dataframe(self, name: str, df: pd.DataFrame, index: bool = True) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                df.to_csv(handle, index=index)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        self.outputs[path.stem] = str(path)
        return path

    def finish(self, result: ValidationResult) -> ValidationResult:
        manifest = {
            "validation_name": result.name,
            "status": result.status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": result.outputs,
            "summary": result.summary,
            "issues": result.issues,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        self.json("manifest.json", manifest)
        return result
