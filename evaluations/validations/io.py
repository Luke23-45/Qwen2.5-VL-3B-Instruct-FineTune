from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from omegaconf import DictConfig

from evaluations.validations.core.outputs import ValidationOutputWriter, utc_run_id


def ensure_output_dir(root: str | Path, run_id: str, validation_name: str) -> Path:
    path = Path(root).expanduser().resolve() / run_id / validation_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=True)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def write_manifest(
    output_dir: Path,
    *,
    validation_name: str,
    cfg: DictConfig,
    status: str,
    outputs: dict[str, str],
    summary: dict[str, Any] | None = None,
) -> Path:
    writer = ValidationOutputWriter(output_dir.parent.parent, output_dir.parent.name, validation_name, cfg)
    writer.outputs.update(outputs)
    from evaluations.validations.core.contracts import ValidationResult

    writer.finish(
        ValidationResult(
            name=validation_name,
            status=status,
            output_dir=str(output_dir),
            summary=summary or {},
            outputs=outputs,
        )
    )
    return output_dir / "manifest.json"
