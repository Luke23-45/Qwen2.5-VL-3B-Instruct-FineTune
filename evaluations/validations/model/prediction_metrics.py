from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluations.validations.core.contracts import ValidationResult  # noqa: E402
from evaluations.validations.core.outputs import ValidationOutputWriter, utc_run_id  # noqa: E402
from evaluations.validations.data import load_qwen_records, resolve_split_json  # noqa: E402
from evaluations.validations.metrics import classification_report, confusion_matrix  # noqa: E402
from evaluations.validations.model.parsing import predicted_canonical_label  # noqa: E402
from evaluations.validations.model.requests import inference_request, prediction_row  # noqa: E402
from inference.src.model import QwenVLInferenceEngine  # noqa: E402
from inference.src.model.utils.batches import batched  # noqa: E402


logger = logging.getLogger(__name__)
VALIDATION_NAME = "prediction_metrics"
PREDICTION_FIELDS = [
    "id",
    "image",
    "split",
    "expected_label",
    "predicted_label",
    "correct",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "target_answer",
    "prediction_text",
]


def run(cfg: DictConfig, run_id: str | None = None) -> ValidationResult:
    run_id = run_id or cfg.run_id or utc_run_id()
    writer = ValidationOutputWriter(cfg.output.root_dir, run_id, VALIDATION_NAME, cfg)
    split = str(cfg.prediction_metrics.split)
    records = _records_for_split(cfg, split)
    engine = QwenVLInferenceEngine.from_config(cfg.inference)
    batch_size = int(engine.batch_size)
    logger.info("Prediction validation batch_size=%s split=%s samples=%s", batch_size, split, len(records))

    rows = []
    total_batches = (len(records) + batch_size - 1) // batch_size
    for record_batch in tqdm(batched(records, batch_size), total=total_batches, desc=f"validating {split}"):
        requests = [inference_request(record) for record in record_batch]
        results = engine.predict_batch(requests)
        rows.extend(_prediction_rows(record_batch, results, split))

    metrics = _metrics(rows, split, str(resolve_split_json(cfg, split)))
    writer.json("metrics.json", metrics)
    writer.jsonl("predictions.jsonl", rows)
    writer.csv("predictions.csv", rows, PREDICTION_FIELDS)
    writer.dataframe("confusion_matrix.csv", confusion_matrix(
        [row["expected_label"] for row in rows],
        [row["predicted_label"] for row in rows],
    ))

    result = ValidationResult(
        name=VALIDATION_NAME,
        status="passed",
        output_dir=str(writer.output_dir),
        summary=metrics,
        outputs=dict(writer.outputs),
    )
    return writer.finish(result)


def _records_for_split(cfg: DictConfig, split: str) -> list[dict]:
    records = load_qwen_records(resolve_split_json(cfg, split))
    limit = cfg.prediction_metrics.get("limit")
    if limit:
        records = records[: int(limit)]
    if not records:
        raise ValueError(f"No records found for split={split}")
    return records


def _prediction_rows(records: list[dict], results: list, split: str) -> list[dict]:
    rows = []
    for record, result in zip(records, results):
        row = prediction_row(record, result, predicted_canonical_label(result.text))
        row["split"] = split
        rows.append(row)
    return rows


def _metrics(rows: list[dict], split: str, split_json: str) -> dict:
    report = classification_report(
        [row["expected_label"] for row in rows],
        [row["predicted_label"] for row in rows],
    )
    report["split"] = split
    report["split_json"] = split_json
    return report


@hydra.main(version_base=None, config_path="../config", config_name="validation")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper()))
    run(cfg)


if __name__ == "__main__":
    main()
