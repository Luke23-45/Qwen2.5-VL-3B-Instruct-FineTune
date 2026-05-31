from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluations.validations.data import canonical_label_from_record, target_answer, user_prompt
from inference.src.model import InferenceRequest, InferenceResult


def inference_request(record: dict[str, Any]) -> InferenceRequest:
    return InferenceRequest(
        image=Path(str(record["image"])).expanduser().resolve(),
        prompt=user_prompt(record),
        sample_id=record.get("id"),
        metadata={"expected_label": canonical_label_from_record(record), "target_answer": target_answer(record)},
    )


def prediction_row(record: dict[str, Any], result: InferenceResult, predicted_label: str) -> dict[str, Any]:
    expected_label = canonical_label_from_record(record) or "__missing_label__"
    return {
        "id": record.get("id"),
        "image": record.get("image"),
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "correct": expected_label == predicted_label,
        "target_answer": target_answer(record),
        "prediction_text": result.text,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
