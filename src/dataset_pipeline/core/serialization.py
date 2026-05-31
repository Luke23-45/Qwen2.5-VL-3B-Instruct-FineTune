from __future__ import annotations

import json

import pandas as pd


SERIALIZED_COLLECTION_COLUMNS = (
    "quality_flags",
    "warning_flags",
    "lineage",
    "annotation_refs",
    "source_metadata",
)


def restore_serialized_columns(frame: pd.DataFrame, columns: tuple[str, ...] = SERIALIZED_COLLECTION_COLUMNS) -> pd.DataFrame:
    restored = frame.copy()
    for column in columns:
        if column not in restored.columns:
            continue
        default = [] if column not in {"annotation_refs", "source_metadata"} else {}
        restored[column] = restored[column].apply(
            lambda value: json.loads(value) if isinstance(value, str) and value[:1] in "[{" else default
        )
    return restored
