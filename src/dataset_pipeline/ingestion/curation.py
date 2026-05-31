from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import pandas as pd

from ..core.models import ProjectConfig


ACTIVE_DUPLICATE_STATUSES = {"unique", "exact_keeper", "near_keeper"}
BLOCKING_QUALITY_FLAGS = {
    "corrupt_or_unreadable",
    "empty_file",
    "scan_error",
}


def _items(value: Any) -> list[str]:
    return value if isinstance(value, list) else []


def _is_candidate(row: pd.Series) -> bool:
    return (
        row.get("label_status") == "mapped"
        and row.get("duplicate_status") in ACTIVE_DUPLICATE_STATUSES
        and not BLOCKING_QUALITY_FLAGS.intersection(_items(row.get("quality_flags")))
    )


def _score(row: pd.Series, config: ProjectConfig) -> tuple:
    source = config.sources[row["source_name"]]
    warning_count = len(_items(row.get("warning_flags")))
    quality_count = len(_items(row.get("quality_flags")))
    area = (row.get("image_width") or 0) * (row.get("image_height") or 0)
    return (
        int(source.source_trust == "high"),
        int(source.field_realistic),
        source.source_priority,
        -warning_count,
        -quality_count,
        area,
        row.get("canonical_sample_id") or "",
    )


def _cap_for_label(config: ProjectConfig, row: pd.Series) -> int:
    curation = config.curation or {}
    per_label_caps = curation.get("per_label_caps", {})
    label = row.get("canonical_label")
    if label in per_label_caps:
        return int(per_label_caps[label])

    crop_caps = curation.get("per_crop_caps", {})
    crop = row.get("crop_name")
    if crop in crop_caps:
        return int(crop_caps[crop])

    source_cfg = config.sources[row["source_name"]]
    if source_cfg.max_samples_per_class:
        return int(source_cfg.max_samples_per_class)

    return int(curation.get("max_samples_per_class", 800))


def apply_strict_curation(config: ProjectConfig, manifest_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mark only the strongest capped samples as trainable.

    The full manifest is preserved for auditability; split generation later ignores rows
    where ``curation_status`` is not ``selected``.
    """

    df = manifest_df.copy()
    if df.empty:
        df["curation_status"] = pd.Series(dtype="object")
        return df, {"selected_rows": 0, "rejected_rows": 0, "per_label_selected": {}}

    if not bool(config.curation.get("enabled", True)):
        df["curation_status"] = "selected"
        return df, {
            "selected_rows": int(len(df)),
            "rejected_rows": 0,
            "per_label_selected": df["canonical_label"].value_counts(dropna=True).to_dict(),
        }

    df["curation_status"] = "rejected_not_candidate"
    candidate_mask = df.apply(_is_candidate, axis=1)
    candidate_df = df[candidate_mask].copy()

    selected_ids: set[str] = set()
    rng = random.Random(config.seed)
    for label, group in candidate_df.groupby("canonical_label", dropna=True):
        if group.empty:
            continue
        first_row = group.iloc[0]
        cap = _cap_for_label(config, first_row)
        min_edge = int(config.curation.get("min_image_edge_px", 128))
        group = group[
            (group["image_width"].fillna(0) >= min_edge)
            & (group["image_height"].fillna(0) >= min_edge)
        ].copy()
        if group.empty:
            continue

        grouped_by_source: defaultdict[str, list[dict]] = defaultdict(list)
        for row in group.to_dict(orient="records"):
            grouped_by_source[row["source_name"]].append(row)

        for rows in grouped_by_source.values():
            rng.shuffle(rows)
            rows.sort(key=lambda item: _score(pd.Series(item), config), reverse=True)

        source_order = sorted(
            grouped_by_source,
            key=lambda name: (
                config.sources[name].source_priority,
                int(config.sources[name].field_realistic),
                int(config.sources[name].source_trust == "high"),
                name,
            ),
            reverse=True,
        )

        label_ids = set(group["canonical_sample_id"])
        label_selected = 0
        while label_selected < cap:
            progressed = False
            for source_name in source_order:
                rows = grouped_by_source[source_name]
                if not rows:
                    continue
                selected_id = rows.pop(0)["canonical_sample_id"]
                if selected_id in label_ids and selected_id not in selected_ids:
                    selected_ids.add(selected_id)
                    label_selected += 1
                progressed = True
                if label_selected >= cap:
                    break
            if not progressed:
                break

    df.loc[candidate_mask, "curation_status"] = "rejected_over_cap"
    df.loc[df["canonical_sample_id"].isin(selected_ids), "curation_status"] = "selected"

    summary = {
        "selected_rows": int((df["curation_status"] == "selected").sum()),
        "rejected_rows": int((df["curation_status"] != "selected").sum()),
        "per_label_selected": df.loc[df["curation_status"] == "selected", "canonical_label"].value_counts().to_dict(),
        "per_label_rejected": df.loc[df["curation_status"] != "selected", "canonical_label"].value_counts().to_dict(),
    }
    return df, summary
