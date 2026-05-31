from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..core.models import ProjectConfig, ValidationIssue


REQUIRED_COLUMNS = {
    "canonical_sample_id",
    "dataset_version",
    "source_name",
    "source_id",
    "original_path",
    "crop_name",
    "disease_name",
    "canonical_label",
    "split",
    "group_id",
    "exact_hash",
    "phash",
    "duplicate_status",
    "quality_flags",
    "warning_flags",
    "image_width",
    "image_height",
    "file_format",
    "label_status",
}


def validate_manifest(config: ProjectConfig, manifest_df: pd.DataFrame, locked_test_reference: Path | None = None) -> dict:
    issues: list[ValidationIssue] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(manifest_df.columns))
    if missing_columns:
        issues.append(ValidationIssue("missing_columns", "critical", "Manifest is missing required columns.", {"columns": missing_columns}))

    if manifest_df.empty:
        issues.append(ValidationIssue("empty_manifest", "critical", "Manifest is empty."))
    else:
        corrupt_rows = manifest_df["quality_flags"].apply(lambda items: "corrupt_or_unreadable" in (items or []) or "scan_error" in (items or []))
        if corrupt_rows.any():
            issues.append(ValidationIssue("corrupt_rows_present", "critical", "Manifest contains corrupt or unreadable samples.", {"count": int(corrupt_rows.sum())}))

        training_rows = manifest_df["split"].isin(["train", "val"])
        unmapped_training = manifest_df.loc[training_rows, "label_status"].ne("mapped").sum()
        if unmapped_training:
            issues.append(ValidationIssue("unmapped_training_rows", "critical", "Train/val rows contain unmapped labels.", {"count": int(unmapped_training)}))

        sample_leakage = manifest_df.groupby("canonical_sample_id")["split"].nunique().gt(1).sum()
        if sample_leakage:
            issues.append(ValidationIssue("sample_leakage", "critical", "Sample IDs appear in multiple splits.", {"count": int(sample_leakage)}))

        dup_df = manifest_df[manifest_df["split"].isin(["train", "val", "test"])].copy()
        dup_df["dup_group"] = dup_df["duplicate_of"].fillna(dup_df["canonical_sample_id"])
        duplicate_leakage_count = int((dup_df.groupby("dup_group")["split"].nunique() > 1).sum())
        if duplicate_leakage_count:
            issues.append(ValidationIssue("duplicate_leakage", "critical", "Duplicate groups cross split boundaries.", {"count": duplicate_leakage_count}))

        grouped = manifest_df[manifest_df["split"].isin(["train", "val", "test"])].assign(
            effective_group=manifest_df.loc[manifest_df["split"].isin(["train", "val", "test"]), "group_id"].fillna(
                manifest_df.loc[manifest_df["split"].isin(["train", "val", "test"]), "canonical_sample_id"]
            )
        )
        group_leakage = int((grouped.groupby("effective_group")["split"].nunique() > 1).sum())
        if group_leakage:
            issues.append(ValidationIssue("group_leakage", "critical", "Known groups cross split boundaries.", {"count": group_leakage}))

        split_counts = manifest_df["split"].value_counts().to_dict()
        for split_name in ("train", "val", "test"):
            if split_counts.get(split_name, 0) == 0:
                issues.append(ValidationIssue("missing_split", "critical", f"{split_name} split is empty."))

        mapped = manifest_df[manifest_df["label_status"] == "mapped"]
        class_counts = mapped["canonical_label"].value_counts()
        threshold = int(config.split.get("minimum_class_count_warning", 5))
        underfilled = class_counts[class_counts < threshold]
        if not underfilled.empty:
            issues.append(ValidationIssue("class_imbalance_warning", "warning", "Some classes have very low counts.", {"classes": underfilled.to_dict()}))

        tiny_images = ((manifest_df["image_width"].fillna(0) < 64) | (manifest_df["image_height"].fillna(0) < 64)).sum()
        if tiny_images:
            issues.append(ValidationIssue("tiny_images", "warning", "Some images are smaller than 64px on one edge.", {"count": int(tiny_images)}))

        if locked_test_reference and locked_test_reference.exists():
            reference_df = pd.read_csv(locked_test_reference)
            current = sorted(manifest_df.loc[manifest_df["is_locked_test"], "canonical_sample_id"].tolist())
            previous = sorted(reference_df["canonical_sample_id"].tolist())
            if current != previous:
                issues.append(ValidationIssue("locked_holdout_changed", "critical", "Locked holdout differs from the reference manifest."))

    return {
        "status": "failed" if any(issue.severity == "critical" for issue in issues) else "passed",
        "issues": [asdict(issue) for issue in issues],
        "summary": {
            "total_rows": int(len(manifest_df)),
            "split_counts": manifest_df["split"].value_counts().to_dict() if not manifest_df.empty else {},
            "mapped_rows": int((manifest_df["label_status"] == "mapped").sum()) if not manifest_df.empty else 0,
        },
    }
