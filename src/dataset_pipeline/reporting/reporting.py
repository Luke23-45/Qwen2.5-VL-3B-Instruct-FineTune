from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_validation_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Dataset Validation Report",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Issues", ""])
    if not report.get("issues"):
        lines.append("- No validation issues found.")
    else:
        for issue in report["issues"]:
            lines.append(f"- [{issue['severity'].upper()}] {issue['code']}: {issue['message']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scan_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Raw Data Scan Summary",
        "",
        f"- total_images: {summary.get('total_images', 0)}",
        f"- missing_label_count: {summary.get('missing_label_count', 0)}",
        f"- corrupt_count: {summary.get('corrupt_count', 0)}",
        "",
        "## Per Source Counts",
        "",
    ]
    for key, value in summary.get("per_source_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Per Label Counts", ""])
    for key, value in summary.get("per_label_counts", {}).items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset_card(manifest_df: pd.DataFrame, validation_report: dict, dataset_version: str) -> str:
    sources = ", ".join(sorted(manifest_df["source_name"].unique().tolist())) if not manifest_df.empty else "none"
    crops = ", ".join(sorted(manifest_df["crop_name"].dropna().unique().tolist())) if not manifest_df.empty else "none"
    classes = int(manifest_df["canonical_label"].dropna().nunique()) if not manifest_df.empty else 0
    unresolved = int((manifest_df["label_status"] != "mapped").sum()) if not manifest_df.empty else 0
    duplicate_removed = int(manifest_df["duplicate_status"].isin(["exact_removed", "near_removed"]).sum()) if not manifest_df.empty else 0

    recommendations = []
    if unresolved:
        recommendations.append(f"Review {unresolved} unmapped or ambiguous rows before expanding training scope.")
    if duplicate_removed:
        recommendations.append(f"Audit {duplicate_removed} removed duplicates for false-positive near-duplicate matches.")
    if validation_report["status"] != "passed":
        recommendations.append("Do not promote this dataset build to training until critical validation issues are resolved.")
    if not recommendations:
        recommendations.append("Dataset build is suitable for classification training with manifest-driven loaders.")

    lines = [
        f"# Dataset Card: {dataset_version}",
        "",
        "## Overview",
        "",
        f"- Included sources: {sources}",
        f"- Crop coverage: {crops}",
        f"- Canonical classes: {classes}",
        f"- Total samples: {len(manifest_df)}",
        "",
        "## Known Biases",
        "",
        "- Mixed source conditions remain, including lab-background imagery from PlantVillage-style sources.",
        "- Classification-first curation is implemented; detection and segmentation metadata are preserved but not exported as training labels yet.",
        "",
        "## Holdout Policy",
        "",
        "- Locked test samples are drawn from holdout-eligible field-realistic sources first.",
        "- Duplicate clusters and known groups are kept within a single split to reduce leakage.",
        "",
        "## Leakage Controls",
        "",
        "- Exact duplicate detection uses SHA-256 file hashes.",
        "- Near-duplicate detection uses perceptual hashing with deterministic retention rules.",
        "- Split generation is group-aware and duplicate-aware.",
        "",
        "## Manual Review",
        "",
        f"- Unresolved labels needing review: {unresolved}",
        f"- Removed duplicate rows to audit: {duplicate_removed}",
        "",
        "## Recommended Training Usage",
        "",
    ]
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    return "\n".join(lines)
