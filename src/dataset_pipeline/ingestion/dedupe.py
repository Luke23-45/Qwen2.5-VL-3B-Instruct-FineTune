from __future__ import annotations

import pandas as pd

from ..core.models import ProjectConfig


def _phash_distance(left: str | None, right: str | None) -> int | None:
    if not left or not right or len(left) != len(right):
        return None
    left_int = int(left, 16)
    right_int = int(right, 16)
    return (left_int ^ right_int).bit_count()


def _retention_score(row: pd.Series, config: ProjectConfig) -> tuple:
    source_cfg = config.sources[row["source_name"]]
    annotation_priority = {"segmentation": 3, "detection": 2, "classification": 1}
    return (
        int(row["label_status"] == "mapped"),
        int(source_cfg.field_realistic and config.dedupe.get("prefer_field_realistic_sources", True)),
        annotation_priority.get(row["annotation_type"], 0),
        source_cfg.source_priority,
        row.get("image_width") or 0,
        row.get("image_height") or 0,
        -(len(row.get("quality_flags") or [])),
        row.get("canonical_sample_id") or "",
    )


def apply_deduplication(config: ProjectConfig, manifest_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = manifest_df.copy()
    if df.empty:
        return df, {"exact_duplicate_clusters": [], "near_duplicate_clusters": [], "lineage": []}

    df["duplicate_of"] = df["duplicate_of"].astype("object")
    df["duplicate_cluster"] = None
    exact_clusters: list[list[str]] = []
    lineage_rows: list[dict] = []

    for _, group in df.dropna(subset=["exact_hash"]).groupby("exact_hash"):
        if len(group) < 2:
            continue
        sorted_group = sorted(group.to_dict(orient="records"), key=lambda row: _retention_score(pd.Series(row), config), reverse=True)
        keeper = sorted_group[0]["canonical_sample_id"]
        cluster_ids = [row["canonical_sample_id"] for row in sorted_group]
        exact_clusters.append(cluster_ids)
        for row in sorted_group:
            mask = df["canonical_sample_id"] == row["canonical_sample_id"]
            df.loc[mask, "duplicate_cluster"] = keeper
            if row["canonical_sample_id"] == keeper:
                df.loc[mask, "duplicate_status"] = "exact_keeper"
            else:
                df.loc[mask, "duplicate_status"] = "exact_removed"
                df.loc[mask, "duplicate_of"] = keeper
                lineage_rows.append({"kept": keeper, "removed": row["canonical_sample_id"], "reason": "exact_duplicate"})

    threshold = int(config.dedupe.get("near_duplicate_hamming_threshold", 8))
    candidate_df = df[df["duplicate_status"].isin(["unique", "exact_keeper"])].copy()
    near_clusters: list[list[str]] = []
    candidate_rows = candidate_df.to_dict(orient="records")
    visited: set[str] = set()

    for idx, left in enumerate(candidate_rows):
        left_id = left["canonical_sample_id"]
        if left_id in visited:
            continue
        cluster = [left]
        for right in candidate_rows[idx + 1:]:
            right_id = right["canonical_sample_id"]
            if right_id in visited:
                continue
            if left["canonical_label"] and right["canonical_label"] and left["canonical_label"] != right["canonical_label"]:
                continue
            distance = _phash_distance(left.get("phash"), right.get("phash"))
            if distance is not None and distance <= threshold:
                cluster.append(right)
                visited.add(right_id)
        if len(cluster) < 2:
            continue
        sorted_cluster = sorted(cluster, key=lambda row: _retention_score(pd.Series(row), config), reverse=True)
        keeper = sorted_cluster[0]["canonical_sample_id"]
        near_clusters.append([row["canonical_sample_id"] for row in sorted_cluster])
        for row in sorted_cluster:
            mask = df["canonical_sample_id"] == row["canonical_sample_id"]
            if row["canonical_sample_id"] == keeper and df.loc[mask, "duplicate_status"].iat[0] == "unique":
                df.loc[mask, "duplicate_status"] = "near_keeper"
            elif row["canonical_sample_id"] != keeper:
                df.loc[mask, "duplicate_status"] = "near_removed"
                df.loc[mask, "duplicate_of"] = keeper
                current = df.loc[mask, "warning_flags"].iat[0] or []
                df.loc[mask, "warning_flags"] = [*current, "near_duplicate"]
                lineage_rows.append({"kept": keeper, "removed": row["canonical_sample_id"], "reason": "near_duplicate"})

    return df, {
        "exact_duplicate_clusters": exact_clusters,
        "near_duplicate_clusters": near_clusters,
        "lineage": lineage_rows,
    }
