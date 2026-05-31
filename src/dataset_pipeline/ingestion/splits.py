from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from ..core.models import ProjectConfig


def _effective_group_id(row: pd.Series) -> str:
    return row.get("group_id") or row.get("duplicate_of") or row["canonical_sample_id"]


def _assign_groups(group_stats: list[dict], ratio: float, seed: int) -> set[str]:
    target = max(1, round(sum(item["size"] for item in group_stats) * ratio)) if group_stats else 0
    rng = random.Random(seed)
    shuffled = group_stats[:]
    rng.shuffle(shuffled)
    selected: set[str] = set()
    label_counts: defaultdict[str, int] = defaultdict(int)
    total = 0

    for item in sorted(shuffled, key=lambda x: (x["field_priority"], x["size"]), reverse=True):
        if total >= target:
            break
        labels = item["labels"]
        scarcity_bonus = sum(1 for label in labels if label_counts[label] == 0)
        if scarcity_bonus == 0 and total >= target * 0.8:
            continue
        selected.add(item["group_id"])
        total += item["size"]
        for label in labels:
            label_counts[label] += 1
    return selected


def _ensure_validation_coverage(df: pd.DataFrame, active_df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped = active_df.groupby("canonical_label")
    for label, label_df in grouped:
        if label_df.empty:
            continue
        current_val = result.loc[result["canonical_sample_id"].isin(label_df["canonical_sample_id"]), "split"].eq("val").any()
        distinct_groups = label_df["effective_group_id"].nunique()
        if current_val or distinct_groups < 2:
            continue
        candidate_groups = []
        for group_id, group in label_df.groupby("effective_group_id"):
            group_sample_ids = set(group["canonical_sample_id"])
            split_name = result.loc[result["canonical_sample_id"].isin(group_sample_ids), "split"].mode().iat[0]
            if split_name != "train":
                continue
            candidate_groups.append((len(group_sample_ids), sorted(group_sample_ids), group_id))
        if not candidate_groups:
            continue
        _, sample_ids, _ = sorted(candidate_groups, key=lambda item: (item[0], item[1]))[0]
        result.loc[result["canonical_sample_id"].isin(sample_ids), "split"] = "val"
    return result


def create_split_manifests(config: ProjectConfig, manifest_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = manifest_df.copy()
    if df.empty:
        return df, {"train": 0, "val": 0, "test": 0}

    active_df = df[
        (df["label_status"] == "mapped")
        & (df["duplicate_status"].isin(["unique", "exact_keeper", "near_keeper"]))
        & (~df["quality_flags"].apply(lambda items: "corrupt_or_unreadable" in (items or [])))
    ].copy()
    if "curation_status" in active_df.columns:
        active_df = active_df[active_df["curation_status"] == "selected"].copy()

    active_df["effective_group_id"] = active_df.apply(_effective_group_id, axis=1)
    group_rows = []
    for group_id, group in active_df.groupby("effective_group_id"):
        field_priority = max(1 if config.sources[name].field_realistic else 0 for name in group["source_name"].tolist())
        holdout_eligible = all(config.sources[name].holdout_eligible for name in group["source_name"].tolist())
        group_rows.append({
            "group_id": group_id,
            "size": len(group),
            "labels": sorted(group["canonical_label"].dropna().unique().tolist()),
            "field_priority": field_priority if holdout_eligible else 0,
            "holdout_eligible": holdout_eligible,
        })

    locked_ratio = float(config.split.get("locked_test_ratio", 0.15))
    val_ratio = float(config.split.get("validation_ratio", 0.15))
    holdout_groups = _assign_groups([row for row in group_rows if row["holdout_eligible"]], locked_ratio, config.seed)
    if not holdout_groups and group_rows and locked_ratio > 0:
        holdout_groups = _assign_groups(group_rows, locked_ratio, config.seed)
        
    remaining_groups = [row for row in group_rows if row["group_id"] not in holdout_groups]
    if not remaining_groups and holdout_groups:
        holdout_groups.remove(sorted(holdout_groups)[-1])
        remaining_groups = [row for row in group_rows if row["group_id"] not in holdout_groups]
    val_groups = _assign_groups(remaining_groups, val_ratio, config.seed + 1)
    if len(val_groups) == len(remaining_groups) and val_groups:
        val_groups.remove(sorted(val_groups)[-1])

    df["split"] = "excluded"
    df["is_locked_test"] = False
    for group_id in holdout_groups:
        target_ids = set(active_df.loc[active_df["effective_group_id"] == group_id, "canonical_sample_id"])
        df.loc[df["canonical_sample_id"].isin(target_ids), "split"] = "test"
        df.loc[df["canonical_sample_id"].isin(target_ids), "is_locked_test"] = True

    for group_id in val_groups:
        target_ids = set(active_df.loc[active_df["effective_group_id"] == group_id, "canonical_sample_id"])
        remaining = df["canonical_sample_id"].isin(target_ids) & (df["split"] == "excluded")
        df.loc[remaining, "split"] = "val"

    remaining_train = active_df["canonical_sample_id"].isin(df.loc[df["split"] == "excluded", "canonical_sample_id"])
    train_ids = set(active_df.loc[remaining_train, "canonical_sample_id"])
    df.loc[df["canonical_sample_id"].isin(train_ids), "split"] = "train"
    df = _ensure_validation_coverage(df, active_df)

    return df, df["split"].value_counts().to_dict()
