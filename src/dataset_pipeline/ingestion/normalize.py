from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from ..core.models import ManifestRecord, ProjectConfig
from ..utils.hashing import stable_id


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


class TaxonomyNormalizer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.crops = config.taxonomy.get("crops", {})
        self.global_aliases = {
            source_name: {key.lower(): value for key, value in aliases.items()}
            for source_name, aliases in config.taxonomy.get("global_aliases", {}).items()
        }
        self.lookup = self._build_lookup()

    def _build_lookup(self) -> dict[str, list[tuple[str, str, str]]]:
        lookup: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for crop_name, crop_data in self.crops.items():
            crop_aliases = {_slug(crop_name), *[_slug(alias) for alias in crop_data.get("aliases", [])]}
            for disease_name, disease_data in crop_data.get("diseases", {}).items():
                canonical_label = disease_data["canonical_label"]
                disease_aliases = {_slug(disease_name), *[_slug(alias) for alias in disease_data.get("aliases", [])]}
                for crop_token in crop_aliases:
                    for disease_token in disease_aliases:
                        lookup[f"{crop_token}::{disease_token}"].append((crop_name, disease_name, canonical_label))
                for disease_token in disease_aliases:
                    lookup[disease_token].append((crop_name, disease_name, canonical_label))
        return lookup

    def normalize_label(self, source_name: str, source_label: str | None) -> tuple[str | None, str | None, str | None, str]:
        if not source_label:
            return None, None, None, "unmapped"

        source_label_key = source_label.strip().lower()
        source_override = self.global_aliases.get(source_name, {})
        if source_label_key in source_override:
            entry = source_override[source_label_key]
            crop_name = entry["crop"]
            disease_name = entry["disease"]
            canonical_label = self.crops[crop_name]["diseases"][disease_name]["canonical_label"]
            return crop_name, disease_name, canonical_label, "mapped"

        slug_label = _slug(source_label.replace("___", " "))
        matches = self.lookup.get(slug_label, [])
        if len(matches) == 1:
            return (*matches[0], "mapped")
        if len(matches) > 1:
            return None, None, None, "ambiguous"

        for crop_name, crop_data in self.crops.items():
            crop_tokens = {_slug(crop_name), *[_slug(alias) for alias in crop_data.get("aliases", [])]}
            for crop_token in crop_tokens:
                prefix = f"{crop_token}_"
                if not slug_label.startswith(prefix):
                    continue
                disease_candidate = slug_label[len(prefix):]
                disease_matches = []
                for disease_name, disease_data in crop_data.get("diseases", {}).items():
                    disease_aliases = {_slug(disease_name), *[_slug(alias) for alias in disease_data.get("aliases", [])]}
                    if disease_candidate in disease_aliases:
                        disease_matches.append((crop_name, disease_name, disease_data["canonical_label"]))
                if len(disease_matches) == 1:
                    return (*disease_matches[0], "mapped")
                if len(disease_matches) > 1:
                    return None, None, None, "ambiguous"

        tokens = [token for token in slug_label.split("_") if token]
        matches = []
        for crop_name, crop_data in self.crops.items():
            crop_tokens = {_slug(crop_name), *[_slug(alias) for alias in crop_data.get("aliases", [])]}
            if not crop_tokens.intersection(tokens):
                continue
            for disease_name, disease_data in crop_data.get("diseases", {}).items():
                disease_tokens = {_slug(disease_name), *[_slug(alias) for alias in disease_data.get("aliases", [])]}
                if disease_tokens.intersection(tokens):
                    matches.append((crop_name, disease_name, disease_data["canonical_label"]))

        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            return (*unique_matches[0], "mapped")
        if len(unique_matches) > 1:
            return None, None, None, "ambiguous"
        return None, None, None, "unmapped"


def build_manifest(config: ProjectConfig, scan_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    normalizer = TaxonomyNormalizer(config)
    rows: list[dict] = []
    audit_rows: list[dict] = []
    created_at = pd.Timestamp.utcnow().isoformat()

    for record in scan_df.to_dict(orient="records"):
        crop_name, disease_name, canonical_label, label_status = normalizer.normalize_label(
            source_name=record["source_name"],
            source_label=record.get("source_label"),
        )
        source_config = config.sources[record["source_name"]]
        if crop_name and source_config.allowed_crops and crop_name not in source_config.allowed_crops:
            crop_name, disease_name, canonical_label, label_status = None, None, None, "filtered_disallowed_crop"
        canonical_sample_id = stable_id(config.dataset_version, record["source_name"], record["source_id"])
        manifest = ManifestRecord(
            canonical_sample_id=canonical_sample_id,
            dataset_version=config.dataset_version,
            source_name=record["source_name"],
            source_id=record["source_id"],
            source_split_original=record.get("source_split_original"),
            original_path=record["original_path"],
            relative_path=record["relative_path"],
            crop_name=crop_name,
            disease_name=disease_name,
            canonical_label=canonical_label,
            task_type="classification",
            annotation_type="classification",
            split="unassigned",
            is_locked_test=False,
            group_id=record.get("group_hint"),
            exact_hash=record.get("exact_hash"),
            phash=record.get("phash"),
            duplicate_status="unique",
            duplicate_of=None,
            quality_flags=list(record.get("quality_flags", [])),
            warning_flags=list(record.get("warning_flags", [])),
            image_width=record.get("image_width"),
            image_height=record.get("image_height"),
            aspect_ratio=record.get("aspect_ratio"),
            file_size_bytes=record.get("file_size_bytes"),
            file_format=record.get("file_format"),
            label_status=label_status,
            lineage=[record["source_id"]],
            notes=record.get("scan_error"),
            created_at=created_at,
            annotation_refs=record.get("annotation_refs", {}),
        )
        if record.get("scan_status") != "ok":
            manifest.quality_flags.append("scan_error")
        rows.append(manifest.as_dict())
        audit_rows.append({
            "source_name": record["source_name"],
            "source_id": record["source_id"],
            "source_label": record.get("source_label"),
            "crop_name": crop_name,
            "disease_name": disease_name,
            "canonical_label": canonical_label,
            "label_status": label_status,
        })

    import dataclasses
    manifest_cols = [f.name for f in dataclasses.fields(ManifestRecord)]
    audit_cols = ["source_name", "source_id", "source_label", "crop_name", "disease_name", "canonical_label", "label_status"]
    
    manifest_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=manifest_cols)
    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame(columns=audit_cols)
    summary = {
        "total_rows": int(len(manifest_df)),
        "mapped_rows": int((manifest_df["label_status"] == "mapped").sum()) if not manifest_df.empty else 0,
        "unmapped_rows": int((manifest_df["label_status"] == "unmapped").sum()) if not manifest_df.empty else 0,
        "ambiguous_rows": int((manifest_df["label_status"] == "ambiguous").sum()) if not manifest_df.empty else 0,
        "filtered_disallowed_crop_rows": int((manifest_df["label_status"] == "filtered_disallowed_crop").sum()) if not manifest_df.empty else 0,
        "canonical_labels": sorted(audit_df["canonical_label"].dropna().unique().tolist()) if not audit_df.empty else [],
        "audit": audit_df,
    }
    return manifest_df, summary
