from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..dataset_store import build_dataset_archives, update_stage_latest_links, write_latest_dataset_pointer
from ..reporting.analyze import generate_analytics
from ..ingestion.curation import apply_strict_curation
from ..ingestion.dedupe import apply_deduplication
from ..export.export import export_dataframe, save_config_snapshot, write_build_metadata
from ..ingestion.ingest import download_sources
from ..ingestion.normalize import build_manifest
from ..reporting.reporting import build_dataset_card, write_scan_markdown, write_validation_markdown
from ..ingestion.scan import build_scan_summary, scan_sources
from ..core.serialization import restore_serialized_columns
from ..ingestion.splits import create_split_manifests
from ..utils.files import read_json, write_json
from ..utils.stage import write_stage_metadata
from ..reporting.validate import validate_manifest
from ..export.vlm_export import export_qwen_vl_sft


FULL_PIPELINE_STAGES = [
    "download_sources.py",
    "scan_raw_data.py",
    "build_manifest.py",
    "deduplicate.py",
    "curate_dataset.py",
    "create_splits.py",
    "validate_dataset.py",
    "export_vlm_sft.py",
    "analyze_dataset.py",
    "export_dataset_card.py",
    "package_dataset.py",
]


def run_download_stage(runtime) -> None:
    statuses = download_sources(runtime.config)
    write_json(runtime.dirs["interim"] / "source_download_status.json", statuses)
    write_build_metadata(runtime.config, runtime.dirs["processed"])
    write_stage_metadata(runtime.dirs["interim"], "download_sources", {"config_hash": runtime.config.config_hash, "outputs": statuses})
    manual_required = [
        name for name, result in statuses.items() 
        if result.get("status") == "manual_required"
    ]
    if manual_required:
        runtime.logger.error("The following sources require manual download/placement: %s", ", ".join(manual_required))
        for name in manual_required:
            runtime.logger.error(" - %s: %s", name, statuses[name].get("message", "Check config."))
        raise SystemExit(1)
        
    runtime.logger.info("Recorded source availability for %s sources.", len(statuses))


def run_scan_stage(runtime) -> None:
    scan_df, per_source = scan_sources(runtime.config)
    outputs = export_dataframe(runtime.config, scan_df, runtime.dirs["interim"] / "scan_manifest")
    source_outputs = {}
    for source_name, frame in per_source.items():
        source_outputs[source_name] = export_dataframe(runtime.config, frame, runtime.dirs["interim"] / f"scan_manifest_{source_name}")
    summary = build_scan_summary(scan_df)
    write_json(runtime.dirs["reports"] / "scan_summary.json", summary)
    write_scan_markdown(summary, runtime.dirs["reports"] / "scan_summary.md")
    write_stage_metadata(runtime.dirs["interim"], "scan_raw_data", {"config_hash": runtime.config.config_hash, "outputs": outputs | {"per_source": source_outputs}})
    runtime.logger.info("Scanned %s images across %s sources.", len(scan_df), len(per_source))


def run_manifest_stage(runtime) -> None:
    scan_df = restore_serialized_columns(pd.read_csv(runtime.dirs["interim"] / "scan_manifest.csv", low_memory=False))
    manifest_df, summary = build_manifest(runtime.config, scan_df)
    outputs = export_dataframe(runtime.config, manifest_df, runtime.dirs["manifests"] / "canonical_manifest")
    audit_df = summary.pop("audit")
    export_dataframe(runtime.config, audit_df, runtime.dirs["reports"] / "label_mapping_audit")
    write_json(runtime.dirs["reports"] / "label_mapping_summary.json", summary)
    write_json(runtime.dirs["processed"] / "canonical_class_map.json", {
        crop_name: {
            disease_name: disease_data["canonical_label"]
            for disease_name, disease_data in crop_data.get("diseases", {}).items()
        }
        for crop_name, crop_data in runtime.config.taxonomy.get("crops", {}).items()
    })
    write_stage_metadata(runtime.dirs["manifests"], "build_manifest", {"config_hash": runtime.config.config_hash, "outputs": outputs})
    runtime.logger.info("Built canonical manifest with %s rows.", len(manifest_df))


def run_dedupe_stage(runtime) -> None:
    manifest_df = restore_serialized_columns(pd.read_csv(runtime.dirs["manifests"] / "canonical_manifest.csv", low_memory=False))
    deduped_df, summary = apply_deduplication(runtime.config, manifest_df)
    outputs = export_dataframe(runtime.config, deduped_df, runtime.dirs["processed"] / "canonical_manifest_deduped")
    write_json(runtime.dirs["reports"] / "deduplication_report.json", summary)
    lineage_df = pd.DataFrame(summary["lineage"]) if summary["lineage"] else pd.DataFrame(columns=["kept", "removed", "reason"])
    export_dataframe(runtime.config, lineage_df, runtime.dirs["reports"] / "dedupe_lineage")
    export_dataframe(runtime.config, deduped_df[deduped_df["duplicate_status"] == "near_removed"], runtime.dirs["reports"] / "near_duplicate_warnings")
    write_stage_metadata(runtime.dirs["processed"], "deduplicate", {"config_hash": runtime.config.config_hash, "outputs": outputs})
    runtime.logger.info("Deduplication complete. Removed %s rows.", int(deduped_df["duplicate_status"].isin(["exact_removed", "near_removed"]).sum()))


def run_curation_stage(runtime) -> None:
    deduped_df = restore_serialized_columns(pd.read_csv(runtime.dirs["processed"] / "canonical_manifest_deduped.csv", low_memory=False))
    curated_df, summary = apply_strict_curation(runtime.config, deduped_df)
    outputs = export_dataframe(runtime.config, curated_df, runtime.dirs["processed"] / "canonical_manifest_curated")
    write_json(runtime.dirs["reports"] / "curation_report.json", summary)
    write_stage_metadata(runtime.dirs["processed"], "curate_dataset", {"config_hash": runtime.config.config_hash, "outputs": outputs})
    runtime.logger.info("Curation selected %s rows and rejected %s rows.", summary["selected_rows"], summary["rejected_rows"])


def run_split_stage(runtime) -> None:
    curated_path = runtime.dirs["processed"] / "canonical_manifest_curated.csv"
    manifest_path = curated_path if curated_path.exists() else runtime.dirs["processed"] / "canonical_manifest_deduped.csv"
    manifest_df = restore_serialized_columns(pd.read_csv(manifest_path, low_memory=False))
    split_df, summary = create_split_manifests(runtime.config, manifest_df)
    outputs = export_dataframe(runtime.config, split_df, runtime.dirs["splits"] / "manifest_with_splits")
    for split_name in ("train", "val", "test"):
        export_dataframe(runtime.config, split_df[split_df["split"] == split_name], runtime.dirs["splits"] / f"{split_name}_manifest")
    export_dataframe(runtime.config, split_df[split_df["is_locked_test"]], runtime.dirs["splits"] / "test_holdout_locked")
    write_json(runtime.dirs["reports"] / "split_summary.json", summary)
    write_stage_metadata(runtime.dirs["splits"], "create_splits", {"config_hash": runtime.config.config_hash, "outputs": outputs})
    runtime.logger.info("Created splits: %s", summary)


def run_validation_stage(runtime) -> None:
    split_df = restore_serialized_columns(pd.read_csv(runtime.dirs["splits"] / "manifest_with_splits.csv", low_memory=False))
    locked_reference = runtime.dirs["splits"] / "test_holdout_locked.csv"
    report = validate_manifest(runtime.config, split_df, locked_reference if locked_reference.exists() else None)
    write_json(runtime.dirs["reports"] / "validation_report.json", report)
    write_validation_markdown(report, runtime.dirs["reports"] / "validation_report.md")
    write_stage_metadata(runtime.dirs["reports"], "validate_dataset", {"config_hash": runtime.config.config_hash, "status": report["status"]})
    runtime.logger.info("Validation status: %s", report["status"])
    if report["status"] != "passed":
        raise SystemExit(1)


def run_analysis_stage(runtime) -> None:
    split_df = restore_serialized_columns(pd.read_csv(runtime.dirs["splits"] / "manifest_with_splits.csv", low_memory=False))
    outputs = generate_analytics(split_df, runtime.dirs["reports"] / "analytics")
    write_json(runtime.dirs["reports"] / "analytics_outputs.json", outputs)
    write_stage_metadata(runtime.dirs["reports"], "analyze_dataset", {"config_hash": runtime.config.config_hash, "outputs": outputs})
    runtime.logger.info("Analytics generated.")


def run_dataset_card_stage(runtime) -> None:
    split_df = pd.read_csv(runtime.dirs["splits"] / "manifest_with_splits.csv", low_memory=False)
    validation_report = read_json(runtime.dirs["reports"] / "validation_report.json")
    card = build_dataset_card(split_df, validation_report, runtime.config.dataset_version)
    (runtime.dirs["reports"] / "dataset_card.md").write_text(card, encoding="utf-8")
    runtime.logger.info("Dataset card written.")


def run_vlm_export_stage(runtime) -> None:
    split_df = restore_serialized_columns(pd.read_csv(runtime.dirs["splits"] / "manifest_with_splits.csv", low_memory=False))
    output_dir = runtime.dirs["final"] / "qwen25_vl_sft"
    template_dir = runtime.config.paths.root / "data" / "templates" / "vlm"
    outputs = export_qwen_vl_sft(
        runtime.config,
        split_df,
        output_dir=output_dir,
        template_dir=template_dir,
        image_root=runtime.config.paths.root,
    )
    write_stage_metadata(runtime.dirs["processed"], "export_vlm_sft", {"config_hash": runtime.config.config_hash, "outputs": outputs})
    runtime.logger.info("Exported Qwen2.5-VL SFT data to %s.", output_dir)


def run_package_stage(runtime) -> None:
    archive_formats = runtime.config.export.get("archive_formats", ["tar.gz", "zip"])
    include_stage_dirs = runtime.config.export.get("artifact_stage_dirs", ["final", "manifests", "splits", "reports"])
    outputs = build_dataset_archives(
        runtime.config,
        runtime.dirs,
        archive_formats=archive_formats,
        included_stage_dirs=include_stage_dirs,
    )
    write_stage_metadata(
        runtime.dirs["artifacts"],
        "package_dataset",
        {"config_hash": runtime.config.config_hash, "outputs": outputs, "stage_dirs": include_stage_dirs},
    )
    runtime.logger.info("Packaged dataset artifacts: %s", outputs)


def run_full_pipeline(runtime, force: bool, config_path: str | Path) -> None:
    save_config_snapshot(runtime.config, runtime.dirs["processed"] / "config_snapshot")
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "dataset_pipeline"
    for stage in FULL_PIPELINE_STAGES:
        runtime.logger.info("Running stage: %s", stage)
        cmd = [sys.executable, str(scripts_dir / stage), "--config", str(Path(config_path).resolve())]
        if force:
            cmd.append("--force")
        env = os.environ.copy()
        env["KV_DATASET_VERSION"] = runtime.config.dataset_version
        subprocess.run(cmd, check=True, cwd=runtime.config.paths.root, env=env)

    update_stage_latest_links(runtime.dirs, runtime.logger)
    write_latest_dataset_pointer(runtime.config, runtime.dirs)
    runtime.logger.info("Updated dataset latest pointers.")
