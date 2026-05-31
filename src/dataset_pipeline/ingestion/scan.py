from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..core.models import ProjectConfig, ScanRecord
from ..sources import build_adapter
from ..utils.hashing import perceptual_hash, sha256_file


def _scan_image(record) -> ScanRecord:
    path = Path(record.original_path)
    record_dict = asdict(record)
    quality_flags: list[str] = []
    warning_flags: list[str] = []
    if path.stat().st_size == 0:
        return ScanRecord(**record_dict, scan_status="error", scan_error="empty_file", quality_flags=["empty_file"])

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if min(width, height) < 64:
                warning_flags.append("very_small_image")
            aspect_ratio = round(width / height, 6) if height else None
            if aspect_ratio and (aspect_ratio > 3 or aspect_ratio < 0.33):
                warning_flags.append("extreme_aspect_ratio")
            return ScanRecord(
                **record_dict,
                image_width=width,
                image_height=height,
                aspect_ratio=aspect_ratio,
                file_size_bytes=path.stat().st_size,
                file_format=(image.format or path.suffix.lstrip(".")).lower(),
                image_mode=image.mode,
                exact_hash=sha256_file(path),
                phash=perceptual_hash(image.convert("RGB")),
                quality_flags=quality_flags,
                warning_flags=warning_flags,
                scan_status="ok",
                scan_error=None,
            )
    except (UnidentifiedImageError, OSError) as exc:
        quality_flags.append("corrupt_or_unreadable")
        return ScanRecord(
            **record_dict,
            file_size_bytes=path.stat().st_size if path.exists() else None,
            quality_flags=quality_flags,
            warning_flags=warning_flags,
            scan_status="error",
            scan_error=str(exc),
        )


def scan_sources(config: ProjectConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    all_records: list[dict] = []
    per_source: dict[str, pd.DataFrame] = {}

    import dataclasses
    columns = [f.name for f in dataclasses.fields(ScanRecord)]
    
    for source_name, source_config in config.sources.items():
        if not source_config.enabled:
            continue
        adapter = build_adapter(source_config)
        scanned = [asdict(_scan_image(record)) for record in adapter.scan_records()]
        frame = pd.DataFrame(scanned) if scanned else pd.DataFrame(columns=columns)
        per_source[source_name] = frame
        if not frame.empty:
            all_records.extend(scanned)

    final_df = pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=columns)
    return final_df, per_source


def build_scan_summary(scan_df: pd.DataFrame) -> dict:
    if scan_df.empty:
        return {
            "total_images": 0,
            "per_source_counts": {},
            "per_label_counts": {},
            "missing_label_count": 0,
            "corrupt_count": 0,
            "resolution_stats": {},
        }

    ok_df = scan_df[scan_df["scan_status"] == "ok"].copy()
    return {
        "total_images": int(len(scan_df)),
        "per_source_counts": scan_df["source_name"].value_counts().to_dict(),
        "per_label_counts": scan_df["source_label"].fillna("__missing__").value_counts().to_dict(),
        "missing_label_count": int(scan_df["source_label"].isna().sum()),
        "corrupt_count": int((scan_df["scan_status"] != "ok").sum()),
        "resolution_stats": {
            "width": ok_df["image_width"].describe().dropna().to_dict() if not ok_df.empty else {},
            "height": ok_df["image_height"].describe().dropna().to_dict() if not ok_df.empty else {},
            "aspect_ratio": ok_df["aspect_ratio"].describe().dropna().to_dict() if not ok_df.empty else {},
        },
    }
