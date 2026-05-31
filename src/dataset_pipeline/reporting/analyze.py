from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_bar(series: pd.Series, path: Path, title: str, xlabel: str = "", ylabel: str = "count") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_hist(series: pd.Series, path: Path, title: str, bins: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def generate_analytics(manifest_df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "class_distribution": manifest_df["canonical_label"].fillna("__missing__").value_counts(),
        "crop_distribution": manifest_df["crop_name"].fillna("__missing__").value_counts(),
        "source_distribution": manifest_df["source_name"].value_counts(),
        "split_distribution": manifest_df["split"].value_counts(),
    }
    for name, series in tables.items():
        series.rename_axis("key").reset_index(name="count").to_csv(output_dir / f"{name}.csv", index=False)

    _save_bar(tables["class_distribution"], output_dir / "class_distribution.png", "Class Distribution")
    _save_bar(tables["source_distribution"], output_dir / "source_distribution.png", "Source Distribution")
    _save_bar(tables["split_distribution"], output_dir / "split_distribution.png", "Split Distribution")
    _save_hist(manifest_df["image_width"], output_dir / "width_distribution.png", "Image Width Distribution")
    _save_hist(manifest_df["image_height"], output_dir / "height_distribution.png", "Image Height Distribution")
    _save_hist(manifest_df["aspect_ratio"], output_dir / "aspect_ratio_distribution.png", "Aspect Ratio Distribution")

    duplicate_counts = manifest_df["duplicate_status"].value_counts().rename_axis("duplicate_status").reset_index(name="count")
    duplicate_counts.to_csv(output_dir / "duplicate_counts.csv", index=False)
    quality_df = manifest_df.explode("quality_flags")
    quality_df["quality_flags"] = quality_df["quality_flags"].fillna("__none__")
    quality_df["quality_flags"].value_counts().rename_axis("quality_flag").reset_index(name="count").to_csv(
        output_dir / "quality_flag_counts.csv", index=False
    )
    manifest_df.groupby(["crop_name", "source_name"]).size().rename("count").reset_index().to_csv(
        output_dir / "per_crop_source_counts.csv", index=False
    )

    return {
        "tables": {name: str(output_dir / f"{name}.csv") for name in tables},
        "plots": {
            "class_distribution": str(output_dir / "class_distribution.png"),
            "source_distribution": str(output_dir / "source_distribution.png"),
            "split_distribution": str(output_dir / "split_distribution.png"),
            "width_distribution": str(output_dir / "width_distribution.png"),
            "height_distribution": str(output_dir / "height_distribution.png"),
            "aspect_ratio_distribution": str(output_dir / "aspect_ratio_distribution.png"),
        },
    }
