from __future__ import annotations

from pathlib import Path

import pandas as pd


class ManifestDataset:
    def __init__(self, manifest_path: str | Path, split: str | None = None, crops: list[str] | None = None, sources: list[str] | None = None):
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.suffix == ".parquet":
            self.df = pd.read_parquet(self.manifest_path)
        else:
            self.df = pd.read_csv(self.manifest_path)
        if split:
            self.df = self.df[self.df["split"] == split].copy()
        if crops:
            self.df = self.df[self.df["crop_name"].isin(crops)].copy()
        if sources:
            self.df = self.df[self.df["source_name"].isin(sources)].copy()
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        return {
            "image_path": row["original_path"],
            "canonical_label": row["canonical_label"],
            "crop_name": row["crop_name"],
            "source_name": row["source_name"],
            "metadata": row.to_dict(),
        }
