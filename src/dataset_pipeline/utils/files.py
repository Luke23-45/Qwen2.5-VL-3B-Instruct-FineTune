from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METADATA_EXTENSIONS = {".json", ".csv", ".txt", ".xml"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_dataframe(df: pd.DataFrame, base_path: Path, write_csv: bool = True, write_parquet: bool = True) -> dict[str, str]:
    outputs: dict[str, str] = {}
    base_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_df = df.copy()
    for column in serializable_df.columns:
        if serializable_df[column].dtype == "object":
            serializable_df[column] = serializable_df[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
    if write_csv:
        csv_path = base_path.with_suffix(".csv")
        serializable_df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
    if write_parquet:
        parquet_path = base_path.with_suffix(".parquet")
        serializable_df.to_parquet(parquet_path, index=False)
        outputs["parquet"] = str(parquet_path)
    return outputs


def create_archive(input_root: Path, archive_path: Path, archive_format: str) -> Path:
    import tarfile
    import zipfile
    
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in input_root.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(input_root))
        return archive_path

    if archive_format == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(input_root, arcname=input_root.name)
        return archive_path

    raise ValueError(f"Unsupported archive format: {archive_format}")


def extract_archive(archive_path: Path, destination: Path) -> Path:
    import tarfile
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    extract_root = destination / f".extract-{archive_path.stem.replace('.', '-')}"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    suffixes = archive_path.suffixes
    if suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_root)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_root)
    else:
        raise ValueError(f"Unsupported archive for extraction: {archive_path}")

    children = [child for child in extract_root.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_root
