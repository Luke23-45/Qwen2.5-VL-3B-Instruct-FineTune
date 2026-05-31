from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.models import ProjectConfig
from .prompts import choose_template, load_prompt_templates, render_answer, render_instruction
from ..utils.files import ensure_dir, write_json


def _record(row: pd.Series, template: dict[str, Any], image_path: str) -> dict[str, Any]:
    instruction = render_instruction(template, row["crop_name"], row["disease_name"], row["canonical_label"])
    answer = render_answer(template, row["crop_name"], row["disease_name"], row["canonical_label"])
    return {
        "id": row["canonical_sample_id"],
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{instruction}"},
            {"from": "gpt", "value": answer},
        ],
        "metadata": {
            "split": row["split"],
            "crop_name": row["crop_name"],
            "disease_name": row["disease_name"],
            "canonical_label": row["canonical_label"],
            "template_id": template["id"],
            "source_name": row["source_name"],
        },
    }


def export_qwen_vl_sft(
    config: ProjectConfig,
    split_df: pd.DataFrame,
    output_dir: str | Path,
    template_dir: str | Path = "data/templates/vlm",
    image_root: str | Path | None = None,
) -> dict[str, str]:
    output_root = ensure_dir(Path(output_dir))
    templates = load_prompt_templates(template_dir)
    selected = split_df[
        split_df["split"].isin(["train", "val", "test"])
        & (split_df["label_status"] == "mapped")
        & (split_df["duplicate_status"].isin(["unique", "exact_keeper", "near_keeper"]))
    ].copy()
    if "curation_status" in selected.columns:
        selected = selected[selected["curation_status"] == "selected"].copy()

    outputs: dict[str, str] = {}
    summary: dict[str, Any] = {"total": int(len(selected)), "splits": {}}

    for split_name in ("train", "val", "test"):
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, group in selected.groupby("split"):
        records = []
        for _, row in group.sort_values("canonical_sample_id").iterrows():
            template = choose_template(templates, row["canonical_sample_id"])
            
            src_path = Path(str(row["original_path"]))
            img_filename = f"{row['canonical_sample_id']}{src_path.suffix}"
            dst_path = output_root / "images" / split_name / img_filename
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                
            image_rel_path = f"images/{split_name}/{img_filename}"
            records.append(_record(row, template, image_rel_path))

        json_path = output_root / f"qwen25_vl_{split_name}.json"
        jsonl_path = output_root / f"qwen25_vl_{split_name}.jsonl"
        write_json(json_path, records)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        outputs[f"{split_name}_json"] = str(json_path)
        outputs[f"{split_name}_jsonl"] = str(jsonl_path)
        summary["splits"][split_name] = len(records)

    write_json(output_root / "qwen25_vl_sft_summary.json", summary)
    outputs["summary"] = str(output_root / "qwen25_vl_sft_summary.json")

    return outputs
