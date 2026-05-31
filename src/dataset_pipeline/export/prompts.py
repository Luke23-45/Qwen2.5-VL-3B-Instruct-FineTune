from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TEMPLATE_DIR = Path("data/templates/vlm")


def load_prompt_templates(template_dir: str | Path = DEFAULT_TEMPLATE_DIR) -> list[dict[str, Any]]:
    root = Path(template_dir)
    templates: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not payload.get("id") or not payload.get("instruction"):
            raise ValueError(f"Invalid prompt template: {path}")
        templates.append(payload)
    if not templates:
        raise FileNotFoundError(f"No prompt templates found in {root.resolve()}")
    return templates


def choose_template(templates: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(templates)
    return templates[index]


def render_instruction(template: dict[str, Any], crop_name: str, disease_name: str, canonical_label: str) -> str:
    return str(template["instruction"]).format(
        crop_name=crop_name,
        disease_name=disease_name.replace("_", " "),
        canonical_label=canonical_label,
    )


def render_answer(template: dict[str, Any], crop_name: str, disease_name: str, canonical_label: str) -> str:
    answer = template.get("answer")
    if answer:
        return str(answer).format(
            crop_name=crop_name,
            disease_name=disease_name.replace("_", " "),
            canonical_label=canonical_label,
        )
    return (
        "{"
        f'"crop":"{crop_name}",'
        f'"disease":"{disease_name}",'
        f'"canonical_label":"{canonical_label}",'
        '"confidence_label":"high"'
        "}"
    )
