from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluations.validations.core.contracts import ValidationResult  # noqa: E402
from evaluations.validations.core.outputs import ValidationOutputWriter, utc_run_id  # noqa: E402
from evaluations.validations.datasets.roots import safe_resolve_dataset_roots  # noqa: E402
from evaluations.validations.datasets.scanner import (  # noqa: E402
    DatasetScanner,
    category_summary,
    image_records_dataframe,
    json_records_dataframe,
)


logger = logging.getLogger(__name__)
VALIDATION_NAME = "dataset_validation"


def run(cfg: DictConfig, run_id: str | None = None) -> ValidationResult:
    run_id = run_id or cfg.run_id or utc_run_id()
    writer = ValidationOutputWriter(cfg.output.root_dir, run_id, VALIDATION_NAME, cfg)
    roots, root_issues = safe_resolve_dataset_roots(cfg)
    scanner = DatasetScanner(check_decode=bool(cfg.dataset_validation.check_image_decode))
    result = scanner.scan(roots)
    issues = root_issues + result.issues

    images_df = image_records_dataframe(result.image_records)
    json_df = json_records_dataframe(result.json_records)
    summary = category_summary(result.image_records, result.json_records)
    summary["roots"] = [{"name": root.name, "kind": root.kind, "path": str(root.path)} for root in roots]
    summary["issue_count"] = len(issues)

    writer.json("summary.json", summary)
    writer.json("issues.json", issues)
    writer.dataframe("image_inventory.csv", images_df, index=False)
    writer.dataframe("qwen_records.csv", json_df, index=False)

    status = "passed" if not any(issue.get("severity") == "error" for issue in issues) else "failed"
    validation_result = ValidationResult(
        name=VALIDATION_NAME,
        status=status,
        output_dir=str(writer.output_dir),
        summary=summary,
        outputs=dict(writer.outputs),
        issues=issues,
    )
    return writer.finish(validation_result)


@hydra.main(version_base=None, config_path="config", config_name="validation")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper()))
    run(cfg)


if __name__ == "__main__":
    main()
