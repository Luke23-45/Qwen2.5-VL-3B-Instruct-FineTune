from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime
from dataset_pipeline.workflows.workflows import run_validation_stage


def main() -> None:
    parser = common_parser("Validate the prepared dataset manifests.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "validate_dataset.log")
    run_validation_stage(runtime)


if __name__ == "__main__":
    main()
