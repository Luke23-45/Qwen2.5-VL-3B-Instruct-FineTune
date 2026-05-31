from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime, should_skip_stage
from dataset_pipeline.workflows.workflows import run_split_stage


def main() -> None:
    parser = common_parser("Create deterministic train/val/test splits.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "create_splits.log")

    if should_skip_stage(runtime, "splits", "create_splits", args.force):
        runtime.logger.info("Reusing existing split outputs.")
        return

    run_split_stage(runtime)


if __name__ == "__main__":
    main()
