from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime, should_skip_stage
from dataset_pipeline.workflows.workflows import run_package_stage


def main() -> None:
    parser = common_parser("Package the finalized dataset into portable archives.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "package_dataset.log")
    if should_skip_stage(runtime, "artifacts", "package_dataset", args.force):
        runtime.logger.info("Reusing existing packaged dataset artifacts.")
        return
    run_package_stage(runtime)


if __name__ == "__main__":
    main()
