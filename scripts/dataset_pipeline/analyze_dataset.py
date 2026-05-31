from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime, should_skip_stage
from dataset_pipeline.workflows.workflows import run_analysis_stage


def main() -> None:
    parser = common_parser("Generate dataset analytics and plots.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "analyze_dataset.log")

    if should_skip_stage(runtime, "reports", "analyze_dataset", args.force):
        runtime.logger.info("Reusing existing dataset analytics.")
        return

    run_analysis_stage(runtime)


if __name__ == "__main__":
    main()
