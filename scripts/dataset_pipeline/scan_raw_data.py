from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime, should_skip_stage
from dataset_pipeline.workflows.workflows import run_scan_stage


def main() -> None:
    parser = common_parser("Scan raw datasets and compute integrity metadata.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "scan_raw_data.log")

    if should_skip_stage(runtime, "interim", "scan_raw_data", args.force):
        runtime.logger.info("Reusing existing raw scan outputs.")
        return

    run_scan_stage(runtime)


if __name__ == "__main__":
    main()
