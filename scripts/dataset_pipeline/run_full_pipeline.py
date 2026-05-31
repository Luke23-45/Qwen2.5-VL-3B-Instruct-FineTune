from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime
from dataset_pipeline.workflows.workflows import run_full_pipeline


def main() -> None:
    parser = common_parser("Run the full dataset build pipeline.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "run_full_pipeline.log")
    run_full_pipeline(runtime, args.force, args.config)


if __name__ == "__main__":
    main()
