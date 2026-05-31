from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime
from dataset_pipeline.workflows.workflows import run_curation_stage


def main() -> None:
    parser = common_parser("Apply strict quality and per-class caps before splitting.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "curate_dataset.log")
    run_curation_stage(runtime)


if __name__ == "__main__":
    main()
