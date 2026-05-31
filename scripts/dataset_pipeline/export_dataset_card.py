from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime
from dataset_pipeline.workflows.workflows import run_dataset_card_stage


def main() -> None:
    parser = common_parser("Export a dataset card markdown report.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "export_dataset_card.log")
    run_dataset_card_stage(runtime)


if __name__ == "__main__":
    main()
