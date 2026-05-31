from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime
from dataset_pipeline.workflows.workflows import run_vlm_export_stage


def main() -> None:
    parser = common_parser("Export Qwen2.5-VL SFT conversations from curated splits.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "export_vlm_sft.log")
    run_vlm_export_stage(runtime)


if __name__ == "__main__":
    main()
