from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime, should_skip_stage
from dataset_pipeline.workflows.workflows import run_download_stage


def main() -> None:
    parser = common_parser("Download or register raw sources.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "download_sources.log")

    if should_skip_stage(runtime, "interim", "download_sources", args.force):
        runtime.logger.info("Reusing existing source download metadata.")
        return

    run_download_stage(runtime)


if __name__ == "__main__":
    main()
