from __future__ import annotations

from _bootstrap import setup_src_path

setup_src_path()

from dataset_pipeline.core.cli import common_parser
from dataset_pipeline.core.runtime import build_stage_runtime, should_skip_stage
from dataset_pipeline.workflows.workflows import run_manifest_stage


def main() -> None:
    parser = common_parser("Build the canonical manifest from scan results.")
    args = parser.parse_args()
    runtime = build_stage_runtime(args.config, "build_manifest.log")

    if should_skip_stage(runtime, "manifests", "build_manifest", args.force):
        runtime.logger.info("Reusing existing manifest outputs.")
        return

    run_manifest_stage(runtime)


if __name__ == "__main__":
    main()
