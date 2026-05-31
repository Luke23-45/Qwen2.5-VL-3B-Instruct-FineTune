from __future__ import annotations

import argparse


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="data/configs/pipeline.yaml", help="Path to the pipeline YAML config.")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if stage metadata matches.")
    return parser
