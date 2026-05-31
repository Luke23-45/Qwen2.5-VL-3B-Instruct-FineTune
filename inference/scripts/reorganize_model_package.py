from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "src" / "model"


@dataclass(frozen=True, slots=True)
class MoveSpec:
    source: str
    destination: str


MOVE_SPECS = (
    MoveSpec("__init__.py", "__init__.py"),
    MoveSpec("batches.py", "utils/batches.py"),
    MoveSpec("config.py", "config/settings.py"),
    MoveSpec("defaults.py", "config/defaults.py"),
    MoveSpec("engine.py", "core/engine.py"),
    MoveSpec("generation.py", "generation/generator.py"),
    MoveSpec("infer.py", "cli/infer.py"),
    MoveSpec("loader.py", "loading/loader.py"),
    MoveSpec("messages.py", "generation/messages.py"),
    MoveSpec("performance.py", "config/performance.py"),
    MoveSpec("runtime.py", "config/runtime.py"),
    MoveSpec("types.py", "core/types.py"),
    MoveSpec("vision.py", "generation/vision.py"),
)

PACKAGE_DIRS = (
    "cli",
    "config",
    "core",
    "generation",
    "loading",
    "utils",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorganize inference/src/model into subpackages.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the directory creation and file moves. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def ensure_within_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(MODEL_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes model root: {resolved}") from exc
    return resolved


def validate_manifest() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    seen_sources: set[Path] = set()
    seen_destinations: set[Path] = set()

    for spec in MOVE_SPECS:
        source = ensure_within_root(MODEL_ROOT / spec.source)
        destination = ensure_within_root(MODEL_ROOT / spec.destination)
        if source in seen_sources:
            raise ValueError(f"Duplicate source in manifest: {source}")
        if destination in seen_destinations:
            raise ValueError(f"Duplicate destination in manifest: {destination}")
        if spec.source != "__init__.py" and not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        seen_sources.add(source)
        seen_destinations.add(destination)
        moves.append((source, destination))

    for package_dir in PACKAGE_DIRS:
        ensure_within_root(MODEL_ROOT / package_dir)

    return moves


def print_plan(moves: list[tuple[Path, Path]]) -> None:
    print(f"Model root: {MODEL_ROOT}")
    print("Planned package directories:")
    for package_dir in PACKAGE_DIRS:
        print(f"  - {package_dir}/")
    print("Planned moves:")
    for source, destination in moves:
        if source == destination:
            print(f"  - keep {source.relative_to(MODEL_ROOT)}")
            continue
        print(f"  - {source.relative_to(MODEL_ROOT)} -> {destination.relative_to(MODEL_ROOT)}")


def create_packages() -> None:
    for package_dir in PACKAGE_DIRS:
        package_path = ensure_within_root(MODEL_ROOT / package_dir)
        package_path.mkdir(parents=True, exist_ok=True)
        init_file = package_path / "__init__.py"
        init_file.touch(exist_ok=True)


def apply_moves(moves: list[tuple[Path, Path]]) -> None:
    create_packages()
    for source, destination in moves:
        if source == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        source.replace(destination)


def main() -> None:
    args = parse_args()
    moves = validate_manifest()
    print_plan(moves)
    if not args.execute:
        print("Dry run only. Re-run with --execute to apply the manifest.")
        return
    apply_moves(moves)
    print("Reorganization complete.")


if __name__ == "__main__":
    main()
