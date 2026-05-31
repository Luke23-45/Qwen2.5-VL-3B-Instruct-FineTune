from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root(file_path: str, levels: int) -> Path:
    root = Path(file_path).resolve().parents[levels]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root
