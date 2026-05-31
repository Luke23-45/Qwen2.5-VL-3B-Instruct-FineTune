from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(root: Path) -> Iterator[Path]:
    if not root.exists() or not root.is_dir():
        return

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                        yield Path(entry.path)
        except OSError:
            continue
