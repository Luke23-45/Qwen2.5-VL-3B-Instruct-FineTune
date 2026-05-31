from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import imagehash
except ImportError:  # pragma: no cover - exercised indirectly in environments without ImageHash
    imagehash = None
from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    if imagehash is not None:
        return str(imagehash.phash(image))

    resized = image.convert("L").resize((8, 8))
    pixels = list(resized.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def stable_id(*parts: str) -> str:
    payload = "::".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]
