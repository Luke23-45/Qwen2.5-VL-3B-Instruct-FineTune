from __future__ import annotations

from itertools import islice
from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


def batched(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
