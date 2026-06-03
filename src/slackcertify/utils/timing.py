"""Lightweight timing utilities."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["Timer", "timed"]


@dataclass(slots=True)
class Timer:
    """Hold the wall-clock and CPU elapsed times of a measured block.

    Use via the :func:`timed` context manager.
    """

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0


@contextmanager
def timed(label: str | None = None) -> Iterator[Timer]:
    """Context manager that records wall-clock and CPU time of its block.

    Examples
    --------
    >>> import time
    >>> with timed("noop") as t:
    ...     pass
    >>> t.wall_seconds >= 0.0 and t.cpu_seconds >= 0.0
    True
    """
    timer = Timer()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        yield timer
    finally:
        timer.wall_seconds = time.perf_counter() - wall_start
        timer.cpu_seconds = time.process_time() - cpu_start
        _ = label  # accepted for callers that emit log messages around the block
