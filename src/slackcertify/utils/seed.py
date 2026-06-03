"""Deterministic seed plumbing.

These helpers exist so every experiment in the paper can be reproduced
bit-for-bit from a single ``--seed`` value: :func:`set_global_seeds`
fixes Python's ``random`` and NumPy's legacy global seed; for new code,
prefer :func:`derive_seed` to mint per-rollout / per-agent seeds via
SHA-256.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

__all__ = ["derive_seed", "set_global_seeds"]


def set_global_seeds(seed: int) -> None:
    """Fix the Python ``random`` and NumPy legacy global seeds.

    Examples
    --------
    >>> set_global_seeds(0)
    >>> import random
    >>> random.randint(0, 1000) == random.Random(0).randint(0, 1000)
    True
    """
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


def derive_seed(parent: int, label: str) -> int:
    """Return a 32-bit deterministic seed derived from ``parent`` and ``label``.

    The hash mixes the parent integer with the UTF-8 bytes of the label
    via SHA-256 and returns the first 32 bits as an unsigned integer.
    Suitable for seeding ``np.random.default_rng``.

    Examples
    --------
    >>> a = derive_seed(0, "rollout-0")
    >>> b = derive_seed(0, "rollout-0")
    >>> c = derive_seed(0, "rollout-1")
    >>> a == b and a != c
    True
    """
    h = hashlib.sha256(f"{parent}:{label}".encode()).digest()
    return int.from_bytes(h[:4], "big")
