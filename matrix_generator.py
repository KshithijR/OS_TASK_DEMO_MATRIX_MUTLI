"""Matrix generation utilities."""

import numpy as np


def generate_matrices(size: int, seed: int = 42, low: int = 1, high: int = 10):
    """Generate two size x size integer matrices using a reproducible seed.

    Values are drawn from [low, high) -- default range 1..9, matching the
    assignment's requirement for small, easy-to-verify-by-hand integers.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(low, high, size=(size, size), dtype=np.int64)
    b = rng.integers(low, high, size=(size, size), dtype=np.int64)
    return a, b
