from __future__ import annotations

import numpy as np


def shannon_entropy(arr: np.ndarray, bins: int = 256) -> float:
    """Compute histogram-based Shannon entropy in bits.

    Flattens the array, builds a histogram over `bins` equal-width bins,
    normalises to a probability distribution, then computes -sum(p * log2(p))
    over non-zero bins.
    """
    flat = arr.flatten().astype(np.float64)
    counts, _ = np.histogram(flat, bins=bins)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))
