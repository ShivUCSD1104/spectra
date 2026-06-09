from __future__ import annotations

import numpy as np


def sparsity_fraction(arr: np.ndarray) -> dict:
    """Return exact-zero and near-zero fractions of an array.

    Returns:
        {
            "exact_zero":    fraction of values exactly equal to 0,
            "near_zero_1e6": fraction of values with abs(x) < 1e-6,
        }
    """
    total = arr.size
    if total == 0:
        return {"exact_zero": 0.0, "near_zero_1e6": 0.0}
    exact = float(np.sum(arr == 0)) / total
    near = float(np.sum(np.abs(arr) < 1e-6)) / total
    return {"exact_zero": exact, "near_zero_1e6": near}
