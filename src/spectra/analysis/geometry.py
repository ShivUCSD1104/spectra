from __future__ import annotations

import numpy as np


def intrinsic_dim_estimate(S: np.ndarray) -> int:
    """Count singular values above 1% of the largest (simple threshold method)."""
    if S.size == 0:
        return 0
    threshold = 0.01 * S[0]
    return int(np.sum(S > threshold))


def participation_ratio(S: np.ndarray) -> float:
    """Participation ratio: (sum(S))^2 / sum(S^2). Alias of effective_rank."""
    S = S[S > 0]
    if S.size == 0:
        return 0.0
    return float(np.sum(S) ** 2 / np.sum(S ** 2))
