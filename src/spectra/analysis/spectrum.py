from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds


def randomized_svd_top_k(
    arr: np.ndarray, k: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top-k singular triplets (U, S, Vt) of a 2D matrix.

    Uses scipy.sparse.linalg.svds, which computes a partial SVD efficiently.
    Results are returned in descending order of singular values.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    k = min(k, min(arr.shape) - 1)
    A = arr.astype(np.float64)
    U, S, Vt = svds(A, k=k)
    # svds returns in ascending order — reverse to descending
    idx = np.argsort(S)[::-1]
    return U[:, idx], S[idx], Vt[idx, :]


def spectral_decay_rate(S: np.ndarray) -> float:
    """Fit an exponential decay to the singular value curve and return the rate.

    Models S[i] ~ S[0] * exp(-rate * i). Fits via linear regression on log(S).
    Values near 1.0 indicate fast decay (strong low-rank structure).
    Values near 0.0 indicate slow decay (dense spectrum).
    """
    S = S[S > 0]
    if S.size < 2:
        return 0.0
    S_norm = S / S[0]
    log_S = np.log(np.clip(S_norm, 1e-12, None))
    indices = np.arange(len(log_S), dtype=np.float64)
    # Least-squares fit: log_S ~ -rate * i
    rate = float(-np.polyfit(indices, log_S, 1)[0])
    # Normalise to [0, 1]: rate=0 → 0.0 (flat), rate→∞ → 1.0 (instant drop)
    return float(1.0 - np.exp(-rate))


def effective_rank(S: np.ndarray) -> float:
    """Participation ratio: (sum(S))^2 / sum(S^2)."""
    S = S[S > 0]
    if S.size == 0:
        return 0.0
    return float(np.sum(S) ** 2 / np.sum(S ** 2))


def condition_number(S: np.ndarray) -> float:
    """S[0] / S[-1], guarded against division by zero."""
    S = S[S > 0]
    if S.size == 0:
        return 0.0
    return float(S[0] / S[-1]) if S[-1] != 0 else float("inf")
