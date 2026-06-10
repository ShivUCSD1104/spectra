from __future__ import annotations

import numpy as np

from spectra.analysis.spectrum import spectral_decay_rate


def svd_compress(
    arr: np.ndarray,
    rank: int,
    cached_svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[dict, dict]:
    """Compress a 2D matrix via truncated SVD.

    Uses cached (U, S, Vt) if provided and rank fits; otherwise computes
    a fresh full SVD via np.linalg.svd.

    Returns:
        arrays:   {"__U": U_r, "__S": S_r, "__Vt": Vt_r}
        metadata: {rank_used, rank_full, spectrum_decay_rate,
                   reconstruction_error_mse, reconstruction_error_relative,
                   storage_type, lossless, ...}
    """
    if arr.ndim != 2:
        raise ValueError(f"svd_compress requires a 2D array, got shape {arr.shape}")

    arr64 = arr.astype(np.float64)

    # Use cache if rank fits within cached truncation
    if cached_svd is not None and len(cached_svd[1]) >= rank:
        U_full, S_full, Vt_full = cached_svd
    else:
        U_full, S_full, Vt_full = np.linalg.svd(arr64, full_matrices=False)

    rank_full = len(S_full)
    rank = min(rank, rank_full)

    U_r  = U_full[:, :rank]
    S_r  = S_full[:rank]
    Vt_r = Vt_full[:rank, :]

    reconstructed = svd_reconstruct(U_r, S_r, Vt_r)
    diff = arr64 - reconstructed
    mse  = float(np.mean(diff ** 2))
    frob_orig = float(np.linalg.norm(arr64))
    rel  = float(np.linalg.norm(diff) / (frob_orig + 1e-12))

    decay = spectral_decay_rate(S_full)

    arrays = {
        "__U":  U_r.astype(np.float32),
        "__S":  S_r.astype(np.float32),
        "__Vt": Vt_r.astype(np.float32),
    }
    size_stored = sum(a.nbytes for a in arrays.values())
    metadata = {
        "storage_type": "svd",
        "original_shape": list(arr.shape),
        "original_dtype": str(arr.dtype),
        "lossless": False,
        "reconstruction_method": "U @ diag(S) @ Vt",
        "rank_used": rank,
        "rank_full": rank_full,
        "spectrum_decay_rate": decay,
        "reconstruction_error_mse": mse,
        "reconstruction_error_relative": rel,
        "size_original_bytes": arr.nbytes,
        "size_stored_bytes": size_stored,
        "compression_ratio": arr.nbytes / (size_stored + 1e-12),
    }
    return arrays, metadata


def svd_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Reconstruct a matrix from SVD factors: U @ diag(S) @ Vt."""
    return U @ np.diag(S) @ Vt


def find_rank_for_tolerance(
    S: np.ndarray,
    arr: np.ndarray,
    tolerance: float,
) -> int:
    """Find the smallest rank k where SVD relative reconstruction error < tolerance.

    Uses the Eckart-Young theorem with the TRUE Frobenius norm of arr as the
    energy denominator (not sum(S**2), which is only the partial SVD energy):

        relative_error(k) = sqrt(1 - sum(S[:k]**2) / ||arr||_F**2)

    S must be sorted in descending order (standard SVD convention).
    Falls back to len(S) if no rank within the captured singular values satisfies
    the tolerance — the caller should treat this as "SVD not viable at this k".
    """
    # Use the true Frobenius norm of the original array as the energy denominator.
    # S is a partial SVD (top-k values only); using sum(S**2) would underestimate
    # the total spectral energy for matrices with broadly distributed spectra
    # (e.g. BERT attention weights), causing the required rank to be massively
    # under-estimated and the tolerance guarantee to be violated.
    total_energy = float(np.linalg.norm(arr.ravel()) ** 2)
    if total_energy == 0:
        return 1

    # Energy captured by keeping the top-k singular values (cumulative from rank 1)
    captured_energy = np.cumsum(S ** 2)
    # Relative error at rank k = sqrt(1 - captured(k) / total)
    rel_errors = np.sqrt(np.maximum(0.0, 1.0 - captured_energy / total_energy))

    # rel_errors[i] = error when using rank i+1.
    # Find first index where error drops below tolerance → rank = index + 1.
    passing = np.where(rel_errors < tolerance)[0]
    if len(passing) == 0:
        return len(S)  # even full captured rank is insufficient (matrix needs more than k SVs)
    return max(1, int(passing[0]) + 1)
