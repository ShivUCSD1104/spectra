from __future__ import annotations

import numpy as np


def sparsify(arr: np.ndarray, threshold: float) -> tuple[dict, dict]:
    """Zero out values with abs(x) < threshold, then store in COO format.

    Returns:
        arrays:   {"__indices": ndarray, "__values": ndarray}
        metadata: {original_shape, original_dtype, sparsity_fraction,
                   threshold, lossless, storage_type}
    """
    zeroed = arr.copy()
    zeroed[np.abs(zeroed) < threshold] = 0.0

    indices = np.argwhere(zeroed != 0)          # shape (nnz, ndim)
    values = zeroed[zeroed != 0]                # shape (nnz,)

    sparsity_frac = 1.0 - (values.size / arr.size) if arr.size > 0 else 1.0

    arrays = {
        "__indices": indices.astype(np.int64),
        "__values": values,
    }
    metadata = {
        "storage_type": "sparse_coo",
        "original_shape": list(arr.shape),
        "original_dtype": str(arr.dtype),
        "sparsity_fraction": sparsity_frac,
        "threshold": threshold,
        "lossless": threshold == 0.0,
        "reconstruction_method": "reconstruct dense from COO indices + values",
    }
    return arrays, metadata


def reconstruct_coo(
    indices: np.ndarray,
    values: np.ndarray,
    shape: tuple,
    dtype=None,
) -> np.ndarray:
    """Reconstruct a dense array from COO indices and values."""
    out = np.zeros(shape, dtype=dtype if dtype is not None else values.dtype)
    if values.size > 0:
        out[tuple(indices.T)] = values
    return out
