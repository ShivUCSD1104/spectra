from __future__ import annotations

import numpy as np


def isotropic_deviatoric_split(arr: np.ndarray) -> dict:
    """Decompose a 2D square matrix into isotropic and deviatoric parts.

    The isotropic component is (trace / n) * I.
    The deviatoric component is arr minus the isotropic part.

    Returns:
        {
            "isotropic_norm":  Frobenius norm of the isotropic component,
            "deviatoric_norm": Frobenius norm of the deviatoric component,
        }
    """
    if arr.ndim != 2:
        raise ValueError(f"isotropic_deviatoric_split requires a 2D array, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"isotropic_deviatoric_split requires a square matrix, got {arr.shape}")
    n = arr.shape[0]
    iso_scalar = np.trace(arr) / n
    isotropic = iso_scalar * np.eye(n, dtype=arr.dtype)
    deviatoric = arr - isotropic
    return {
        "isotropic_norm": float(np.linalg.norm(isotropic, "fro")),
        "deviatoric_norm": float(np.linalg.norm(deviatoric, "fro")),
    }
