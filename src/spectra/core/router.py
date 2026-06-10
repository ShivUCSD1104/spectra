from __future__ import annotations

import numpy as np

from spectra.analysis.entropy import shannon_entropy
from spectra.analysis.sparsity import sparsity_fraction
from spectra.analysis.spectrum import randomized_svd_top_k, spectral_decay_rate
from spectra.core.artifact import Artifact, TensorRecord
from spectra.transforms.factorize import find_rank_for_tolerance
from spectra.transforms.quantize import int8_safe


def route_tensor(
    record: TensorRecord,
    artifact: Artifact,
    tolerance: float = 0.01,
    wavelet: bool = False,
    spatial_dims: int = 3,
    min_size_bytes: int = 0,
    no_factorize: bool = False,
    no_quantize: bool = False,
    lossless_only: bool = False,
) -> dict:
    """Determine the optimal compression strategy for a single tensor.

    Returns a routing decision dict:
        {
            "strategy":       str,          # e.g. "svd", "quantize_fp16", ...
            "params":         dict,         # strategy-specific parameters
            "estimated_ratio": float,       # estimated compression ratio
            "reason":         str,          # human-readable explanation
        }
    """
    arr = record.data
    name = record.name

    # Below min_size threshold — skip
    if arr.nbytes < min_size_bytes:
        return {
            "strategy": "dense",
            "params": {},
            "estimated_ratio": 1.0,
            "reason": f"below min_size ({arr.nbytes} < {min_size_bytes} bytes)",
        }

    # ── 1D tensors ────────────────────────────────────────────────────────────
    if arr.ndim == 1:
        if lossless_only:
            return {
                "strategy": "quantize_fp16",
                "params": {},
                "estimated_ratio": 2.0,
                "reason": "1D tensor, lossless fp16",
            }
        return {
            "strategy": "quantize_fp16",
            "params": {},
            "estimated_ratio": 2.0,
            "reason": "1D tensor — fp16 always safe",
        }

    # ── 2D tensors ────────────────────────────────────────────────────────────
    if arr.ndim == 2:
        # Get or compute SVD (top 64)
        if name in artifact._svd_cache:
            cached = artifact._svd_cache[name]
            if isinstance(cached, tuple):
                U, S, Vt = cached
            else:
                S = cached  # legacy: just S stored
                U, Vt = None, None
        else:
            k = min(64, min(arr.shape) - 1)
            if k >= 1:
                U, S, Vt = randomized_svd_top_k(arr.astype(np.float64), k=k)
                artifact._svd_cache[name] = (U, S, Vt)
            else:
                S = np.array([1.0])
                U, Vt = None, None
                artifact._svd_cache[name] = (U, S, Vt)

        decay = spectral_decay_rate(S)

        # SVD routing (unless disabled or lossless-only)
        # Two triggers: fast exponential decay OR strong energy concentration (sharp rank cliff)
        if not no_factorize and not lossless_only:
            rank = find_rank_for_tolerance(S, arr, tolerance)
            svd_size = (arr.shape[0] * rank + rank + rank * arr.shape[1]) * 4  # float32
            ratio = arr.nbytes / (svd_size + 1e-12)
            energy_fraction = float(np.sum(S[:rank] ** 2) / (np.sum(S ** 2) + 1e-12))
            rank_fraction = rank / min(arr.shape)

            svd_candidate = (
                decay > 0.85                       # fast exponential decay
                or (rank_fraction < 0.3 and ratio > 2.0)  # sharp rank cliff → good compression
            )
            if svd_candidate and ratio > 2.0:
                return {
                    "strategy": "svd",
                    "params": {"rank": rank},
                    "estimated_ratio": ratio,
                    "reason": (
                        f"fast spectral decay ({decay:.2f}), SVD rank={rank}"
                        if decay > 0.85
                        else f"low intrinsic rank ({rank}/{min(arr.shape)}), SVD rank={rank}"
                    ),
                }
            # SVD not worth it — fall through to quantize

        # Entropy-based quantization routing
        if not no_quantize:
            h = shannon_entropy(arr)
            if not lossless_only and h < 5.0 and int8_safe(arr):
                return {
                    "strategy": "quantize_int8",
                    "params": {},
                    "estimated_ratio": 4.0,
                    "reason": f"low entropy ({h:.2f} bits), int8 safe",
                }

            # Sparsity routing
            sp = sparsity_fraction(arr)
            if not lossless_only and sp["near_zero_1e6"] > 0.5:
                nnz = arr.size * (1 - sp["near_zero_1e6"])
                coo_size = nnz * (arr.ndim * 8 + 4)  # int64 indices + float32 values
                ratio = arr.nbytes / (coo_size + 1e-12)
                return {
                    "strategy": "sparse_coo",
                    "params": {"threshold": 1e-6},
                    "estimated_ratio": ratio,
                    "reason": f"sparse ({sp['near_zero_1e6']*100:.0f}% near-zero)",
                }

            return {
                "strategy": "quantize_fp16",
                "params": {},
                "estimated_ratio": 2.0,
                "reason": "dense 2D tensor — fp16 conservative",
            }

        # no_quantize=True and SVD not routed → dense passthrough
        return {
            "strategy": "dense",
            "params": {},
            "estimated_ratio": 1.0,
            "reason": "quantize disabled and SVD not applicable",
        }

    # ── 3D+ tensors (Tucker deferred to Phase 14) ─────────────────────────────
    if not no_quantize:
        return {
            "strategy": "quantize_fp16",
            "params": {},
            "estimated_ratio": 2.0,
            "reason": f"{arr.ndim}D tensor — Tucker not yet implemented, fp16 fallback",
        }

    return {
        "strategy": "dense",
        "params": {},
        "estimated_ratio": 1.0,
        "reason": "quantize disabled, Tucker not yet implemented",
    }
