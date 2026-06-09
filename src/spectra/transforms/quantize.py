from __future__ import annotations

import numpy as np

from spectra.analysis.entropy import shannon_entropy


def quantize_fp16(arr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Cast array to float16 and return (quantized, metadata).

    Metadata keys:
        original_dtype, fp16_overflow_detected, lossless, storage_type
    """
    fp16_overflow = bool(np.any(np.abs(arr) > 65504))
    with np.errstate(over="ignore"):
        quantized = arr.astype(np.float16)
    metadata = {
        "storage_type": "quantized_fp16",
        "original_dtype": str(arr.dtype),
        "fp16_overflow_detected": fp16_overflow,
        "lossless": not fp16_overflow,
        "reconstruction_method": "cast_to_original_dtype",
    }
    return quantized, metadata


def quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Quantize array to int8 with per-tensor affine scale/zero_point.

    scale     = (max - min) / 255
    zero_point = -round(min / scale) - 128   (maps min → -128, max → +127)

    Metadata keys:
        original_dtype, scale, zero_point, lossless, storage_type
    """
    mn = float(arr.min())
    mx = float(arr.max())

    if mx == mn:
        # Constant tensor — map everything to 0
        scale = 1.0
        zero_point = 0
        quantized = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = (mx - mn) / 255.0
        zero_point = int(round(-mn / scale)) - 128
        zero_point = int(np.clip(zero_point, -128, 127))
        q = np.round(arr / scale + zero_point + 128).clip(0, 255).astype(np.uint8)
        quantized = (q.astype(np.int16) - 128).astype(np.int8)

    metadata = {
        "storage_type": "quantized_int8",
        "original_dtype": str(arr.dtype),
        "scale": scale,
        "zero_point": zero_point,
        "lossless": False,
        "reconstruction_method": "dequantize: x * scale + zero_point",
    }
    return quantized, metadata


def dequantize_int8(
    arr: np.ndarray,
    scale: float,
    zero_point: int,
    original_dtype,
) -> np.ndarray:
    """Reverse int8 quantization: (arr.astype(float32) * scale) + zero_point."""
    return ((arr.astype(np.float32) + 128 - zero_point - 128) * scale).astype(original_dtype)


def int8_safe(arr: np.ndarray) -> bool:
    """Return True if the array is safe to quantize to int8.

    Criteria (from spec):
        entropy < 5.0 bits  AND  dynamic_range < 20.0
    where dynamic_range = max(abs(x)) / mean(abs(x)).
    """
    h = shannon_entropy(arr)
    mean_abs = float(np.mean(np.abs(arr)))
    if mean_abs == 0:
        return True
    dynamic_range = float(np.max(np.abs(arr))) / mean_abs
    return h < 5.0 and dynamic_range < 20.0
