from __future__ import annotations

import warnings
import zipfile

_LEVEL_RANGES: dict[str, tuple[int, int] | None] = {
    "zstd": (1, 22),
    "gzip": (1, 9),
    "zlib": (1, 9),
    "xz":   (0, 9),
    "none": None,
}

_DEFAULTS: dict[str, int | None] = {
    "zstd": 3,
    "gzip": 6,
    "zlib": 6,
    "xz":   6,
    "none": None,
}


def get_compressor(name: str) -> int:
    """Map a compression method name to a zipfile compression constant.

    "zstd" falls back to "zlib" with a warning if zstandard is not installed.
    Returns a zipfile.ZIP_* constant.
    """
    name = name.lower()
    if name == "zstd":
        try:
            import zstandard  # noqa: F401
            return zipfile.ZIP_DEFLATED  # zstd handled externally; zip shell uses DEFLATED
        except ImportError:
            warnings.warn(
                "zstandard package not installed — falling back to zlib compression.",
                stacklevel=2,
            )
            return zipfile.ZIP_DEFLATED
    elif name in ("zlib", "gzip"):
        return zipfile.ZIP_DEFLATED
    elif name == "xz":
        return zipfile.ZIP_LZMA
    elif name == "none":
        return zipfile.ZIP_STORED
    else:
        raise ValueError(
            f"Unknown compression method '{name}'. "
            f"Supported: {', '.join(sorted(_LEVEL_RANGES))}"
        )


def clamp_level(method: str, level: int | None) -> int | None:
    """Validate and clamp compression level to the algorithm's valid range.

    Warns if the level was out of range. Returns None for method 'none'.
    """
    method = method.lower()
    if method not in _LEVEL_RANGES:
        raise ValueError(f"Unknown method '{method}'")

    valid = _LEVEL_RANGES[method]
    if valid is None:
        return None  # "none" has no level

    if level is None:
        return _DEFAULTS[method]

    lo, hi = valid
    if level < lo or level > hi:
        clamped = max(lo, min(hi, level))
        warnings.warn(
            f"Compression level {level} out of range for '{method}' "
            f"(valid {lo}–{hi}); clamping to {clamped}.",
            stacklevel=2,
        )
        return clamped
    return level
