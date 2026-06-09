from __future__ import annotations

import re

_UNITS = ["B", "KB", "MB", "GB", "TB"]
_UNIT_BYTES = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
}


def fmt_bytes(n: int) -> str:
    """Return a human-readable byte count string (e.g. 67108864 → '64.0 MB')."""
    value = float(n)
    for unit in _UNITS[:-1]:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} {_UNITS[-1]}"


def parse_size(s: str) -> int:
    """Parse a human-readable size string into bytes (e.g. '10MB' → 10485760)."""
    s = s.strip()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)", s)
    if not match:
        raise ValueError(f"Cannot parse size: '{s}'. Expected format like '10MB', '1.5GB'.")
    number, suffix = match.group(1), match.group(2).lower()
    if suffix not in _UNIT_BYTES:
        raise ValueError(f"Unknown size unit '{suffix}'. Supported: B, KB, MB, GB, TB.")
    return int(float(number) * _UNIT_BYTES[suffix])
