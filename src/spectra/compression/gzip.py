from __future__ import annotations

import gzip as _gzip


def compress(data: bytes, level: int = 6) -> bytes:
    return _gzip.compress(data, compresslevel=level)


def decompress(data: bytes) -> bytes:
    return _gzip.decompress(data)
