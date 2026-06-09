from __future__ import annotations

import lzma


def compress(data: bytes, level: int = 6) -> bytes:
    return lzma.compress(data, preset=level)


def decompress(data: bytes) -> bytes:
    return lzma.decompress(data)
