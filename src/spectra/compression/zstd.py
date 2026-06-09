from __future__ import annotations


def compress(data: bytes, level: int = 3) -> bytes:
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard is required for zstd compression. "
            "Install it with: uv add zstandard"
        )
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def decompress(data: bytes) -> bytes:
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard is required for zstd decompression. "
            "Install it with: uv add zstandard"
        )
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)
