from __future__ import annotations

import io
import json
import zipfile
from typing import Optional

import numpy as np

from spectra.compression.registry import clamp_level, get_compressor


def pack(
    tensors: dict[str, np.ndarray],
    manifest: dict,
    out_path: str,
    compression: str = "zstd",
    level: Optional[int] = None,
) -> None:
    """Pack tensors and manifest into a .stz archive.

    Structure:
        out_path (zip)
        ├── tensors.npz
        └── manifest.json

    The zip's compression method is the binary compression layer.
    For zstd: tensors.npz is compressed with zstd externally, then stored
    uncompressed inside the zip (ZIP_STORED), because Python's zipfile does
    not natively support zstd. For all other methods, the zip compression
    constant is used directly.
    """
    level = clamp_level(compression, level)

    # Serialise tensors.npz into memory
    npz_buf = io.BytesIO()
    np.savez(npz_buf, **tensors)
    npz_bytes = npz_buf.getvalue()

    # Serialise manifest.json
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    # Apply binary compression
    compression_lower = compression.lower()
    if compression_lower == "zstd":
        from spectra.compression.zstd import compress as zstd_compress
        npz_bytes = zstd_compress(npz_bytes, level=level or 3)
        zip_method = zipfile.ZIP_STORED
    elif compression_lower == "gzip":
        from spectra.compression.gzip import compress as gz_compress
        npz_bytes = gz_compress(npz_bytes, level=level or 6)
        zip_method = zipfile.ZIP_STORED
    elif compression_lower == "xz":
        from spectra.compression.xz import compress as xz_compress
        npz_bytes = xz_compress(npz_bytes, level=level or 6)
        zip_method = zipfile.ZIP_STORED
    elif compression_lower == "zlib":
        zip_method = zipfile.ZIP_DEFLATED
    else:  # "none"
        zip_method = zipfile.ZIP_STORED

    with zipfile.ZipFile(out_path, "w", compression=zip_method) as zf:
        zf.writestr("tensors.npz", npz_bytes)
        zf.writestr("manifest.json", manifest_bytes)


def unpack_manifest(path: str) -> dict:
    """Read only manifest.json from a .stz archive. Does not load tensors.npz."""
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read("manifest.json").decode("utf-8"))


def unpack_tensors(
    path: str,
    keys: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """Load tensors.npz from a .stz archive and return requested keys.

    Detects whether tensors.npz was compressed externally (zstd/gzip/xz)
    by reading the binary_compression_method from the manifest.
    """
    with zipfile.ZipFile(path, "r") as zf:
        manifest_raw = json.loads(zf.read("manifest.json").decode("utf-8"))
        npz_bytes = zf.read("tensors.npz")

    method = manifest_raw.get("global_stats", {}).get("binary_compression_method", "zlib")

    if method == "zstd":
        from spectra.compression.zstd import decompress as zstd_decomp
        npz_bytes = zstd_decomp(npz_bytes)
    elif method == "gzip":
        from spectra.compression.gzip import decompress as gz_decomp
        npz_bytes = gz_decomp(npz_bytes)
    elif method == "xz":
        from spectra.compression.xz import decompress as xz_decomp
        npz_bytes = xz_decomp(npz_bytes)
    # zlib and none: zipfile already handled decompression

    archive = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
    if keys is None:
        return {k: archive[k] for k in archive.files}
    return {k: archive[k] for k in keys if k in archive.files}
