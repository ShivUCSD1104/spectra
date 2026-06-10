from __future__ import annotations

import os

import numpy as np

from spectra.core.artifact import Artifact, TensorRecord
from spectra.formats.stz import unpack_manifest, unpack_tensors
from spectra.transforms.quantize import dequantize_int8
from spectra.transforms.sparsify import reconstruct_coo

_NOT_YET = {"svd", "tucker", "tucker_wavelet", "svd_wavelet"}


def load_stz(path: str) -> Artifact:
    """Load a .stz archive into an Artifact, reconstructing each tensor.

    Uses the same dispatch logic as the `extract` command.
    Reconstructed tensors are cast back to their original dtype.
    """
    abs_path = os.path.abspath(path)
    manifest = unpack_manifest(abs_path)
    tensor_meta = manifest["tensors"]

    # Collect all keys needed
    needed_keys: list[str] = []
    for meta in tensor_meta.values():
        needed_keys.extend(meta.get("keys", []))

    raw = unpack_tensors(abs_path, keys=needed_keys)

    tensors: dict[str, TensorRecord] = {}

    for name, meta in tensor_meta.items():
        storage = meta["storage_type"]

        if storage in _NOT_YET:
            raise NotImplementedError(
                f"Cannot load tensor '{name}': storage_type '{storage}' "
                "requires SVD/Tucker support (Phase 13)."
            )

        orig_dtype = np.dtype(meta["original_dtype"])

        if storage == "dense":
            arr = raw[meta["keys"][0]].astype(orig_dtype)

        elif storage == "quantized_fp16":
            arr = raw[meta["keys"][0]].astype(orig_dtype)

        elif storage == "quantized_int8":
            q = raw[meta["keys"][0]]
            arr = dequantize_int8(q, meta["scale"], meta["zero_point"], orig_dtype)

        elif storage == "sparse_coo":
            indices = raw[f"{name}__indices"]
            values  = raw[f"{name}__values"]
            shape   = tuple(meta["original_shape"])
            arr = reconstruct_coo(indices, values, shape, dtype=orig_dtype)

        else:
            raise ValueError(f"Unknown storage_type '{storage}' for tensor '{name}'.")

        tensors[name] = TensorRecord(
            name=name,
            data=arr,
            original_dtype=orig_dtype,
            source_file=abs_path,
        )

    return Artifact(
        tensors=tensors,
        source_path=abs_path,
        source_format="stz",
    )
