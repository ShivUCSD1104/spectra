from __future__ import annotations

import os

import numpy as np

from spectra.core.artifact import Artifact, TensorRecord


def load_npy(path: str) -> Artifact:
    abs_path = os.path.abspath(path)
    arr = np.load(abs_path, allow_pickle=False)
    name = os.path.splitext(os.path.basename(abs_path))[0]
    record = TensorRecord(
        name=name,
        data=arr,
        original_dtype=arr.dtype,
        source_file=abs_path,
    )
    return Artifact(
        tensors={name: record},
        source_path=abs_path,
        source_format="npy",
    )


def load_npz(path: str) -> Artifact:
    abs_path = os.path.abspath(path)
    archive = np.load(abs_path, allow_pickle=False)
    tensors: dict[str, TensorRecord] = {}
    for key in archive.files:
        arr = archive[key]
        tensors[key] = TensorRecord(
            name=key,
            data=arr,
            original_dtype=arr.dtype,
            source_file=abs_path,
        )
    return Artifact(
        tensors=tensors,
        source_path=abs_path,
        source_format="npz",
    )


def load(path: str) -> Artifact:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_npy(path)
    elif ext == ".npz":
        return load_npz(path)
    elif ext == ".stz":
        from spectra.loaders.stz_loader import load_stz
        return load_stz(path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Supported: .npy, .npz, .stz")
