from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TensorRecord:
    name: str
    data: np.ndarray
    original_dtype: np.dtype
    source_file: str


@dataclass
class Artifact:
    tensors: dict[str, TensorRecord]
    source_path: str
    source_format: str
    _svd_cache: dict = field(default_factory=dict, repr=False, compare=False)
