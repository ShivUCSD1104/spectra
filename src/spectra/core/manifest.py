from __future__ import annotations

import json
from datetime import datetime, timezone

from spectra import __version__
from spectra.core.artifact import Artifact


def build_manifest(
    artifact: Artifact,
    transform_records: list[dict],
    global_stats: dict,
    binary_method: str,
) -> dict:
    """Assemble the full manifest dict from an Artifact and per-tensor records.

    transform_records: one dict per tensor, containing all per-tensor metadata
                       produced by the transform step (storage_type, keys, etc.)
    global_stats:      pre-computed global size stats (original_size_bytes, etc.)
    binary_method:     name of the binary compression algorithm used.
    """
    tensors_section: dict[str, dict] = {}
    for rec in transform_records:
        name = rec["name"]
        tensors_section[name] = {k: v for k, v in rec.items() if k != "name"}

    manifest = {
        "spectra_version": __version__,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": artifact.source_path,
        "source_format": artifact.source_format,
        "global_stats": {
            **global_stats,
            "binary_compression_method": binary_method,
        },
        "tensors": tensors_section,
    }
    return manifest


def write_manifest(manifest: dict) -> str:
    """Serialise manifest to a JSON string."""
    return json.dumps(manifest, indent=2)


def read_manifest(json_str: str) -> dict:
    """Deserialise a manifest from a JSON string."""
    return json.loads(json_str)
