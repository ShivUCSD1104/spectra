from __future__ import annotations

import csv
import fnmatch
import json
import os
import sys
from typing import List, Optional

import numpy as np
import typer
from rich import print as rprint
from rich.console import Console

from spectra.analysis.decomposition import isotropic_deviatoric_split
from spectra.analysis.entropy import shannon_entropy
from spectra.analysis.geometry import intrinsic_dim_estimate, participation_ratio
from spectra.analysis.sparsity import sparsity_fraction
from spectra.analysis.spectrum import (
    condition_number,
    effective_rank,
    randomized_svd_top_k,
    spectral_decay_rate,
)
from spectra.core.manifest import build_manifest
from spectra.core.router import route_tensor
from spectra.formats.stz import pack, unpack_manifest, unpack_tensors
from spectra.loaders.numpy_loader import load
from spectra.transforms.factorize import svd_compress, svd_reconstruct, find_rank_for_tolerance
from spectra.transforms.quantize import quantize_fp16, quantize_int8, dequantize_int8
from spectra.transforms.sparsify import sparsify, reconstruct_coo
from spectra.utils.display import make_tensor_table, print_summary_block
from spectra.utils.units import fmt_bytes, parse_size

app = typer.Typer(help="Spectra — tensor analysis and compression toolkit.")
console = Console()

# ── helpers ───────────────────────────────────────────────────────────────────


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _mode_wise_ranks(arr: np.ndarray, tolerance: float = 0.01) -> list[int]:
    """Estimate Tucker rank per mode via unfolding + SVD energy threshold."""
    ranks = []
    for mode in range(arr.ndim):
        unfolded = np.reshape(np.moveaxis(arr, mode, 0), (arr.shape[mode], -1))
        k = min(64, min(unfolded.shape) - 1)
        if k < 1:
            ranks.append(arr.shape[mode])
            continue
        _, S, _ = randomized_svd_top_k(unfolded.astype(np.float64), k=k)
        energy = np.cumsum(S**2) / (np.sum(S**2) + 1e-12)
        rank = int(np.searchsorted(energy, 1.0 - tolerance**2) + 1)
        ranks.append(min(rank, arr.shape[mode]))
    return ranks


def _build_recommendation(
    arr: np.ndarray,
    entropy_val: float,
    sparsity: dict,
    decay: float | None,
    ndim: int,
    tucker_ranks: list[int] | None,
) -> str:
    parts: list[str] = []

    # fp16 safety
    fp16_overflow = bool(np.any(np.abs(arr) > 65504))
    if not fp16_overflow:
        parts.append("fp16 safe")

    # int8 safety
    dynamic_range = float(np.max(np.abs(arr)) / (np.mean(np.abs(arr)) + 1e-12))
    if entropy_val < 5.0 and dynamic_range < 20.0:
        parts.append("int8 safe")

    # SVD candidate (2D)
    if decay is not None and decay > 0.85:
        parts.append(f"SVD candidate (decay={decay:.2f})")

    # Tucker candidate (3D+)
    if tucker_ranks is not None:
        orig_size = arr.size
        tucker_size = np.prod(tucker_ranks) + sum(
            arr.shape[m] * tucker_ranks[m] for m in range(arr.ndim)
        )
        ratio = orig_size / (tucker_size + 1e-12)
        if ratio > 2.0:
            parts.append(f"Tucker {tucker_ranks} → {ratio:.1f}x")

    # Sparsify candidate
    if sparsity["near_zero_1e6"] > 0.5:
        parts.append(f"sparse ({sparsity['near_zero_1e6']*100:.0f}% near-zero)")

    return ", ".join(parts) if parts else "dense, leave as-is"


def _analyze_tensor(name: str, arr: np.ndarray, source_file: str) -> dict:
    """Run the full analysis pipeline on one tensor. Returns a flat stats dict."""
    size_bytes = arr.nbytes
    params = arr.size

    sp = sparsity_fraction(arr)
    h = shannon_entropy(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    p1, p99 = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))

    record: dict = {
        "name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "params": params,
        "size_bytes": size_bytes,
        "sparsity": sp["near_zero_1e6"],
        "exact_zero": sp["exact_zero"],
        "entropy": h,
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "p1": p1,
        "p99": p99,
        # 2D-specific
        "decay_rate": None,
        "effective_rank": None,
        "intrinsic_dim": None,
        "condition_number": None,
        "isotropic_norm": None,
        "deviatoric_norm": None,
        "svd_S": None,
        # 3D+-specific
        "tucker_ranks": None,
        "mode_ranks": None,
    }

    if arr.ndim == 2:
        k = min(64, min(arr.shape) - 1)
        if k >= 1:
            U, S, Vt = randomized_svd_top_k(arr.astype(np.float64), k=k)
            record["svd_S"] = S
            record["decay_rate"] = spectral_decay_rate(S)
            record["effective_rank"] = effective_rank(S)
            record["intrinsic_dim"] = intrinsic_dim_estimate(S)
            record["condition_number"] = condition_number(S)
            if arr.shape[0] == arr.shape[1]:
                iso = isotropic_deviatoric_split(arr.astype(np.float64))
                record["isotropic_norm"] = iso["isotropic_norm"]
                record["deviatoric_norm"] = iso["deviatoric_norm"]

    elif arr.ndim >= 3:
        mode_ranks = _mode_wise_ranks(arr)
        record["mode_ranks"] = mode_ranks
        record["tucker_ranks"] = mode_ranks  # initial estimate = mode-wise ranks at tol=0.01

    record["recommendation"] = _build_recommendation(
        arr,
        h,
        sp,
        record["decay_rate"],
        arr.ndim,
        record["tucker_ranks"],
    )
    return record


# ── inspect command ───────────────────────────────────────────────────────────

SORT_KEYS = {"size", "entropy", "rank", "sparsity", "name"}
DEPTH_OPTS = {"summary", "full"}
FORMAT_OPTS = {"table", "csv", "json"}


@app.command()
def inspect(
    file: str = typer.Argument(..., help="Path to .npy, .npz, .pt, or .pth file"),
    tensor: Optional[str] = typer.Option(None, "--tensor", help="Inspect a single named tensor only"),
    sort: str = typer.Option("size", "--sort", help="Sort by: size, entropy, rank, sparsity, name"),
    top: Optional[int] = typer.Option(None, "--top", help="Show only top N tensors"),
    depth: str = typer.Option("summary", "--depth", help="summary | full"),
    format: str = typer.Option("table", "--format", help="table | csv | json"),
):
    """Profile every tensor in an artifact. No output file. Read-only."""

    if sort not in SORT_KEYS:
        typer.echo(f"--sort must be one of: {', '.join(sorted(SORT_KEYS))}", err=True)
        raise typer.Exit(1)
    if depth not in DEPTH_OPTS:
        typer.echo(f"--depth must be one of: {', '.join(sorted(DEPTH_OPTS))}", err=True)
        raise typer.Exit(1)
    if format not in FORMAT_OPTS:
        typer.echo(f"--format must be one of: {', '.join(sorted(FORMAT_OPTS))}", err=True)
        raise typer.Exit(1)

    # Load
    try:
        artifact = load(file)
    except Exception as e:
        typer.echo(f"Error loading file: {e}", err=True)
        raise typer.Exit(1)

    # Filter to single tensor if requested
    tensor_items = list(artifact.tensors.items())
    if tensor is not None:
        tensor_items = [(k, v) for k, v in tensor_items if k == tensor]
        if not tensor_items:
            typer.echo(f"Tensor '{tensor}' not found in {file}", err=True)
            raise typer.Exit(1)

    # Analyse
    records: list[dict] = []
    with console.status("[bold green]Analysing tensors…"):
        for name, trec in tensor_items:
            r = _analyze_tensor(name, trec.data, trec.source_file)
            # Cache SVD for later reuse by router
            if r["svd_S"] is not None:
                artifact._svd_cache[name] = r["svd_S"]
            records.append(r)

    # Sort
    sort_fn = {
        "size":     lambda r: -r["size_bytes"],
        "entropy":  lambda r: -r["entropy"],
        "rank":     lambda r: -(r["effective_rank"] or 0),
        "sparsity": lambda r: -r["sparsity"],
        "name":     lambda r: r["name"],
    }[sort]
    records.sort(key=sort_fn)

    if top is not None:
        records = records[:top]

    # Header
    total_params = sum(r["params"] for r in records)
    total_bytes = sum(r["size_bytes"] for r in records)
    rprint(
        f"\n[bold]File:[/bold] {file}  |  "
        f"[bold]{len(records)}[/bold] tensors  |  "
        f"[bold]{_fmt_params(total_params)}[/bold] parameters  |  "
        f"[bold]{fmt_bytes(total_bytes)}[/bold]\n"
    )

    # Render
    if format == "json":
        output = []
        for r in records:
            row = {k: v for k, v in r.items() if k != "svd_S"}
            row["shape"] = list(row["shape"])
            output.append(row)
        print(json.dumps(output, indent=2))

    elif format == "csv":
        fieldnames = ["name", "shape", "dtype", "size_bytes", "sparsity",
                      "entropy", "decay_rate", "effective_rank", "condition_number",
                      "recommendation"]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["shape"] = "x".join(str(d) for d in r["shape"])
            writer.writerow(row)

    else:  # table (default)
        table = make_tensor_table(records)
        rprint(table)

        if depth == "full":
            for r in records:
                rprint(f"\n[bold cyan]{r['name']}[/bold cyan]  {r['shape']}  {r['dtype']}")
                rprint(f"  size       : {fmt_bytes(r['size_bytes'])}  ({r['params']:,} params)")
                rprint(f"  sparsity   : exact={r['exact_zero']*100:.1f}%  near-zero={r['sparsity']*100:.1f}%")
                rprint(f"  entropy    : {r['entropy']:.3f} bits")
                rprint(f"  stats      : mean={r['mean']:.4f}  std={r['std']:.4f}  "
                       f"min={r['min']:.4f}  max={r['max']:.4f}  "
                       f"p1={r['p1']:.4f}  p99={r['p99']:.4f}")
                if r["decay_rate"] is not None:
                    rprint(f"  decay rate : {r['decay_rate']:.4f}")
                    rprint(f"  eff. rank  : {r['effective_rank']:.2f}")
                    rprint(f"  intrinsic  : {r['intrinsic_dim']}")
                    rprint(f"  cond. num  : {r['condition_number']:.2e}")
                if r["isotropic_norm"] is not None:
                    rprint(f"  iso norm   : {r['isotropic_norm']:.4f}")
                    rprint(f"  dev norm   : {r['deviatoric_norm']:.4f}")
                if r["mode_ranks"] is not None:
                    rprint(f"  mode ranks : {r['mode_ranks']}")
                rprint(f"  recommend  : {r['recommendation']}")

    # Summary block (table format only)
    if format == "table":
        svd_tucker = [r for r in records if r["decay_rate"] is not None and r["decay_rate"] > 0.85
                      or r["tucker_ranks"] is not None]
        quant_only = [r for r in records if r not in svd_tucker]
        leave = [r for r in quant_only if r["sparsity"] < 0.1 and r["entropy"] > 5.0
                 and (r["decay_rate"] is None or r["decay_rate"] <= 0.85)]
        quant_only = [r for r in quant_only if r not in leave]

        print_summary_block({
            "svd_tucker_count":  len(svd_tucker),
            "svd_tucker_bytes":  sum(r["size_bytes"] for r in svd_tucker),
            "quantize_count":    len(quant_only),
            "quantize_bytes":    sum(r["size_bytes"] for r in quant_only),
            "leave_alone_count": len(leave),
            "leave_alone_bytes": sum(r["size_bytes"] for r in leave),
        })


# ── helpers shared by transform / compress ───────────────────────────────────

def _tensor_matches(name: str, select: Optional[str], exclude: Optional[str]) -> bool:
    if select is not None and not fnmatch.fnmatch(name, select):
        return False
    if exclude is not None and fnmatch.fnmatch(name, exclude):
        return False
    return True


def _passthrough_record(name: str, arr: np.ndarray) -> dict:
    return {
        "name": name,
        "storage_type": "dense",
        "original_shape": list(arr.shape),
        "original_dtype": str(arr.dtype),
        "lossless": True,
        "reconstruction_method": "load directly",
        "size_original_bytes": arr.nbytes,
        "size_stored_bytes": arr.nbytes,
        "compression_ratio": 1.0,
        "keys": [name],
    }


def _default_out(file: str) -> str:
    base = os.path.splitext(file)[0]
    return base + ".stz"


def _reconstruction_errors(original: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    orig64 = original.astype(np.float64).ravel()
    rest64 = restored.astype(np.float64).ravel()
    mse = float(np.mean((orig64 - rest64) ** 2))
    norm_orig = float(np.linalg.norm(orig64))
    rel = float(np.linalg.norm(orig64 - rest64) / (norm_orig + 1e-12))
    return mse, rel


def _print_transform_report(
    results: list[dict],
    out_path: Optional[str],
    dry_run: bool,
) -> None:
    rprint("\n[bold]Transform Report[/bold]")
    rprint("─" * 65)

    total_original = 0
    total_stored = 0
    max_rel_err = 0.0
    transformed_count = 0

    for r in results:
        shape_str = "x".join(str(d) for d in r["shape"])
        rprint(f"\n[cyan]{r['name']}[/cyan]  {r['dtype']} [{shape_str}]")

        if r.get("skipped"):
            rprint(f"  → Skipped ({r['reason']})")
        else:
            rprint(f"  → {r['transform']}")
            orig_str = fmt_bytes(r["size_original"])
            stor_str = fmt_bytes(r["size_stored"])
            ratio = r["size_original"] / (r["size_stored"] + 1e-12)
            rprint(f"  → {orig_str} → {stor_str}  ({ratio:.1f}x)")
            if r.get("mse") is not None:
                rprint(f"  → MSE: {r['mse']:.4f}  |  Relative error: {r['rel_err']*100:.2f}%")
            transformed_count += 1
            max_rel_err = max(max_rel_err, r.get("rel_err") or 0.0)

        total_original += r["size_original"]
        total_stored += r.get("size_stored", r["size_original"])

    rprint("\n[bold]Summary[/bold]")
    rprint(f"  Tensors transformed:  {transformed_count} / {len(results)}")
    rprint(f"  Original size:        {fmt_bytes(total_original)}")
    rprint(f"  Stored size:          {fmt_bytes(total_stored)}")
    overall = total_original / (total_stored + 1e-12)
    rprint(f"  Overall ratio:        {overall:.1f}x")
    if max_rel_err > 0:
        rprint(f"  Max relative error:   {max_rel_err*100:.2f}%")
    if not dry_run and out_path:
        rprint(f"  Written to: [green]{out_path}[/green]")
    elif dry_run:
        rprint("  [yellow](dry run — nothing written)[/yellow]")


# ── transform command ─────────────────────────────────────────────────────────

@app.command()
def transform(
    file: str = typer.Argument(..., help="Input file (.npy, .npz)"),
    out: Optional[str] = typer.Option(None, "--out", help="Output .stz path (default: <input>.stz)"),
    quantize: Optional[str] = typer.Option(None, "--quantize", help="fp16 | int8"),
    factorize: Optional[str] = typer.Option(None, "--factorize", help="svd | tucker (Phase 13)"),
    rank: Optional[int] = typer.Option(None, "--rank", help="Fixed SVD rank"),
    sparsify_threshold: Optional[float] = typer.Option(None, "--sparsify", help="Zero out abs(x) < threshold"),
    select: Optional[str] = typer.Option(None, "--select", help="Glob pattern to select tensors"),
    exclude: Optional[str] = typer.Option(None, "--exclude", help="Glob pattern to exclude tensors"),
    min_size: str = typer.Option("0", "--min-size", help="Skip tensors smaller than this (e.g. 1MB)"),
    skip_1d: bool = typer.Option(True, "--skip-1d/--no-skip-1d", help="Skip 1D tensors for --factorize"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Show plan, write nothing"),
    report: bool = typer.Option(True, "--report/--no-report", help="Print per-tensor report"),
    binary_compress: str = typer.Option("zstd", "--binary-compress", help="zstd | gzip | xz | zlib | none"),
    binary_level: Optional[int] = typer.Option(None, "--binary-level", help="Compression level"),
):
    """Apply explicit transforms to all tensors (or a selection). Outputs a .stz archive."""

    # --factorize tucker not yet available
    if factorize is not None and factorize not in ("svd",):
        typer.echo(
            f"Error: --factorize {factorize} is not yet implemented. "
            "Supported: svd. Tucker is added in Phase 14.",
            err=True,
        )
        raise typer.Exit(1)

    if factorize == "svd" and rank is None:
        typer.echo("Error: --factorize svd requires --rank N", err=True)
        raise typer.Exit(1)

    if quantize is not None and quantize not in ("fp16", "int8"):
        typer.echo("--quantize must be fp16 or int8", err=True)
        raise typer.Exit(1)

    min_size_bytes = parse_size(min_size)
    out_path = out or _default_out(file)

    # Load
    try:
        artifact = load(file)
    except Exception as e:
        typer.echo(f"Error loading file: {e}", err=True)
        raise typer.Exit(1)

    # Process each tensor
    output_tensors: dict[str, np.ndarray] = {}
    transform_records: list[dict] = []
    report_rows: list[dict] = []

    for name, trec in artifact.tensors.items():
        arr = trec.data
        orig_bytes = arr.nbytes

        # Filter
        if not _tensor_matches(name, select, exclude):
            output_tensors[name] = arr
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "skipped": True, "reason": "not selected",
            })
            transform_records.append(_passthrough_record(name, arr))
            continue

        if orig_bytes < min_size_bytes:
            output_tensors[name] = arr
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "skipped": True, "reason": "below --min-size",
            })
            transform_records.append(_passthrough_record(name, arr))
            continue

        # Apply sparsify first (if set)
        if sparsify_threshold is not None:
            coo_arrays, coo_meta = sparsify(arr, sparsify_threshold)
            for suffix, sub_arr in coo_arrays.items():
                output_tensors[f"{name}{suffix}"] = sub_arr
            stored_bytes = sum(a.nbytes for a in coo_arrays.values())
            rec = {
                "name": name,
                **coo_meta,
                "size_original_bytes": orig_bytes,
                "size_stored_bytes": stored_bytes,
                "compression_ratio": orig_bytes / (stored_bytes + 1e-12),
                "reconstruction_error_mse": 0.0,
                "reconstruction_error_relative": 0.0,
                "keys": [f"{name}{s}" for s in coo_arrays],
            }
            transform_records.append(rec)
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "size_stored": stored_bytes,
                "transform": f"sparsify threshold={sparsify_threshold}",
                "mse": 0.0, "rel_err": 0.0,
            })
            # Use sparsified arr for any further transform
            arr = arr.copy()
            arr[np.abs(arr) < sparsify_threshold] = 0.0
            if quantize is None:
                continue

        # Apply SVD factorize
        if factorize == "svd" and arr.ndim == 2:
            cached = artifact._svd_cache.get(name)
            svd_arrays, svd_meta = svd_compress(arr, rank, cached_svd=cached)
            for suffix, sub_arr in svd_arrays.items():
                output_tensors[f"{name}{suffix}"] = sub_arr
            stored_bytes = sum(a.nbytes for a in svd_arrays.values())
            rec = {
                "name": name,
                **svd_meta,
                "keys": [f"{name}{s}" for s in svd_arrays],
                "strategy_reason": f"explicit --factorize svd --rank {rank}",
            }
            transform_records.append(rec)
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "size_stored": stored_bytes,
                "transform": f"SVD rank={rank}",
                "mse": svd_meta["reconstruction_error_mse"],
                "rel_err": svd_meta["reconstruction_error_relative"],
            })
            continue

        elif factorize == "svd" and arr.ndim != 2:
            # skip non-2D for SVD
            output_tensors[name] = arr
            transform_records.append(_passthrough_record(name, arr))
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "skipped": True,
                "reason": f"SVD skipped ({arr.ndim}D tensor)",
            })
            continue

        # Apply quantize
        if quantize == "fp16":
            q_arr, q_meta = quantize_fp16(arr)
            stored_bytes = q_arr.nbytes
            restored = q_arr.astype(arr.dtype)
            mse, rel = _reconstruction_errors(arr, restored)
            output_tensors[name] = q_arr
            rec = {
                "name": name,
                **q_meta,
                "original_shape": list(arr.shape),
                "size_original_bytes": orig_bytes,
                "size_stored_bytes": stored_bytes,
                "compression_ratio": orig_bytes / (stored_bytes + 1e-12),
                "reconstruction_error_mse": mse,
                "reconstruction_error_relative": rel,
                "keys": [name],
            }
            transform_records.append(rec)
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "size_stored": stored_bytes,
                "transform": "quantize fp16",
                "mse": mse, "rel_err": rel,
            })

        elif quantize == "int8":
            q_arr, q_meta = quantize_int8(arr)
            stored_bytes = q_arr.nbytes
            restored = dequantize_int8(q_arr, q_meta["scale"], q_meta["zero_point"], arr.dtype)
            mse, rel = _reconstruction_errors(arr, restored)
            output_tensors[name] = q_arr
            rec = {
                "name": name,
                **q_meta,
                "original_shape": list(arr.shape),
                "size_original_bytes": orig_bytes,
                "size_stored_bytes": stored_bytes,
                "compression_ratio": orig_bytes / (stored_bytes + 1e-12),
                "reconstruction_error_mse": mse,
                "reconstruction_error_relative": rel,
                "keys": [name],
            }
            transform_records.append(rec)
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "size_stored": stored_bytes,
                "transform": "quantize int8",
                "mse": mse, "rel_err": rel,
            })

        elif sparsify_threshold is None:
            # No transform specified — pass through
            output_tensors[name] = arr
            transform_records.append(_passthrough_record(name, arr))
            report_rows.append({
                "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                "size_original": orig_bytes, "skipped": True, "reason": "no transform specified",
            })

    # Global stats
    total_orig  = sum(r.get("size_original_bytes", r.get("size_original", 0))
                      for r in transform_records)
    total_stored = sum(r.get("size_stored_bytes", r.get("size_original_bytes", 0))
                       for r in transform_records)

    global_stats = {
        "total_tensors": len(artifact.tensors),
        "total_parameters": sum(t.data.size for t in artifact.tensors.values()),
        "original_size_bytes": total_orig,
        "tensor_transformed_size_bytes": total_stored,
        "compressed_size_bytes": total_stored,  # updated after pack
        "compression_ratio_tensor_aware": total_orig / (total_stored + 1e-12),
        "compression_ratio_binary": 1.0,
        "compression_ratio_total": total_orig / (total_stored + 1e-12),
    }

    manifest = build_manifest(artifact, transform_records, global_stats, binary_method=binary_compress)

    if report:
        _print_transform_report(report_rows, out_path if not dry_run else None, dry_run)

    if not dry_run:
        pack(output_tensors, manifest, out_path, compression=binary_compress, level=binary_level)
        final_size = os.path.getsize(out_path)
        if report:
            rprint(f"  Final .stz size: [bold]{fmt_bytes(final_size)}[/bold]")


@app.command()
def compress(
    file: str = typer.Argument(..., help="Input file (.npy, .npz, .stz)"),
    out: Optional[str] = typer.Option(None, "--out", help="Output .stz path"),
    tolerance: float = typer.Option(0.01, "--tolerance", help="Max relative reconstruction error"),
    lossless_only: bool = typer.Option(False, "--lossless-only/--no-lossless-only"),
    no_factorize: bool = typer.Option(False, "--no-factorize/--factorize"),
    no_quantize: bool = typer.Option(False, "--no-quantize/--quantize"),
    wavelet: bool = typer.Option(False, "--wavelet/--no-wavelet", help="Wavelet preconditioning (Phase 15)"),
    wavelet_basis: str = typer.Option("db4", "--wavelet-basis"),
    spatial_dims: int = typer.Option(3, "--spatial-dims"),
    min_size: str = typer.Option("0", "--min-size"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    report: bool = typer.Option(True, "--report/--no-report"),
    report_file: Optional[str] = typer.Option(None, "--report-file"),
    binary_compress: str = typer.Option("zstd", "--binary-compress"),
    binary_level: Optional[int] = typer.Option(None, "--binary-level"),
):
    """Automatically analyze each tensor and route it to the optimal compression strategy."""

    min_size_bytes = parse_size(min_size)
    out_path = out or _default_out(file)

    try:
        artifact = load(file)
    except Exception as e:
        typer.echo(f"Error loading file: {e}", err=True)
        raise typer.Exit(1)

    output_tensors: dict[str, np.ndarray] = {}
    transform_records: list[dict] = []
    report_rows: list[dict] = []

    with console.status("[bold green]Routing tensors…"):
        for name, trec in artifact.tensors.items():
            arr = trec.data
            orig_bytes = arr.nbytes

            decision = route_tensor(
                trec, artifact,
                tolerance=tolerance,
                wavelet=wavelet,
                spatial_dims=spatial_dims,
                min_size_bytes=min_size_bytes,
                no_factorize=no_factorize,
                no_quantize=no_quantize,
                lossless_only=lossless_only,
            )
            strategy = decision["strategy"]
            params   = decision["params"]

            if strategy == "svd":
                rank = params["rank"]
                cached = artifact._svd_cache.get(name)
                svd_arrays, svd_meta = svd_compress(arr, rank, cached_svd=cached)
                for suffix, sub_arr in svd_arrays.items():
                    output_tensors[f"{name}{suffix}"] = sub_arr
                stored_bytes = svd_meta["size_stored_bytes"]
                rec = {
                    "name": name, **svd_meta,
                    "keys": [f"{name}{s}" for s in svd_arrays],
                    "strategy_reason": decision["reason"],
                }
                transform_records.append(rec)
                report_rows.append({
                    "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                    "size_original": orig_bytes, "size_stored": stored_bytes,
                    "transform": f"SVD rank={rank}  ({decision['reason']})",
                    "mse": svd_meta["reconstruction_error_mse"],
                    "rel_err": svd_meta["reconstruction_error_relative"],
                })

            elif strategy == "quantize_fp16":
                q_arr, q_meta = quantize_fp16(arr)
                output_tensors[name] = q_arr
                stored_bytes = q_arr.nbytes
                rec = {
                    "name": name, **q_meta,
                    "original_shape": list(arr.shape),
                    "size_original_bytes": orig_bytes,
                    "size_stored_bytes": stored_bytes,
                    "compression_ratio": orig_bytes / (stored_bytes + 1e-12),
                    "reconstruction_error_mse": 0.0,
                    "reconstruction_error_relative": 0.0,
                    "keys": [name],
                    "strategy_reason": decision["reason"],
                }
                transform_records.append(rec)
                restored = q_arr.astype(arr.dtype)
                mse, rel = _reconstruction_errors(arr, restored)
                report_rows.append({
                    "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                    "size_original": orig_bytes, "size_stored": stored_bytes,
                    "transform": f"fp16  ({decision['reason']})",
                    "mse": mse, "rel_err": rel,
                })

            elif strategy == "quantize_int8":
                q_arr, q_meta = quantize_int8(arr)
                output_tensors[name] = q_arr
                stored_bytes = q_arr.nbytes
                restored = dequantize_int8(q_arr, q_meta["scale"], q_meta["zero_point"], arr.dtype)
                mse, rel = _reconstruction_errors(arr, restored)
                rec = {
                    "name": name, **q_meta,
                    "original_shape": list(arr.shape),
                    "size_original_bytes": orig_bytes,
                    "size_stored_bytes": stored_bytes,
                    "compression_ratio": orig_bytes / (stored_bytes + 1e-12),
                    "reconstruction_error_mse": mse,
                    "reconstruction_error_relative": rel,
                    "keys": [name],
                    "strategy_reason": decision["reason"],
                }
                transform_records.append(rec)
                report_rows.append({
                    "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                    "size_original": orig_bytes, "size_stored": stored_bytes,
                    "transform": f"int8  ({decision['reason']})",
                    "mse": mse, "rel_err": rel,
                })

            elif strategy == "sparse_coo":
                threshold = params.get("threshold", 1e-6)
                coo_arrays, coo_meta = sparsify(arr, threshold)
                for suffix, sub_arr in coo_arrays.items():
                    output_tensors[f"{name}{suffix}"] = sub_arr
                stored_bytes = sum(a.nbytes for a in coo_arrays.values())
                rec = {
                    "name": name, **coo_meta,
                    "size_original_bytes": orig_bytes,
                    "size_stored_bytes": stored_bytes,
                    "compression_ratio": orig_bytes / (stored_bytes + 1e-12),
                    "reconstruction_error_mse": 0.0,
                    "reconstruction_error_relative": 0.0,
                    "keys": [f"{name}{s}" for s in coo_arrays],
                    "strategy_reason": decision["reason"],
                }
                transform_records.append(rec)
                report_rows.append({
                    "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                    "size_original": orig_bytes, "size_stored": stored_bytes,
                    "transform": f"sparse COO  ({decision['reason']})",
                    "mse": 0.0, "rel_err": 0.0,
                })

            else:  # dense passthrough
                output_tensors[name] = arr
                transform_records.append(_passthrough_record(name, arr))
                report_rows.append({
                    "name": name, "shape": arr.shape, "dtype": str(arr.dtype),
                    "size_original": orig_bytes, "skipped": True,
                    "reason": decision["reason"],
                })

    # Global stats
    total_orig   = sum(r.get("size_original_bytes", r.get("size_original", 0)) for r in transform_records)
    total_stored = sum(r.get("size_stored_bytes", r.get("size_original_bytes", 0)) for r in transform_records)
    global_stats = {
        "total_tensors": len(artifact.tensors),
        "total_parameters": sum(t.data.size for t in artifact.tensors.values()),
        "original_size_bytes": total_orig,
        "tensor_transformed_size_bytes": total_stored,
        "compressed_size_bytes": total_stored,
        "compression_ratio_tensor_aware": total_orig / (total_stored + 1e-12),
        "compression_ratio_binary": 1.0,
        "compression_ratio_total": total_orig / (total_stored + 1e-12),
    }
    manifest = build_manifest(artifact, transform_records, global_stats, binary_method=binary_compress)

    if report:
        _print_transform_report(report_rows, out_path if not dry_run else None, dry_run)

    if report_file:
        ext = os.path.splitext(report_file)[1].lower()
        with open(report_file, "w") as f:
            if ext == ".json":
                json.dump({"rows": report_rows, "global_stats": global_stats}, f, indent=2)
            else:
                for r in report_rows:
                    f.write(f"{r['name']}: {r.get('transform', 'skipped')}\n")

    if not dry_run:
        pack(output_tensors, manifest, out_path, compression=binary_compress, level=binary_level)
        final_size = os.path.getsize(out_path)
        if report:
            rprint(f"  Final .stz size: [bold]{fmt_bytes(final_size)}[/bold]")


@app.command()
def extract(
    file: str = typer.Argument(..., help="Path to .stz archive"),
    out: Optional[str] = typer.Option(None, "--out", help="Output path (default: <input>.npz)"),
    format: str = typer.Option("npz", "--format", help="npz | npy (single tensor only)"),
    tensor: Optional[str] = typer.Option(None, "--tensor", help="Extract a single named tensor"),
    original_dtype: bool = typer.Option(True, "--original-dtype/--no-original-dtype",
                                        help="Cast back to original dtype on extract"),
    report: bool = typer.Option(False, "--report/--no-report", help="Show reconstruction report"),
):
    """Reconstruct tensors from a .stz archive."""

    if format not in ("npz", "npy"):
        typer.echo("--format must be npz or npy", err=True)
        raise typer.Exit(1)

    # Default output path
    out_path = out or (os.path.splitext(file)[0] + ".npz")

    # Load manifest and raw arrays
    try:
        manifest = unpack_manifest(file)
    except Exception as e:
        typer.echo(f"Error reading manifest: {e}", err=True)
        raise typer.Exit(1)

    tensor_meta = manifest["tensors"]

    # Filter to single tensor if requested
    if tensor is not None:
        if tensor not in tensor_meta:
            typer.echo(f"Tensor '{tensor}' not found in archive.", err=True)
            raise typer.Exit(1)
        tensor_meta = {tensor: tensor_meta[tensor]}

    # Load all raw arrays needed for the requested tensors
    needed_keys: list[str] = []
    for meta in tensor_meta.values():
        needed_keys.extend(meta.get("keys", []))

    try:
        raw = unpack_tensors(file, keys=needed_keys)
    except Exception as e:
        typer.echo(f"Error loading tensors: {e}", err=True)
        raise typer.Exit(1)

    # Reconstruct each tensor
    _NOT_YET = {"tucker", "tucker_wavelet", "svd_wavelet"}

    reconstructed: dict[str, np.ndarray] = {}
    report_rows: list[dict] = []

    for name, meta in tensor_meta.items():
        storage = meta["storage_type"]

        if storage in _NOT_YET:
            typer.echo(
                f"Error: storage_type '{storage}' for tensor '{name}' "
                "requires SVD/Tucker (Phase 13). Cannot extract yet.",
                err=True,
            )
            raise typer.Exit(1)

        orig_dtype = np.dtype(meta["original_dtype"]) if original_dtype else None

        if storage == "dense":
            arr = raw[meta["keys"][0]]
            if orig_dtype is not None:
                arr = arr.astype(orig_dtype)

        elif storage == "quantized_fp16":
            arr = raw[meta["keys"][0]]
            if orig_dtype is not None:
                arr = arr.astype(orig_dtype)

        elif storage == "quantized_int8":
            q = raw[meta["keys"][0]]
            arr = dequantize_int8(q, meta["scale"], meta["zero_point"],
                                  orig_dtype if orig_dtype is not None else np.float32)

        elif storage == "svd":
            U  = raw[f"{name}__U"].astype(np.float64)
            S  = raw[f"{name}__S"].astype(np.float64)
            Vt = raw[f"{name}__Vt"].astype(np.float64)
            arr = svd_reconstruct(U, S, Vt)
            if orig_dtype is not None:
                arr = arr.astype(orig_dtype)

        elif storage == "sparse_coo":
            indices = raw[f"{name}__indices"]
            values  = raw[f"{name}__values"]
            shape   = tuple(meta["original_shape"])
            arr = reconstruct_coo(indices, values, shape,
                                  dtype=orig_dtype if orig_dtype is not None else values.dtype)

        else:
            typer.echo(f"Warning: unknown storage_type '{storage}' for '{name}', skipping.", err=True)
            continue

        reconstructed[name] = arr

        if report:
            report_rows.append({
                "name": name,
                "storage": storage,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "lossless": meta.get("lossless", True),
                "mse": meta.get("reconstruction_error_mse"),
                "rel": meta.get("reconstruction_error_relative"),
            })

    # Write output
    if format == "npy":
        if len(reconstructed) != 1:
            typer.echo("--format npy requires exactly one tensor (use --tensor NAME)", err=True)
            raise typer.Exit(1)
        arr = next(iter(reconstructed.values()))
        np.save(out_path if out_path.endswith(".npy") else out_path.replace(".npz", ".npy"), arr)
        final_path = out_path if out_path.endswith(".npy") else out_path.replace(".npz", ".npy")
    else:
        np.savez(out_path, **reconstructed)
        final_path = out_path

    # Report
    if report:
        rprint("\n[bold]Extraction Report[/bold]")
        rprint("─" * 55)
        for r in report_rows:
            shape_str = "x".join(str(d) for d in r["shape"])
            lossless_str = "[green]lossless[/green]" if r["lossless"] else "[yellow]lossy[/yellow]"
            rprint(f"\n[cyan]{r['name']}[/cyan]  {r['dtype']} [{shape_str}]  {lossless_str}")
            rprint(f"  storage: {r['storage']}")
            if not r["lossless"] and r["mse"] is not None:
                rprint(f"  MSE: {r['mse']:.6f}  |  Relative error: {r['rel']*100:.3f}%")

    rprint(f"\nExtracted {len(reconstructed)} tensor(s) → [green]{final_path}[/green]")


@app.command()
def info(
    file: str = typer.Argument(..., help="Path to .stz archive"),
    json_out: bool = typer.Option(False, "--json/--no-json", help="Raw JSON output"),
    tensor: Optional[str] = typer.Option(None, "--tensor", help="Info for one tensor only"),
):
    """Display the manifest of a .stz archive without decompressing tensors."""

    try:
        manifest = unpack_manifest(file)
    except Exception as e:
        typer.echo(f"Error reading archive: {e}", err=True)
        raise typer.Exit(1)

    # ── raw JSON mode ─────────────────────────────────────────────────────────
    if json_out:
        if tensor is not None:
            entry = manifest.get("tensors", {}).get(tensor)
            if entry is None:
                typer.echo(f"Tensor '{tensor}' not found.", err=True)
                raise typer.Exit(1)
            print(json.dumps({tensor: entry}, indent=2))
        else:
            print(json.dumps(manifest, indent=2))
        return

    # ── single tensor info ────────────────────────────────────────────────────
    if tensor is not None:
        entry = manifest.get("tensors", {}).get(tensor)
        if entry is None:
            typer.echo(f"Tensor '{tensor}' not found in archive.", err=True)
            raise typer.Exit(1)
        rprint(f"\n[bold cyan]{tensor}[/bold cyan]")
        for k, v in entry.items():
            rprint(f"  {k:<35} {v}")
        return

    # ── full summary display ──────────────────────────────────────────────────
    gs = manifest.get("global_stats", {})
    tensors_meta = manifest.get("tensors", {})
    created = manifest.get("created_at", "")[:10]  # date only
    source = os.path.basename(manifest.get("source_file", ""))
    version = manifest.get("spectra_version", "?")

    rprint(f"\n[bold]Spectra Archive:[/bold] {os.path.basename(file)}")
    rprint(f"Created: {created}  |  Source: {source}  |  Version: {version}")

    # Storage summary
    orig_bytes     = gs.get("original_size_bytes", 0)
    tensor_bytes   = gs.get("tensor_transformed_size_bytes", orig_bytes)
    compressed     = gs.get("compressed_size_bytes", tensor_bytes)
    ratio_tensor   = gs.get("compression_ratio_tensor_aware", orig_bytes / (tensor_bytes + 1e-12))
    ratio_binary   = gs.get("compression_ratio_binary", tensor_bytes / (compressed + 1e-12))
    ratio_total    = gs.get("compression_ratio_total", orig_bytes / (compressed + 1e-12))
    binary_method  = gs.get("binary_compression_method", "?")
    total_tensors  = gs.get("total_tensors", len(tensors_meta))

    rprint(f"\n[bold]Storage Summary[/bold]")
    rprint(f"  {total_tensors} tensors")
    rprint(f"  Original:                {fmt_bytes(orig_bytes):>10}")
    rprint(f"  After tensor transforms: {fmt_bytes(tensor_bytes):>10}  ({ratio_tensor:.1f}x)")
    rprint(f"  After binary ({binary_method}):  {fmt_bytes(compressed):>10}  ({ratio_binary:.1f}x)")
    rprint(f"  Total ratio:             {ratio_total:.1f}x")

    # Strategy breakdown
    from collections import defaultdict
    by_storage: dict[str, list] = defaultdict(list)
    for name, meta in tensors_meta.items():
        by_storage[meta.get("storage_type", "dense")].append(meta)

    rprint(f"\n[bold]Strategy Breakdown[/bold]")
    for storage, entries in sorted(by_storage.items(), key=lambda x: -len(x[1])):
        total_orig = sum(e.get("size_original_bytes", 0) for e in entries)
        rprint(f"  {storage:<18} {len(entries):>4} tensors  ({fmt_bytes(total_orig)} original)")

    # Lossy / lossless counts + max MSE
    lossy    = [m for m in tensors_meta.values() if not m.get("lossless", True)]
    lossless = [m for m in tensors_meta.values() if m.get("lossless", True)]
    all_mse  = [m.get("reconstruction_error_mse", 0.0) for m in tensors_meta.values()
                if m.get("reconstruction_error_mse") is not None]

    rprint(f"\nLossy tensors: {len(lossy)}  |  Lossless tensors: {len(lossless)}")
    if all_mse:
        rprint(f"Max stored error (MSE): {max(all_mse):.6f}")
