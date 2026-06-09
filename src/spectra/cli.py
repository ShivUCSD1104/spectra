from __future__ import annotations

import csv
import json
import sys
from typing import Optional

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
from spectra.loaders.numpy_loader import load
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


# ── stub commands (implemented in later phases) ───────────────────────────────

@app.command()
def transform(file: str = typer.Argument(...)):
    """Apply explicit transforms to tensors. (Phase 9)"""
    typer.echo("transform: not yet implemented", err=True)
    raise typer.Exit(1)


@app.command()
def compress(file: str = typer.Argument(...)):
    """Auto-route each tensor to optimal compression. (Phase 13)"""
    typer.echo("compress: not yet implemented", err=True)
    raise typer.Exit(1)


@app.command()
def extract(file: str = typer.Argument(...)):
    """Reconstruct tensors from a .stz archive. (Phase 10)"""
    typer.echo("extract: not yet implemented", err=True)
    raise typer.Exit(1)


@app.command()
def info(file: str = typer.Argument(...)):
    """Display manifest of a .stz without decompressing. (Phase 11)"""
    typer.echo("info: not yet implemented", err=True)
    raise typer.Exit(1)
