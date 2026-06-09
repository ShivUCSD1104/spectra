from __future__ import annotations

from typing import Iterable

from rich.progress import track as rich_track
from rich.table import Table
from rich import print as rprint

from spectra.utils.units import fmt_bytes


def make_tensor_table(records: list[dict]) -> Table:
    """Build the Rich table for inspect output.

    Each record dict must have keys:
        name, shape, dtype, size_bytes, sparsity, entropy, recommendation
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Tensor", style="cyan", no_wrap=False)
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Size", justify="right")
    table.add_column("Sparsity", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("Recommendation")

    for r in records:
        shape_str = "x".join(str(d) for d in r["shape"])
        if len(shape_str) > 10:
            shape_str = shape_str[:9] + ".."
        sparsity = f'{r["sparsity"] * 100:.1f}%' if r.get("sparsity") is not None else "—"
        entropy = f'{r["entropy"]:.1f} bit' if r.get("entropy") is not None else "—"
        table.add_row(
            r["name"],
            shape_str,
            str(r["dtype"]),
            fmt_bytes(r["size_bytes"]),
            sparsity,
            entropy,
            r.get("recommendation", ""),
        )

    return table


def print_summary_block(stats: dict) -> None:
    """Print the Compression Potential Summary footer.

    Expected keys in stats:
        svd_tucker_count, svd_tucker_bytes,
        quantize_count, quantize_bytes,
        leave_alone_count, leave_alone_bytes,
        estimated_compressed_bytes (optional), tolerance (optional)
    """
    rprint("\n[bold]Compression Potential Summary[/bold]")
    rprint(
        f"  SVD/Tucker candidates:  {stats['svd_tucker_count']:>5} tensors"
        f"  ({fmt_bytes(stats['svd_tucker_bytes'])})"
    )
    rprint(
        f"  Quantization only:      {stats['quantize_count']:>5} tensors"
        f"  ({fmt_bytes(stats['quantize_bytes'])})"
    )
    rprint(
        f"  Leave alone:            {stats['leave_alone_count']:>5} tensors"
        f"  ({fmt_bytes(stats['leave_alone_bytes'])})"
    )
    if stats.get("estimated_compressed_bytes") is not None:
        tol = stats.get("tolerance", 0.01)
        est = fmt_bytes(stats["estimated_compressed_bytes"])
        total = stats["svd_tucker_bytes"] + stats["quantize_bytes"] + stats["leave_alone_bytes"]
        ratio = total / stats["estimated_compressed_bytes"] if stats["estimated_compressed_bytes"] else 0
        rprint(f"\n  Estimated compressed size at tolerance {tol}:  ~{est}  ({ratio:.1f}x)")


def progress_track(iterable: Iterable, description: str = "Processing") -> Iterable:
    """Thin wrapper around rich.progress.track."""
    return rich_track(iterable, description=description)
