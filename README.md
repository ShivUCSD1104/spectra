# Spectra

A command-line tool for analyzing, transforming, and compressing tensor artifacts (`.npy`, `.npz`). Spectra profiles every tensor individually — measuring sparsity, entropy, and spectral structure — then routes each one to the optimal compression strategy (quantization, sparse COO storage, or truncated SVD decomposition). Every transform is audited in a per-tensor report and a machine-readable manifest stored alongside the data.

Designed for data scientists who need to shrink model weights, activation checkpoints, or any numerical array collection without black-box compression that hides what was done or how much was lost.

---

## Installation

Requires Python 3.12+, managed with [uv](https://github.com/astral-sh/uv).

```bash
uv add "git+https://github.com/ShivUCSD1104/spectra.git"
```

Or clone and install in editable mode:

```bash
git clone https://github.com/ShivUCSD1104/spectra.git
cd spectra
uv sync
uv run spectra --help
```

Optional extras:

```bash
uv add "spectra[torch]"    # enable .pt / .pth loading (requires PyTorch)
uv add "spectra[wavelet]"  # enable wavelet preconditioning (requires PyWavelets)
```

---

## Commands

```
spectra inspect   <file>          Profile every tensor. Read-only.
spectra compress  <file>          Auto-route each tensor to optimal compression.
spectra transform <file>          Apply an explicit transform strategy.
spectra extract   <file.stz>      Reconstruct tensors from a .stz archive.
spectra info      <file.stz>      Show archive manifest without decompressing.
```

---

## `spectra inspect`

Profile every tensor in an artifact without writing anything. Reports shape, dtype, size, sparsity, Shannon entropy, and — for 2D matrices — a full spectral analysis including singular value decay rate, effective rank, intrinsic dimension, and condition number.

```
spectra inspect <file> [OPTIONS]
```

**Inputs:** `.npy`, `.npz`, `.stz`

**Outputs:** Terminal table, CSV, or JSON to stdout. Nothing written to disk.

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tensor NAME` | str | all | Inspect only the named tensor |
| `--sort FIELD` | str | `size` | Sort order: `size`, `entropy`, `rank`, `sparsity`, `name` |
| `--top N` | int | all | Show only the top N tensors after sorting |
| `--depth LEVEL` | str | `summary` | `summary` (table only) or `full` (table + per-tensor detail block) |
| `--format FORMAT` | str | `table` | `table` (rich), `csv`, or `json` |

### Metrics computed per tensor

| Metric | Description |
|--------|-------------|
| **shape / dtype** | Array dimensions and storage type |
| **params** | Total element count |
| **size** | Memory footprint in bytes |
| **sparsity (exact)** | Fraction of values exactly equal to zero |
| **sparsity (near-zero)** | Fraction of values with \|x\| < 1e-6 |
| **entropy** | Shannon entropy in bits over a 256-bin histogram of values |
| **decay rate** | *(2D only)* Rate of exponential falloff of singular values, in [0, 1] |
| **effective rank** | *(2D only)* Participation ratio: (ΣS)² / Σ(S²) |
| **intrinsic dim** | *(2D only)* Count of singular values above 1% of the largest |
| **condition number** | *(2D only)* S[0] / S[-1] — sensitivity to numerical noise |
| **isotropic / deviatoric norm** | *(square 2D only)* Decomposition into scalar + traceless parts |
| **mode-wise ranks** | *(3D+ only)* Estimated Tucker rank per mode at 1% tolerance |
| **recommendation** | Human-readable summary of what transforms are applicable |

### SVD analysis detail

For 2D tensors, Spectra computes the top-64 singular triplets via `scipy.sparse.linalg.svds` (a partial SVD — much faster than full SVD for large matrices). The **spectral decay rate** is derived by fitting a line to log(S/S[0]) as a function of index, then normalizing: `rate = 1 - exp(-slope)`. A rate near 1.0 means singular values drop sharply (strong low-rank structure). A rate near 0.0 means the spectrum is flat (dense, information-rich).

### Examples

```bash
# Basic table
spectra inspect model.npz

# Sort by entropy, show only top 5 tensors
spectra inspect model.npz --sort entropy --top 5

# Inspect one tensor with full detail block
spectra inspect model.npz --tensor attention.weight --depth full

# Export as JSON for scripting
spectra inspect model.npz --format json > analysis.json

# Export as CSV
spectra inspect model.npz --format csv > analysis.csv

# Inspect tensors previously compressed into a .stz
spectra inspect model.stz
```

---

## `spectra compress`

The intelligent command. Automatically analyzes every tensor and routes it to the best compression strategy using the built-in routing engine. Produces a `.stz` archive with a full audit manifest.

```
spectra compress <file> [OPTIONS]
```

**Inputs:** `.npy`, `.npz`, `.stz`

**Output:** `.stz` archive (default: `<input>.stz`)

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out PATH` | path | `<input>.stz` | Output archive path |
| `--tolerance FLOAT` | float | `0.01` | Max acceptable relative reconstruction error per tensor (1% = 0.01) |
| `--lossless-only` | bool | false | Only apply lossless transforms (fp16 unless values exceed ±65504) |
| `--no-factorize` | bool | false | Disable SVD routing; quantize-only mode |
| `--no-quantize` | bool | false | Disable quantization routing; factorize-only mode |
| `--min-size SIZE` | str | `0` | Skip tensors smaller than this size (e.g. `1MB`, `512KB`) |
| `--dry-run` | bool | false | Print routing decisions and report; write nothing to disk |
| `--report / --no-report` | bool | true | Print per-tensor transform report to terminal |
| `--report-file PATH` | path | none | Write report to a `.json` or `.txt` file |
| `--binary-compress METHOD` | str | `zstd` | Binary compression applied after tensor transforms: `zstd`, `gzip`, `xz`, `zlib`, `none` |
| `--binary-level N` | int | method default | Compression level (see table below) |

### Routing engine

The router analyzes each tensor and selects a strategy based on its structure. The decision tree is:

```
if tensor.nbytes < min_size:
    → dense passthrough (no change)

if tensor.ndim == 1:
    → quantize_fp16
      Rationale: 1D tensors are bias vectors, position encodings, etc.
      fp16 is always lossless for values within ±65504 and halves the size.

if tensor.ndim == 2:
    Compute top-64 singular values (cached from inspect if available).
    Compute spectral decay rate and find the smallest SVD rank k where
    relative error < tolerance (using Eckart-Young theorem).

    SVD candidate if:
        decay_rate > 0.85        (fast exponential decay of singular values)
        OR rank_k / min(shape) < 0.3 AND compression_ratio(k) > 2.0
                                 (sharp rank cliff: few singular values explain
                                  most energy, even if they are similar in magnitude)
    if SVD candidate AND compression_ratio > 2.0:
        → SVD rank k
          Stored as float32 U, S, Vt factors.
          Reconstruction: U @ diag(S) @ Vt

    elif entropy < 5.0 AND int8_safe:
        → quantize_int8
          int8_safe: entropy < 5.0 bits AND dynamic_range < 20.0
          Rationale: low-entropy tensors have concentrated value distributions
          that map well onto 256 discrete levels.

    elif near_zero_fraction > 0.50:
        → sparse_coo
          Rationale: more than half of values are near-zero (|x| < 1e-6);
          COO format stores only non-zero indices + values.

    else:
        → quantize_fp16
          Conservative fallback for dense, high-entropy 2D tensors.

if tensor.ndim >= 3:
    → quantize_fp16
      Rationale: Tucker decomposition (Phase 14) not yet implemented.
      fp16 is always a safe 2x reduction.
```

**Why these thresholds?**

- `decay_rate > 0.85` — An exponential fit slope of 0.85 on the normalized singular value curve corresponds to roughly 85% energy loss per step. At this rate the matrix is compressible with small rank. Values below 0.85 indicate that many singular values are significant and truncation would be lossy.
- `rank_fraction < 0.3` — If the tolerance-satisfying rank is less than 30% of the smaller dimension, SVD will almost always exceed 2x compression. This catches matrices with a sharp spectral cliff (e.g. a true rank-10 matrix in a 256×256 space) that the exponential decay metric misses because the retained singular values are not themselves fast-decaying.
- `compression_ratio > 2.0` — The break-even for SVD storage (m×k + k + k×n float32) vs. the original (m×n float32). Below 2× it is not worth the reconstruction overhead.
- `entropy < 5.0 bits` — Uniform float32 noise has entropy ≈ 8 bits. Values below 5 bits indicate a distribution that is clustered enough to be accurately represented with only 256 levels.
- `dynamic_range < 20.0` — Defined as max(|x|) / mean(|x|). A ratio above 20 means outliers would dominate the int8 quantization scale, causing large errors on the common values.
- `near_zero_fraction > 0.50` — COO storage costs `nnz × (ndim × 8 + 4)` bytes. Break-even vs. dense is at ~50% sparsity for a typical 2D float32 tensor, assuming float32 values and int64 indices.

### Examples

```bash
# Auto-compress at 1% tolerance (default)
spectra compress model.npz

# 5% tolerance — more aggressive, smaller files
spectra compress model.npz --tolerance 0.05 --out model_compressed.stz

# Only quantize, no SVD
spectra compress model.npz --no-factorize

# Preview routing decisions without writing
spectra compress model.npz --dry-run

# Ignore tensors smaller than 1 MB
spectra compress model.npz --min-size 1MB

# Save report as JSON
spectra compress model.npz --report-file report.json

# Use xz binary compression for maximum space savings (slow)
spectra compress model.npz --binary-compress xz --binary-level 9

# Lossless only (fp16 — safe for values within ±65504)
spectra compress model.npz --lossless-only
```

---

## `spectra transform`

Apply an explicit, user-specified transform to all (or selected) tensors. Unlike `compress`, you choose the strategy; Spectra applies it uniformly.

```
spectra transform <file> [OPTIONS]
```

**Inputs:** `.npy`, `.npz`, `.stz`

**Output:** `.stz` archive (default: `<input>.stz`)

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out PATH` | path | `<input>.stz` | Output archive path |
| `--quantize MODE` | str | none | `fp16` or `int8` quantization |
| `--factorize svd` | str | none | SVD factorization (requires `--rank`) |
| `--rank N` | int | none | Fixed SVD rank (required with `--factorize svd`) |
| `--sparsify THRESHOLD` | float | none | Zero out values with \|x\| < threshold before storing as COO |
| `--select GLOB` | str | none | Glob pattern to select tensors (e.g. `attention.*`) |
| `--exclude GLOB` | str | none | Glob pattern to exclude tensors (e.g. `*.bias`) |
| `--min-size SIZE` | str | `0` | Skip tensors below this size |
| `--skip-1d / --no-skip-1d` | bool | true | Skip 1D tensors when `--factorize` is set |
| `--dry-run` | bool | false | Show plan; write nothing |
| `--report / --no-report` | bool | true | Print per-tensor report |
| `--binary-compress METHOD` | str | `zstd` | Binary compression: `zstd`, `gzip`, `xz`, `zlib`, `none` |
| `--binary-level N` | int | method default | Compression level |

### Transform modes

**`--quantize fp16`**
Casts every value to float16. Float32 → float16 halves the byte count. Lossless for values within ±65504; values outside this range are clipped to ±inf. The manifest records `fp16_overflow_detected: true` if any value exceeds the representable range.

**`--quantize int8`**
Per-tensor affine quantization. Computes `scale = (max - min) / 255` and `zero_point` such that the minimum value maps to -128 and the maximum to +127. Stores as int8 (4× compression from float32). Reconstruction formula stored in manifest: `x ≈ q * scale + zero_point`. Always lossy; error depends on value distribution.

**`--factorize svd --rank N`**
Truncated SVD at a fixed rank N, applied to all 2D tensors. Stores three float32 arrays per tensor: `U` (m×k), `S` (k,), `Vt` (k×n). Reconstruction: `U @ diag(S) @ Vt`. Non-2D tensors are passed through unmodified. Compression ratio: `(m×n) / (m×k + k + k×n)`.

**`--sparsify THRESHOLD`**
Zeros out all values with |x| < threshold, then encodes as COO (coordinate list): int64 indices of shape (nnz, ndim) and float32 values of shape (nnz,). Lossless when threshold = 0. Can be combined with `--quantize` to first sparsify, then quantize the remaining values.

### Selection

`--select` and `--exclude` use Python's `fnmatch` shell-style glob patterns:

```bash
# Only transform attention weight matrices
spectra transform model.npz --quantize fp16 --select "attention*"

# Transform everything except embedding layers
spectra transform model.npz --quantize fp16 --exclude "embed*"

# Skip tensors smaller than 100 KB
spectra transform model.npz --quantize int8 --min-size 100KB
```

### Examples

```bash
# fp16 quantize everything
spectra transform model.npz --quantize fp16

# int8 quantize all weight matrices (not biases)
spectra transform model.npz --quantize int8 --exclude "*.bias"

# SVD at rank 32 for all 2D tensors
spectra transform model.npz --factorize svd --rank 32

# Sparsify: zero out values smaller than 1e-4
spectra transform model.npz --sparsify 1e-4

# Sparsify + quantize the non-zero values
spectra transform model.npz --sparsify 1e-4 --quantize fp16

# Dry run to preview what would happen
spectra transform model.npz --quantize int8 --dry-run

# Chain on an existing .stz file
spectra transform previous.stz --quantize fp16
```

---

## `spectra extract`

Reconstruct tensors from a `.stz` archive, reversing all stored transforms. Handles all storage types: `dense`, `quantized_fp16`, `quantized_int8`, `svd`, `sparse_coo`.

```
spectra extract <file.stz> [OPTIONS]
```

**Inputs:** `.stz`

**Output:** `.npz` (default) or `.npy` (single tensor)

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out PATH` | path | `<input>.npz` | Output file path |
| `--format FORMAT` | str | `npz` | `npz` (all tensors) or `npy` (single tensor only) |
| `--tensor NAME` | str | all | Extract only the named tensor |
| `--original-dtype / --no-original-dtype` | bool | true | Cast back to the dtype recorded at compress time |
| `--report` | bool | false | Print per-tensor reconstruction report |

### Reconstruction by storage type

| Storage type | Reconstruction method |
|---|---|
| `dense` | Direct load; cast to original dtype |
| `quantized_fp16` | Cast float16 → original dtype |
| `quantized_int8` | `q * scale + zero_point`, cast to original dtype |
| `svd` | `U.astype(float64) @ diag(S) @ Vt`, cast to original dtype |
| `sparse_coo` | Place COO values at COO indices into a zero-filled dense array |

**Note:** `--no-original-dtype` leaves tensors in their stored dtype (e.g. float16 or int8) rather than casting back to float32. Useful for memory-constrained environments.

### Examples

```bash
# Reconstruct all tensors to recovered.npz
spectra extract model.stz --out recovered.npz

# Extract one tensor to a .npy file
spectra extract model.stz --tensor attention.weight --format npy

# Extract without restoring original dtype (keep as float16)
spectra extract model.stz --no-original-dtype

# Show reconstruction report (storage type and error per tensor)
spectra extract model.stz --report
```

---

## `spectra info`

Display the manifest of a `.stz` archive. Reads only `manifest.json` from the zip; never decompresses `tensors.npz`. Sub-second even for large archives.

```
spectra info <file.stz> [OPTIONS]
```

**Inputs:** `.stz`

**Outputs:** Terminal summary or raw JSON to stdout.

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tensor NAME` | str | none | Show manifest entry for one tensor only |
| `--json` | bool | false | Print raw JSON manifest (or single tensor entry) |

### Output sections (default mode)

- **Archive header** — filename, creation date, source file, Spectra version
- **Storage summary** — original size, size after tensor transforms (e.g. quantization/SVD), size after binary compression, each ratio
- **Strategy breakdown** — how many tensors used each storage type, and total original bytes per group
- **Quality summary** — count of lossy vs. lossless tensors; maximum stored MSE across all tensors

### Examples

```bash
# Full manifest summary
spectra info model.stz

# Info for one tensor
spectra info model.stz --tensor attention.weight

# Raw manifest JSON (pipeable to jq)
spectra info model.stz --json | jq '.tensors | keys'

# Just the global stats
spectra info model.stz --json | jq '.global_stats'
```

---

## Binary compression options

After tensor-level transforms, Spectra applies a second binary compression pass over the packed `tensors.npz`. The binary layer can be tuned independently of the tensor strategy.

| Method | Level range | Default level | Characteristics |
|--------|-------------|---------------|-----------------|
| `zstd` | 1–22 | 3 | Best speed/ratio tradeoff; default |
| `gzip` | 1–9 | 6 | Universal compatibility |
| `xz` | 0–9 | 6 | Highest compression ratio; slowest |
| `zlib` | 1–9 | 6 | Built into Python's zipfile module |
| `none` | — | — | No binary compression; fastest extraction |

Out-of-range levels are clamped with a warning rather than erroring. The method used is recorded in `manifest.json → global_stats.binary_compression_method` so extraction is always automatic.

---

## The `.stz` format

A `.stz` (Spectral Tensor Zip) file is a standard ZIP archive containing exactly two entries:

```
archive.stz
├── tensors.npz      ← all transformed tensor arrays, optionally binary-compressed
└── manifest.json    ← metadata, routing decisions, error metrics
```

**`manifest.json` structure:**

```json
{
  "spectra_version": "0.1.0",
  "created_at": "2026-06-09T...",
  "source_file": "model.npz",
  "source_format": "npz",
  "global_stats": {
    "total_tensors": 4,
    "total_parameters": 85000,
    "original_size_bytes": 340000,
    "tensor_transformed_size_bytes": 42000,
    "compressed_size_bytes": 38000,
    "compression_ratio_tensor_aware": 8.1,
    "compression_ratio_binary": 1.1,
    "compression_ratio_total": 8.9,
    "binary_compression_method": "zstd"
  },
  "tensors": {
    "attention.weight": {
      "storage_type": "svd",
      "original_shape": [256, 256],
      "original_dtype": "float32",
      "lossless": false,
      "reconstruction_method": "U @ diag(S) @ Vt",
      "rank_used": 10,
      "rank_full": 256,
      "spectrum_decay_rate": 0.21,
      "reconstruction_error_mse": 0.0,
      "reconstruction_error_relative": 0.0,
      "keys": ["attention.weight__U", "attention.weight__S", "attention.weight__Vt"],
      "strategy_reason": "low intrinsic rank (10/256), SVD rank=10"
    },
    "output.bias": {
      "storage_type": "quantized_fp16",
      "original_dtype": "float32",
      "fp16_overflow_detected": false,
      "lossless": true,
      "reconstruction_method": "cast_to_original_dtype",
      "keys": ["output.bias"]
    }
  }
}
```

**Array key naming conventions inside `tensors.npz`:**

| Storage type | Keys stored |
|---|---|
| `dense` | `<name>` |
| `quantized_fp16` | `<name>` |
| `quantized_int8` | `<name>` |
| `svd` | `<name>__U`, `<name>__S`, `<name>__Vt` |
| `sparse_coo` | `<name>__indices`, `<name>__values` |

`.stz` files are readable by any ZIP tool (e.g. `unzip -l model.stz`) and the manifest is always plain JSON — no custom binary headers or proprietary structures.

---

## Analysis module reference

| Module | Functions |
|--------|-----------|
| `analysis.sparsity` | `sparsity_fraction(arr)` → `{exact_zero, near_zero_1e6}` |
| `analysis.entropy` | `shannon_entropy(arr, bins=256)` → float (bits) |
| `analysis.spectrum` | `randomized_svd_top_k(arr, k=64)`, `spectral_decay_rate(S)`, `effective_rank(S)`, `condition_number(S)` |
| `analysis.geometry` | `intrinsic_dim_estimate(S)`, `participation_ratio(S)` |
| `analysis.decomposition` | `isotropic_deviatoric_split(arr)` → `{isotropic_norm, deviatoric_norm, ...}` |
| `transforms.quantize` | `quantize_fp16`, `quantize_int8`, `dequantize_int8`, `int8_safe` |
| `transforms.sparsify` | `sparsify(arr, threshold)`, `reconstruct_coo(indices, values, shape)` |
| `transforms.factorize` | `svd_compress(arr, rank, cached_svd)`, `svd_reconstruct(U, S, Vt)`, `find_rank_for_tolerance(S, arr, tol)` |
| `core.router` | `route_tensor(record, artifact, tolerance, ...)` |
| `formats.stz` | `pack(tensors, manifest, path, ...)`, `unpack_manifest(path)`, `unpack_tensors(path, keys)` |

---

## Size notation

The `--min-size` flag accepts human-readable sizes:

| Input | Meaning |
|---|---|
| `1MB` | 1,000,000 bytes |
| `1MiB` | 1,048,576 bytes |
| `512KB` | 512,000 bytes |
| `0` | 0 bytes (no threshold) |
