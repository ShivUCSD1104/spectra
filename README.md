# Spectra

A command-line tool for analyzing, transforming, and compressing tensor
artifacts (.npy, .npz, .pt). Profiles every tensor individually — measuring
sparsity, entropy, and spectral structure — then routes each to the optimal
compression strategy (quantization, SVD, Tucker decomposition) with a full
audit trail in the output.

## Usage

# Inspect every tensor in an archive

spectra inspect model.npz

# Automatically compress with tolerance 1%

spectra compress model.npz --tolerance 0.01

# Explicit fp16 quantization

spectra transform model.npz --quantize fp16 --out model.stz

# Reconstruct from a .stz archive

spectra extract model.stz --out recovered.npz

# Read compression stats without decompressing tensor data

spectra info model.stz
