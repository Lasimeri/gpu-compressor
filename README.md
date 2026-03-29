# gpu-compressor

GPU-accelerated file compression and decompression using NVIDIA [nvCOMP](https://developer.nvidia.com/nvcomp).

## Features

- **GPU-accelerated compression** via nvCOMP batched APIs
- **Gdeflate** (GPU DEFLATE) and **Zstd** (GPU Zstandard) algorithms
- **Dual GPU auto-detection** — automatically uses both GPUs when available
- **Streaming pipeline** — parallel read/compress/write threads for maximum throughput
- **Integrity verification** — GPU-accelerated BLAKE3 hashing with automatic verification on decompress
- **Multi-file compression** — async pipeline processing multiple files concurrently
- **Directory compression** — recursive directory traversal with structure preservation

## Requirements

- NVIDIA GPU (Compute Capability 8.0+, tested on RTX 3090 / 3090 Ti)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 12.x
- [nvCOMP](https://developer.nvidia.com/nvcomp) 3.x
- Rust toolchain (stable)
- clang (for bindgen)

### Install nvCOMP

```bash
# Arch Linux (if AUR package available)
yay -S nvcomp

# Manual installation
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-3.0.6-archive.tar.xz
tar -xf nvcomp-linux-x86_64-3.0.6-archive.tar.xz
sudo cp -r nvcomp-linux-x86_64-3.0.6-archive/include/* /usr/local/include/
sudo cp -r nvcomp-linux-x86_64-3.0.6-archive/lib/* /usr/local/lib/
sudo ldconfig
```

## Build

```bash
cargo build --release
```

## Usage

### Compress a file (Zstd, default)

```bash
gpu-compressor compress -i file.bin -o file.bin.nvzs
```

### Compress with Gdeflate

```bash
gpu-compressor compress -i file.bin -o file.bin.nvgd -a gdeflate
```

### Decompress

```bash
gpu-compressor decompress -i file.bin.nvzs -o file.bin
```

### Compress a directory

```bash
gpu-compressor compress -i /path/to/dir -o /path/to/output
```

### Multi-file compression

```bash
gpu-compressor compress-multi -i file1.bin file2.bin -o file1.nvzs file2.nvzs
```

### GPU BLAKE3 hash

```bash
gpu-compressor hash -i file.bin
```

### Select GPU device

All commands accept `-d <device_id>` (default: 0). Zstd compression auto-detects and uses dual GPUs when available.

## Algorithms

| Algorithm | Extension | Chunk Size | Best For |
|-----------|-----------|------------|----------|
| **Zstd** (default) | `.nvzs` | 4 MB | General purpose, high ratio |
| **Gdeflate** | `.nvgd` | 64 KB | DEFLATE compatibility |

Zstd compression wraps data in a tar archive containing the original file and a BLAKE3 hash for integrity verification on decompress.

## File Formats

| Magic | Format | Description |
|-------|--------|-------------|
| `NVZS` | Zstd compressed | Header + chunk size table + compressed chunks |
| `NVGD` | Gdeflate compressed | Header + chunk size table + compressed chunks |
| `NVMC` | Multi-chunk | Wrapper for Gdeflate files >2 GB |

## Architecture

```
src/
  main.rs              CLI entry point and command dispatch
  cli.rs               Clap argument definitions
  constants.rs         Chunk sizes, buffer sizes, limits
  cuda.rs              GPU detection and CUDA utilities
  blake3.rs            GPU-accelerated BLAKE3 hashing
  format.rs            Pipeline message types
  compress_gdeflate.rs Gdeflate batched compression
  compress_zstd.rs     Zstd streaming compression
  pipeline.rs          Single GPU streaming pipeline
  pipeline_dual.rs     Dual GPU streaming pipeline
  dispatch.rs          Compression routing and directory handling
  multi.rs             Multi-file async compression
  decompress.rs        All decompression paths
  nvcomp_bindings.rs   Auto-generated nvCOMP FFI bindings
```

The BLAKE3 implementation is a custom CUDA kernel (`blake3.cu`) compiled to PTX at build time. Each GPU thread processes one 1 KB BLAKE3 chunk for maximum parallelism.

## License

[MIT](LICENSE)
