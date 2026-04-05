# gpu-compressor

GPU-accelerated file compression and decompression using NVIDIA [nvCOMP](https://developer.nvidia.com/nvcomp).

## Features

- **GPU-accelerated compression** via nvCOMP batched APIs
- **Gdeflate** (GPU DEFLATE) and **Zstd** (GPU Zstandard) algorithms
- **Custom GPU Zstd kernel** — LZ77 match finding + FSE encoding, RFC 8878 compliant
- **Compression levels** — level 0 (nvCOMP fast), level 1 (custom lazy matching), level 2 (custom optimal)
- **Dual GPU auto-detection** — automatically uses both GPUs when available
- **Streaming pipeline** — parallel read/compress/write threads for maximum throughput
- **Integrity verification** — GPU-accelerated BLAKE3 hashing with automatic verification on decompress
- **Multi-file compression** — async pipeline processing multiple files concurrently
- **Directory compression** — recursive directory traversal with structure preservation
- **Legacy compatibility** — auto-detects and decompresses old NVZS files with legacy BLAKE3 hashing

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

### Compress with custom GPU Zstd kernel

```bash
# Level 1: lazy matching (better ratio on repetitive data)
gpu-compressor compress -i file.bin -o file.bin.nvzs -l 1

# Level 2: optimal matching (highest ratio, slower)
gpu-compressor compress -i file.bin -o file.bin.nvzs -l 2
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

| Algorithm | Level | Extension | Chunk Size | Best For |
|-----------|-------|-----------|------------|----------|
| **Zstd** (default) | 0 | `.nvzs` | 4 MB | General purpose, fast (nvCOMP) |
| **Zstd custom** | 1 | `.nvzs` | 64 KB sub-chunks | Better ratio, lazy matching |
| **Zstd custom** | 2 | `.nvzs` | 64 KB sub-chunks | Best ratio, optimal matching |
| **Gdeflate** | — | `.nvgd` | 64 KB | DEFLATE compatibility |

Zstd compression wraps data in a tar archive containing the original file and a BLAKE3 hash for integrity verification on decompress. Custom Zstd levels (1-2) use a bespoke CUDA kernel implementing LZ77 match finding with shared-memory hash tables and RFC 8878-compliant FSE encoding. Achieves ~5.3x compression on repetitive data with safe 1x fallback on random data.

## File Formats

| Magic | Format | Description |
|-------|--------|-------------|
| `NVZS` | Zstd compressed | Header + chunk size table + compressed chunks |
| `NVGD` | Gdeflate compressed | Header + chunk size table + compressed chunks |
| `NVMC` | Multi-chunk | Wrapper for Gdeflate files >2 GB |

## Architecture

```
src/
  main.rs                CLI entry point and command dispatch
  cli.rs                 Clap argument definitions
  constants.rs           Chunk sizes, buffer sizes, limits, embedded PTX
  cuda.rs                GPU detection and CUDA utilities
  blake3.rs              GPU-accelerated BLAKE3 hashing (spec + legacy modes)
  format.rs              Pipeline message types
  compress_gdeflate.rs   Gdeflate batched compression via nvCOMP
  compress_zstd.rs       Zstd streaming compression via nvCOMP (level 0)
  compress_zstd_custom.rs  Custom GPU Zstd kernel orchestrator (levels 1-2)
  pipeline.rs            Single GPU streaming pipeline
  pipeline_dual.rs       Dual GPU streaming pipeline
  dispatch.rs            Compression routing, directory handling, level dispatch
  multi.rs               Multi-file async compression
  decompress.rs          All decompression paths with format auto-detection
  nvcomp_bindings.rs     Auto-generated nvCOMP FFI bindings

blake3.cu              CUDA BLAKE3 hashing kernel (1KB/thread, tree reduction)
zstd_compress.cu       Custom CUDA Zstd kernel (LZ77 + FSE, RFC 8878)
```

Two custom CUDA kernels are compiled to PTX at build time and embedded into the binary:
- **blake3.cu** — each GPU thread processes one 1 KB BLAKE3 chunk; tree reduction combines chunk hashes
- **zstd_compress.cu** — LZ77 match finding with 14-bit shared-memory hash tables, lazy/optimal matching, and FSE encoding per RFC 8878

See [DESIGN.md](DESIGN.md) for detailed architecture documentation.

## License

[MIT](LICENSE)
