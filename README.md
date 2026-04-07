# gpu-compressor

GPU-accelerated file compression and decompression using NVIDIA [nvCOMP](https://developer.nvidia.com/nvcomp) and custom CUDA kernels.

## Features

- **Zstd compression** — L0 (nvCOMP batched API, dual GPU), L1 (GPU lazy match + CPU FSE), L2 (GPU optimal match + CPU FSE)
- **LZMA2 compression** — L0 (multi-threaded liblzma, 8 workers, configurable preset 0-9), L1+ (async GPU match finding + N CPU encoder threads)
- **Dual GPU auto-detection** — even/odd chunk interleaving for Zstd L0; LZMA2 L1+ distributes sub-block jobs across both GPUs via MPMC channel
- **Streaming pipeline** — reader → GPU match finder → sub-block job queue → CPU encoder pool → collector → writer
- **Dictionary size control** — `--dict-size` sets LZMA2 pipeline chunk size (dictionary window); stored in host RAM, not VRAM
- **VRAM efficient** — ~552 MB per pipeline chunk; fits 4 GB GPUs
- **Custom CUDA kernels** — `zstd_match_find` (LZ77 with shared-memory hash table), `lzma2_match_find` (HC4 hash chain), compiled to PTX at build time and embedded in the binary

## Requirements

- NVIDIA GPU with Compute Capability 8.0+ (tested on RTX 3090 / 3090 Ti)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 12+, including `nvcc`
- [nvCOMP](https://developer.nvidia.com/nvcomp) 3.x headers and libraries
- Rust stable toolchain
- clang (required by bindgen for FFI generation)
- libzstd (for CPU FSE encoding in Zstd L1/L2)
- liblzma (for LZMA2 L0 and decompression, loaded via dlopen at runtime)

### Install nvCOMP

```bash
# Arch Linux (AUR)
yay -S nvcomp

# Manual
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

Build steps performed automatically:
1. bindgen generates `nvcomp_bindings.rs` from nvCOMP C headers
2. nvcc compiles `zstd_compress.cu` → `zstd_compress.ptx` (sm_86, -O3)
3. nvcc compiles `lzma2_match_find.cu` → `lzma2_match_find.ptx` (sm_86, -O3)
4. PTX files are embedded into the binary via `include_str!`
5. Cargo links `libnvcomp`, `libcudart`, `libzstd`

Binary location: `target/release/gpu-compressor`

## Usage

```
gpu-compressor compress -i <INPUT>... [-o OUTPUT] [-a zstd|lzma2] [-d DEVICE] [-l LEVEL] [--chunk-size BYTES] [--dict-size MB]
gpu-compressor decompress -i <INPUT>... [-o OUTPUT] [-d DEVICE]
```

### Compress (Zstd, default)

```bash
gpu-compressor compress -i file.bin
# Output: file.bin.nvzs
```

### Compress with LZMA2

```bash
gpu-compressor compress -i file.bin -a lzma2
# Output: file.bin.nvlz

# With custom dictionary size (default 128 MB)
gpu-compressor compress -i file.bin -a lzma2 --dict-size 256
```

### Compress with levels

```bash
gpu-compressor compress -i file.bin -l 1      # Zstd L1: GPU lazy match + CPU FSE
gpu-compressor compress -i file.bin -l 2      # Zstd L2: GPU optimal match + CPU FSE
gpu-compressor compress -i file.bin -a lzma2 -l 1   # LZMA2 L1: async GPU+CPU pipeline
```

### Decompress

```bash
gpu-compressor decompress -i file.bin.nvzs    # → file.bin
gpu-compressor decompress -i file.bin.nvlz    # → file.bin
```

### Compress multiple files

```bash
gpu-compressor compress -i file1.bin file2.bin file3.bin
```

## Algorithms

| Algorithm | Level | Extension | Pipeline |
|-----------|-------|-----------|----------|
| Zstd | 0 | `.nvzs` | nvCOMP batched API; dual GPU auto |
| Zstd | 1 | `.nvzs` | GPU LZ77 depth 16 → CPU ZSTD_compressSequences |
| Zstd | 2 | `.nvzs` | GPU LZ77 depth 64 → CPU ZSTD_compressSequences |
| LZMA2 | 0 | `.nvlz` | liblzma raw encoder, 8 workers, preset 0-9 |
| LZMA2 | 1+ | `.nvlz` | GPU HC4 match finding (1-2 GPUs) → N CPU encoder threads (liblzma preset 9) |

### LZMA2 L1+ Pipeline

```
Reader thread
  → [bounded channel, 4 slots]
  → GPU match finding thread(s) [1 per GPU, round-robin via MPMC]
  → [bounded sub-block job queue, N*4 slots]
  → N CPU encoder threads [each grabs 64KB sub-blocks independently]
  → Collector thread [reassembles sub-blocks into chunks by chunk_index]
  → [bounded compressed channel, 8 slots]
  → Writer thread [BTreeMap reorder, NVLZ header + size table + data]
```

Both GPUs run match finding concurrently and feed the shared CPU encoder pool. CPU encoder count is capped at `min(num_cpus, 8)`.

## Benchmarks

Windows Server 2003 ISO (613 MB):

| Algorithm | Level | Ratio | Time |
|-----------|-------|-------|------|
| Zstd | L0 GPU | 91.6% | 3.1s |
| LZMA2 | L0 CPU 8-way | 91.2% | 12.3s |
| LZMA2 | L1 GPU+CPU | 91.4% | 164.5s |

LZMA2 L1 ratio slightly worse than L0: 64KB sub-block independence prevents cross-block dictionary matching.

## File Formats

Both formats share the same 28-byte header structure:

```
[0:4]    Magic ("NVZS" or "NVLZ")
[4:12]   Original size (u64 LE)
[12:20]  Chunk size (u64 LE)
[20:28]  Number of chunks (u64 LE)
[28:]    N × 8 bytes: compressed size per chunk (u64 LE)
[28+N*8:] Compressed chunk data
```

NVZS chunks are one or more Zstd frames. NVLZ chunks are raw LZMA2 streams (L0: single stream; L1+: concatenated 64KB sub-block streams).

## Architecture

```
src/
  compress_lzma2.rs           liblzma FFI via dlopen
  compress_lzma2_custom.rs    GPU match finding + range coder + LZMA2 block assembly
  compress_zstd.rs            nvCOMP Zstd L0
  compress_zstd_custom.rs     GPU match finding + CPU FSE encoding
  pipeline_lzma2.rs           LZMA2 streaming pipeline (multi-GPU, async)
  pipeline_lzma2_dual.rs      Legacy dual GPU LZMA2 (not used, kept for reference)
  pipeline.rs                 Zstd single GPU pipeline
  pipeline_dual.rs            Zstd dual GPU pipeline
  cuda.rs                     GPU detection (uses nvcomp_bindings cudaDeviceProp)
  dispatch.rs                 Algorithm/level routing
  decompress.rs               NVZS + NVLZ decompression

lzma2_match_find.cu           HC4 hash chain CUDA kernel
zstd_compress.cu              Zstd LZ77 + FSE CUDA kernels
blake3.cu                     BLAKE3 kernel (compiled but not loaded)
```

## Known Issues

- `cuda-runtime-sys 0.3.0-alpha.1` has incorrect `cudaDeviceProp` size (712 vs 1008 bytes for CUDA 13.x). Worked around in `cuda.rs` by using the struct from `nvcomp_bindings`.
- LZMA2 custom range coder has match encoding edge cases — verification fallback to raw blocks ensures correctness.
- LZMA2 L1 compression ratio slightly worse than L0 due to 64KB sub-block independence (no cross-block dictionary).

## License

[MIT](LICENSE)
