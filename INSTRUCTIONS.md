# gpu-compressor usage guide

## overview

gpu-compressor uses NVIDIA GPUs to compress and decompress files. Two algorithms are supported:

- **Zstd** — output format `.nvzs`, decompressed by nvCOMP
- **LZMA2** — output format `.nvlz`, decompressed by liblzma

Both support multiple compression levels. Dual GPU is auto-detected for Zstd L0 and LZMA2 L1+.

---

## installation

### prerequisites

- NVIDIA GPU, Compute Capability 8.0+ (Ampere or newer, tested on RTX 3090 / 3090 Ti)
- CUDA Toolkit 12+ including `nvcc` (typically at `/opt/cuda` or `/usr/local/cuda`)
- nvCOMP 3.x headers and shared library
- Rust stable toolchain
- clang (required by bindgen)
- libzstd (for Zstd L1/L2 CPU FSE encoding)
- liblzma (runtime only, loaded via dlopen — must be installed but not linked at build time)

### install nvCOMP

**Arch Linux (AUR):**
```bash
yay -S nvcomp
```

**Manual:**
```bash
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-3.0.6-archive.tar.xz
tar -xf nvcomp-linux-x86_64-3.0.6-archive.tar.xz
sudo cp -r nvcomp-linux-x86_64-3.0.6-archive/include/* /usr/local/include/
sudo cp -r nvcomp-linux-x86_64-3.0.6-archive/lib/* /usr/local/lib/
sudo ldconfig
```

### build

```bash
cargo build --release
```

Build compiles two CUDA kernels to PTX, generates nvCOMP FFI bindings via bindgen, and links against nvcomp, cudart, and zstd. PTX files are embedded in the binary.

Binary: `target/release/gpu-compressor`

---

## basic usage

### compress a file

```bash
# Zstd (default, fastest)
gpu-compressor compress -i file.bin
# → file.bin.nvzs

# LZMA2
gpu-compressor compress -i file.bin -a lzma2
# → file.bin.nvlz
```

### decompress a file

```bash
# Auto-detects NVZS or NVLZ from magic bytes
gpu-compressor decompress -i file.bin.nvzs
# → file.bin

gpu-compressor decompress -i file.bin.nvlz
# → file.bin
```

### specify output path

```bash
gpu-compressor compress -i file.bin -o /backups/file.nvzs
gpu-compressor decompress -i file.bin.nvzs -o /restored/file.bin
```

### compress multiple files

```bash
gpu-compressor compress -i a.bin b.bin c.bin
# → a.bin.nvzs  b.bin.nvzs  c.bin.nvzs
```

### compress a directory

```bash
gpu-compressor compress -i /data/models/ -o /compressed/models/
```

All files under `/data/models/` are compressed recursively. Directory structure is preserved under the output path. Each file becomes an independent `.nvzs` file. LZMA2 directory compression is not supported.

---

## compression levels

### Zstd levels

| Level | Match Finding | Entropy Coding | Notes |
|-------|---------------|----------------|-------|
| 0 (default) | nvCOMP internal | nvCOMP internal | Fastest; dual GPU auto-enabled |
| 1 | GPU LZ77, depth 16 | CPU ZSTD_compressSequences | Better ratio; single GPU |
| 2 | GPU LZ77, depth 64 | CPU ZSTD_compressSequences | Best ratio; single GPU |

```bash
gpu-compressor compress -i file.bin -l 1
gpu-compressor compress -i file.bin -l 2
```

Zstd L1/L2: GPU kernel finds LZ77 matches across 64 KB sub-chunks in parallel. Sequences encoded into Zstd frames by libzstd on CPU. Falls back to `ZSTD_compress(level=1)` per sub-chunk on sequence encoding failure.

### LZMA2 levels

| Level | Pipeline | Notes |
|-------|----------|-------|
| 0 (default) | liblzma raw encoder, 8 workers, preset specified by `-l` | `-l 0` through `-l 9` map to liblzma presets |
| 1+ | Async GPU match finding → N CPU encoder threads (liblzma preset 9) | Both GPUs if available |

```bash
# L0 with preset 6
gpu-compressor compress -i file.bin -a lzma2 -l 6

# L1: async GPU+CPU pipeline
gpu-compressor compress -i file.bin -a lzma2 -l 1
```

**Dictionary size** (`--dict-size`): sets the LZMA2 pipeline chunk size (= dictionary window). Larger values improve compression ratio on files with long-range redundancy. Dictionary is stored in host RAM, not VRAM.

```bash
gpu-compressor compress -i file.bin -a lzma2 --dict-size 256
```

Default: 128 MB. Max limited by available RAM.

---

## GPU selection

### default

Device 0 is used by default. Zstd L0 auto-detects GPU count; if 2+ GPUs are present, both are used automatically.

### select a specific GPU

```bash
gpu-compressor compress -i file.bin -d 1
```

### force single GPU

```bash
CUDA_VISIBLE_DEVICES=0 gpu-compressor compress -i file.bin
```

### dual GPU behavior

**Zstd L0:** Reader distributes chunks alternately — even to GPU 0, odd to GPU 1. Writer reorders results by chunk index.

**LZMA2 L1+:** Both GPUs run HC4 match finding and push sub-block jobs into a shared MPMC job queue. CPU encoder pool drains independently. Round-robin distribution is implicit via crossbeam MPMC channel.

**Zstd L1/L2 and LZMA2 L0:** Single GPU only.

---

## VRAM requirements

| Mode | VRAM per GPU |
|------|-------------|
| Zstd L0 | ~16 MB active per chunk (8 MB in + 8 MB out) |
| Zstd L1/L2 | ~64 KB per sub-chunk + GPU match output buffers |
| LZMA2 L0 | No GPU usage |
| LZMA2 L1+ | ~552 MB per pipeline chunk (fits 4 GB GPUs) |

---

## output naming

### compression

| Input | Algorithm | `-o` | Output |
|-------|-----------|------|--------|
| `file.bin` | zstd | omitted | `file.bin.nvzs` |
| `file.bin` | lzma2 | omitted | `file.bin.nvlz` |
| `file.bin` | zstd | `/out/file.nvzs` | `/out/file.nvzs` |
| `file.bin` | zstd | `/out/` (dir) | `/out/file.bin.nvzs` |
| `a.bin b.bin` | zstd | omitted | `a.bin.nvzs b.bin.nvzs` |
| `a.bin b.bin` | zstd | `/out/` (dir) | `/out/a.bin.nvzs /out/b.bin.nvzs` |

### decompression

| Input | `-o` | Output |
|-------|------|--------|
| `file.bin.nvzs` | omitted | `file.bin` |
| `file.bin.nvlz` | omitted | `file.bin` |
| `file.compressed` | omitted | `file.compressed.out` |
| `file.bin.nvzs` | `/out/file.bin` | `/out/file.bin` |
| `file.bin.nvzs` | `/out/` (dir) | `/out/file.bin` |

---

## file formats

### NVZS (Zstd)

```
[0:4]    "NVZS" magic
[4:12]   Original file size (u64 LE)
[12:20]  Chunk size (u64 LE, 8388608 = 8 MB)
[20:28]  Number of chunks N (u64 LE)
[28:]    N × 8 bytes: compressed size per chunk (u64 LE)
[28+N*8:] Compressed chunk data
```

L0 chunks: single Zstd frame. L1/L2 chunks: multiple concatenated Zstd frames (one per 64 KB sub-chunk).

### NVLZ (LZMA2)

```
[0:4]    "NVLZ" magic
[4:12]   Original file size (u64 LE)
[12:20]  Chunk size (u64 LE — equals dict-size for L0, 65536 for L1+)
[20:28]  Number of chunks N (u64 LE)
[28:]    N × 8 bytes: compressed size per chunk (u64 LE)
[28+N*8:] Raw LZMA2 streams (one per chunk)
```

L0: each chunk is a single raw LZMA2 stream. L1+: each chunk is N concatenated 64 KB sub-block streams, each independently decompressible by `lzma_raw_decoder`.

---

## troubleshooting

### "No NVIDIA GPUs detected"

- Verify: `nvidia-smi`
- Verify CUDA runtime: `ldconfig -p | grep libcudart`
- If using `CUDA_VISIBLE_DEVICES`, ensure it is not empty

### "Unable to generate nvCOMP bindings" (build error)

- Install nvCOMP headers: `ls /usr/local/include/nvcomp.h`
- Install clang: `pacman -S clang` or `apt install libclang-dev`

### "Failed to compile zstd_compress.cu" or "lzma2_match_find.cu"

- Verify nvcc: `/opt/cuda/bin/nvcc --version`
- Kernels target `sm_86` (RTX 30xx). For other GPUs edit `--gpu-architecture=sm_XX` in `build.rs`
- Supported: sm_80 (A100), sm_86 (RTX 3090), sm_89 (RTX 4090), sm_90 (H100)

### "Failed to open liblzma" (LZMA2 runtime error)

- Install liblzma: `pacman -S xz` or `apt install liblzma-dev`
- Verify: `ldconfig -p | grep liblzma`
- liblzma is loaded via dlopen at runtime — not a link-time dependency

### LZMA2 decompression produces garbled output

- NVLZ chunks are independent raw LZMA2 streams — partial truncation corrupts all subsequent chunks
- Verify `original_size` in header matches expected output size

### compression produces no size reduction on random/encrypted data

- Zstd L0 will produce output slightly larger than input (frame header overhead)
- Zstd L1/L2 falls back to `ZSTD_compress(level=1)` per sub-chunk if GPU sequences yield no benefit
- LZMA2 on random data: expect near 1:1 ratio or slight expansion
- Use Zstd L0 for incompressible data to minimize overhead

### TUI output garbled in non-terminal environments

- TUI uses ANSI escape codes for cursor movement
- Redirect stderr to suppress: `2>/dev/null` or `2>log.txt`
