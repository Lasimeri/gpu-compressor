# gpu-compressor usage guide

## overview

gpu-compressor uses NVIDIA GPUs to compress and decompress files via the nvCOMP library and custom CUDA kernels. It supports two algorithms (Zstd and Gdeflate), three Zstd compression levels (nvCOMP fast, custom lazy, custom optimal), automatic dual-GPU detection, streaming pipelines for large files, and GPU-accelerated BLAKE3 integrity verification.

## installation

### prerequisites

- NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- CUDA Toolkit 12.x (`/opt/cuda` or `/usr/local/cuda`)
- nvCOMP 3.x headers and libraries
- Rust stable toolchain
- clang (required by bindgen for FFI generation)

### install nvCOMP

**Arch Linux (AUR):**
```
yay -S nvcomp
```

**Manual:**
```
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-3.0.6-archive.tar.xz
tar -xf nvcomp-linux-x86_64-3.0.6-archive.tar.xz
sudo cp -r nvcomp-linux-x86_64-3.0.6-archive/include/* /usr/local/include/
sudo cp -r nvcomp-linux-x86_64-3.0.6-archive/lib/* /usr/local/lib/
sudo ldconfig
```

### build

```
cargo build --release
```

The binary is at `target/release/gpu-compressor`. The build process:
1. Generates Rust FFI bindings from nvCOMP C headers via bindgen
2. Compiles `blake3.cu` to PTX via nvcc (targeting sm_86)
3. Compiles `zstd_compress.cu` to PTX via nvcc (targeting sm_86, maxrregcount=64)
4. Embeds both PTX files into the binary at compile time
5. Links against `libnvcomp` and `libcudart`

The release profile uses `codegen-units = 1` to prevent CUDA FFI optimization bugs.

### install (Arch Linux)

```
makepkg -si
```

This installs the binary to `/usr/bin/gpu-compressor`.

## commands

### compress

Compress a single file.

```
gpu-compressor compress -i <input> -o <output> [-a <algorithm>] [-d <device>] [-l <level>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Input file or directory |
| `-o` | required | Output file or directory |
| `-a` | `zstd` | Algorithm: `zstd` or `gdeflate` |
| `-d` | `0` | CUDA device ID |
| `-l` | `0` | Compression level (Zstd only): 0, 1, or 2 |

#### compression levels (Zstd only)

| Level | Engine | Matching | Sub-chunk | Throughput | Ratio |
|-------|--------|----------|-----------|------------|-------|
| 0 | nvCOMP batched API | — | 4 MB | Highest | Good |
| 1 | Custom CUDA kernel | Lazy (depth 16) | 64 KB | Moderate | Better on repetitive data |
| 2 | Custom CUDA kernel | Optimal (depth 64) | 64 KB | Lower | Best on repetitive data |

- Level 0 uses NVIDIA's nvCOMP library directly — fastest, good general-purpose ratio
- Levels 1-2 use a custom CUDA kernel (`zstd_compress.cu`) implementing LZ77 match finding with shared-memory hash tables and RFC 8878-compliant FSE encoding
- Custom levels achieve ~5.3x compression on repetitive data, with safe 1x raw-block fallback on random/incompressible data
- All levels produce NVZS-format output decompressible by nvCOMP's standard Zstd decompressor

#### Zstd compression behavior

Creates an NVZS archive containing:
- A tar of the original file + its BLAKE3 hash (`.blake3` sidecar)
- Compressed in 4 MB streaming chunks (level 0) or 64 KB sub-chunks within 4 MB frames (levels 1-2)
- Automatic dual-GPU if 2+ GPUs detected (level 0 only — even/odd chunk interleaving)
- Produces a `.blake3` sidecar for the compressed output

#### Gdeflate compression behavior

Creates an NVGD archive:
- 64 KB chunks compressed in 128 MB batches
- Files >2 GB automatically split into 128 MB file chunks (NVMC wrapper format)
- No tar wrapping, no hash sidecar
- Compression level fixed at 5 (maximum)

#### directory input

If `-i` is a directory, all files are recursively discovered, hashed, then compressed individually with directory structure preserved under the output path.

```
# zstd (default, recommended)
gpu-compressor compress -i data.bin -o data.bin.nvzs

# zstd with custom lazy matching
gpu-compressor compress -i data.bin -o data.bin.nvzs -l 1

# zstd with custom optimal matching
gpu-compressor compress -i data.bin -o data.bin.nvzs -l 2

# gdeflate
gpu-compressor compress -i data.bin -o data.bin.nvgd -a gdeflate

# compress to directory (auto-names output)
gpu-compressor compress -i data.bin -o /backups/

# compress entire directory
gpu-compressor compress -i /data/ -o /compressed/
```

### compress-multi

Compress multiple files concurrently via async pipeline.

```
gpu-compressor compress-multi -i <file1> <file2> ... -o <out1> <out2> ... [-a zstd] [-d 0] [-l 0] [--chunk-size 134217728]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Input files (space-separated) or a directory |
| `-o` | optional | Output files (space-separated); auto-generated if omitted |
| `-a` | `zstd` | Algorithm (only `zstd` supported for multi) |
| `-d` | `0` | CUDA device ID |
| `-l` | `0` | Compression level: 0, 1, or 2 |
| `--chunk-size` | `134217728` | Chunk size in bytes (default 128 MB) |

Processes 2 files concurrently. Chunks from multiple files are batched together on the GPU for maximum utilization.

```
gpu-compressor compress-multi -i a.bin b.bin c.bin -o a.nvzs b.nvzs c.nvzs
```

Directory input:
```
gpu-compressor compress-multi -i /data/
```

### decompress

Decompress a single file. Algorithm is auto-detected from the file header (NVZS, NVGD, or NVMC magic bytes).

```
gpu-compressor decompress -i <input> -o <output> [-d <device>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Compressed input file |
| `-o` | required | Decompressed output path |
| `-d` | `0` | CUDA device ID |

**Zstd decompression:**
1. Reads NVZS header and compressed chunk sizes
2. Decompresses each chunk on GPU in micro-batches (1 chunk at a time to minimize VRAM)
3. Reconstructs the tar archive
4. Extracts the original file and `.blake3` hash
5. Verifies integrity by re-hashing the decompressed file against the stored hash
6. Falls back to legacy BLAKE3 mode if standard hash doesn't match (backward compat with old .nvzs files)
7. Cleans up temporary tar

**Gdeflate decompression:** loads all compressed chunks, decompresses in a single GPU batch, writes output.

**NVMC decompression:** processes each 128 MB sub-archive independently, concatenates results.

```
gpu-compressor decompress -i data.bin.nvzs -o /restored/data.bin
gpu-compressor decompress -i data.bin.nvgd -o /restored/data.bin
```

### decompress-multi

Decompress multiple files sequentially.

```
gpu-compressor decompress-multi -i <file1> <file2> -o <out1> <out2> [-d 0]
```

### hash

Compute a GPU-accelerated BLAKE3 hash.

```
gpu-compressor hash -i <input> [-d <device>]
```

Files are streamed in 4 MB batches to the GPU. Each 1 KB BLAKE3 chunk is processed by a dedicated CUDA thread. A tree reduction kernel combines chunk hashes into the final 256-bit hash.

```
gpu-compressor hash -i data.bin
```

Output:
```
  blake3: 381909b9480185c1268658eab4e6503187e84fdae1515d9854c5d840d277e9fe
```

## algorithms

### zstd (default)

#### level 0 — nvCOMP

GPU-accelerated Zstandard via nvCOMP's batched Zstd API.

- Chunk size: 4 MB
- Pipeline: 3-thread (read/compress/write) or 4-thread (dual GPU)
- Integrity: BLAKE3 hash embedded in tar, verified on decompress
- Best for: general purpose, maximum throughput
- File extension: `.nvzs`

#### level 1 — custom lazy matching

Custom CUDA kernel with LZ77 lazy match finding.

- Sub-chunk size: 64 KB (within 4 MB streaming chunks)
- Hash table: 16K entries, 14-bit, shared memory
- Search depth: 16 (lazy — checks if next position has a better match)
- Encoding: RFC 8878 FSE with predefined tables; RLE-mode for uniform-symbol chunks
- Fallback: raw block wrapping for incompressible data
- Best for: repetitive data where nvCOMP's fast mode under-compresses
- File extension: `.nvzs`

#### level 2 — custom optimal matching

Same custom CUDA kernel with deeper search.

- Search depth: 64 (optimal — evaluates more match candidates)
- Higher compression ratio at the cost of throughput
- Same FSE encoding and fallback behavior as level 1
- Best for: maximum compression ratio on compressible data
- File extension: `.nvzs`

### gdeflate

GPU-accelerated DEFLATE via nvCOMP's batched Gdeflate API.

- Chunk size: 64 KB
- Batch size: 128 MB on GPU
- Compression level: 5 (maximum)
- File size limit: ~2 GB per chunk (auto-splits larger files via NVMC)
- Best for: DEFLATE compatibility
- File extension: `.nvgd`

## file formats

### NVZS (Zstd compressed)

```
offset  size    field
0       4       magic: "NVZS"
4       8       original tar size (u64 LE)
12      8       chunk size (u64 LE, typically 4194304)
20      8       num chunks (u64 LE)
28      N*8     compressed sizes array (u64 LE each)
28+N*8  ...     compressed chunk data (concatenated)
```

The "original" data is a tar archive containing:
- The input file with its original filename
- A `.blake3` file containing the hex-encoded BLAKE3 hash

For custom Zstd levels (1-2), each 4 MB chunk contains multiple concatenated Zstd frames (one per 64 KB sub-chunk). These are valid Zstd and decompress identically via nvCOMP's standard API.

### NVGD (Gdeflate compressed)

```
offset  size    field
0       4       magic: "NVGD"
4       8       original size (u64 LE)
12      8       chunk size (u64 LE, typically 65536)
20      8       num chunks (u64 LE)
28      N*8     compressed sizes array (u64 LE each)
28+N*8  ...     compressed chunk data (concatenated)
```

### NVMC (multi-chunk wrapper)

Used when Gdeflate input exceeds ~2 GB. Wraps multiple NVGD archives:

```
offset  size    field
0       4       magic: "NVMC"
4       8       original file size (u64 LE)
12      8       num file chunks (u64 LE)
20      ...     concatenated NVGD archives (one per 128 MB chunk)
```

## GPU behavior

### device selection

- `-d 0` selects GPU 0 (default)
- Zstd level 0 auto-detects GPU count and uses dual-GPU when 2+ are available
- Zstd levels 1-2 use single GPU (custom kernel pipeline)
- BLAKE3 hashing uses GPU 0 regardless of `-d` flag
- `CUDA_VISIBLE_DEVICES` environment variable can restrict visible GPUs

### dual GPU mode

When 2+ GPUs are detected during Zstd level 0 compression:
- Reader thread distributes 4 MB tar chunks alternately to GPU 0 (even indices) and GPU 1 (odd indices)
- Each GPU compresses independently via nvCOMP
- Writer thread reorders results via BTreeMap and writes sequentially
- ~1.5-1.8x throughput vs single GPU (limited by PCIe bandwidth)

To force single GPU:
```
CUDA_VISIBLE_DEVICES=0 gpu-compressor compress -i file.bin -o file.nvzs
```

### VRAM usage

- Zstd streaming (level 0): ~8 MB per active chunk (4 MB uncompressed + 4 MB compressed buffer)
- Zstd custom (levels 1-2): ~64 KB per sub-chunk + match output buffers + hash table in shared memory
- Gdeflate batched: up to 128 MB batch + compressed buffers + temp workspace
- BLAKE3: ~4 MB per batch + chunk hash output arrays
- Multi-file: scales with batch size (default 9 chunks per GPU batch)

## integrity verification

Zstd compression automatically:
1. Hashes the input file with GPU BLAKE3 before compression
2. Embeds the hash inside the tar archive as a `.blake3` sidecar file
3. Hashes the compressed output and writes a `.blake3` sidecar alongside it

Zstd decompression automatically:
1. Decompresses the tar archive
2. Extracts the original file and stored hash
3. Re-hashes the decompressed file on GPU
4. Compares against the stored hash
5. If mismatch: retries with legacy BLAKE3 mode (for files compressed with older versions)
6. Fails with an error if both standard and legacy hashes don't match

Gdeflate does not include integrity verification.

## examples

### compress a large dataset
```
gpu-compressor compress -i /data/model_weights.bin -o /backups/model_weights.nvzs
```

### compress with better ratio (custom kernel)
```
gpu-compressor compress -i /data/model_weights.bin -o /backups/model_weights.nvzs -l 1
```

### compress an entire directory
```
gpu-compressor compress -i /data/training/ -o /backups/training/ -a gdeflate
```

### batch compress multiple files
```
gpu-compressor compress-multi \
    -i file1.bin file2.bin file3.bin \
    -o file1.nvzs file2.nvzs file3.nvzs
```

### batch compress with custom level
```
gpu-compressor compress-multi \
    -i file1.bin file2.bin file3.bin \
    -o file1.nvzs file2.nvzs file3.nvzs \
    -l 2
```

### verify a file hash
```
gpu-compressor hash -i /data/model_weights.bin
```

### decompress and verify
```
gpu-compressor decompress -i /backups/model_weights.nvzs -o /restored/model_weights.bin
```
Integrity is checked automatically. If the hash doesn't match, the command fails.

### force single GPU
```
CUDA_VISIBLE_DEVICES=0 gpu-compressor compress -i large.bin -o large.nvzs
```

## troubleshooting

### "No NVIDIA GPUs detected"
- Verify `nvidia-smi` shows your GPU
- Ensure CUDA toolkit is installed and `libcudart.so` is on the library path

### "Unable to generate nvCOMP bindings"
- Install nvCOMP headers: `ls /usr/local/include/nvcomp.h` or `/usr/include/nvcomp.h`
- Install clang: `pacman -S clang` or `apt install libclang-dev`

### "Failed to compile blake3.cu" or "Failed to compile zstd_compress.cu"
- Ensure nvcc is available: `/opt/cuda/bin/nvcc --version`
- Both kernels target sm_86 (Ampere). For older GPUs, edit `build.rs` to change `--gpu-architecture=sm_XX`

### segfault on exit after successful compression
- Known issue with CUDA context cleanup during async runtime teardown
- The operation completed successfully; the exit code is non-zero but output is valid
- Mitigated with `std::process::exit(0)` in current builds

### decompression fails with "status 1000"
- nvcomp error code 1000 = `nvcompErrorCudaError`
- Usually indicates corrupted compressed data or VRAM exhaustion
- Check `nvidia-smi` for available GPU memory

### decompression hash mismatch on old files
- Files compressed with earlier versions used a non-spec BLAKE3 chunk flag
- The decompressor automatically retries with legacy BLAKE3 mode
- If both modes fail, the file is genuinely corrupted
