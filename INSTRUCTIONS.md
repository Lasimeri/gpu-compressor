# gpu-compressor usage guide

## overview

gpu-compressor uses NVIDIA GPUs to compress and decompress files via the nvCOMP library. It supports two algorithms (Zstd and Gdeflate), automatic dual-GPU detection, streaming pipelines for large files, and GPU-accelerated BLAKE3 integrity verification.

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
3. Links against `libnvcomp` and `libcudart`

### install (Arch Linux)

```
makepkg -si
```

This installs the binary to `/usr/bin/gpu-compressor`.

## commands

### compress

Compress a single file.

```
gpu-compressor compress -i <input> -o <output> [-a <algorithm>] [-d <device>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Input file or directory |
| `-o` | required | Output file or directory |
| `-a` | `zstd` | Algorithm: `zstd` or `gdeflate` |
| `-d` | `0` | CUDA device ID |

**Zstd compression** creates an NVZS archive containing:
- A tar of the original file + its BLAKE3 hash (`.blake3` sidecar)
- Compressed in 4MB streaming chunks
- Automatic dual-GPU if 2+ GPUs detected (even/odd chunk interleaving)
- Produces a `.blake3` sidecar for the compressed output

**Gdeflate compression** creates an NVGD archive:
- 64KB chunks compressed in 128MB batches
- Files >2GB automatically split into 128MB file chunks (NVMC wrapper format)
- No tar wrapping, no hash sidecar

**Directory input:** if `-i` is a directory, all files are recursively discovered, hashed, then compressed individually with directory structure preserved.

```
# zstd (default, recommended)
gpu-compressor compress -i data.bin -o data.bin.nvzs

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
gpu-compressor compress-multi -i <file1> <file2> ... -o <out1> <out2> ... [-a zstd] [-d 0] [--chunk-size 134217728]
```

Processes 2 files concurrently. Chunks from multiple files are batched together on the GPU for maximum utilization. Only supports Zstd.

```
gpu-compressor compress-multi -i a.bin b.bin c.bin -o a.nvzs b.nvzs c.nvzs
```

Directory input also works:
```
gpu-compressor compress-multi -i /data/
```

### decompress

Decompress a single file. Algorithm is auto-detected from the file header (NVZS, NVGD, or NVMC magic bytes).

```
gpu-compressor decompress -i <input> -o <output> [-d <device>]
```

**Zstd decompression:**
1. Reads NVZS header and compressed chunk sizes
2. Decompresses each chunk on GPU in micro-batches
3. Reconstructs the tar archive
4. Extracts the original file and `.blake3` hash
5. Verifies integrity by re-hashing the decompressed file against the stored hash
6. Cleans up temporary tar

**Gdeflate decompression:** loads all compressed chunks, decompresses in a single GPU batch, writes output.

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

Files are streamed in 4MB batches to the GPU. Each 1KB BLAKE3 chunk is processed by a dedicated CUDA thread. A tree reduction kernel combines chunk hashes into the final 256-bit hash.

```
gpu-compressor hash -i data.bin
```

Output:
```
  blake3: 381909b9480185c1268658eab4e6503187e84fdae1515d9854c5d840d277e9fe
```

## algorithms

### zstd (default)

GPU-accelerated Zstandard via nvCOMP's batched Zstd API.

- Chunk size: 4MB
- Pipeline: 3-thread (read/compress/write) or 4-thread (dual GPU)
- Integrity: BLAKE3 hash embedded in tar, verified on decompress
- Best for: general purpose, high compression ratio
- File extension: `.nvzs`

### gdeflate

GPU-accelerated DEFLATE via nvCOMP's batched Gdeflate API.

- Chunk size: 64KB
- Batch size: 128MB on GPU
- Compression level: 5 (maximum)
- File size limit: ~2GB per chunk (auto-splits larger files)
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

Used when Gdeflate input exceeds ~2GB. Wraps multiple NVGD archives:

```
offset  size    field
0       4       magic: "NVMC"
4       8       original file size (u64 LE)
12      8       num file chunks (u64 LE)
20      ...     concatenated NVGD archives (one per 128MB chunk)
```

## gpu behavior

### device selection

- `-d 0` selects GPU 0 (default)
- Zstd compression auto-detects GPU count and uses dual-GPU when 2+ are available
- BLAKE3 hashing always uses GPU 0 regardless of `-d` flag
- `CUDA_VISIBLE_DEVICES` environment variable can restrict visible GPUs

### dual gpu mode

When 2+ GPUs are detected during Zstd compression:
- Reader thread distributes 4MB tar chunks alternately to GPU 0 (even indices) and GPU 1 (odd indices)
- Each GPU compresses independently
- Writer thread reorders results via BTreeMap and writes sequentially
- ~1.5-1.8x throughput vs single GPU (limited by PCIe bandwidth)

To force single GPU:
```
CUDA_VISIBLE_DEVICES=0 gpu-compressor compress -i file.bin -o file.nvzs
```

### vram usage

- Zstd streaming: ~8MB per active chunk (4MB uncompressed + 4MB compressed buffer)
- Gdeflate batched: up to 128MB batch + compressed buffers + temp workspace
- BLAKE3: ~4MB per batch + chunk hash output arrays
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
5. Fails with an error if hashes don't match

Gdeflate does not include integrity verification.

## examples

### compress a large dataset
```
gpu-compressor compress -i /data/model_weights.bin -o /backups/model_weights.nvzs
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

### verify a file hash
```
gpu-compressor hash -i /data/model_weights.bin
```

### decompress and verify
```
gpu-compressor decompress -i /backups/model_weights.nvzs -o /restored/model_weights.bin
```
Integrity is checked automatically. If the hash doesn't match, the command fails.

### force single gpu
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

### "Failed to compile blake3.cu"
- Ensure nvcc is available: `/opt/cuda/bin/nvcc --version`
- The kernel targets sm_86 (Ampere). For older GPUs, edit `build.rs` line: `--gpu-architecture=sm_XX`

### segfault on exit after successful compression
- Known issue with CUDA context cleanup during async runtime teardown
- The operation completed successfully; the exit code is non-zero but output is valid
- Mitigated with `std::process::exit(0)` in current builds

### decompression fails with "status 1000"
- nvcomp error code 1000 = `nvcompErrorCudaError`
- Usually indicates corrupted compressed data or VRAM exhaustion
- Check `nvidia-smi` for available GPU memory
