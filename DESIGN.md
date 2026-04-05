# gpu-compressor design document

## project overview

gpu-compressor is a GPU-accelerated file compression tool built in Rust. It leverages NVIDIA's nvCOMP library for batched GPU compression/decompression and includes two custom CUDA kernels — one for BLAKE3 hashing and one for Zstd compression. The tool supports single and dual GPU operation, streaming pipelines, multi-file async processing, and directory-recursive compression with integrity verification.

**Language:** Rust (2021 edition) + CUDA C++  
**Runtime:** Tokio (async I/O, multi-threaded) + raw threads (pipeline stages)  
**GPU interface:** cudarc (high-level) + cuda-runtime-sys (low-level FFI) + nvCOMP (compression API)  
**Build:** Cargo + bindgen (FFI generation) + nvcc (CUDA compilation)

## module dependency graph

```
                          ┌──────────┐
                          │ main.rs  │
                          └────┬─────┘
                               │
              ┌────────────────┼────────────────────────┐
              │                │                        │
         ┌────▼────┐    ┌─────▼──────┐          ┌──────▼──────┐
         │ cli.rs  │    │dispatch.rs │          │decompress.rs│
         └─────────┘    └─────┬──────┘          └──────┬──────┘
                              │                        │
           ┌──────────┬───────┼──────────┬─────────────┤
           │          │       │          │             │
     ┌─────▼────┐ ┌───▼────┐ │  ┌───────▼────────┐   │
     │pipeline. │ │pipeline│ │  │compress_zstd   │   │
     │  rs      │ │_dual.rs│ │  │  _custom.rs    │   │
     └─────┬────┘ └───┬────┘ │  └───────┬────────┘   │
           │          │      │          │             │
     ┌─────▼──────────▼──────▼──────────▼─────────────▼─────┐
     │                  shared modules                       │
     │  blake3.rs  compress_zstd.rs  compress_gdeflate.rs   │
     │  cuda.rs    constants.rs      format.rs              │
     │  multi.rs   nvcomp_bindings.rs                       │
     └──────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   CUDA kernels     │
                    │  blake3.ptx        │
                    │  zstd_compress.ptx │
                    └────────────────────┘
```

## module descriptions

### entry point

**main.rs** — Parses CLI via clap, routes to the appropriate handler based on subcommand (Compress, CompressMulti, Decompress, DecompressMulti, Hash). Uses `tokio::main` for the async runtime. Force-exits via `std::process::exit(0)` after completion to avoid CUDA driver segfaults during tokio shutdown.

### CLI

**cli.rs** — Defines the argument schema using clap derive macros.

- `Algorithm` enum: `Gdeflate` | `Zstd` (accepts aliases: "deflate", "zstandard")
- `Commands` enum: five subcommands with typed parameters
- Compression level (`-l`): u32, default 0. Only meaningful for Zstd.
- Chunk size (`--chunk-size`): usize for compress-multi, default 128 MB

### compression dispatch

**dispatch.rs** — Central routing layer. Determines which compression pipeline to invoke based on algorithm, level, file size, and GPU count.

Decision tree:
```
compress_file(input, output, algo, device, level)
  │
  ├─ algo = Zstd
  │   ├─ level > 0 → compress_file_custom_zstd()
  │   │                (custom CUDA kernel, single GPU)
  │   ├─ gpu_count >= 2 → compress_file_streaming_dual_gpu()
  │   │                    (nvCOMP, dual GPU pipeline)
  │   └─ gpu_count == 1 → compress_file_streaming_zstd()
  │                        (nvCOMP, single GPU pipeline)
  │
  ├─ algo = Gdeflate
  │   ├─ size > 2 GB → compress_large_file_in_chunks()
  │   │                 (NVMC wrapper, 128 MB sub-files)
  │   └─ size ≤ 2 GB → compress_buffer_gdeflate()
  │                      (single-shot buffered)
  │
  └─ directory input → compress_directory()
                        Phase 1: hash all files (GPU BLAKE3)
                        Phase 2: compress each file (above logic)
```

**expand_directory_inputs()** — WalkDir traversal that maps input paths to output paths preserving directory structure, appending algorithm-specific extensions.

### streaming pipelines

#### pipeline.rs — single GPU (Zstd level 0)

Three-thread producer-consumer pipeline connected by bounded crossbeam channels:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Tar Generator   │────▶│  GPU Compressor  │────▶│     Writer       │
│  (read thread)   │     │  (compress thread)│    │  (write thread)  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
    Reads input file       nvCOMP Zstd batch         Writes NVZS header
    Builds tar archive     compress per chunk         + chunk sizes table
    Streams 4 MB chunks    on GPU device              + compressed data
    Includes .blake3                                  + .blake3 sidecar
```

A fourth stats thread monitors atomic counters (read bytes, GPU bytes, write bytes) and displays live throughput via indicatif spinners.

**Backpressure:** channels are bounded to `PIPELINE_QUEUE_SIZE` (16). If the writer falls behind, the compressor blocks; if the compressor falls behind, the reader blocks.

**Buffer sizes:** 8 MB BufReader and BufWriter for SSD throughput saturation.

#### pipeline_dual.rs — dual GPU (Zstd level 0)

Six-thread pipeline with chunk interleaving:

```
                          ┌─────────────────┐
                     ┌───▶│  GPU 0 Thread   │───┐
                     │    │  (even chunks)  │   │
┌──────────────┐     │    └─────────────────┘   │    ┌──────────────┐
│ Tar Generator│─────┤                          ├───▶│    Writer    │
│ (reader)     │     │    ┌─────────────────┐   │    │ (reorderer)  │
└──────────────┘     └───▶│  GPU 1 Thread   │───┘    └──────────────┘
                          │  (odd chunks)   │
                          └─────────────────┘
```

- Reader distributes via `batch_index % 2` to separate channels per GPU
- Both GPU threads compress via nvCOMP and send to a shared writer channel
- Writer maintains a `BTreeMap<usize, CompressedChunk>` for reordering
- Writes chunks sequentially by tracking `next_chunk_to_write`

### compression engines

#### compress_zstd.rs — nvCOMP Zstd (level 0)

Thin wrapper around nvCOMP's batched Zstd API for single-chunk operation:
1. Allocate device memory for one 4 MB chunk
2. Call `nvcompBatchedZstdCompressAsync` 
3. Synchronize and download compressed result
4. Returns `(Vec<Vec<u8>>, Vec<usize>)` — compressed sub-chunks and their sizes

Used by both single and dual GPU pipelines per-chunk.

#### compress_zstd_custom.rs — custom CUDA kernel (levels 1-2)

Orchestrates the custom Zstd CUDA kernel pipeline:

```
Input 4 MB chunk
  │
  ├─ Split into 64 KB sub-chunks
  │
  ├─ Level 0 fallback → zstd_compress_raw kernel
  │   (wraps each sub-chunk as a Zstd raw frame)
  │
  └─ Level 1+ → two-pass pipeline:
      │
      ├─ Pass 1: zstd_match_find kernel
      │   - LZ77 match finding per sub-chunk
      │   - 14-bit shared-memory hash table (16K entries)
      │   - Lazy matching (depth 16) or optimal (depth 64)
      │   - Output: sequence array + literal bytes per sub-chunk
      │
      └─ Pass 2: zstd_encode_block kernel
          - FSE encoding of sequences + literals
          - Predefined FSE tables from RFC 8878
          - RLE-mode for uniform-symbol chunks
          - Output: valid Zstd compressed frames
          - Fallback: raw block if compressed >= original
```

Each sub-chunk becomes an independent Zstd frame. Multiple frames are concatenated to form the 4 MB chunk payload stored in NVZS format. nvCOMP's standard Zstd decompressor handles multi-frame chunks natively.

#### compress_gdeflate.rs — nvCOMP Gdeflate

Buffered (non-streaming) compression:
1. Split input into 64 KB chunks
2. Batch chunks into 128 MB GPU batches
3. Per batch: upload → nvCOMP Gdeflate compress → download
4. Write NVGD header + chunk size table + concatenated compressed data

### multi-file compression

**multi.rs** — Tokio-based async pipeline for concurrent multi-file processing.

```
┌────────────┐    ┌────────────┐
│ Reader 1   │───▶│            │    ┌────────────────┐    ┌──────────┐
│ (file 1)   │    │  Shared    │───▶│ GPU Compressor │───▶│ Writer 1 │
└────────────┘    │  Channel   │    │ (blocking task) │   └──────────┘
┌────────────┐    │            │    │ 9 chunks/batch  │   ┌──────────┐
│ Reader 2   │───▶│ (file_idx, │    └────────────────┘───▶│ Writer 2 │
│ (file 2)   │    │  chunk)    │                          └──────────┘
└────────────┘    └────────────┘
```

- Max 2 files processed concurrently
- Reader tasks are async (tokio::fs)
- GPU compressor is a blocking task that batches 9 chunks from any file
- Per-file writer tasks receive chunks via dedicated channels
- Interleaving chunks from multiple files maximizes GPU utilization

### decompression

**decompress.rs** — Format auto-detection via 4-byte magic header:

| Magic | Handler | Strategy |
|-------|---------|----------|
| `NVZS` | `decompress_file_zstd_streaming()` | Micro-batch (1 chunk), tar extract, BLAKE3 verify |
| `NVGD` | `decompress_file_gdeflate()` | Load all chunks, single GPU batch decompress |
| `NVMC` | `decompress_multi_chunk_file()` | Process each sub-archive independently |

**Zstd decompression detail:**
1. Read header → parse chunk count and sizes
2. For each chunk: read compressed → upload to GPU → nvCOMP decompress → download → append to temp .tar
3. Extract tar → recover original file + `.blake3` sidecar
4. Hash decompressed file on GPU → compare to stored hash
5. If mismatch: retry with `blake3_hash_file_legacy()` (backward compat for old BLAKE3 chunk flag bug)
6. Clean up temp files

### GPU utilities

**cuda.rs** — Safe wrapper around `cudaGetDeviceCount` and `cudaGetDeviceProperties`. Returns `Vec<i32>` of device IDs. Properly scoped unsafe blocks with error checking on every CUDA call.

**nvcomp_bindings.rs** — Auto-generated via bindgen from `wrapper.h` at build time. Includes all nvCOMP API functions for Gdeflate, Zstd, Bitcomp, and LZ4.

### shared types and constants

**format.rs** — Pipeline message types:
- `PipelineMsg::Chunk { data: Vec<u8>, chunk_index: usize }` — a chunk to compress
- `PipelineMsg::Done` — signal end of input
- `CompressedChunk` — compressed output with sub-chunk data and sizes

**constants.rs** — All tunable parameters:
| Constant | Value | Purpose |
|----------|-------|---------|
| `CHUNK_SIZE` | 64 KB | Gdeflate chunk size |
| `ZSTD_CHUNK_SIZE` | 4 MB | Zstd streaming chunk size |
| `BATCH_SIZE` | 128 MB | Gdeflate GPU batch size |
| `MAX_FILE_CHUNK_SIZE` | 128 MB | Multi-chunk file split size |
| `PIPELINE_QUEUE_SIZE` | 16 | Channel backpressure depth |
| `BLAKE3_CHUNK_SIZE` | 1 KB | Per-thread GPU hash chunk |
| `GDEFLATE_MAX_SIZE` | ~2 GB | Gdeflate single-file limit |
| `CUSTOM_ZSTD_CHUNK_SIZE` | 64 KB | Custom kernel sub-chunk size |
| `CUSTOM_ZSTD_SEARCH_DEPTH_LAZY` | 16 | Level 1 match search depth |

PTX kernels (`blake3.ptx`, `zstd_compress.ptx`) are embedded via `include_str!` at compile time.

## CUDA kernel design

### blake3.cu — GPU BLAKE3 hashing

Two kernels:

**`blake3_hash_chunks`** — per-chunk hashing
- One thread per 1 KB BLAKE3 chunk
- 256 threads/block, dynamic grid sizing
- Each thread: loads 1 KB, runs BLAKE3 compression function (7 rounds) on two 64-byte blocks
- Flags: `CHUNK_START` on block 0 only (fixed in v2; legacy mode sets it on all blocks)
- Output: 32-byte chaining value per chunk
- Constants (IV, permutation schedule) in CUDA constant memory

**`blake3_reduce_tree`** — tree reduction
- Pairwise combines chunk hashes: hash[2i] + hash[2i+1] → parent[i]
- Runs iteratively (log₂ N stages) until one hash remains
- Each stage: parent flag set, child hashes concatenated, compressed to new chaining value
- Final output: 256-bit BLAKE3 hash

**Legacy mode:** A `u32 legacy` parameter is passed to `blake3_hash_chunks`. When set, applies `CHUNK_START` to all blocks (matching the bug in v1). This allows decompression to verify hashes from old .nvzs files.

### zstd_compress.cu — custom Zstd compression

Three kernels implementing RFC 8878 (Zstandard) compliant compression:

**`zstd_compress_raw`** — raw frame wrapping
- One block per sub-chunk
- Wraps each 64 KB input as a Zstd raw frame (no compression)
- Header: magic (0xFD2FB528) + frame descriptor + FCS + raw block header + data
- Used as fallback when match finding produces no benefit

**`zstd_match_find`** — LZ77 match finding
- One block per 64 KB sub-chunk, thread 0 does sequential scanning
- **Hash table:** 16K entries in shared memory, 14-bit multiplicative hash on 4-byte windows
- **Algorithm:** scans input sequentially, probes hash table for matches
  - Extends matches forward to determine match length
  - Lazy matching: checks if next position has a longer match (configurable depth)
  - Tracks 3 repeat offsets per RFC 8878
- **Output per sub-chunk:**
  - `Sequence` array: (literal_length, match_length, offset) triples
  - Literal bytes: raw bytes not covered by matches
  - `ChunkResult`: counts (compressed size, num sequences, num literals)

**`zstd_encode_block`** — FSE encoding
- One block per sub-chunk, thread 0 does sequential encoding
- **Input:** sequences + literals from match finder
- **Bitstream:** 64-bit accumulator with forward bit writing
- **Encoding pipeline:**
  1. Write Zstd frame header (magic, FCS, single segment flag)
  2. Encode literals section (raw mode — literals copied directly)
  3. Encode sequences section:
     - Build frequency tables for literal lengths (LL), match lengths (ML), offsets (OF)
     - Normalize to power-of-2 table sizes
     - Spread symbols via FSE table construction
     - Encode sequences in reverse order (per RFC 8878)
  4. Finalize block (block header with compressed size, last block flag)
- **Predefined tables:** RFC 8878 Appendix A predefined FSE distributions for LL, ML, OF
- **RLE mode:** when all symbols in a category are identical, uses RLE encoding (1-byte table)
- **Raw fallback:** if compressed block ≥ original size, falls back to raw block

**Data structures:**
```c
struct Sequence {
    uint32_t literal_length;
    uint32_t match_length;
    uint32_t offset;
};

struct ChunkResult {
    uint32_t compressed_size;
    uint32_t num_sequences;
    uint32_t num_literals;
};

struct BitStream {
    uint8_t* buffer;
    uint32_t byte_pos;
    uint64_t accumulator;
    uint32_t bits_in_acc;
};

struct DynFSE {
    int16_t next_state[MAX_TABLE_SIZE];
    uint8_t symbol[MAX_TABLE_SIZE];
    uint8_t num_bits[MAX_TABLE_SIZE];
    uint32_t table_log;
    uint32_t table_size;
};
```

## file format specifications

### NVZS format (Zstd compressed)

```
┌────────────────────────────────────────────┐
│ Header (28 bytes)                          │
│   [0:4]   Magic: "NVZS" (4 bytes)         │
│   [4:12]  Original tar size (u64 LE)       │
│   [12:20] Chunk size (u64 LE, 4194304)     │
│   [20:28] Number of chunks N (u64 LE)      │
├────────────────────────────────────────────┤
│ Chunk Size Table (N × 8 bytes)             │
│   [28]    Compressed size of chunk 0 (u64) │
│   [36]    Compressed size of chunk 1 (u64) │
│   ...     ...                              │
│   [28+N*8-8] Compressed size of chunk N-1  │
├────────────────────────────────────────────┤
│ Compressed Data                            │
│   Chunk 0 data (sizes[0] bytes)            │
│   Chunk 1 data (sizes[1] bytes)            │
│   ...                                      │
│   Chunk N-1 data (sizes[N-1] bytes)        │
└────────────────────────────────────────────┘
```

The decompressed payload is a tar archive containing:
- `<original_filename>` — the input file
- `<original_filename>.blake3` — hex-encoded 256-bit BLAKE3 hash

For custom Zstd levels, each chunk may contain multiple concatenated Zstd frames (one per 64 KB sub-chunk), all individually valid per RFC 8878.

### NVGD format (Gdeflate compressed)

Same structure as NVZS but with:
- Magic: `"NVGD"`
- Chunk size: typically 65536 (64 KB)
- Original size: raw file size (no tar wrapping)

### NVMC format (multi-chunk wrapper)

```
┌────────────────────────────────────────────┐
│ Header (20 bytes)                          │
│   [0:4]   Magic: "NVMC" (4 bytes)         │
│   [4:12]  Original file size (u64 LE)      │
│   [12:20] Number of sub-archives M (u64)   │
├────────────────────────────────────────────┤
│ Sub-archive 0 (complete NVGD)              │
│   NVGD header + chunks for bytes 0..128MB  │
├────────────────────────────────────────────┤
│ Sub-archive 1 (complete NVGD)              │
│   NVGD header + chunks for next 128MB      │
├────────────────────────────────────────────┤
│ ...                                        │
└────────────────────────────────────────────┘
```

## build system

**build.rs** performs three operations:

1. **Bindgen** — parses `wrapper.h` (which includes nvcomp.h and algorithm headers) → generates `nvcomp_bindings.rs` in the output directory. Configured with:
   - `no_copy`, `no_debug` on all types
   - Include paths: `/opt/cuda/include`, `/usr/local/cuda/include`

2. **CUDA compilation** — compiles two kernels via nvcc:
   - `blake3.cu → blake3.ptx` — flags: `-ptx -O3 --use_fast_math --gpu-architecture=sm_86`
   - `zstd_compress.cu → zstd_compress.ptx` — same flags + `--maxrregcount=64`
   - PTX output placed alongside source files for `include_str!` embedding

3. **Linking** — tells Cargo to link `nvcomp` and `cudart` from standard CUDA library paths

**Release profile:** `codegen-units = 1` prevents LLVM from splitting codegen, which can cause CUDA FFI optimization bugs where function calls across codegen units get incorrectly optimized.

## concurrency model

The project uses three concurrency mechanisms:

| Mechanism | Where | Purpose |
|-----------|-------|---------|
| **std::thread** | pipeline.rs, pipeline_dual.rs | Pipeline stages (read/compress/write) |
| **tokio tasks** | multi.rs, main.rs | Async file I/O, multi-file orchestration |
| **crossbeam channels** | pipeline.rs, pipeline_dual.rs | Bounded backpressure between pipeline stages |

**Pipeline threading:**
- Single GPU: 3 worker threads + 1 stats thread
- Dual GPU: reader + GPU0 + GPU1 + writer + stats = 5 threads
- Multi-file: tokio runtime with blocking GPU task + async reader/writer tasks per file

**Synchronization:**
- `AtomicUsize` counters for lock-free stats collection
- Bounded channels (depth 16) for backpressure
- `BTreeMap` in dual-GPU writer for chunk reordering
- GPU operations are inherently serialized per CUDA stream (default stream used)

## data flow diagrams

### single-file Zstd compression (level 0)

```
Input File
    │
    ▼
┌─────────────────────┐
│ Hash (GPU BLAKE3)   │ ← Computes hash before compression
└─────────┬───────────┘
          │ hash string
          ▼
┌─────────────────────┐
│ Build tar archive   │ ← file + file.blake3 → tar stream
│ Stream 4 MB chunks  │
└─────────┬───────────┘
          │ PipelineMsg::Chunk
          ▼
┌─────────────────────┐
│ nvCOMP Zstd batch   │ ← compress_chunk_zstd() per chunk
│ compress on GPU     │
└─────────┬───────────┘
          │ CompressedChunk
          ▼
┌─────────────────────┐
│ Write NVZS file     │ ← header + size table + data
│ Write .blake3 hash  │ ← hash of compressed output
└─────────────────────┘
```

### single-file Zstd compression (levels 1-2)

```
Input File
    │
    ▼
┌─────────────────────┐
│ Read entire file     │
│ Build tar in memory  │ ← file + blake3 hash → tar bytes
└─────────┬───────────┘
          │ tar bytes
          ▼
┌─────────────────────┐
│ Split into 4 MB     │
│ streaming chunks     │
└─────────┬───────────┘
          │ per chunk:
          ▼
┌─────────────────────────────────────┐
│ compress_chunk_zstd_custom()        │
│                                     │
│  Split chunk → 64 KB sub-chunks     │
│       │                             │
│       ▼                             │
│  zstd_match_find kernel             │
│  (LZ77 + lazy/optimal matching)     │
│       │                             │
│       ▼                             │
│  zstd_encode_block kernel           │
│  (FSE encoding → Zstd frames)      │
│       │                             │
│       ▼                             │
│  Concatenated Zstd frames           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────┐
│ Write NVZS file     │ ← same format as level 0
└─────────────────────┘
```

### decompression (Zstd)

```
NVZS File
    │
    ▼
┌─────────────────────┐
│ Read magic + header  │ ← detect format
│ Parse chunk sizes    │
└─────────┬───────────┘
          │ per chunk:
          ▼
┌─────────────────────┐
│ nvCOMP Zstd batch   │ ← micro-batch (1 chunk)
│ decompress on GPU   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Reconstruct tar     │ ← write to temp .tar file
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Extract tar         │ ← recover file + .blake3
│ or detect raw file  │ ← (auto-detect tar vs raw content)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ GPU BLAKE3 hash     │ ← hash decompressed file
│ Compare to stored   │
│ Retry legacy mode   │ ← if mismatch, try old flag behavior
└─────────────────────┘
```

## error handling

- `anyhow::Result` throughout — all errors propagate with context
- CUDA errors checked after every FFI call (device set, malloc, memcpy, kernel launch, sync)
- nvCOMP status codes mapped to descriptive error messages
- GPU detection failure is non-fatal for `detect_gpus()` (returns empty vec)
- Decompression hash mismatch triggers legacy retry before final failure
- Tar extraction handles both tar-wrapped and raw content (auto-detection)

## changelog

### v0.1.0 — initial release (ca566e7)

14-module Rust codebase refactored from monolithic main.rs:
- Zstd: 4 MB streaming chunks, auto dual-GPU, tar+blake3 integrity
- Gdeflate: 64 KB batched chunks, 128 MB GPU batches, >2 GB multi-chunk (NVMC)
- BLAKE3: custom CUDA kernel, 1 KB/thread, tree reduction
- CLI: compress, compress-multi, decompress, decompress-multi, hash
- Streaming pipeline with crossbeam channels and backpressure
- Multi-file async pipeline via tokio
- Directory compression with structure preservation

### v0.1.0+fixes — bug fixes + custom Zstd kernel (c372516)

**Phase A — critical bug fixes:**
- Fix Bitcomp/Zstd mismatch in compress-multi (was using wrong nvCOMP API)
- Fix NVZS header format in multi-file writer (68B → 28B standard header)
- Add tar auto-detection in decompressor for raw vs tar-wrapped content
- Fix BLAKE3 CHUNK_START flag (was set on all blocks, now only block 0 per spec)
- Add legacy BLAKE3 mode for backward compatibility with old .nvzs files
- Properly scope unsafe blocks in cuda.rs with error checking

**Phase B — custom GPU Zstd compression (`--level` flag):**
- New `zstd_compress.cu`: LZ77 match finder with lazy matching, 14-bit hash table in shared memory, repeat offset tracking
- New `compress_zstd_custom.rs`: Rust orchestrator for custom kernel pipeline
- RLE-mode FSE encoding for uniform-symbol chunks (validated by CPU zstd)
- Raw block fallback for mixed-symbol/incompressible chunks
- Full round-trip verified: compress → nvCOMP decompress → BLAKE3 integrity
- 5.3x compression on repetitive data, safe 1x fallback on random data

**Build fixes:**
- Add `codegen-units = 1` to release profile (fixes CUDA FFI optimization bug)
- Compile `zstd_compress.cu` to PTX alongside `blake3.cu`
