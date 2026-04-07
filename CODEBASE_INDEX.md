# gpu-compressor Codebase Index

> Generated 2026-04-07. Documents every source file, structure, function, constant, and
> data flow in the project. Intended as a complete reference — no need to read source to
> understand the system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [File Tree](#2-file-tree)
3. [Module Dependency Graph](#3-module-dependency-graph)
4. [Data Flow](#4-data-flow)
5. [File-by-File Analysis](#5-file-by-file-analysis)
   - [src/main.rs](#srcmainrs)
   - [src/cli.rs](#srcclirsrs)
   - [src/constants.rs](#srcconstantsrs)
   - [src/cuda.rs](#srccudars)
   - [src/format.rs](#srcformatrs)
   - [src/compress_zstd.rs](#srccompress_zstdrs)
   - [src/compress_zstd_custom.rs](#srccompress_zstd_customrs)
   - [src/compress_lzma2.rs](#srccompress_lzma2rs)
   - [src/compress_lzma2_custom.rs](#srccompress_lzma2_customrs)
   - [src/pipeline.rs](#srcpipeliners)
   - [src/pipeline_dual.rs](#srcpipeline_dualrs)
   - [src/pipeline_lzma2.rs](#srcpipeline_lzma2rs)
   - [src/pipeline_lzma2_dual.rs](#srcpipeline_lzma2_dualrs)
   - [src/dispatch.rs](#srcdispatchrs)
   - [src/multi.rs](#srcmultirs)
   - [src/decompress.rs](#srcdecompressrs)
   - [src/tui.rs](#srctuirs)
   - [src/nvcomp_bindings.rs](#srcnvcomp_bindingsrs)
6. [CUDA Kernels](#6-cuda-kernels)
7. [File Format Specifications](#7-file-format-specifications)
8. [Build System](#8-build-system)
9. [CLI Interface](#9-cli-interface)
10. [Key Algorithms](#10-key-algorithms)
11. [Error Handling Patterns](#11-error-handling-patterns)
12. [Configuration and Constants](#12-configuration-and-constants)
13. [Concurrency Model](#13-concurrency-model)

---

## 1. Project Overview

**gpu-compressor** is a GPU-accelerated file compression/decompression tool written in
Rust with CUDA C++ kernels. It uses NVIDIA's [nvCOMP](https://developer.nvidia.com/nvcomp)
library for hardware-accelerated Zstandard (Zstd) compression, and ships two custom CUDA
kernels compiled to PTX at build time:

- `lzma2_match_find.cu` — HC4 hash chain GPU match finder for LZMA2 L1+
- `zstd_compress.cu` — Custom LZ77 match finder + (inactive) GPU FSE encoder

### Tech Stack

| Layer | Technology |
|---|---|
| Language | Rust 2021 + CUDA C++ |
| Async runtime | Tokio (multi-thread) |
| Sync threading | `std::thread` + `crossbeam-channel` |
| GPU interface (high-level) | `cudarc` crate |
| GPU interface (low-level FFI) | `cuda-runtime-sys` crate |
| GPU compression library | NVIDIA nvCOMP 3.x |
| FFI bindings generation | `bindgen` |
| CUDA compilation | `nvcc` targeting sm_86 (Ampere) |
| CLI parsing | `clap` 4.4 with derive macros |
| Error propagation | `anyhow` |
| Directory traversal | `walkdir` |

---

## 2. File Tree

```
gpu-compressor/
├── src/
│   ├── main.rs                      CLI entry point, command dispatch
│   ├── cli.rs                       Clap argument definitions
│   ├── constants.rs                 All tunable parameters + embedded PTX
│   ├── cuda.rs                      GPU detection utilities
│   ├── format.rs                    Pipeline message types
│   ├── compress_zstd.rs             nvCOMP Zstd single-chunk compression (L0)
│   ├── compress_zstd_custom.rs      GPU match finding + CPU FSE encoding (L1/L2)
│   ├── compress_lzma2.rs            liblzma FFI via dlopen (L0 + decompression)
│   ├── compress_lzma2_custom.rs     GPU HC4 match finding + Rust range coder (L1+)
│   ├── pipeline.rs                  Zstd single GPU streaming pipeline
│   ├── pipeline_dual.rs             Zstd dual GPU streaming pipeline
│   ├── pipeline_lzma2.rs            LZMA2 pipeline (multi-GPU, async, all levels)
│   ├── pipeline_lzma2_dual.rs       Legacy LZMA2 dual GPU pipeline (unused, reference)
│   ├── dispatch.rs                  Algorithm/level routing + directory handling
│   ├── multi.rs                     Multi-file async Zstd L0 compression
│   ├── decompress.rs                All decompression paths (NVZS + NVLZ)
│   ├── tui.rs                       ANSI TUI progress display
│   └── nvcomp_bindings.rs           Auto-generated nvCOMP FFI bindings
├── blake3.cu                        BLAKE3 CUDA kernel (compiles, not loaded — dead code)
├── blake3.ptx                       Compiled PTX
├── zstd_compress.cu                 Custom Zstd CUDA kernel (match find active; FSE not used)
├── zstd_compress.ptx                Compiled PTX
├── zstd_compress.cu.modified        Modified kernel variant (reference)
├── zstd_compress.cu.original        Original kernel iteration (reference)
├── zstd_compress_original.cu        Prior iteration (reference)
├── zstd_compress_prototype.cu       Earlier prototype (reference)
├── lzma2_match_find.cu              LZMA2 HC4 CUDA kernel
├── lzma2_match_find.ptx             Compiled PTX
├── build.rs                         bindgen + nvcc PTX compilation + library linking
├── wrapper.h                        nvCOMP header include for bindgen
├── Cargo.toml                       Rust manifest
├── Cargo.lock                       Locked dependencies
├── install_nvcomp.sh                nvCOMP install helper script
└── PKGBUILD                         Arch Linux package build script
```

---

## 3. Module Dependency Graph

```
                        ┌──────────┐
                        │ main.rs  │
                        └────┬─────┘
                             │
            ┌────────────────┼───────────────────────┐
            │                │                       │
       ┌────▼────┐    ┌──────▼──────┐        ┌──────▼──────┐
       │ cli.rs  │    │ dispatch.rs │        │decompress.rs│
       └─────────┘    └──────┬──────┘        └──────┬──────┘
                             │                      │
        ┌────────────────────┼──────────────┐       │
        │          │         │              │       │
  ┌─────▼────┐ ┌───▼────┐ ┌─▼─────────────┐│       │
  │pipeline. │ │pipeline│ │pipeline_lzma2 ││       │
  │  rs      │ │_dual.rs│ │   .rs         ││       │
  └─────┬────┘ └───┬────┘ └──────┬────────┘│       │
        │          │             │          │       │
  ┌─────▼──────────▼─────────────▼──────────▼───────▼──────┐
  │                       shared modules                    │
  │  compress_zstd.rs   compress_zstd_custom.rs             │
  │  compress_lzma2.rs  compress_lzma2_custom.rs            │
  │  cuda.rs  constants.rs  format.rs  tui.rs               │
  │  multi.rs  nvcomp_bindings.rs                           │
  └──────────────────────────────────────────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │    CUDA kernels    │
                   │  zstd_compress.ptx │
                   │  lzma2_match_find.ptx │
                   └────────────────────┘
```

---

## 4. Data Flow

### Zstd L0, single GPU

```
Input file
  → Reader thread (8 MB chunks, PipelineMsg::Chunk)
  → [bounded channel, 2 slots]
  → Compressor thread (compress_chunk_zstd → nvCOMP batched Zstd)
  → [bounded channel, 4 slots]
  → Writer thread (NVZS header + size table + chunk data)
```

### Zstd L0, dual GPU

```
Input file
  → Reader thread (8 MB chunks)
  → even chunks → GPU 0 thread → CompressedChunk → Writer (BTreeMap reorder)
  → odd chunks  → GPU 1 thread → CompressedChunk ↗
```

### Zstd L1/L2, single GPU

```
Input file
  → Reader (8 MB chunks)
  → Compressor thread:
      chunk → 64 KB sub-chunks
      GPU: zstd_match_find kernel (all sub-chunks in parallel)
      CPU: ZSTD_compressSequences per sub-chunk → concatenated Zstd frames
  → Writer (NVZS)
```

### LZMA2 L0 (multi-threaded liblzma)

```
Input file
  → Reader thread (dict_size_mb chunks)
  → [bounded channel, 4 slots]
  → N CPU workers (min(num_cpus, 8)), each calls compress_chunk_lzma2()
  → [bounded channel, 8 slots]
  → Writer thread (NVLZ header + size table + raw LZMA2 streams)
```

### LZMA2 L1+ (async GPU→CPU pipeline)

```
Input file
  → Reader thread (dict_size_mb chunks, 4-slot channel)
  → GPU thread(s) [1 per GPU, MPMC read channel]:
      gpu_find_matches() → sub-block jobs per chunk
  → [bounded job queue, num_cpus*4 slots]
  → N CPU encoder threads [Arc<Receiver>, min(num_cpus, 8)]:
      encode_single_sub_block() → (chunk_idx, sub_idx, total_subs, block)
  → [result channel, num_cpus*4 slots]
  → Collector thread:
      HashMap<chunk_idx, Vec<(sub_idx, block)>> → when complete → CompressedChunk
  → [bounded compressed channel, 8 slots]
  → Writer thread (BTreeMap reorder, NVLZ header + per-sub-block size table + data)
```

### Decompression (NVZS)

```
NVZS file → magic check → 28-byte header (original_size, chunk_size, num_chunks)
  → N×8 size table → per chunk:
      read compressed bytes → GPU upload → nvcompBatchedZstdDecompressAsync → download → write
```

### Decompression (NVLZ)

```
NVLZ file → magic check → 28-byte header
  → N×8 size table → per chunk:
      read compressed bytes → decompress_chunk_lzma2() → write
```

---

## 5. File-by-File Analysis

---

### src/main.rs

Entry point. `#[tokio::main] async fn main()`.

**Helpers:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `auto_compress_output_zstd` | `(&PathBuf) -> PathBuf` | Appends `.nvzs` extension |
| `auto_compress_output_lzma2` | `(&PathBuf) -> PathBuf` | Appends `.nvlz` extension |
| `auto_decompress_output` | `(&PathBuf) -> PathBuf` | Strips `.nvzs`/`.nvlz`; appends `.out` for unknowns |
| `resolve_output` | `(&PathBuf, &Option<PathBuf>, fn, bool) -> PathBuf` | Resolves output path: None→auto, dir→auto inside dir, file+single→use directly, file+multi→auto |

**Compress routing:**
- `is_lzma2 = algorithm == "lzma2"`
- Single directory input → `compress_directory()` (Zstd only; LZMA2 returns error)
- `is_lzma2` → sequential loop over `compress_file_lzma2()`
- Zstd, multi-file, L0 → `compress_multi_files_async()`
- Otherwise → sequential loop over `compress_file()`

**Decompress routing:**
- Sequential loop over `decompress_file()` for each input

---

### src/cli.rs

Clap derive macro definitions.

```rust
struct Args { command: Commands }

enum Commands {
    Compress {
        input: Vec<PathBuf>,         // -i, required, multi
        output: Option<PathBuf>,     // -o, optional
        algorithm: String,           // -a, default "zstd"
        device: i32,                 // -d, default 0
        level: u32,                  // -l, default 0
        chunk_size: usize,           // --chunk-size, default 134217728 (128 MiB)
        dict_size: u32,              // --dict-size, default 128 (MiB)
    },
    Decompress {
        input: Vec<PathBuf>,         // -i, required, multi
        output: Option<PathBuf>,     // -o, optional
        device: i32,                 // -d, default 0
    },
}
```

---

### src/constants.rs

All compile-time tunable parameters and embedded PTX.

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `ZSTD_CHUNK_SIZE` | `usize` | 8388608 (8 MB) | Zstd pipeline chunk size |
| `PTX_ZSTD_COMPRESS` | `&str` | embedded | `include_str!("../zstd_compress.ptx")` |
| `CUSTOM_ZSTD_CHUNK_SIZE` | `usize` | 65536 (64 KB) | Zstd sub-chunk for GPU match finding |
| `CUSTOM_ZSTD_SEARCH_DEPTH_LAZY` | `usize` | 16 | L1 match search depth |
| `LZMA2_CHUNK_SIZE` | `usize` | 8388608 | Reference constant (actual chunk = dict_size_mb * 1024^2) |
| `LZMA2_CUSTOM_CHUNK_SIZE` | `usize` | 65536 | LZMA2 sub-block size for GPU kernel |
| `PTX_LZMA2_MATCH_FIND` | `&str` | embedded | `include_str!("../lzma2_match_find.ptx")` |
| `LZMA2_MAX_MATCHES_PER_POS` | `usize` | 8 | Max match candidates per position |
| `LZMA2_HC4_SEARCH_DEPTH` | `u32` | 32 | GPU HC4 chain probe depth |
| `LZMA2_DICT_SIZE` | `u32` | 16777216 | (dead_code) Phase 2 dict size placeholder |
| `LZMA2_DEFAULT_PRESET` | `u32` | 6 | (dead_code) Default liblzma preset |

---

### src/cuda.rs

**`detect_gpus() -> Result<Vec<i32>>`**

Calls `cudaGetDeviceCount`, then `cudaGetDeviceProperties` for each device. Uses `cudaDeviceProp` from `nvcomp_bindings` (1008 bytes, CUDA 13.x correct) instead of `cuda-runtime-sys` (712 bytes, incorrect for CUDA 13.x — 296-byte buffer overflow). Prints device names to stderr. Returns `Vec<i32>` of device IDs.

---

### src/format.rs

```rust
pub(crate) enum PipelineMsg {
    Chunk { data: Vec<u8>, chunk_index: usize },
    Done,
}

pub(crate) struct CompressedChunk {
    pub chunks: Vec<Vec<u8>>,           // one entry per sub-chunk/sub-block
    pub chunk_index: usize,             // (#[allow(dead_code)] — used by writer for BTreeMap key)
    pub compressed_sizes: Vec<u64>,     // size of each element in chunks
}
```

`CompressedChunk.chunks` may contain:
- A single element for L0 (one raw stream per chunk)
- Multiple elements for L1+ (one per 64 KB sub-block)

---

### src/compress_zstd.rs

**`compress_chunk_zstd(chunk_data: &[u8], device_id: i32) -> Result<(Vec<Vec<u8>>, Vec<usize>)>`**

Thin wrapper around nvCOMP batched Zstd API. Flow:
1. `cudaSetDevice`
2. Allocate device memory for input chunk
3. `cudaMemcpy` host → device
4. Allocate device arrays for pointer-of-pointers (single chunk: batch size 1)
5. `nvcompBatchedZstdGetMaxCompressedSize` → allocate output buffer
6. `nvcompBatchedZstdCompressAsync` on CUDA stream 0
7. `cudaStreamSynchronize`
8. `cudaMemcpy` device → host for compressed size, then for data
9. Free all device memory
10. Return `(vec![compressed_data], vec![size])`

---

### src/compress_zstd_custom.rs

**`compress_chunk_zstd_custom(chunk_data: &[u8], device_id: i32, level: u32) -> Result<(Vec<Vec<u8>>, Vec<usize>)>`**

Search depth: L1 → `CUSTOM_ZSTD_SEARCH_DEPTH_LAZY` (16), L2 → 64, other → 1.

Flow:
1. Split `chunk_data` into 64 KB sub-chunks (`num_sub_chunks = div_ceil(total, 65536)`)
2. Load `zstd_match_find` from `PTX_ZSTD_COMPRESS` via cudarc
3. Allocate device buffers for all sub-chunks concatenated
4. Upload sub-chunk data + metadata (sizes, offsets, search_depth)
5. Launch kernel: grid `(num_sub_chunks, 1, 1)`, block `(256, 1, 1)`
6. Sync + download sequences array (`seq_data`) and per-sub-chunk counts (`seq_counts`)
7. For each sub-chunk:
   - Convert GPU sequences to `ZstdSequence { offset, lit_length, match_length, rep }`
   - Call `ZSTD_compressSequences(cctx, dst, capacity, seqs, n_seqs, src, src_size)`
   - On `ZSTD_isError`: fall back to `ZSTD_compress(src, src_size, level=1)`
8. Return `(sub_chunk_frames, sizes)`

**FFI symbols linked:** `ZSTD_createCCtx`, `ZSTD_freeCCtx`, `ZSTD_CCtx_reset`, `ZSTD_CCtx_setParameter`, `ZSTD_compressSequences`, `ZSTD_isError`, `ZSTD_getErrorName`, `ZSTD_compressBound`, `ZSTD_compress`

---

### src/compress_lzma2.rs

liblzma FFI via `dlopen("liblzma.so.5")`. Not a link-time dependency — loaded at runtime.

**Struct layouts (verified against liblzma 5.8.3 x86_64):**

```
LzmaStream: next_in, avail_in, total_in, next_out, avail_out, total_out, _rest[88]
LzmaOptionsLzma: dict_size, _pad0, preset_dict, preset_dict_size, lc, lp, pb, mode,
                 nice_len, mf, depth, _reserved[64]
LzmaFilter: id: u64, options: *mut LzmaOptionsLzma
```

**Functions loaded:**

| dlsym symbol | Rust type alias |
|---|---|
| `lzma_raw_encoder` | `FnLzmaRawEncoder` |
| `lzma_raw_decoder` | `FnLzmaRawDecoder` |
| `lzma_code` | `FnLzmaCode` |
| `lzma_end` | `FnLzmaEnd` |
| `lzma_lzma_preset` | `FnLzmaPreset` |

**`compress_chunk_lzma2(chunk_data, _dict_size, preset) -> Result<(Vec<Vec<u8>>, Vec<usize>)>`**
- `lzma_lzma_preset(&mut opts, preset.min(9))`
- `lzma_raw_encoder` → `lzma_code(LZMA_FINISH)` → `lzma_end`
- Output capacity: `len + len/8 + 1024`
- Returns `(vec![out_buf], vec![size])`
- Note: `_dict_size` parameter is passed but not used; liblzma uses the preset-configured dict size

**`decompress_chunk_lzma2(compressed, decompressed_size) -> Result<Vec<u8>>`**
- 64 MB dict window hardcoded in opts
- `lzma_raw_decoder` → `lzma_code(LZMA_FINISH)` → `lzma_end`
- Accepts `LZMA_STREAM_END` or `LZMA_OK` as success (partial completion)

---

### src/compress_lzma2_custom.rs

GPU HC4 match finding + custom Rust LZMA range coder.

**`RangeCoder` struct:**
```
range: u32, cache: u8, cache_size: u64, low: u64, output: Vec<u8>
```
Methods: `new()`, `shift_low()`, `encode_bit(&mut prob, bit)`, `encode_bit_tree(probs, num_bits, value)`, `encode_bit_tree_reverse(probs, num_bits, value)`, `flush() -> Vec<u8>`

**`MatchResults` struct (internal):**
```
match_distances: Vec<u32>, match_lengths: Vec<u32>, match_counts: Vec<u32>,
sub_block_size: usize, max_matches: usize, num_sub_blocks: usize
```

**`gpu_find_matches(data: &[u8], device_id: i32) -> Result<MatchResults>`**
- Loads `lzma2_match_find` from `PTX_LZMA2_MATCH_FIND` via cudarc
- Sub-block size = `LZMA2_CUSTOM_CHUNK_SIZE` (64 KB)
- Grid: `(num_sub_blocks, 1, 1)`, block: kernel-defined
- Passes `LZMA2_HC4_SEARCH_DEPTH`, `LZMA2_MAX_MATCHES_PER_POS`
- Downloads: distances `[n_bytes * max_matches]`, lengths `[n_bytes * max_matches]`, counts `[n_bytes]`
- Returns `MatchResults`

**`encode_single_sub_block(sub_data, distances, lengths, counts, max_matches) -> Vec<u8>`**
- Full LZMA probability model (literal, match, rep0/1/2/3, length probs)
- Match encoding edge cases: fallback to raw block on verification failure
- Returns raw LZMA2 sub-block stream

---

### src/pipeline.rs

**`compress_file_streaming_zstd(input_path, output_path, device_id, _quiet, level) -> Result<()>`**

3-thread pipeline. `PIPELINE_QUEUE_SIZE = 2`.

- **Reader thread:** reads 8 MB chunks, sends `PipelineMsg::Chunk`, then `PipelineMsg::Done`. Chunk size: `CUSTOM_ZSTD_CHUNK_SIZE` (64 KB) for level > 0, `ZSTD_CHUNK_SIZE` (8 MB) for L0.
- **Compressor thread:** `if level > 0 { compress_chunk_zstd_custom(...) } else { compress_chunk_zstd(...) }`. Sends `CompressedChunk`.
- **Writer thread:** Writes NVZS header (placeholder sizes), then chunks as received. After all chunks: seek back, write actual size table.
- **Stats thread:** Spawned separately; polls TUI counters every 200ms until `stats_done` flag.

Writer uses `BTreeMap` for ordering even in single-GPU mode (ensures deterministic output if backpressure reorders).

---

### src/pipeline_dual.rs

**`compress_file_streaming_dual_gpu(input_path, output_path, gpu0, gpu1, _quiet) -> Result<()>`**

5-thread pipeline. Zstd L0 only (no `level` parameter — always calls `compress_chunk_zstd`).

- **Reader thread:** sends even-indexed chunks to `read_tx0`, odd-indexed chunks to `read_tx1`. Sends `Done` to both.
- **GPU 0 thread:** drains `read_rx0`, calls `compress_chunk_zstd(data, gpu0)`, sends to shared `compress_tx`.
- **GPU 1 thread:** drains `read_rx1`, calls `compress_chunk_zstd(data, gpu1)`, sends to shared `compress_tx`.
- **Writer thread:** receives from shared `compress_rx`, inserts into `BTreeMap<usize, CompressedChunk>`, writes in order, writes NVZS format.
- **Stats thread:** 200ms TUI polling with `chunk_gpu0` and `chunk_gpu1` counters.

---

### src/pipeline_lzma2.rs

**`compress_file_streaming_lzma2(input_path, output_path, gpu_ids, _quiet, level, dict_size_mb) -> Result<()>`**

Handles all LZMA2 levels. Accepts `gpu_ids: &[i32]` — all detected GPUs passed in.

`pipeline_chunk_size = dict_size_mb as usize * 1024 * 1024`

`mode_label`: `"lzma2 L{level} ({N} gpu + {M} cpu, {dict}MB dict)"` for L1+; `"lzma2 preset {preset} ({M} cpu, {dict}MB dict)"` for L0.

**L0 path:**
- N workers = `min(num_cpus, 8)`, each `compress_chunk_lzma2(data, dict_size_bytes, preset)`
- Shared `read_rx` cloned across workers (MPMC)
- `compress_tx` cloned across workers, dropped after spawn
- Writer receives from `compress_rx`, writes NVLZ with chunk_size = `pipeline_chunk_size`

**L1+ path:**
- `SubBlockJob` struct: `sub_data, match_distances, match_lengths, match_counts, max_matches, chunk_index, sub_index, total_subs`
- GPU threads: one per GPU ID; each consumes `PipelineMsg::Chunk` from shared MPMC `read_rx`, calls `gpu_find_matches()`, splits into `SubBlockJob`s and sends to `job_tx`
- CPU encoders: `num_encoders = min(num_cpus, 8)`, each holds `Arc<Receiver<SubBlockJob>>`, calls `encode_single_sub_block()`
- Result channel: `(chunk_index: usize, sub_index: usize, total_subs: usize, block: Vec<u8>)`
- Collector: `HashMap<chunk_index, Vec<(sub_index, block)>>`; when `subs.len() == total_subs`, sorts by sub_index, assembles `CompressedChunk`, sends to writer
- Writer: `BTreeMap<usize, CompressedChunk>` reorder; NVLZ chunk_size field = `LZMA2_CUSTOM_CHUNK_SIZE` (64 KB); writes per-sub-block sizes into the size table

**Channel bounds:**
- read channel: 4 slots
- job queue: `num_cpus * 4` slots
- result channel: `num_cpus * 4` slots
- compressed channel: 8 slots

**Supervisor thread:** joins GPU threads → then CPU encoders → then collector. Each join in sequence; completion of GPU threads causes `job_tx` drop → CPU encoders drain; CPU encoder completion causes `result_tx` drop → collector drains.

---

### src/pipeline_lzma2_dual.rs

**Status: unused.** Earlier dual GPU design with explicit even/odd chunk interleaving. Superseded by `pipeline_lzma2.rs` which handles multi-GPU via MPMC channels. Kept for reference.

Not imported in `dispatch.rs` (commented out: `// LZMA2 dual GPU pipeline superseded by multi-threaded single pipeline`).

---

### src/dispatch.rs

**`expand_directory_inputs(inputs, outputs, extension) -> Result<(Vec<PathBuf>, Vec<PathBuf>)>`**
- If single directory input: WalkDir traversal → `(input_file, output_dir/relative_path + extension)`
- Otherwise: pass through as-is

**`compress_file(input, output, device_id, level) -> Result<()>`**
- Calls `compress_file_impl(input, output, device_id, false, level)`

**`compress_file_impl(input, output, device_id, quiet, level) -> Result<()>`**
- Auto-generates output filename if `output.is_dir()`
- `detect_gpus()` → `available_gpus`
- `available_gpus.len() >= 2 && level == 0` → `compress_file_streaming_dual_gpu(gpu0, gpu1)`
- otherwise → `compress_file_streaming_zstd(device_id, level)`

**`compress_directory(input_dir, output_dir, device_id, level) -> Result<()>`**
- WalkDir, collects `(input, file_size, output)` tuples
- Sequential `compress_file_impl()` per file
- Prints summary: files, input GB, output GB, ratio, time, throughput

**`compress_file_lzma2(input, output, _device_id, level, dict_size_mb) -> Result<()>`**
- Auto-generates output if `output.is_dir()`
- `detect_gpus()` → `available_gpus`
- Calls `compress_file_streaming_lzma2(input, output, &available_gpus, false, level, dict_size_mb)`
- Note: all available GPUs are passed in; pipeline handles multi-GPU internally

---

### src/multi.rs

**`compress_multi_files_async(inputs, outputs, device_id, chunk_size) -> Result<()>`** (async)

Tokio async multi-file Zstd L0 pipeline. Only invoked for Zstd L0 with 2+ inputs.

**`compress_buffer_zstd_multi(chunks: &[Vec<u8>], device_id: i32) -> Result<Vec<Vec<u8>>>`**
- nvCOMP batched Zstd: takes a slice of chunks, uploads all to GPU, compresses as a single batch
- Returns compressed chunk data (one `Vec<u8>` per input chunk)

**`compress_file_async(input, output, device_id, chunk_size) -> Result<()>`** (async)
- Reads file in `chunk_size` blocks using `tokio::fs`
- Calls blocking `compress_buffer_zstd_multi` via `tokio::task::spawn_blocking`
- Writes NVZS format

---

### src/decompress.rs

**`decompress_file(input_path, output_path, device_id) -> Result<()>`**
- Reads 4-byte magic
- `b"NVZS"` → `decompress_file_zstd_streaming()`
- `b"NVLZ"` → `decompress_file_lzma2_streaming()`
- Other → error

**`decompress_file_zstd_streaming(input, output, device_id) -> Result<()>`**
1. Re-open file, read 28-byte header (skipping magic at [0:4])
2. `original_size (u64 LE) = header[4:12]`
3. `chunk_size (u64 LE) = header[12:20]`
4. `num_chunks (u64 LE) = header[20:28]`
5. Read `num_chunks * 8` bytes size table
6. Per chunk: read `size_table[i]` bytes → GPU upload → `nvcompBatchedZstdDecompressAsync` → sync → download → write
7. Last chunk: decompressed size = `original_size - (num_chunks-1) * chunk_size`

**`decompress_file_lzma2_streaming(input, output) -> Result<()>`**
1. Same header parsing
2. Per chunk: read `size_table[i]` bytes → `decompress_chunk_lzma2(compressed, decompressed_size)` → write
3. Last chunk size computed same way

---

### src/tui.rs

**`TuiState` struct fields:**
```
file_size: u64
bytes_read: Arc<AtomicU64>
bytes_compressed: Arc<AtomicU64>
bytes_written: Arc<AtomicU64>
chunk_gpu0: Arc<AtomicU64>
chunk_gpu1: Option<Arc<AtomicU64>>
gpu_devices: Vec<i32>
mode_label: String
input_path: String
output_path: String
start_time: Instant
quiet: bool
num_lines: usize          // for cursor-up on redraw
first_draw: bool          // suppresses cursor-up on initial render
```

**`TuiState::new(...) -> Self`** — constructor, sets `start_time = Instant::now()`, `first_draw = true`.

**`TuiState::draw(&mut self)`** — if `!quiet`: cursor-up `num_lines` (unless `first_draw`), render stats lines via `eprint!`. Tracks `num_lines` for next redraw.

**`TuiState::print_summary(&mut self, total_compressed: u64, is_dual: bool)`** — prints final stats: ratio, throughput, elapsed time.

---

### src/nvcomp_bindings.rs

Auto-generated by bindgen from `wrapper.h`. Contains:
- `cudaDeviceProp` struct (1008 bytes, correct for CUDA 13.x)
- nvCOMP API types and functions for Zstd:
  - `nvcompBatchedZstdCompressOpts_t`
  - `nvcompBatchedZstdGetMaxCompressedSize`
  - `nvcompBatchedZstdCompressAsync`
  - `nvcompBatchedZstdDecompressAsync`
  - `nvcompBatchedZstdGetDecompressSizeAsync`
  - `nvcompStatus_t` enum

---

## 6. CUDA Kernels

### zstd_compress.cu

**`zstd_match_find`** — active; loaded by `compress_zstd_custom.rs`
- Grid: `(num_sub_chunks, 1, 1)`, Block: `(256, 1, 1)`
- Thread 0 in each block performs sequential scan; other threads idle
- Shared memory: 16K-entry hash table (14-bit multiplicative hash on 4-byte windows)
- Lazy matching: configurable `search_depth` — at each position, checks if next position gives a longer match
- Tracks repeat offsets per RFC 8878
- Output per sub-chunk: sequence array `(lit_len u32, match_len_minus3 u32, offset u32)` + `seq_count`
- Build flags: `-ptx -O3 --use_fast_math --gpu-architecture=sm_86 --maxrregcount=64`

**`zstd_compress_raw`** — not loaded; wraps sub-chunks as raw Zstd frames without compression

**`zstd_encode_block`** — not loaded; GPU FSE encoder with known bug in FSE_Compressed mode; CPU path used instead

### lzma2_match_find.cu

**`lzma2_match_find`** — active; loaded by `compress_lzma2_custom.rs`
- Grid: `(num_sub_blocks, 1, 1)`
- HC4 (hash + chain depth 4) match finder
- Parameters: `LZMA2_HC4_SEARCH_DEPTH` (32) probes per position
- Output per position: up to `LZMA2_MAX_MATCHES_PER_POS` (8) candidate matches (offset, length)
- Build flags: `-ptx -O3 --use_fast_math --gpu-architecture=sm_86`

### blake3.cu

Dead code. Compiles to `blake3.ptx`, no Rust module loads or launches it. Remnant from a prior version (tar + BLAKE3 integrity was removed during Gdeflate/BLAKE3 stripping).

---

## 7. File Format Specifications

### NVZS (Zstd)

```
Offset  Size  Field
0       4     Magic: b"NVZS"
4       8     original_size: u64 LE (raw input bytes, no tar wrapping)
12      8     chunk_size: u64 LE (8388608 = 8 MB)
20      8     num_chunks: u64 LE
28      N×8   compressed_sizes: [u64 LE; N] (one per chunk)
28+N×8  Σ     chunk data:
                L0: single Zstd frame per chunk (nvCOMP output)
                L1/L2: multiple concatenated Zstd frames per chunk (one per 64 KB sub-chunk)
```

nvCOMP decompressor handles multi-frame input natively.

### NVLZ (LZMA2)

```
Offset  Size  Field
0       4     Magic: b"NVLZ"
4       8     original_size: u64 LE (raw input bytes)
12      8     chunk_size: u64 LE
                L0: dict_size_mb * 1048576
                L1+: 65536 (LZMA2_CUSTOM_CHUNK_SIZE)
20      8     num_chunks: u64 LE
28      N×8   compressed_sizes: [u64 LE; N]
                L0: size of each pipeline chunk's raw LZMA2 stream
                L1+: size of each 64 KB sub-block's encoded stream
28+N×8  Σ     chunk data:
                L0: single raw LZMA2 stream per pipeline chunk
                L1+: concatenated 64 KB sub-block LZMA2 streams
```

Both formats: `num_chunks = div_ceil(original_size, chunk_size)`. Last chunk decompressed size = `original_size - (num_chunks-1) * chunk_size`.

---

## 8. Build System

**build.rs** — three steps:

**1. Bindgen:**
```rust
bindgen::Builder::default()
    .header("wrapper.h")
    .clang_arg("-I/opt/cuda/include")
    .clang_arg("-I/usr/local/cuda/include")
    .generate()
    .write_to_file("src/nvcomp_bindings.rs")
```

**2. CUDA kernel compilation:**
```
nvcc -ptx -O3 --use_fast_math --gpu-architecture=sm_86 --maxrregcount=64
     zstd_compress.cu -o zstd_compress.ptx

nvcc -ptx -O3 --use_fast_math --gpu-architecture=sm_86
     lzma2_match_find.cu -o lzma2_match_find.ptx

nvcc -ptx ... blake3.cu -o blake3.ptx   [dead code]
```

PTX files placed alongside `.cu` sources for `include_str!` embedding.

**3. Linking:**
```
cargo:rustc-link-lib=nvcomp
cargo:rustc-link-lib=cudart
cargo:rustc-link-lib=zstd
```
liblzma is NOT linked here — loaded via dlopen at runtime.

**Cargo.toml profile:**
```toml
[profile.release]
codegen-units = 1   # prevents LLVM codegen split → avoids CUDA FFI optimization bugs
```

---

## 9. CLI Interface

```
gpu-compressor compress -i <INPUT>... [-o OUTPUT] [-a zstd|lzma2] [-d DEVICE] [-l LEVEL]
                        [--chunk-size BYTES] [--dict-size MB]

gpu-compressor decompress -i <INPUT>... [-o OUTPUT] [-d DEVICE]
```

| Flag | Default | Scope |
|------|---------|-------|
| `-i` | (required) | both |
| `-o` | auto | both |
| `-a` | `zstd` | compress |
| `-d` | `0` | both |
| `-l` | `0` | compress |
| `--chunk-size` | 134217728 | compress, Zstd L0 multi-file only |
| `--dict-size` | 128 | compress, LZMA2 only |

Output naming: see CLI.md.

---

## 10. Key Algorithms

### Zstd L0 (nvCOMP)

NVIDIA's nvCOMP library handles all match finding and entropy coding. Black-box GPU implementation. Produces standard Zstd frames. Single chunk per GPU call.

### Zstd L1/L2 (GPU LZ77 + CPU FSE)

1. **GPU:** `zstd_match_find` kernel — 14-bit hash table in shared memory, lazy matching with configurable depth. Per thread: sequential scan of 64 KB, greedy/lazy match selection.
2. **CPU:** `ZSTD_compressSequences` — takes GPU-found LZ77 sequences, performs FSE entropy coding via libzstd, produces a valid Zstd frame.

### LZMA2 L0 (liblzma multi-threaded)

liblzma raw encoder with configurable preset (0-9). Pipeline chunk size equals dictionary window (`dict_size_mb`). 8 parallel worker threads each process independent chunks; no cross-chunk dictionary sharing at the pipeline level.

### LZMA2 L1+ (GPU HC4 + CPU range coding)

1. **GPU:** `lzma2_match_find` kernel — HC4 hash chain (hash + chain depth 4), 32 probes per position, up to 8 match candidates per position. Parallelized across 64 KB sub-blocks.
2. **CPU:** Custom Rust LZMA range coder — full LZMA probability model, arithmetic coding per IGE Pavlov's LZMA SDK reference. Match selection from GPU candidates; edge-case fallback to raw blocks.

Trade-off: GPU match finding is fast (ms), CPU range coding is slow (seconds per chunk). Async pipeline lets GPU run ahead; bounded job queue prevents unbounded memory use.

---

## 11. Error Handling Patterns

- `anyhow::Result` throughout; `.context("...")` for FFI calls
- CUDA errors: every FFI call checked; non-success → `Err(anyhow::anyhow!(...))`
- nvCOMP status codes: checked after async operations
- Thread panics: `.join().map_err(|_| anyhow::anyhow!("... thread panicked"))??`
- Zstd L1/L2: `ZSTD_isError(ret) != 0` → `ZSTD_compress(level=1)` fallback per sub-chunk
- LZMA2 range coder: match encoding edge cases → raw block fallback
- liblzma: decompression accepts `LZMA_STREAM_END` or `LZMA_OK` (handles partial completion)

---

## 12. Configuration and Constants

All in `src/constants.rs`. To tune:

- **Zstd chunk size:** `ZSTD_CHUNK_SIZE` (default 8 MB). Larger = fewer GPU launches, more VRAM.
- **Zstd sub-chunk size:** `CUSTOM_ZSTD_CHUNK_SIZE` (default 64 KB). Must match kernel assumption.
- **Zstd L1 search depth:** `CUSTOM_ZSTD_SEARCH_DEPTH_LAZY` (default 16). Higher = better ratio, slower.
- **LZMA2 sub-block size:** `LZMA2_CUSTOM_CHUNK_SIZE` (default 64 KB). Must match kernel.
- **LZMA2 HC4 depth:** `LZMA2_HC4_SEARCH_DEPTH` (default 32). Higher = better ratio, slower GPU.
- **LZMA2 matches per position:** `LZMA2_MAX_MATCHES_PER_POS` (default 8).
- **GPU architecture:** `sm_86` in `build.rs`. Change for other GPU generations.

Runtime parameters (CLI flags):
- `--dict-size`: LZMA2 pipeline chunk size / dictionary window (default 128 MB)
- `--chunk-size`: Zstd multi-file chunk size (default 128 MiB, only affects `multi.rs` path)

---

## 13. Concurrency Model

| Mechanism | Location | Purpose |
|-----------|----------|---------|
| `std::thread` | pipeline*.rs | All pipeline stages |
| `tokio::task::spawn_blocking` | multi.rs | GPU work from async context |
| `crossbeam_channel::bounded` | pipeline*.rs | Inter-stage backpressure queues |
| `Arc<crossbeam_channel::Receiver>` (MPMC) | pipeline_lzma2.rs | Shared CPU encoder pool drain |
| `Arc<AtomicU64>` | tui.rs, pipelines | Lock-free stats for TUI |
| `BTreeMap<usize, CompressedChunk>` | pipeline writers | In-order chunk reassembly |
| `HashMap<usize, Vec<(usize, Vec<u8>)>>` | collector (L1+) | Sub-block reassembly |

**Thread counts by mode:**

| Mode | Threads |
|------|---------|
| Zstd L0 single GPU | reader + compressor + writer + stats = 4 |
| Zstd L0 dual GPU | reader + GPU0 + GPU1 + writer + stats = 5 |
| Zstd L1/L2 | reader + compressor + writer + stats = 4 |
| LZMA2 L0 | reader + N workers (≤8) + supervisor + writer + stats = N+4 |
| LZMA2 L1+ single GPU | reader + GPU + N encoders (≤8) + collector + supervisor + writer + stats = N+6 |
| LZMA2 L1+ dual GPU | reader + 2×GPU + N encoders (≤8) + collector + supervisor + writer + stats = N+7 |

**Backpressure chain (LZMA2 L1+):**
1. Reader blocked by 4-slot read channel (GPU can't keep up → reader stalls)
2. GPU blocked by N*4-slot job queue (CPU encoders drain)
3. CPU encoders blocked by N*4-slot result channel (collector drains)
4. Collector blocked by 8-slot compressed channel (writer drains)

This prevents unbounded memory accumulation across all pipeline stages.
