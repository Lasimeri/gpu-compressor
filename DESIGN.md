# gpu-compressor design document

## project overview

gpu-compressor is a GPU-accelerated file compression tool built in Rust. It supports two algorithms — Zstd and LZMA2 — each with multiple levels that progressively shift work from library routines to custom GPU kernels and custom entropy coders. It handles single and dual GPU operation, streaming pipelines with crossbeam backpressure, multi-file async processing, and directory-recursive compression.

**Language:** Rust (2021 edition) + CUDA C++
**Runtime:** Tokio (async I/O, multi-file orchestration) + raw threads (pipeline stages)
**GPU interface:** cudarc (high-level kernel launch) + cuda-runtime-sys (FFI) + nvCOMP (compression API)
**Build:** Cargo + bindgen (FFI generation) + nvcc (CUDA PTX compilation)

## module dependency graph

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
  ┌─────▼────┐ ┌───▼────┐ ┌─▼────────────┐ │       │
  │pipeline. │ │pipeline│ │pipeline_lzma2│ │       │
  │  rs      │ │_dual.rs│ │   .rs        │ │       │
  └─────┬────┘ └───┬────┘ └──────┬───────┘ │       │
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

Note: `pipeline_lzma2_dual.rs` exists but is not used — `pipeline_lzma2.rs` handles multi-GPU via MPMC channels internally.

## module descriptions

### entry point

**main.rs** — Parses CLI via clap, routes to the appropriate handler. Uses `tokio::main`. Two subcommands: `Compress` and `Decompress`. Algorithm selection (`zstd` vs `lzma2`) is a string argument checked with `algorithm == "lzma2"`. `dict_size` flag present in CLI struct and passed through to `compress_file_lzma2`. Multi-file Zstd L0 routes to the async `compress_multi_files_async` path; all other cases iterate sequentially.

### CLI

**cli.rs** — Clap derive macro definitions.

- `Commands` enum: `Compress` and `Decompress`
- `Compress` fields: `input: Vec<PathBuf>`, `output: Option<PathBuf>`, `algorithm: String` (default `"zstd"`), `device: i32` (default `0`), `level: u32` (default `0`), `chunk_size: usize` (default 128 MiB), `dict_size: u32` (default `128` MiB)
- `Decompress` fields: `input: Vec<PathBuf>`, `output: Option<PathBuf>`, `device: i32` (default `0`)
- Output naming helpers: `auto_compress_output_zstd` appends `.nvzs`; `auto_compress_output_lzma2` appends `.nvlz`; `auto_decompress_output` strips `.nvzs`/`.nvlz`, appends `.out` for unknowns

### compression dispatch

**dispatch.rs** — Central routing.

```
compress_file(input, output, device, level)      [Zstd]
  │
  ├─ level == 0 AND gpus >= 2 → compress_file_streaming_dual_gpu()
  └─ otherwise → compress_file_streaming_zstd()
      (level > 0 forces single GPU regardless of GPU count)

compress_file_lzma2(input, output, device, level, dict_size_mb) [LZMA2]
  │
  └─ compress_file_streaming_lzma2(gpu_ids=all_available, ...)
      (single pipeline handles both single and multi-GPU via MPMC internals)

compress_directory(input_dir, output_dir, device, level)
  → WalkDir traverse + compress_file_impl() per file [Zstd only]
```

`expand_directory_inputs()` — WalkDir traversal mapping directory input to individual files, preserving relative paths, appending the appropriate extension.

`compress_file_impl()` — Resolves output path, detects GPU count, routes to streaming pipeline.

### streaming pipelines

#### pipeline.rs — single GPU Zstd

3-thread structure: reader → compressor → writer, connected by bounded crossbeam channels.

Level routing in the compressor thread: `if level > 0 { compress_chunk_zstd_custom(...) } else { compress_chunk_zstd(...) }`.

#### pipeline_dual.rs — dual GPU Zstd (L0 only)

```
Reader → even chunks → GPU 0 thread → CompressedChunk → Writer (BTreeMap reorder)
       → odd chunks  → GPU 1 thread → CompressedChunk ↗
```

Reader distributes by `chunk_index % 2` to per-GPU channels. Writer uses `BTreeMap<usize, CompressedChunk>` to reorder before writing.

#### pipeline_lzma2.rs — LZMA2 (single + multi-GPU, all levels)

Handles both L0 and L1+ internally. Accepts `gpu_ids: &[i32]` — all detected GPUs are passed in.

**L0 path:**
```
Reader thread → [bounded channel, 4 slots] → N CPU workers (liblzma, min(num_cpus, 8))
  → [bounded compressed channel, 8 slots] → Writer thread
```

Each worker pulls whole chunks from the read channel and calls `compress_chunk_lzma2()`. Pipeline chunk size = `dict_size_mb * 1024 * 1024` (dictionary window equals chunk size).

**L1+ path (async GPU→CPU pipeline):**
```
Reader thread
  → [bounded channel, 4 slots]
  → GPU match finding thread(s) [1 per GPU, each consumes PipelineMsg::Chunk via MPMC]
      → sub-block jobs [bounded job queue, num_cpus*4 slots]
  → N CPU encoder threads [min(num_cpus, 8), each grabs SubBlockJobs independently]
      → (chunk_index, sub_index, total_subs, encoded_block) [result channel]
  → Collector thread [HashMap pending, reassembles sub-blocks into CompressedChunks]
      → [bounded compressed channel, 8 slots]
  → Writer thread [BTreeMap reorder, NVLZ header + size table + data]
```

Key property: GPU runs ahead of CPU encoders. The bounded job queue (N*4 slots) provides backpressure. CPU encoders are truly parallel — each grabs one 64 KB sub-block at a time from the shared Arc<Receiver>. Multiple GPUs each have their own thread consuming from the shared MPMC read channel (crossbeam allows multiple receivers).

Writer uses `BTreeMap<usize, CompressedChunk>` to reorder out-of-order completions. For L1+, NVLZ `chunk_size` field is set to `LZMA2_CUSTOM_CHUNK_SIZE` (64 KB); for L0, it's the pipeline chunk size.

#### pipeline_lzma2_dual.rs — legacy, unused

Earlier dual GPU design. Superseded by `pipeline_lzma2.rs` which handles multi-GPU natively via MPMC. Kept for reference only.

### compression engines

#### compress_zstd.rs — nvCOMP Zstd (L0)

Thin wrapper around nvCOMP's batched Zstd API for a single chunk. Allocates device memory, calls `nvcompBatchedZstdCompressAsync`, synchronizes, downloads result.

#### compress_zstd_custom.rs — GPU match finding + CPU FSE (L1/L2)

Two-pass per chunk:

1. **GPU pass:** `zstd_match_find` kernel across all 64 KB sub-chunks in parallel. LZ77 with 14-bit shared-memory hash table (16K entries). Search depth 16 (L1) or 64 (L2). Output: sequence array (lit_len, match_len_minus3, offset) + seq_counts.

2. **CPU pass:** `ZSTD_compressSequences` (libzstd FFI) per sub-chunk. Converts GPU sequences to `ZstdSequence` structs, produces a valid Zstd frame. Falls back to `ZSTD_compress(level=1)` on error.

Sub-chunk frames are concatenated; nvCOMP's decompressor handles multi-frame input natively.

#### compress_lzma2.rs — liblzma FFI (L0)

liblzma loaded at runtime via `dlopen("liblzma.so.5")`. Resolves `lzma_raw_encoder`, `lzma_code`, `lzma_end`, `lzma_lzma_preset`. Each chunk encoded as a raw LZMA2 stream at the specified preset (0-9). Also provides `decompress_chunk_lzma2` used by `decompress.rs`. The `_dict_size` parameter is currently unused (passed through but liblzma uses the preset-configured dict size).

#### compress_lzma2_custom.rs — GPU HC4 + range coder (L1+)

Two functions: `gpu_find_matches` and `encode_single_sub_block`.

`gpu_find_matches(data, device_id)` → `MatchResults`:
- Loads `lzma2_match_find` PTX kernel via cudarc
- Grid: one block per 64 KB sub-block; HC4 (hash chain depth 4) match finding
- LZMA2_HC4_SEARCH_DEPTH = 32 probes per position, LZMA2_MAX_MATCHES_PER_POS = 8 candidates
- Returns: `match_distances: Vec<u32>`, `match_lengths: Vec<u32>`, `match_counts: Vec<u32>`, `sub_block_size`, `max_matches`, `num_sub_blocks`

`encode_single_sub_block(sub_data, distances, lengths, counts, max_matches)` → `Vec<u8>`:
- Custom Rust LZMA range coder (full LZMA probability model)
- `RangeCoder` struct: range/cache/cache_size/low arithmetic encoding
- `encode_bit`, `encode_bit_tree`, `encode_bit_tree_reverse`
- Match encoding edge cases handled with verification fallback to raw blocks

### multi-file compression

**multi.rs** — Tokio async pipeline for concurrent Zstd L0 compression.
- Max 2 files concurrently
- Reader tasks: async `tokio::fs`
- GPU compressor: blocking task
- Per-file writer tasks receive via dedicated channels
- Only used for Zstd L0 with 2+ input files

### decompression

**decompress.rs** — Format detection by 4-byte magic:

| Magic | Handler | Strategy |
|-------|---------|----------|
| `NVZS` | `decompress_file_zstd_streaming()` | nvCOMP batched Zstd, per-chunk GPU decompress |
| `NVLZ` | `decompress_file_lzma2_streaming()` | liblzma raw decoder per chunk |

Both read the 28-byte header (re-reading from position 0 after magic check), read the N×8 size table, then process chunks sequentially.

**Zstd decompression:** chunk → upload to GPU → `nvcompBatchedZstdDecompressAsync` → sync → download → write

**LZMA2 decompression:** chunk → `decompress_chunk_lzma2()` (liblzma raw decoder, 64 MB dict window) → write

No tar wrapping, no integrity verification. Output is raw decompressed bytes.

### GPU utilities

**cuda.rs** — `detect_gpus()` calls `cudaGetDeviceCount` and `cudaGetDeviceProperties`. Uses `cudaDeviceProp` from `nvcomp_bindings` rather than `cuda-runtime-sys` to avoid the 296-byte size mismatch bug in `cuda-runtime-sys 0.3.0-alpha.1` on CUDA 13.x. Returns `Vec<i32>` of device IDs and prints device names to stderr.

**nvcomp_bindings.rs** — Auto-generated via bindgen from `wrapper.h`. Includes nvCOMP API functions for Zstd. The generated `cudaDeviceProp` struct has the correct CUDA 13.x size (1008 bytes).

### shared types and constants

**format.rs:**
- `PipelineMsg::Chunk { data: Vec<u8>, chunk_index: usize }` — chunk to compress
- `PipelineMsg::Done` — end-of-input sentinel
- `CompressedChunk { chunks: Vec<Vec<u8>>, chunk_index: usize, compressed_sizes: Vec<u64> }` — output

**constants.rs:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `ZSTD_CHUNK_SIZE` | 8 MB | Zstd pipeline chunk size |
| `CUSTOM_ZSTD_CHUNK_SIZE` | 64 KB | Zstd sub-chunk for GPU match finding |
| `CUSTOM_ZSTD_SEARCH_DEPTH_LAZY` | 16 | L1 match search depth |
| `LZMA2_CHUNK_SIZE` | 8 MB | (reference; actual chunk size = dict_size_mb * 1024 * 1024) |
| `LZMA2_CUSTOM_CHUNK_SIZE` | 64 KB | LZMA2 sub-block for GPU HC4 |
| `LZMA2_HC4_SEARCH_DEPTH` | 32 | GPU HC4 chain probe depth |
| `LZMA2_MAX_MATCHES_PER_POS` | 8 | Max match candidates per position |
| `LZMA2_DICT_SIZE` | 16 MB | (dead_code, unused) |
| `LZMA2_DEFAULT_PRESET` | 6 | (dead_code, unused) |
| `PTX_ZSTD_COMPRESS` | embedded | `include_str!("../zstd_compress.ptx")` |
| `PTX_LZMA2_MATCH_FIND` | embedded | `include_str!("../lzma2_match_find.ptx")` |

**tui.rs** — `TuiState` struct with `Arc<AtomicU64>` counters (bytes_read, bytes_compressed, bytes_written, chunk_gpu0, chunk_gpu1 optional). Renders via ANSI escape codes; cursor-up on redraw to overwrite previous lines. `quiet` flag suppresses output. `print_summary()` called at end of compression.

## CUDA kernel design

### zstd_compress.cu — Zstd match finding

**`zstd_match_find`** — active, loaded by `compress_zstd_custom.rs`
- Grid: one block per 64 KB sub-chunk; Block: 256 threads (thread 0 does sequential scan, others idle)
- Hash table: 16K entries, 14-bit multiplicative hash on 4-byte windows, in shared memory
- Lazy matching: at each position, checks if next position yields a longer match (configurable `search_depth`)
- Tracks repeat offsets per RFC 8878
- Output per sub-chunk: sequence array (lit_len u32, match_len_minus3 u32, offset u32) + seq_count

**`zstd_compress_raw`** and **`zstd_encode_block`** — present in source, not loaded:
- `zstd_compress_raw`: wraps sub-chunks as raw Zstd frames (no compression)
- `zstd_encode_block`: GPU FSE encoder; has a known bug in FSE_Compressed mode — CPU path via `ZSTD_compressSequences` is used instead

### lzma2_match_find.cu — LZMA2 HC4 match finding

- Grid: one block per 64 KB sub-block
- HC4 (hash + chain depth 4) match finder
- `LZMA2_HC4_SEARCH_DEPTH` probes per position
- Output: per-position match candidates (offset, length pairs), up to `LZMA2_MAX_MATCHES_PER_POS`
- Feeds `encode_single_sub_block` in `compress_lzma2_custom.rs`

### blake3.cu — dead code

Compiles to `blake3.ptx`, no Rust code loads or launches it. Remains from a prior version.

## file format specifications

### NVZS format (Zstd)

```
┌────────────────────────────────────────────┐
│ Header (28 bytes)                          │
│   [0:4]   Magic: "NVZS"                   │
│   [4:12]  Original size (u64 LE)           │
│   [12:20] Chunk size (u64 LE, 8388608)     │
│   [20:28] Number of chunks N (u64 LE)      │
├────────────────────────────────────────────┤
│ Chunk Size Table (N × 8 bytes)             │
│   Compressed size of chunk i (u64 LE each) │
├────────────────────────────────────────────┤
│ Compressed Data                            │
│   Chunk 0: one Zstd frame (L0) or          │
│            multiple concatenated frames    │
│            (one per 64 KB sub-chunk, L1/L2)│
│   Chunk 1 ... Chunk N-1                    │
└────────────────────────────────────────────┘
```

### NVLZ format (LZMA2)

```
┌────────────────────────────────────────────┐
│ Header (28 bytes)                          │
│   [0:4]   Magic: "NVLZ"                   │
│   [4:12]  Original size (u64 LE)           │
│   [12:20] Chunk size (u64 LE)              │
│            L0: dict_size_mb × 1048576      │
│            L1+: 65536 (sub-block size)     │
│   [20:28] Number of chunks N (u64 LE)      │
├────────────────────────────────────────────┤
│ Chunk Size Table (N × 8 bytes)             │
├────────────────────────────────────────────┤
│ Compressed Data                            │
│   L0: each chunk = single raw LZMA2 stream │
│   L1+: each chunk = concatenated 64KB      │
│         sub-block streams                  │
└────────────────────────────────────────────┘
```

The size table for L1+ stores per-sub-block sizes (not per pipeline-chunk sizes). The writer iterates `CompressedChunk.chunks` (one entry per sub-block) and writes each size individually.

## build system

**build.rs** performs three operations:

1. **Bindgen** — parses `wrapper.h` → generates `nvcomp_bindings.rs`. Include paths: `/opt/cuda/include`, `/usr/local/cuda/include`. Generated `cudaDeviceProp` is 1008 bytes (CUDA 13.x correct).

2. **CUDA compilation:**
   - `zstd_compress.cu → zstd_compress.ptx` — flags: `-ptx -O3 --use_fast_math --gpu-architecture=sm_86 --maxrregcount=64`
   - `lzma2_match_find.cu → lzma2_match_find.ptx` — flags: `-ptx -O3 --use_fast_math --gpu-architecture=sm_86`
   - `blake3.cu → blake3.ptx` — compiled but dead code
   - PTX placed alongside source files for `include_str!` embedding

3. **Linking** — `libnvcomp`, `libcudart`, `libzstd`. liblzma is loaded at runtime via dlopen (not linked here).

**Release profile:** `codegen-units = 1` prevents LLVM from splitting codegen, which can cause CUDA FFI optimization bugs.

## concurrency model

| Mechanism | Where | Purpose |
|-----------|-------|---------|
| `std::thread` | pipeline*.rs | Pipeline stages |
| `tokio` tasks | multi.rs, main.rs | Async file I/O, multi-file |
| `crossbeam-channel` bounded | pipeline*.rs | Backpressure between stages |
| `Arc<Receiver>` (shared MPMC) | pipeline_lzma2.rs L1+ | CPU encoder pool sub-block drain |
| `Arc<AtomicU64>` | tui.rs, pipeline*.rs | Lock-free stats for TUI |
| `BTreeMap<usize, CompressedChunk>` | writers | Chunk reordering |
| `HashMap<usize, Vec<(usize, Vec<u8>)>>` | collector | Sub-block reassembly |

Thread counts:
- Zstd single GPU: reader + compressor + writer + TUI stats = 4
- Zstd dual GPU: reader + GPU0 + GPU1 + writer + TUI stats = 5
- LZMA2 L0: reader + N CPU workers (up to 8) + writer + TUI stats = N+3
- LZMA2 L1+: reader + G GPU threads + N CPU encoders + collector + supervisor + writer + TUI stats = G+N+4

## error handling

- `anyhow::Result` throughout; errors propagate with context strings
- CUDA errors checked after every FFI call
- nvCOMP status codes mapped to error messages
- Zstd L1/L2: `ZSTD_compressSequences` failure falls back to `ZSTD_compress(level=1)` per sub-chunk
- LZMA2 custom range coder: match encoding edge cases fall back to raw blocks
- LZMA2 decompression: liblzma error codes surfaced as `anyhow::Error`
- Thread join errors propagated as `anyhow::anyhow!("... panicked")`

## known issues and workarounds

| Issue | Workaround |
|-------|-----------|
| `cuda-runtime-sys 0.3.0-alpha.1` wrong `cudaDeviceProp` size (712 vs 1008 bytes) | Use `cudaDeviceProp` from `nvcomp_bindings` in `cuda.rs` |
| `zstd_encode_block` GPU FSE kernel has FSE_Compressed mode bug | Not loaded; CPU `ZSTD_compressSequences` used instead |
| LZMA2 custom range coder match encoding edge cases | Verification fallback to raw blocks |
| LZMA2 L1 ratio slightly worse than L0 | 64KB sub-block independence prevents cross-block dictionary matching |

## changelog

### v0.1.0 — initial release

- Zstd: nvCOMP batched API, 4 MB streaming chunks, auto dual-GPU, streaming pipeline
- Gdeflate: nvCOMP batched API, 64 KB chunks, 128 MB batches
- BLAKE3: custom CUDA kernel, integrity embedded in tar
- CLI: compress, compress-multi, decompress, decompress-multi, hash subcommands

### current state (2026-04-07)

- Added LZMA2: L0 (liblzma multi-threaded), L1+ (async GPU→buffer→CPU pipeline)
- LZMA2 L1+: both GPUs run match finding, feed shared CPU encoder pool via MPMC
- Added `--dict-size` flag: LZMA2 pipeline chunk size (dictionary window), stored in host RAM
- Added NVLZ container format
- Removed Gdeflate (NVGD, NVMC formats gone)
- Removed tar wrapping and BLAKE3 integrity
- Removed hash, compress-multi, decompress-multi CLI subcommands
- Simplified to two subcommands: compress, decompress
- Chunk size 4 MB → 8 MB
- Custom Zstd L1/L2: switched from GPU FSE encoder to CPU ZSTD_compressSequences
- Added TUI ANSI progress display
- `pipeline_lzma2_dual.rs` superseded by multi-GPU-aware `pipeline_lzma2.rs`
