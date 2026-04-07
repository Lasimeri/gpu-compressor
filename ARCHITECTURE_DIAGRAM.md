# gpu-compressor — Architecture Diagram

> Current state as of 2026-04-05. Reflects actual code behavior, not DESIGN.md aspirations.

---

## 1. Top-Level Entry & Dispatch

```
                            ┌──────────────────────┐
                            │   gpu-compressor CLI  │
                            │      (main.rs)        │
                            │   clap parse → Args   │
                            └──────────┬───────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                  ┌─────────────┐            ┌─────────────┐
                  │  Compress   │            │ Decompress  │
                  │  Commands   │            │  Commands   │
                  └──────┬──────┘            └──────┬──────┘
                         │                          │
              ┌──────────┼──────────┐               │
              ▼          ▼          ▼               ▼
         Single Dir  2+ Files   1 File        Sequential
         (walkdir)   (Zstd L0)  (or L>0)      decompress_file()
              │          │          │          per input
              ▼          ▼          ▼               │
         compress_   compress_  compress_     ┌─────┴──────┐
         directory() multi_     file()        │ Magic Byte │
                     files_                   │  Detection │
                     async()                  └─────┬──────┘
                                          ┌────┬────┴────┐
                                          ▼    ▼         ▼
                                        NVZS  NVGD     NVMC
```

---

## 2. Compression Routing (dispatch.rs → compress_file_impl)

```
                        compress_file_impl()
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              Algorithm::Zstd        Algorithm::Gdeflate
                    │                       │
             detect_gpus()          ┌───────┴───────┐
                    │               ▼               ▼
          ┌─────────┴────────┐   ≤ 2GB           > 2GB
          ▼                  ▼     │               │
    2+ GPUs && L=0      1 GPU     ▼               ▼
          │            or L>0   compress_       compress_large_
          ▼                │    buffer_         file_in_chunks()
    pipeline_dual.rs       ▼    gdeflate()     ──────────────
    ────────────────  pipeline.rs  │            Write NVMC hdr
    5 threads:        ────────── Write NVGD     Stream 128MB →
    Reader            4 threads: to file        compress_buffer()
    GPU0 (even)       Reader                    per chunk →
    GPU1 (odd)        GPU                       append NVGD
    Writer            Writer
    Stats             Stats
```

---

## 3. Zstd Level 0 — Single GPU Pipeline (pipeline.rs)

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Single GPU Pipeline                              │
    │                                                                         │
    │  ┌──────────┐  bounded(2)  ┌──────────┐  bounded(4)  ┌──────────┐     │
    │  │  Reader   │────────────▶│   GPU    │────────────▶│  Writer   │     │
    │  │  Thread   │ PipelineMsg │  Thread  │ Compressed  │  Thread   │     │
    │  └──────────┘  ::Chunk     └──────────┘  Chunk      └──────────┘     │
    │       │         (8 MB)          │                         │            │
    │  BufReader                      │                   BTreeMap reorder   │
    │  16 MB buf             ┌────────┴────────┐         NVZS header write  │
    │  reads ZSTD_           ▼                 ▼         placeholder sizes   │
    │  CHUNK_SIZE      Level 0:          Level 1/2:      seek-back fix-up   │
    │  per chunk       compress_chunk_   compress_chunk_  BufWriter 16MB    │
    │                  zstd()            zstd_custom()                      │
    │                  (nvCOMP batch     (GPU match find                    │
    │                   of 1)             + CPU FSE)                        │
    │                                                                       │
    │  ┌──────────┐                                                         │
    │  │  Stats   │  200ms loop, reads Arc<AtomicU64> counters              │
    │  │  Thread  │  drives TUI (tui.rs) — progress bar, throughput         │
    │  └──────────┘                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Zstd Level 0 — Dual GPU Pipeline (pipeline_dual.rs)

```
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                         Dual GPU Pipeline                                 │
    │                                                                           │
    │  ┌──────────┐                                                             │
    │  │  Reader   │──┬── even chunks ──▶ bounded(2) ──▶ ┌────────┐            │
    │  │  Thread   │  │                                   │  GPU0  │──┐         │
    │  │          │  │                                   │ Thread │  │         │
    │  │ BufReader │  │                                   └────────┘  │         │
    │  │  16 MB   │  │                                                │         │
    │  └──────────┘  │                                   bounded(4)   │         │
    │                │                                   (shared)     │         │
    │                │                                       │        │         │
    │                │                                       ▼        │         │
    │                │                                  ┌──────────┐  │         │
    │                │                                  │  Writer   │◀─┘         │
    │                │                                  │  Thread   │◀─┐         │
    │                │                                  │ BTreeMap  │  │         │
    │                │                                  │  reorder  │  │         │
    │                │                                  └──────────┘  │         │
    │                │                                                │         │
    │                └── odd chunks  ──▶ bounded(2) ──▶ ┌────────┐   │         │
    │                                                   │  GPU1  │───┘         │
    │                                                   │ Thread │             │
    │                chunk_index % 2                     └────────┘             │
    │                                                                           │
    │  ┌──────────┐  Stats: bytes_read, bytes_compressed, bytes_written,       │
    │  │  Stats   │         chunk_gpu0, chunk_gpu1 (AtomicU64)                 │
    │  └──────────┘                                                             │
    └───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Custom Zstd Kernel Path (Levels 1-2)

```
    8 MB chunk from pipeline
            │
            ▼
    ┌─────────────────────────────────────────────────────────┐
    │  compress_chunk_zstd_custom()  [compress_zstd_custom.rs]│
    │                                                         │
    │  1. Split into N × 64 KB sub-chunks                     │
    │  2. Upload entire chunk to GPU via cudarc                │
    │  3. Allocate GPU buffers: sequences, literals, counts   │
    │                                                         │
    │  ┌──────────────────── GPU ──────────────────────┐      │
    │  │  zstd_match_find kernel                       │      │
    │  │  grid=(N sub-chunks), block=(256 threads)     │      │
    │  │                                               │      │
    │  │  Per sub-chunk (thread 0 only):               │      │
    │  │  ┌──────────────────────────────────┐         │      │
    │  │  │  __shared__ hash_table[16384]    │         │      │
    │  │  │  (32 KB shared mem, 16-bit pos)  │         │      │
    │  │  │                                  │         │      │
    │  │  │  while pos + 3 < chunk_size:     │         │      │
    │  │  │    h = HASH4(src+pos)            │         │      │
    │  │  │    check hash match              │         │      │
    │  │  │    check 3 repeat offsets        │         │      │
    │  │  │    lazy match at pos+1           │         │      │
    │  │  │    emit sequence or advance      │         │      │
    │  │  └──────────────────────────────────┘         │      │
    │  │                                               │      │
    │  │  Output → Sequence[], literals[], counts      │      │
    │  └───────────────────────────────────────────────┘      │
    │                        │                                 │
    │                   dtoh copy                              │
    │                        │                                 │
    │  ┌──────────────────── CPU ──────────────────────┐      │
    │  │  Per sub-chunk:                               │      │
    │  │  Convert GPU sequences → ZstdSequence[]       │      │
    │  │  ZSTD_compressSequences() via libzstd FFI     │      │
    │  │                                               │      │
    │  │  On error → fallback: ZSTD_compress(level=1)  │      │
    │  └───────────────────────────────────────────────┘      │
    │                                                         │
    │  Output: Vec<Vec<u8>> — one Zstd frame per sub-chunk    │
    └─────────────────────────────────────────────────────────┘
```

---

## 6. Multi-File Async Pipeline (multi.rs)

```
    ┌──────────────────────────────────────────────────────────────┐
    │  compress_multi_files_async()   [Tokio runtime]              │
    │  Processes files in batches of MAX_CONCURRENT_FILES = 2      │
    │                                                              │
    │  Per batch:                                                  │
    │                                                              │
    │  ┌─────────┐  ┌─────────┐                                   │
    │  │ Reader0 │  │ Reader1 │   tokio::spawn (async)            │
    │  │ AsyncRd │  │ AsyncRd │   128 MB chunks                   │
    │  └────┬────┘  └────┬────┘                                   │
    │       │            │                                         │
    │       └──────┬─────┘                                         │
    │              ▼                                                │
    │     mpsc::channel(32)                                        │
    │     (file_idx, chunk_data, chunk_idx)                        │
    │              │                                                │
    │              ▼                                                │
    │  ┌──────────────────────────────┐                            │
    │  │  GPU Compressor Task         │  spawn_blocking            │
    │  │  Accumulates 9 chunks        │                            │
    │  │  compress_buffer_zstd_multi()│  nvCOMP batch (all 9)      │
    │  │  Routes to per-file writers  │                            │
    │  └──────────┬───────────────────┘                            │
    │       ┌─────┴──────┐                                         │
    │       ▼            ▼                                         │
    │  ┌─────────┐  ┌─────────┐                                   │
    │  │ Writer0 │  │ Writer1 │   tokio::spawn (async)            │
    │  │  NVZS   │  │  NVZS   │   header → data → seek-back      │
    │  └─────────┘  └─────────┘                                   │
    └──────────────────────────────────────────────────────────────┘
```

---

## 7. Decompression Paths

```
    decompress_file()
          │
     Read 4-byte magic
          │
    ┌─────┼──────────────────┐
    ▼     ▼                  ▼
  "NVZS" "NVGD"           "NVMC"
    │     │                  │
    ▼     ▼                  ▼
  Streaming  All-at-once   Multi-chunk wrapper
  per-chunk  GPU batch     ─────────────────
  ──────────  ──────────   Read all to memory
  Read hdr   Read entire   For each NVGD sub-archive:
  Read N     file to mem     locate magic
  comp sizes Parse hdr       parse size
  Per chunk: Upload all      write to temp file
   upload    chunks →        decompress_file(temp)
   nvcomp    nvcomp          append to output
   Zstd      Gdeflate       rm temps
   Decomp    DecompAsync
   Async     Download all
   download  Write output
   write
   free
```

---

## 8. NVZS File Format (Binary Layout)

```
    ┌──────────────────────────────────────────────────┐
    │  Bytes 0-3:   Magic "NVZS"                       │
    │  Bytes 4-11:  original_size (u64 LE)             │
    │  Bytes 12-19: chunk_size (u64 LE)                │
    │               └─ 8 MB (L0) or 64 KB (L1/L2)     │
    │  Bytes 20-27: num_chunks (u64 LE)                │
    │               └─ N = ceil(orig / chunk_size)     │
    ├──────────────────────────────────────────────────┤
    │  Bytes 28 .. 28+N*8:                             │
    │    compressed_sizes[0] (u64 LE)                  │
    │    compressed_sizes[1] (u64 LE)                  │
    │    ...                                           │
    │    compressed_sizes[N-1] (u64 LE)                │
    ├──────────────────────────────────────────────────┤
    │  Compressed chunk data (sequential):             │
    │                                                  │
    │  Level 0: One nvCOMP Zstd frame per slot         │
    │                                                  │
    │  Level 1/2: M concatenated Zstd frames per slot  │
    │  (M = ceil(8MB / 64KB) = 128 sub-chunks each)   │
    │  Each frame is independently RFC 8878 compliant  │
    └──────────────────────────────────────────────────┘
```

---

## 9. Module Dependency Graph

```
                              main.rs
                    ┌──────────┼──────────────────┐
                    ▼          ▼                   ▼
                 cli.rs    dispatch.rs          multi.rs
                            │  │  │               │
                 ┌──────────┘  │  └────────┐      │
                 ▼             ▼            ▼      │
           pipeline.rs   pipeline_dual.rs  cuda.rs │
              │  │  │       │  │  │                │
              │  │  └───┐   │  │  └───┐            │
              ▼  ▼      ▼   ▼  ▼      ▼            ▼
    compress_  compress_  format.rs  tui.rs   nvcomp_bindings.rs
    zstd.rs    zstd_                               ▲
       │       custom.rs                           │
       │         │                          cuda-runtime-sys
       ▼         ▼                          (crate, direct FFI)
    compress_  constants.rs
    gdeflate.rs  │
       │         ├── PTX_ZSTD_COMPRESS (embedded)
       ▼         └── All tunable params
    nvcomp_bindings.rs
       │
       ▼
    bindgen-generated FFI
    (nvcomp.h, gdeflate.h, zstd.h)

    ─── CUDA Kernels (loaded at runtime via cudarc) ───

    blake3.cu ──compile──▶ blake3.ptx ──embed──▶ binary
      (NOT loaded by current Rust code — blake3.rs missing)

    zstd_compress.cu ──compile──▶ zstd_compress.ptx ──embed──▶ binary
      └── zstd_match_find   (USED by compress_zstd_custom.rs)
      └── zstd_compress_raw (NOT loaded by current code)
      └── zstd_encode_block (NOT loaded by current code)
```

---

## 10. Concurrency Architecture

```
    ┌────────────────────────────────────────────────────────────┐
    │                    Tokio Runtime (main)                     │
    │                                                            │
    │  main() async                                              │
    │    ├── compress_directory() ──▶ spawns std::thread pipelines│
    │    ├── compress_multi_files_async() ──▶ tokio tasks         │
    │    │     ├── Reader tasks (tokio::spawn, async I/O)        │
    │    │     ├── GPU task (spawn_blocking, sync CUDA)          │
    │    │     └── Writer tasks (tokio::spawn, async I/O)        │
    │    └── compress_file() ──▶ spawns std::thread pipelines    │
    │                                                            │
    └────────────────────────────────────────────────────────────┘

    Pipeline Threads (std::thread):
    ┌────────┐  crossbeam   ┌────────┐  crossbeam   ┌────────┐
    │ Reader │──bounded(2)─▶│  GPU   │──bounded(4)─▶│ Writer │
    └────────┘              └────────┘              └────────┘
         ▲                       │
         │                  cudaSetDevice()
    BufReader               cudaMalloc/Memcpy
    16 MB                   nvcomp*Async
                            cudaDeviceSync
                            cudaFree

    Stats Thread:
    ┌────────┐
    │ Stats  │  reads Arc<AtomicU64> every 200ms
    │  TUI   │  renders ANSI progress bar to stderr
    └────────┘

    Backpressure:
    Writer slow → compress channel full → GPU blocks
    → read channel full → Reader blocks → disk I/O throttles
```

---

## 11. Known Discrepancies (Code vs Docs)

```
    ┌─────────────────────────────────────────────────────────────┐
    │  DESIGN.md / INSTRUCTIONS.md describe:                      │
    │    ✗ Tar wrapping of compressed output                      │
    │    ✗ BLAKE3 integrity hash embedded in archive              │
    │    ✗ Automatic verification on decompress                   │
    │    ✗ GPU BLAKE3 hash command                                │
    │                                                             │
    │  Current code reality:                                      │
    │    ✗ No tar wrapping — raw compressed data in NVZS          │
    │    ✗ No BLAKE3 hash stored — blake3.rs module missing       │
    │    ✗ No verification on decompress                          │
    │    ✗ blake3.cu compiles to PTX, embedded in binary,         │
    │      but no Rust code loads or calls it                     │
    │                                                             │
    │  zstd_compress.cu contains 3 kernels:                       │
    │    ✓ zstd_match_find  — ACTIVE (loaded by Rust)             │
    │    ✗ zstd_compress_raw — EXISTS but NOT loaded              │
    │    ✗ zstd_encode_block — EXISTS but NOT loaded              │
    │      (FSE encoding done on CPU via libzstd instead)         │
    │                                                             │
    │  DESIGN.md says chunk_size = 4 MB                           │
    │  constants.rs says ZSTD_CHUNK_SIZE = 8 MB                   │
    └─────────────────────────────────────────────────────────────┘
```

---

## 12. Data Size Flow (Typical Single-File Zstd L0)

```
    Input File (any size)
         │
    ┌────┴────┐
    │ 8 MB    │  ZSTD_CHUNK_SIZE = 8,388,608
    │ chunks  │  read via BufReader(16 MB)
    └────┬────┘
         │
    ┌────┴────────────────────────┐
    │ Per chunk on GPU:           │
    │   cudaMalloc: ~8 MB input   │
    │   cudaMalloc: ~9 MB output  │  (nvCOMP max output bound)
    │   cudaMalloc: temp buffer   │  (nvCOMP temp size)
    │   cudaMalloc: metadata      │  (~200 bytes)
    │   Total VRAM per chunk:     │
    │     ~17-20 MB transient     │
    └────┬────────────────────────┘
         │
    Compressed chunk (variable size)
    Typically 30-70% of input for compressible data
         │
    ┌────┴────────────────────────┐
    │ Written to NVZS:            │
    │   28 byte header            │
    │   N × 8 byte size table     │
    │   Compressed data           │
    └─────────────────────────────┘
```
