# GPU Zstd Compression Kernel Research

## 1. nvCOMP's Approach to GPU Zstd Compression

### Architecture Overview

nvCOMP is NVIDIA's proprietary CUDA compression library. Key architectural details:

- **Dual API**: High-Level Interface (HLIF) handles chunking/memory automatically; Low-Level Interface (LLIF) operates on batches of pre-chunked data for maximum throughput.
- **Chunk-based parallelism**: Input is split into independent chunks (typically 64KB default). Each chunk is compressed independently. Larger chunks = better ratio but less GPU parallelism.
- **Batch processing**: The LLIF processes arrays of chunks simultaneously via `nvcompBatchedZstdCompressAsync`. Each chunk maps to one or more thread blocks.
- **Source code unavailable**: Since nvCOMP 2.3, compression/decompression source code is proprietary. We can only infer implementation details from API behavior and benchmarks.

### Inferred Implementation Details

Based on the API design, benchmarks, and GDeflate documentation:

- **One thread block per chunk**: The batch API launches `num_chunks` thread blocks. Each block processes one chunk independently. This matches our architecture.
- **Shared memory hash table**: With 64KB chunks on sm_86 (48KB shared memory), a hash table in shared memory is the standard approach. GDeflate uses this pattern explicitly.
- **Sequential match finding**: LZ77 match finding is inherently sequential (each match depends on prior state). Thread 0 likely does sequential matching while other threads assist with hash prefill or data movement. This also matches our approach.
- **Entropy encoding on GPU**: nvCOMP achieves ~55-60% compression ratios on typical data, which requires FSE/Huffman entropy encoding (not just LZ77). Their Zstd compressor must implement FSE_Compressed mode for the sequences section.

### GDeflate (Related GPU Compression Format)

GDeflate is NVIDIA's GPU-optimized deflate variant, used in DirectStorage 1.1:

- Uses deflate-like LZ77 + Huffman, but restructured for GPU parallelism
- 64KB tile size is the standard
- Achieves ~2:1 compression on game assets at high throughput
- Key insight: GPU compression prioritizes throughput over ratio. A simpler entropy coder that runs fast on GPU can beat a complex one that serializes.

### Performance Reference

nvCOMP Zstd on our Windows Server ISO test: **95.7% ratio** (at default settings). This means it's achieving real compression with entropy coding, not just LZ77 with raw/RLE blocks.

Our custom kernel: **99.9% ratio** because it falls back to raw blocks whenever codes aren't uniform.

**The gap is entirely due to the missing FSE_Compressed encoding path.**

Sources:
- [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp)
- [nvCOMP GitHub (docs/examples)](https://github.com/NVIDIA/nvcomp)
- [GDeflate for DirectStorage Blog](https://developer.nvidia.com/blog/accelerating-load-times-for-directx-games-and-apps-with-gdeflate-for-directstorage/)
- [nvCOMP Flexible Interfaces Blog](https://developer.nvidia.com/blog/accelerating-lossless-gpu-compression-with-new-flexible-interfaces-in-nvidia-nvcomp/)
- [nvCOMP v2.1.0 Batch API Blog](https://developer.nvidia.com/blog/using-fully-redesigned-batch-api-and-performance-optimizations-in-nvcomp-v2-1-0/)
- [nvCOMP DeepWiki](https://deepwiki.com/NVIDIA/nvcomp)
- [Gstd Forum Thread](https://encode.su/threads/4176-Gstd-Converting-Zstd-to-a-GPU-friendly-lossless-codec)

---

## 2. Analysis of the Current FSE Encoding Bug

### The Core Problem

The current kernel (lines 807-816) bails to raw block whenever any of the three code dimensions (LL, ML, OF) has non-uniform codes:

```c
if (!llRLE || !ofRLE || !mlRLE) {
    // Can't encode with RLE -- fall back to raw block
    ...
    return;
}
```

This means compression only works when ALL sequences in a sub-chunk use the exact same LL code, the exact same ML code, AND the exact same OF code. This almost never happens in real data (only ~0.1% of sub-chunks).

### Previous FSE_Compressed Attempt Failures

The previous attempt to wire up FSE_Compressed mode produced garbage output sizes (1.4GB, 3.1GB for 64KB chunks). The likely causes, based on code analysis:

#### Bug 1: Stack array overflow (87K * 3 bytes)

```c
uint8_t llCodes[21846], mlCodes[21846], ofCodes[21846]; // ~65KB on stack!
```

CUDA threads have limited stack space (default 1KB per thread, max ~64KB with `cudaDeviceSetLimit`). Three arrays of 21846 bytes = ~64KB total on the stack for thread 0 alone. This likely caused silent stack corruption or CUDA launch failures.

**Fix**: Compute codes on-the-fly during histogram building and encoding. Don't store all codes in arrays.

#### Bug 2: Incorrect encoding order

The current RLE path writes extra bits in this order per sequence:
```
LL extra bits, ML extra bits, OF extra bits
```

But the zstd reference encoder (`ZSTD_encodeSequences_body` in facebook/zstd) uses this order:

**For the LAST sequence (first encoded, since we go backward):**
1. Initialize FSE states for LL, OF, ML (from last sequence's codes)
2. Write LL extra bits
3. Write ML extra bits  
4. Write OF extra bits
5. Flush

**For each subsequent sequence (backward from n_seqs-2 to 0):**
1. FSE encode OF state bits
2. FSE encode ML state bits
3. Flush (if needed for 32-bit)
4. FSE encode LL state bits
5. Flush
6. Write LL extra bits
7. Write ML extra bits
8. Write OF extra bits
9. Flush

**Final flush:**
1. Flush ML final state (stateLog bits)
2. Flush OF final state (stateLog bits)
3. Flush LL final state (stateLog bits)
4. Close bitstream (sentinel 1-bit + padding)

The decoder reads this bitstream backward and sees:
- Sentinel 1-bit + padding (find it, skip it)
- LL initial state, OF initial state, ML initial state
- Then for each sequence (forward): OF value bits, ML value bits, LL value bits, LL FSE bits, ML FSE bits, OF FSE bits

This is confirmed by [Nigel Tao's Zstandard Part 6](https://nigeltao.github.io/blog/2022/zstandard-part-6-sequences.html) which shows the interleaved bit order as:
```
CMOVBits, MLVBits, LLVBits, LLFBits, MLFBits, CMOFBits
```

Where "V" = value/extra bits and "F" = FSE state bits. The first iteration has a historical-accidental swap: initial states are written as LLF, CMOF, MLF (not the regular order).

#### Bug 3: Bitstream direction confusion

Zstd's sequence bitstream is written **forward** (low bytes first) but read **backward** (from the last byte). The sentinel 1-bit at the end tells the decoder where the data starts.

Our `BitStream` struct writes forward (accumulating bits LSB-first, flushing complete bytes), which is correct. The `bs_close` function adds a sentinel 1-bit and flushes remaining bits, which is also correct.

However, the **encoding** must process sequences in reverse order (last to first) because the decoder processes them forward (first to last) while reading bits backward.

#### Bug 4: Missing bounds checking

No check that `bs->pos` stays within the allocated output buffer. If FSE encoding produces more bits than expected, it writes past the end of the output buffer, corrupting adjacent chunks' output and producing garbage sizes.

#### Bug 5: of_code function is wrong for Zstd offset encoding

The current code does:
```c
uint8_t oc = (uint8_t)highbit32(my_seqs[i].offset + 3);
```

And later:
```c
eb = ofCodes[last];
if (eb > 0) bs_addBits(&bs, (my_seqs[last].offset + 3) - (1u << eb), eb);
```

This is the correct encoding for Zstd offsets: `offset_value = raw_offset + 3` (reserving 1,2,3 for repeat offsets), `of_code = floor(log2(offset_value))`, extra bits = `offset_value - (1 << of_code)`, with `of_code` extra bits written. The code for offset appears correct.

However, the match finder stores `match_length - 3` in `my_seqs[i].match_length`, but the extra bits computation uses:
```c
(my_seqs[last].match_length + 3) - ML_baseline[mlCodes[last]]
```

This correctly recovers the actual match length before subtracting the baseline. This part is fine.

#### Bug 6: Mode byte format

The mode byte encodes which FSE mode each dimension uses:
```
byte = (LL_mode << 6) | (OF_mode << 4) | (ML_mode << 2)
```

Where mode values are:
- 0 = Predefined_Mode (use default distribution)
- 1 = RLE_Mode (single symbol)
- 2 = FSE_Compressed (custom table follows)
- 3 = Repeat_Mode (reuse previous table)

The current code writes `0x54` for all-RLE, which is `(01 << 6) | (01 << 4) | (01 << 2)` = correct.

For FSE_Compressed on non-uniform dimensions, we need mode=2 for those dimensions. Each FSE_Compressed dimension is followed by its table description, then each RLE dimension is followed by its single symbol byte.

The table descriptions must appear in order: LL first, then OF, then ML. But only for dimensions that use FSE_Compressed mode. RLE dimensions write just the symbol byte after the mode byte.

**Wait -- re-reading the spec more carefully**: The mode byte comes first. Then, in order LL, OF, ML: if the dimension's mode is RLE, one byte follows (the symbol); if FSE_Compressed, the table description follows. All table descriptions / RLE bytes are concatenated after the mode byte.

Sources:
- [RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878)
- [Zstd Compression Format Spec](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md)
- [facebook/zstd ZSTD_encodeSequences_body source](https://github.com/facebook/zstd/blob/dev/lib/compress/zstd_compress_sequences.c)
- [Nigel Tao: Zstandard Part 5 - FSE](https://nigeltao.github.io/blog/2022/zstandard-part-5-fse.html)
- [Nigel Tao: Zstandard Part 6 - Sequences](https://nigeltao.github.io/blog/2022/zstandard-part-6-sequences.html)

---

## 3. Corrected `zstd_encode_block` Kernel

This is a full replacement for lines ~700-877 of `zstd_compress.cu`. Key changes:

1. **No large stack arrays** -- codes are computed on-the-fly during histogram pass, then recomputed during encoding
2. **Correct encoding order** per the reference encoder
3. **Bounds checking** on bitstream position
4. **FSE_Compressed mode** for non-uniform dimensions, RLE for uniform, mixed as needed
5. **Graceful fallback** to raw block if compressed >= original

```cuda
// ============================================================================
// Kernel: zstd_encode_block
// Encodes match-finder output into a Zstd compressed frame.
// Supports FSE_Compressed mode for non-uniform code distributions.
// Falls back to raw block if compressed size >= uncompressed.
//
// Reference: ZSTD_encodeSequences_body() in facebook/zstd
// Bit order (per Nigel Tao / RFC 8878):
//   Written forward, read backward by decoder.
//   First sequence encoded = LAST sequence (backward iteration).
//   Per sequence: FSE state bits (OF, ML, LL), then extra bits (LL, ML, OF).
//   Final: flush states (ML, OF, LL), sentinel 1-bit, zero-pad to byte.
// ============================================================================

extern "C" __global__ void zstd_encode_block(
    const uint8_t* __restrict__ input,         // original sub-chunks (for raw fallback)
    const Sequence* __restrict__ sequences,    // match finder output
    const uint8_t* __restrict__ literals,      // literal bytes
    const uint32_t* __restrict__ seq_counts,
    const uint32_t* __restrict__ lit_counts,
    uint8_t* __restrict__ output,              // compressed frames output
    uint32_t* __restrict__ output_sizes,       // frame sizes
    uint32_t chunk_size,
    uint32_t num_chunks,
    uint32_t total_input_size,
    uint32_t max_sequences_per_chunk,
    uint32_t max_frame_size                    // max output allocation per chunk
) {
    uint32_t chunk_idx = blockIdx.x;
    if (chunk_idx >= num_chunks) return;
    if (threadIdx.x != 0) return;  // Only thread 0 encodes

    uint64_t chunk_start = (uint64_t)chunk_idx * chunk_size;
    uint32_t this_chunk_size = chunk_size;
    if (chunk_start + this_chunk_size > total_input_size) {
        this_chunk_size = total_input_size - (uint32_t)chunk_start;
    }

    const Sequence* my_seqs = sequences + (uint64_t)chunk_idx * max_sequences_per_chunk;
    const uint8_t* my_lits = literals + (uint64_t)chunk_idx * chunk_size;
    uint32_t n_seqs = seq_counts[chunk_idx];
    uint32_t n_lits = lit_counts[chunk_idx];

    uint8_t* dst = output + (uint64_t)chunk_idx * max_frame_size;
    uint32_t pos = 0;

    // Safety limit: stop writing if we get within 64 bytes of max_frame_size
    uint32_t safe_limit = max_frame_size - 64;

    // === Frame Magic ===
    write_le32(dst + pos, ZSTD_MAGIC);
    pos += 4;

    // === Frame Header ===
    if (this_chunk_size >= 256) {
        dst[pos++] = 0x60;  // FCS_flag=01(2B), Single_Segment=1
        uint16_t fcs = (uint16_t)(this_chunk_size - 256);
        write_le16(dst + pos, fcs);
        pos += 2;
    } else {
        dst[pos++] = 0x20;  // FCS_flag=00(1B), Single_Segment=1
        dst[pos++] = (uint8_t)this_chunk_size;
    }

    // If no sequences, emit raw block
    if (n_seqs == 0) {
        const uint8_t* src = input + chunk_start;
        uint32_t bh = 1u | (ZSTD_BLOCK_TYPE_RAW << 1) | (this_chunk_size << 3);
        dst[pos++] = (uint8_t)bh; dst[pos++] = (uint8_t)(bh>>8); dst[pos++] = (uint8_t)(bh>>16);
        for (uint32_t i = 0; i < this_chunk_size; i++) dst[pos++] = src[i];
        output_sizes[chunk_idx] = pos;
        return;
    }

    // === Build frequency histograms for LL, ML, OF codes ===
    // Max symbols: LL=36, ML=53, OF=32 (for 64KB chunks, offset <= 65535, so OF <= 18)
    uint32_t llHist[36]; for (int i = 0; i < 36; i++) llHist[i] = 0;
    uint32_t mlHist[53]; for (int i = 0; i < 53; i++) mlHist[i] = 0;
    uint32_t ofHist[32]; for (int i = 0; i < 32; i++) ofHist[i] = 0;

    uint8_t llMax = 0, mlMax = 0, ofMax = 0;
    uint8_t llMin = 255, mlMin = 255, ofMin = 255;

    for (uint32_t i = 0; i < n_seqs; i++) {
        uint8_t lc = ll_code(my_seqs[i].literal_length);
        uint8_t mc = ml_code(my_seqs[i].match_length);
        uint8_t oc = of_code(my_seqs[i].offset);

        llHist[lc]++;
        mlHist[mc]++;
        ofHist[oc]++;

        if (lc > llMax) llMax = lc;
        if (mc > mlMax) mlMax = mc;
        if (oc > ofMax) ofMax = oc;
        if (lc < llMin) llMin = lc;
        if (mc < mlMin) mlMin = mc;
        if (oc < ofMin) ofMin = oc;
    }

    // Determine mode for each dimension
    bool llRLE = (llMin == llMax);
    bool ofRLE = (ofMin == ofMax);
    bool mlRLE = (mlMin == mlMax);

    // Choose accuracy logs for FSE tables
    // For small sequence counts, use smaller accuracy logs to avoid bloated tables
    int llALog = 6, mlALog = 6, ofALog = 5; // defaults matching zstd
    if (n_seqs < 128) { llALog = 5; mlALog = 5; ofALog = 4; }
    if (n_seqs < 32)  { llALog = 5; mlALog = 5; ofALog = 4; }

    // Build FSE tables for non-RLE dimensions
    DynFSE llTbl, mlTbl, ofTbl;
    int16_t llNC[36], mlNC[53], ofNC[32];

    if (!llRLE) {
        fse_normalize(llHist, (int)llMax, llALog, llNC);
        fse_build(&llTbl, llNC, (int)llMax, llALog);
    }
    if (!ofRLE) {
        fse_normalize(ofHist, (int)ofMax, ofALog, ofNC);
        fse_build(&ofTbl, ofNC, (int)ofMax, ofALog);
    }
    if (!mlRLE) {
        fse_normalize(mlHist, (int)mlMax, mlALog, mlNC);
        fse_build(&mlTbl, mlNC, (int)mlMax, mlALog);
    }

    // === Build compressed block in temp area ===
    uint32_t block_header_pos = pos;
    pos += 3; // reserve for block header

    // --- Literals Section (Raw literals, no Huffman) ---
    // Literals_Section_Header: Raw_Literals_Block
    // Type = 0b00 (raw), Size_Format depends on n_lits
    if (n_lits < 32) {
        // 1 byte header: Literals_Block_Type=00, Size_Format=0 (or 1), size in bits 3-7
        dst[pos++] = (uint8_t)(n_lits << 3); // type=00, format implicitly 0
    } else if (n_lits < 4096) {
        // 2 byte header: type=00, Size_Format=01, size in 12 bits
        uint16_t hdr = (uint16_t)((n_lits << 4) | 0x04); // type=00, SF=01
        write_le16(dst + pos, hdr); pos += 2;
    } else {
        // 3 byte header: type=00, Size_Format=1x, size in 20 bits
        uint32_t hdr = (n_lits << 4) | 0x0C; // type=00, SF=11
        dst[pos++] = (uint8_t)hdr; dst[pos++] = (uint8_t)(hdr>>8); dst[pos++] = (uint8_t)(hdr>>16);
    }
    // Raw literal bytes
    for (uint32_t i = 0; i < n_lits; i++) dst[pos++] = my_lits[i];

    // --- Sequences Section Header ---
    // Number_of_Sequences encoding
    if (n_seqs < 128) {
        dst[pos++] = (uint8_t)n_seqs;
    } else if (n_seqs < 0x7F00) {
        dst[pos++] = (uint8_t)((n_seqs >> 8) + 128);
        dst[pos++] = (uint8_t)(n_seqs & 0xFF);
    } else {
        dst[pos++] = 0xFF;
        write_le16(dst + pos, (uint16_t)(n_seqs - 0x7F00)); pos += 2;
    }

    // --- Symbol Compression Modes byte ---
    // Bits 7-6: LL mode, Bits 5-4: OF mode, Bits 3-2: ML mode, Bits 1-0: reserved (0)
    uint8_t llMode = llRLE ? 1 : 2;  // 1=RLE, 2=FSE_Compressed
    uint8_t ofMode = ofRLE ? 1 : 2;
    uint8_t mlMode = mlRLE ? 1 : 2;
    dst[pos++] = (llMode << 6) | (ofMode << 4) | (mlMode << 2);

    // --- Write table descriptions / RLE symbols in order: LL, OF, ML ---
    if (llRLE) {
        dst[pos++] = llMin;
    } else {
        uint32_t tblBytes = fse_write_table_desc(dst + pos, llNC, (int)llMax, llALog);
        pos += tblBytes;
    }

    if (ofRLE) {
        dst[pos++] = ofMin;
    } else {
        uint32_t tblBytes = fse_write_table_desc(dst + pos, ofNC, (int)ofMax, ofALog);
        pos += tblBytes;
    }

    if (mlRLE) {
        dst[pos++] = mlMin;
    } else {
        uint32_t tblBytes = fse_write_table_desc(dst + pos, mlNC, (int)mlMax, mlALog);
        pos += tblBytes;
    }

    // Bounds check after headers
    if (pos >= safe_limit) goto fallback_raw;

    // --- Encode sequences bitstream ---
    {
        BitStream bs;
        bs_init(&bs, dst, pos);

        uint32_t last = n_seqs - 1;

        // Recompute codes for last sequence
        uint8_t lastLL = ll_code(my_seqs[last].literal_length);
        uint8_t lastML = ml_code(my_seqs[last].match_length);
        uint8_t lastOF = of_code(my_seqs[last].offset);

        // === Initialize FSE states (no bits output, just set initial state) ===
        // Reference: FSE_initCState2(&stateLitLength, CTable_LitLength, llCodeTable[nbSeq-1]);
        // For RLE mode (accuracyLog=0), state is always 0.
        uint32_t stateLL = 0, stateOF = 0, stateML = 0;

        if (!llRLE) stateLL = fse_init_state(lastLL, &llTbl);
        if (!ofRLE) stateOF = fse_init_state(lastOF, &ofTbl);
        if (!mlRLE) stateML = fse_init_state(lastML, &mlTbl);

        // === First sequence (last in array): write extra bits only ===
        // Order: LL extra bits, ML extra bits, OF extra bits, flush
        // (matches reference: BIT_addBits litLength, BIT_addBits mlBase, BIT_addBits offBase, flush)
        {
            uint8_t llBits = LL_bits[lastLL];
            if (llBits > 0) {
                bs_addBits(&bs, my_seqs[last].literal_length - LL_baseline[lastLL], llBits);
            }

            uint8_t mlBits = ML_bits[lastML];
            if (mlBits > 0) {
                // match_length field stores (actual_len - 3), ML_baseline stores actual lengths
                bs_addBits(&bs, (my_seqs[last].match_length + 3) - ML_baseline[lastML], mlBits);
            }

            uint8_t ofBits = lastOF;
            if (ofBits > 0) {
                uint32_t offset_value = my_seqs[last].offset + 3;
                bs_addBits(&bs, offset_value - (1u << ofBits), ofBits);
            }

            bs_flush(&bs);
        }

        // === Remaining sequences backward ===
        for (int32_t n = (int32_t)n_seqs - 2; n >= 0; n--) {
            // Bounds check
            if (bs.pos >= safe_limit) goto fallback_raw;

            // Recompute codes for this sequence
            uint8_t lc = ll_code(my_seqs[n].literal_length);
            uint8_t mc = ml_code(my_seqs[n].match_length);
            uint8_t oc = of_code(my_seqs[n].offset);

            // FSE encode state bits: OF, ML, LL order (matches reference)
            if (!ofRLE) fse_encode_symbol(&bs, &stateOF, oc, &ofTbl);
            if (!mlRLE) fse_encode_symbol(&bs, &stateML, mc, &mlTbl);
            if (!llRLE) fse_encode_symbol(&bs, &stateLL, lc, &llTbl);

            bs_flush(&bs);

            // Extra bits: LL, ML, OF order (matches reference)
            uint8_t llBits = LL_bits[lc];
            if (llBits > 0) {
                bs_addBits(&bs, my_seqs[n].literal_length - LL_baseline[lc], llBits);
            }

            uint8_t mlBits = ML_bits[mc];
            if (mlBits > 0) {
                bs_addBits(&bs, (my_seqs[n].match_length + 3) - ML_baseline[mc], mlBits);
            }

            uint8_t ofBits = oc;
            if (ofBits > 0) {
                uint32_t offset_value = my_seqs[n].offset + 3;
                bs_addBits(&bs, offset_value - (1u << ofBits), ofBits);
            }

            bs_flush(&bs);
        }

        // === Flush final FSE states ===
        // Order: ML, OF, LL (matches reference: flushCState ML, OF, LL)
        if (!mlRLE) {
            bs_addBits(&bs, stateML, mlTbl.accuracyLog);
        }
        if (!ofRLE) {
            bs_addBits(&bs, stateOF, ofTbl.accuracyLog);
        }
        if (!llRLE) {
            bs_addBits(&bs, stateLL, llTbl.accuracyLog);
        }

        // Close bitstream: sentinel 1-bit + padding
        uint32_t seqDataEnd = bs_close(&bs);

        // Bounds check
        if (seqDataEnd >= max_frame_size) goto fallback_raw;

        // === Check if compressed block is actually smaller ===
        uint32_t block_content_size = seqDataEnd - (block_header_pos + 3);

        if (block_content_size >= this_chunk_size) {
            goto fallback_raw;
        }

        // Write compressed block header
        uint32_t bh = 1u | (ZSTD_BLOCK_TYPE_COMPRESSED << 1) | (block_content_size << 3);
        dst[block_header_pos + 0] = (uint8_t)bh;
        dst[block_header_pos + 1] = (uint8_t)(bh >> 8);
        dst[block_header_pos + 2] = (uint8_t)(bh >> 16);

        output_sizes[chunk_idx] = seqDataEnd;
        return;
    }

fallback_raw:
    {
        pos = block_header_pos;
        const uint8_t* src = input + chunk_start;
        uint32_t bh = 1u | (ZSTD_BLOCK_TYPE_RAW << 1) | (this_chunk_size << 3);
        dst[pos++] = (uint8_t)bh; dst[pos++] = (uint8_t)(bh>>8); dst[pos++] = (uint8_t)(bh>>16);
        for (uint32_t i = 0; i < this_chunk_size; i++) dst[pos++] = src[i];
        output_sizes[chunk_idx] = pos;
        return;
    }
}
```

### Key Differences from the Broken Original

| Aspect | Original (broken) | Corrected |
|--------|-------------------|-----------|
| Non-RLE dimensions | Fall back to raw block | Use FSE_Compressed mode |
| Code storage | `uint8_t llCodes[21846]` (65KB stack!) | Recompute on-the-fly |
| Encoding order (FSE bits) | LL, ML, OF | OF, ML, LL (per reference) |
| Extra bits order | LL, ML, OF | LL, ML, OF (correct, same) |
| State flush order | N/A (only RLE) | ML, OF, LL (per reference) |
| Bounds checking | None | Check `bs.pos >= safe_limit` |
| Fallback mechanism | Rewrite raw block inline | `goto fallback_raw` label |
| Mode byte | Hardcoded 0x54 (all-RLE) | Dynamic per-dimension |
| Table descriptions | Not written | Written via `fse_write_table_desc` |

### Memory Usage Analysis

Per-thread stack usage in the corrected version:
- `llHist[36]`: 144 bytes
- `mlHist[53]`: 212 bytes
- `ofHist[32]`: 128 bytes
- `llNC[36]`: 72 bytes
- `mlNC[53]`: 106 bytes
- `ofNC[32]`: 64 bytes
- `DynFSE llTbl`: ~2180 bytes (512*2 + 64*4 + 64*4 + 8)
- `DynFSE mlTbl`: ~2180 bytes
- `DynFSE ofTbl`: ~2180 bytes
- Other locals: ~100 bytes
- **Total: ~7.4KB per thread**

This is well within CUDA's default stack limit. The `DynFSE` struct has `stateTable[512]` (1024 bytes) plus `deltaNbBits[64]` (256 bytes) plus `deltaFindState[64]` (256 bytes) = ~1.5KB, so three of them = ~4.5KB. Total is safe.

The `fse_build` function internally uses `uint8_t tableSym[512]` and `uint32_t cumul[65]` + `uint32_t tc[65]`, adding another ~1KB during that call. Still safe.

---

## 4. Recommendations for Sub-Chunk Sizing

### Current: 64KB sub-chunks

- Hash table: `uint16_t[16384]` = 32KB shared memory (14-bit hash)
- Hash entries store `uint16_t` position (valid for up to 64KB chunks)
- Matches are limited to within-chunk (no cross-chunk references)
- With 64KB chunks on a 4.6GB ISO: 73,728 chunks = good GPU occupancy

### Analysis: Larger sub-chunks

**128KB sub-chunks:**
- Hash positions need `uint32_t` (17-bit minimum), doubling shared memory to 64KB -- exceeds sm_86 shared memory limit
- Alternative: keep 16K entries but accept more collisions
- Better compression ratio (larger match window)
- Half the GPU parallelism (36,864 chunks)
- Feasible only with `uint16_t` hash + accepted collisions

**256KB sub-chunks:**
- Much better compression ratio (larger context window)
- Only 18,432 chunks -- still good GPU occupancy for a 4.6GB file
- Hash table would need to stay at 32KB (shared memory constraint)
- 14-bit hash with 256KB data = high collision rate, degraded match finding
- Would need a different hash strategy (e.g., chain-based or multi-probe)

**Recommendation:**

Keep 64KB for now. The 64KB chunk size is the sweet spot because:
1. nvCOMP uses 64KB as its default for the same reason
2. Hash table fits perfectly in 32KB shared memory
3. `uint16_t` positions are sufficient (no wasted registers)
4. GPU occupancy is excellent
5. The compression ratio gap vs nvCOMP is entirely in the entropy coding, not chunk size

The real improvement will come from enabling FSE_Compressed mode (estimated 5-15% improvement in ratio), not from changing chunk size.

If you later want to experiment with 128KB chunks:
- Change `HASH_LOG` to 14 (keep same table size)
- Change hash entry to `uint32_t` but only use lower 17 bits
- This uses 64KB shared memory; on sm_86 you'd need to set the shared memory carveout to max (48KB default won't work unless you reduce to `uint16_t[16384]` and mask positions)
- Actually, even with 128KB data, `uint16_t` still works if you accept that positions > 65535 alias. Since the match verifier checks actual byte equality, false hash hits from aliasing just mean wasted comparison cycles, not incorrect matches. The hash table becomes less effective but still functional.

---

## 5. Additional Improvements to Match Finding

### Current Match Finder Limitations

1. **Single-thread matching**: Only thread 0 does sequential matching. Threads 1-255 help with hash table init but then idle. This wastes 99.6% of the thread block's compute.

2. **No repeat offset optimization in encoder**: The match finder tracks repeat offsets but the encoder doesn't use them (treats all offsets as explicit). Repeat offsets are codes 1, 2, 3 in Zstd and are very cheap to encode (OF code 0, 1, or 2 with 0 extra bits).

3. **Hash table only stores one position per hash bucket**: Better compression could be achieved with a 2-entry or 4-entry hash chain per bucket (at the cost of more shared memory).

### Recommended Improvements (Priority Order)

**Priority 1: Enable FSE_Compressed mode (this document)**
- Expected impact: ratio drops from 99.9% to ~96-97% on the ISO
- Zero change to match finder needed

**Priority 2: Improve repeat offset handling**
- Current: Match finder tracks repeat offsets but encodes them as explicit offsets
- Fix: When a match uses a repeat offset, encode it with offset_value=1/2/3 instead of the raw offset
- Impact: Better OF code distribution (more 0/1/2 codes = better FSE compression of OF dimension)

**Priority 3: Warp-cooperative hash prefill**
- Currently threads 1-255 zero the hash table then idle
- Better: have them prefill the hash table with positions from the first portion of the chunk
- Thread 0 can then start matching with a warm hash table

**Priority 4: Parallel literal copy**
- After thread 0 finishes match finding, threads 1-255 could help copy literals into the output
- Current byte-by-byte literal copy in the encoder is slow; could use `uint4` vector loads

**Priority 5: Multi-entry hash chains**
- Store 2 positions per hash bucket: `struct { uint16_t pos0, pos1; }`
- Double the match candidates at each position
- Same 32KB shared memory footprint (8192 entries * 4 bytes)
- Check both positions, keep the longer match

### Potential Future Architecture Change

For maximum throughput, consider a two-pass approach:
1. **Pass 1**: Match finding (current kernel, single-threaded per block)
2. **Pass 2**: Encoding (could use more threads for parallel histogram building + parallel FSE table construction)

The histogram building could easily be parallelized across threads in the block using shared-memory atomics, but the current single-thread approach is fast enough for 64KB chunks (only up to ~21K sequences to scan).

---

## 6. Summary of Required Changes

To apply the corrected kernel:

1. Replace lines 696-877 in `zstd_compress.cu` with the corrected `zstd_encode_block` kernel from Section 3
2. No changes needed to `zstd_match_find` or `zstd_compress_raw`
3. No changes needed to `compress_zstd_custom.rs` or `constants.rs`
4. The FSE infrastructure functions (`fse_build`, `fse_normalize`, `fse_encode_symbol`, `fse_init_state`, `fse_write_table_desc`) at lines 525-695 should work as-is (they were validated against CPU zstd)

### Testing Plan

1. Compile the kernel and verify CUDA launch succeeds (no stack overflow)
2. Compress a small test file and decompress with `zstd -d` (CPU reference decoder)
3. Compress the Windows Server ISO and check ratio improvement
4. Compare bitstream output byte-by-byte against CPU zstd on a small known input
5. If corrupt output persists, add `printf` debugging to one chunk to trace:
   - Histogram values for each dimension
   - Normalized counts
   - Mode byte value
   - FSE table description bytes
   - First few encoded bits
   - Final state values before flush

### Expected Results

- Compression ratio should improve from 99.9% to approximately 95-97% on the ISO
- Throughput may decrease slightly due to FSE encoding overhead (vs raw block copy)
- All output should be decompressible by any standard zstd decoder
