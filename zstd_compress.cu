// Custom GPU Zstd Compression Kernel
// Produces spec-compliant Zstd frames (RFC 8878) decompressible by nvCOMP and CPU zstd.
//
// Architecture: each thread block processes one 64KB sub-chunk.
// - Thread 0: sequential match finding (lazy or greedy)
// - Threads 1-255: hash table prefill + data prefetch
//
// Kernel pipeline per batch:
//   1. zstd_compress_raw   — wrap raw data as Zstd Raw_Block (level 0 fallback / validation)
//   2. zstd_match_find     — LZ77 match finding with hash table in shared memory
//   3. zstd_encode_block   — encode literals + sequences into Zstd compressed block
//
// sm_86 (RTX 3090): 48KB shared memory default, 256 threads/block

#include <stdint.h>
#include <cuda_runtime.h>

// Zstd constants
#define ZSTD_MAGIC 0xFD2FB528u
#define ZSTD_BLOCK_TYPE_RAW 0       // Raw (uncompressed) block
#define ZSTD_BLOCK_TYPE_RLE 1       // RLE block
#define ZSTD_BLOCK_TYPE_COMPRESSED 2 // Compressed block
#define ZSTD_BLOCK_HEADER_SIZE 3

// Match finding constants
#define HASH_LOG 14                  // 16K entries
#define HASH_TABLE_SIZE (1 << HASH_LOG)
#define MIN_MATCH_LEN 3
#define MAX_MATCH_LEN 131074         // Zstd max match length

// Safe unaligned 4-byte load (uint8_t* may not be 4-byte aligned)
__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Sequence representation (output of match finder)
struct Sequence {
    uint32_t literal_length;  // bytes of literals before this match
    uint32_t match_length;    // match length (3+)
    uint32_t offset;          // match offset (1+)
};

// Per-chunk compression output metadata
struct ChunkResult {
    uint32_t compressed_size;   // total frame size
    uint32_t num_sequences;     // number of match sequences found
    uint32_t num_literals;      // total literal bytes
};

// ============================================================================
// Helper: write little-endian values
// ============================================================================

__device__ __forceinline__ void write_le32(uint8_t* dst, uint32_t val) {
    dst[0] = (uint8_t)(val);
    dst[1] = (uint8_t)(val >> 8);
    dst[2] = (uint8_t)(val >> 16);
    dst[3] = (uint8_t)(val >> 24);
}

__device__ __forceinline__ void write_le16(uint8_t* dst, uint16_t val) {
    dst[0] = (uint8_t)(val);
    dst[1] = (uint8_t)(val >> 8);
}

// ============================================================================
// Kernel: zstd_compress_raw
// Wraps each sub-chunk as a Zstd frame with a Raw_Block (uncompressed).
// Used for validation and as fallback when compression doesn't help.
//
// Zstd frame layout for raw block:
//   [4B magic] [1B frame_header_descriptor] [2B frame_content_size]
//   [3B block_header: last=1, type=0(raw), size=N]
//   [NB raw data]
//
// Total overhead: 10 bytes per frame
// ============================================================================

extern "C" __global__ void zstd_compress_raw(
    const uint8_t* __restrict__ input,     // all sub-chunks concatenated
    uint8_t* __restrict__ output,          // output frames (pre-allocated)
    uint32_t* __restrict__ output_sizes,   // compressed size per sub-chunk
    uint32_t chunk_size,                   // sub-chunk size (e.g., 65536)
    uint32_t num_chunks,                   // number of sub-chunks
    uint32_t total_input_size              // total input bytes (last chunk may be smaller)
) {
    uint32_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    // Calculate this chunk's actual size (last chunk may be smaller)
    uint64_t chunk_start = (uint64_t)chunk_idx * chunk_size;
    uint32_t this_chunk_size = chunk_size;
    if (chunk_start + this_chunk_size > total_input_size) {
        this_chunk_size = total_input_size - (uint32_t)chunk_start;
    }

    const uint8_t* src = input + chunk_start;

    // Output position: each chunk gets max_frame_size allocation
    // max_frame_size = 10 (header) + chunk_size (raw data)
    uint32_t max_frame_size = 10 + chunk_size;
    uint8_t* dst = output + (uint64_t)chunk_idx * max_frame_size;
    uint32_t pos = 0;

    // === Zstd Magic Number (4 bytes) ===
    write_le32(dst + pos, ZSTD_MAGIC);
    pos += 4;

    // === Frame Header ===
    // Frame_Header_Descriptor (1 byte):
    //   bits 7-6: Frame_Content_Size_flag = 01 (2 bytes FCS)
    //   bit 5:    Single_Segment_flag = 1 (no window descriptor needed)
    //   bit 3:    reserved = 0
    //   bit 2:    Content_Checksum_flag = 0
    //   bits 1-0: Dictionary_ID_flag = 00
    dst[pos] = 0x60;  // 0b01100000 = FCS_flag=1(2B), Single_Segment=1
    pos += 1;

    // Frame_Content_Size (2 bytes, little-endian) — actual content size
    // Per spec: when FCS_Field_Size=2, value = FCS + 256 if Single_Segment=1
    // Wait — actually per the spec, when FCS_flag bits are 01, FCS_Field_Size = 2,
    // and the value stored is (Frame_Content_Size - 256) for 2-byte encoding...
    // No — re-reading spec: FCS_Field_Size is determined by the flag.
    // For FCS_flag=01: FCS_Field_Size = 2 bytes. The 2-byte field holds the actual
    // Frame_Content_Size value. BUT when Single_Segment_flag=1 and FCS_Field_Size=2,
    // the interpretation is: FCS = value + 256 (to extend the 2-byte range).
    //
    // So to encode this_chunk_size, we write (this_chunk_size - 256).
    // For 65536: write 65280 = 0xFF00
    // For chunks < 256 bytes, we'd need a different FCS encoding.
    //
    // If chunk is small (< 256 bytes), use FCS_flag=00 (1 byte FCS)
    if (this_chunk_size >= 256) {
        // Already wrote 0x60 above
        uint16_t fcs_value = (uint16_t)(this_chunk_size - 256);
        write_le16(dst + pos, fcs_value);
        pos += 2;
    } else {
        // Rewrite descriptor for 1-byte FCS: FCS_flag=00, Single_Segment=1
        dst[pos - 1] = 0x20;  // 0b00100000
        dst[pos] = (uint8_t)this_chunk_size;
        pos += 1;
    }

    // === Block Header (3 bytes) ===
    // Last_Block = 1 (bit 0)
    // Block_Type = 00 (Raw_Block, bits 2-1)
    // Block_Size = this_chunk_size (bits 23-3)
    uint32_t block_header = 1u                           // Last_Block = 1
                          | (ZSTD_BLOCK_TYPE_RAW << 1)   // Block_Type = 0 (raw)
                          | (this_chunk_size << 3);       // Block_Size
    dst[pos + 0] = (uint8_t)(block_header);
    dst[pos + 1] = (uint8_t)(block_header >> 8);
    dst[pos + 2] = (uint8_t)(block_header >> 16);
    pos += 3;

    // === Raw Block Data ===
    for (uint32_t i = 0; i < this_chunk_size; i++) {
        dst[pos + i] = src[i];
    }
    pos += this_chunk_size;

    // Store output size
    output_sizes[chunk_idx] = pos;
}

// ============================================================================
// Kernel: zstd_match_find
// LZ77 match finding with lazy matching.
// Each thread block processes one sub-chunk.
// Thread 0 does sequential matching, other threads prefill hash table.
//
// Shared memory layout (48KB total):
//   [0 .. 32767]  = hash_table: uint16_t[16384] (32KB, 14-bit hash)
//   [32768 .. 48383] = reserved for future use
// ============================================================================

extern "C" __global__ void zstd_match_find(
    const uint8_t* __restrict__ input,     // all sub-chunks concatenated
    Sequence* __restrict__ sequences,      // output: sequences per sub-chunk (pre-allocated)
    uint8_t* __restrict__ literals,        // output: literal bytes per sub-chunk
    uint32_t* __restrict__ seq_counts,     // output: num sequences per sub-chunk
    uint32_t* __restrict__ lit_counts,     // output: num literals per sub-chunk
    uint32_t chunk_size,                   // sub-chunk size (65536)
    uint32_t num_chunks,
    uint32_t total_input_size,
    uint32_t search_depth,                 // 1=greedy, 16=lazy, 64=optimal
    uint32_t max_sequences_per_chunk       // pre-allocated capacity
) {
    uint32_t chunk_idx = blockIdx.x;
    if (chunk_idx >= num_chunks) return;

    // Shared memory: hash table
    __shared__ uint16_t hash_table[HASH_TABLE_SIZE];

    // Initialize hash table to 0 (all threads cooperate)
    for (uint32_t i = threadIdx.x; i < HASH_TABLE_SIZE; i += blockDim.x) {
        hash_table[i] = 0;
    }
    __syncthreads();

    // Only thread 0 does the actual matching
    if (threadIdx.x != 0) return;

    // Calculate this chunk's bounds
    uint64_t chunk_start = (uint64_t)chunk_idx * chunk_size;
    uint32_t this_chunk_size = chunk_size;
    if (chunk_start + this_chunk_size > total_input_size) {
        this_chunk_size = total_input_size - (uint32_t)chunk_start;
    }

    const uint8_t* src = input + chunk_start;

    // Output pointers for this chunk
    Sequence* my_sequences = sequences + (uint64_t)chunk_idx * max_sequences_per_chunk;
    uint8_t* my_literals = literals + (uint64_t)chunk_idx * chunk_size;

    uint32_t pos = 0;
    uint32_t num_seqs = 0;
    uint32_t num_lits = 0;
    uint32_t literal_start = 0;  // start of current literal run

    // Repeat offsets (Zstd tracks last 3)
    uint32_t rep_offsets[3] = {1, 4, 8};

    // Hash function: multiplicative hash
    #define HASH4(p) ((load_u32(p) * 2654435761u) >> (32 - HASH_LOG))

    while (pos + 3 < this_chunk_size) {
        uint32_t h = HASH4(src + pos);
        uint32_t match_pos = hash_table[h];

        // Update hash table with current position
        hash_table[h] = (uint16_t)(pos & 0xFFFF);

        // Check for match
        uint32_t match_offset = pos - match_pos;
        uint32_t match_len = 0;

        if (match_offset > 0 && match_offset <= pos && match_pos < this_chunk_size) {
            // Verify match at hash position
            if (src[match_pos] == src[pos] &&
                src[match_pos + 1] == src[pos + 1] &&
                src[match_pos + 2] == src[pos + 2]) {

                // Extend match
                match_len = 3;
                while (pos + match_len < this_chunk_size &&
                       match_pos + match_len < this_chunk_size &&
                       src[match_pos + match_len] == src[pos + match_len] &&
                       match_len < MAX_MATCH_LEN) {
                    match_len++;
                }
            }
        }

        // Check repeat offsets for potential match
        for (int r = 0; r < 3 && match_len < MIN_MATCH_LEN; r++) {
            if (rep_offsets[r] <= pos) {
                uint32_t rp = pos - rep_offsets[r];
                if (src[rp] == src[pos] && src[rp + 1] == src[pos + 1] && src[rp + 2] == src[pos + 2]) {
                    uint32_t rlen = 3;
                    while (pos + rlen < this_chunk_size && rp + rlen < this_chunk_size &&
                           src[rp + rlen] == src[pos + rlen] && rlen < MAX_MATCH_LEN) {
                        rlen++;
                    }
                    if (rlen >= MIN_MATCH_LEN) {
                        match_len = rlen;
                        match_offset = rep_offsets[r];
                        break;
                    }
                }
            }
        }

        // Lazy matching: check if pos+1 gives a longer match
        if (match_len >= MIN_MATCH_LEN && search_depth > 1 && pos + 4 < this_chunk_size) {
            uint32_t h2 = HASH4(src + pos + 1);
            uint32_t mp2 = hash_table[h2];
            uint32_t off2 = (pos + 1) - mp2;

            if (off2 > 0 && off2 <= (pos + 1) && mp2 < this_chunk_size) {
                if (src[mp2] == src[pos + 1] &&
                    src[mp2 + 1] == src[pos + 2] &&
                    src[mp2 + 2] == src[pos + 3]) {
                    uint32_t len2 = 3;
                    while (pos + 1 + len2 < this_chunk_size &&
                           mp2 + len2 < this_chunk_size &&
                           src[mp2 + len2] == src[pos + 1 + len2] &&
                           len2 < MAX_MATCH_LEN) {
                        len2++;
                    }
                    // Use pos+1 match if it's longer (accounting for the extra literal)
                    if (len2 > match_len + 1) {
                        // Skip current byte (will be emitted as part of literal run
                        // when the match is flushed at lines 311-314)
                        pos++;
                        match_len = len2;
                        match_offset = off2;

                        // Update hash at new position
                        hash_table[h2] = (uint16_t)(pos & 0xFFFF);
                    }
                }
            }
        }

        if (match_len >= MIN_MATCH_LEN && num_seqs < max_sequences_per_chunk) {
            // Emit literals since last match
            uint32_t lit_len = pos - literal_start;
            for (uint32_t i = literal_start; i < pos; i++) {
                my_literals[num_lits++] = src[i];
            }

            // Emit sequence
            my_sequences[num_seqs].literal_length = lit_len;
            my_sequences[num_seqs].match_length = match_len - MIN_MATCH_LEN; // Zstd stores ML - 3
            my_sequences[num_seqs].offset = match_offset;
            num_seqs++;

            // Update repeat offsets
            rep_offsets[2] = rep_offsets[1];
            rep_offsets[1] = rep_offsets[0];
            rep_offsets[0] = match_offset;

            // Update hash table for skipped positions
            for (uint32_t i = 1; i < match_len && pos + i + 3 < this_chunk_size; i++) {
                uint32_t sh = HASH4(src + pos + i);
                hash_table[sh] = (uint16_t)((pos + i) & 0xFFFF);
            }

            pos += match_len;
            literal_start = pos;
        } else {
            pos++;
        }
    }

    // Emit remaining literals (from literal_start through end of chunk)
    for (uint32_t i = literal_start; i < this_chunk_size; i++) {
        my_literals[num_lits++] = src[i];
    }

    // If there are trailing literals after the last sequence, we need to record them
    // as the final literal run. In Zstd, the last "sequence" can have a literal-only
    // component via the Literals_Length of a dummy entry, but the standard way is to
    // just have the literals section be larger than the sum of sequence literal lengths.
    // We store the trailing count so the encoder knows.

    seq_counts[chunk_idx] = num_seqs;
    lit_counts[chunk_idx] = num_lits;

    #undef HASH4
}

// ============================================================================
// Zstd predefined FSE tables (from RFC 8878, Appendix A)
// ============================================================================

// Decoding table entry
struct FSE_decode_t {
    uint8_t  symbol;
    uint8_t  nbBits;
    uint16_t base;
};

// LL predefined decoding table (AccuracyLog=6, 64 entries)
__constant__ FSE_decode_t LL_decTable[64] = {
    {0,4,0},{0,4,16},{1,5,32},{3,5,0},{4,5,0},{6,5,0},{7,5,0},{9,5,0},
    {10,5,0},{12,5,0},{14,6,0},{16,5,0},{18,5,0},{19,5,0},{21,5,0},{22,5,0},
    {24,5,0},{25,5,32},{26,5,0},{27,6,0},{29,6,0},{31,6,0},{0,4,32},{1,4,0},
    {2,5,0},{4,5,32},{5,5,0},{7,5,32},{8,5,0},{10,5,32},{11,5,0},{13,6,0},
    {16,5,32},{17,5,0},{19,5,32},{20,5,0},{22,5,32},{23,5,0},{25,4,0},{25,4,16},
    {26,5,32},{28,6,0},{30,6,0},{0,4,48},{1,4,16},{2,5,32},{3,5,32},{5,5,32},
    {6,5,32},{8,5,32},{9,5,32},{11,5,32},{12,5,32},{15,6,0},{17,5,32},{18,5,32},
    {20,5,32},{21,5,32},{23,5,32},{24,5,32},{35,6,0},{34,6,0},{33,6,0},{32,6,0}
};

// ML predefined decoding table (AccuracyLog=6, 64 entries)
__constant__ FSE_decode_t ML_decTable[64] = {
    {0,6,0},{1,4,0},{2,5,32},{3,5,0},{5,5,0},{6,5,0},{8,5,0},{10,6,0},
    {13,6,0},{16,6,0},{19,6,0},{22,6,0},{25,6,0},{28,6,0},{31,6,0},{33,6,0},
    {35,6,0},{37,6,0},{39,6,0},{41,6,0},{43,6,0},{45,6,0},{1,4,16},{2,4,0},
    {3,5,32},{4,5,0},{6,5,32},{7,5,0},{9,6,0},{12,6,0},{15,6,0},{18,6,0},
    {21,6,0},{24,6,0},{27,6,0},{30,6,0},{32,6,0},{34,6,0},{36,6,0},{38,6,0},
    {40,6,0},{42,6,0},{44,6,0},{1,4,32},{1,4,48},{2,4,16},{4,5,32},{5,5,32},
    {7,5,32},{8,5,32},{11,6,0},{14,6,0},{17,6,0},{20,6,0},{23,6,0},{26,6,0},
    {29,6,0},{52,6,0},{51,6,0},{50,6,0},{49,6,0},{48,6,0},{47,6,0},{46,6,0}
};

// OF predefined decoding table (AccuracyLog=5, 32 entries)
__constant__ FSE_decode_t OF_decTable[32] = {
    {0,5,0},{6,4,0},{9,5,0},{15,5,0},{21,5,0},{3,5,0},{7,4,0},{12,5,0},
    {18,5,0},{23,5,0},{5,5,0},{8,4,0},{14,5,0},{20,5,0},{2,5,0},{7,4,16},
    {11,5,0},{17,5,0},{22,5,0},{4,5,0},{8,4,16},{13,5,0},{19,5,0},{1,5,0},
    {6,4,16},{10,5,0},{16,5,0},{28,5,0},{27,5,0},{26,5,0},{25,5,0},{24,5,0}
};

// LL code baselines and extra bits
__constant__ uint32_t LL_baseline[36] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,28,32,40,48,64,128,256,512,1024,2048,4096,
    8192,16384,32768,65536
};
__constant__ uint8_t LL_bits[36] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,2,2,3,3,4,6,7,8,9,10,11,12,
    13,14,15,16
};

// ML code baselines and extra bits (code 0 = match length 3)
__constant__ uint32_t ML_baseline[53] = {
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
    35,37,39,41,43,47,51,59,67,83,99,131,259,515,1027,2051,
    4099,8195,16387,32771,65539
};
__constant__ uint8_t ML_bits[53] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,2,2,3,3,4,4,5,7,8,9,10,11,
    12,13,14,15,16
};

// ============================================================================
// Code lookup: value → FSE code
// ============================================================================

__device__ __forceinline__ uint32_t highbit32(uint32_t v) {
    return 31 - __clz(v);
}

__device__ uint8_t ll_code(uint32_t lit_length) {
    // LL codes 0-15 map directly
    if (lit_length < 16) return (uint8_t)lit_length;
    if (lit_length < 64) {
        // Codes 16-24 cover values 16-63
        __constant__ static const uint8_t LL_Code[64] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
            16,16,17,17,18,18,19,19,20,20,20,20,21,21,21,21,
            22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,
            24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24
        };
        return LL_Code[lit_length];
    }
    return (uint8_t)(highbit32(lit_length) + 19);
}

__device__ uint8_t ml_code(uint32_t match_length) {
    // match_length is already stored as (actual_match_len - 3) by the match finder
    uint32_t ml = match_length; // mlBase = actual_length - 3
    if (ml < 128) {
        __constant__ static const uint8_t ML_Code[128] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
            32,32,33,33,34,34,35,35,36,36,36,36,37,37,37,37,
            38,38,38,38,38,38,38,38,39,39,39,39,39,39,39,39,
            40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
            41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,
            42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,
            42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42
        };
        return ML_Code[ml];
    }
    return (uint8_t)(highbit32(ml) + 36);
}

__device__ __forceinline__ uint8_t of_code(uint32_t offset) {
    // offset is the raw offset value (>= 1)
    // Zstd offset_value = offset + 3 (to reserve 1,2,3 for repeat offsets)
    // But for simplicity in this first implementation, we encode raw offsets
    // of_code = floor(log2(offset + 3))
    return (uint8_t)highbit32(offset + 3);
}

// ============================================================================
// Bitstream writer (forward, 64-bit accumulator)
// ============================================================================

struct BitStream {
    uint8_t* dst;
    uint32_t pos;       // byte position in output
    uint64_t bits;      // bit accumulator
    uint32_t nbBits;    // bits currently in accumulator
};

__device__ __forceinline__ void bs_init(BitStream* bs, uint8_t* dst, uint32_t startPos) {
    bs->dst = dst;
    bs->pos = startPos;
    bs->bits = 0;
    bs->nbBits = 0;
}

__device__ __forceinline__ void bs_addBits(BitStream* bs, uint64_t value, uint32_t nbBits) {
    if (nbBits == 0) return;
    uint64_t mask = (1ULL << nbBits) - 1;
    bs->bits |= ((value & mask) << bs->nbBits);
    bs->nbBits += nbBits;
}

__device__ __forceinline__ void bs_flush(BitStream* bs) {
    // Write complete bytes from accumulator
    while (bs->nbBits >= 8) {
        bs->dst[bs->pos++] = (uint8_t)(bs->bits);
        bs->bits >>= 8;
        bs->nbBits -= 8;
    }
}

__device__ uint32_t bs_close(BitStream* bs) {
    // Add sentinel 1-bit
    bs_addBits(bs, 1, 1);
    // Flush all complete bytes
    bs_flush(bs);
    // Write remaining partial byte (with zero-padding in upper bits)
    if (bs->nbBits > 0) {
        bs->dst[bs->pos++] = (uint8_t)(bs->bits);
    }
    return bs->pos;
}

// ============================================================================
// Precomputed FSE encoding tables (generated by FSE_buildCTable algorithm
// from the predefined normalized distributions in the Zstd spec)
// ============================================================================

// ============================================================================
// Dynamic FSE table building and encoding (FSE_Compressed mode)
// Validated against CPU zstd -d in /tmp/fse_step1.c
// ============================================================================

// Max table sizes for Zstd sequence encoding
#define MAX_LL_ALOG 9   // max accuracy log for LL
#define MAX_ML_ALOG 9   // max accuracy log for ML
#define MAX_OF_ALOG 8   // max accuracy log for OF
#define MAX_TABLE_SIZE 512 // 1 << MAX_ALOG

// Dynamic FSE encoding table (local to each thread)
struct DynFSE {
    uint16_t stateTable[MAX_TABLE_SIZE];
    uint32_t deltaNbBits[64]; // max symbols per dimension
    int32_t  deltaFindState[64];
    int      accuracyLog;
    int      tableSize;
};

// Build FSE encoding table from normalized counts
// nc[0..maxSym], sum of abs(counts) must equal 1<<aLog
__device__ void fse_build(DynFSE* tbl, int16_t* nc, int maxSym, int aLog) {
    int tSize = 1 << aLog;
    tbl->accuracyLog = aLog;
    tbl->tableSize = tSize;

    // Symbol spreading
    uint8_t tableSym[MAX_TABLE_SIZE];
    for (int i = 0; i < tSize; i++) tableSym[i] = 0;
    int hi = tSize - 1;
    for (int s = 0; s <= maxSym; s++) if (nc[s] == -1) tableSym[hi--] = (uint8_t)s;
    int pos2 = 0, step = (tSize >> 1) + (tSize >> 3) + 3, mask = tSize - 1;
    for (int s = 0; s <= maxSym; s++) {
        if (nc[s] <= 0) continue;
        for (int i = 0; i < nc[s]; i++) {
            tableSym[pos2] = (uint8_t)s;
            pos2 = (pos2 + step) & mask;
            while (pos2 > hi) pos2 = (pos2 + step) & mask;
        }
    }

    // Build stateTable
    // nc == -1 → 1 slot (less-than-one probability), nc == 0 → 0 slots (absent), nc > 0 → nc slots
    uint32_t cumul[65]; cumul[0] = 0;
    for (int s = 0; s <= maxSym; s++) cumul[s+1] = cumul[s] + (nc[s] < 0 ? 1 : (uint32_t)nc[s]);
    uint32_t tc[65];
    for (int s = 0; s <= maxSym + 1; s++) tc[s] = cumul[s];
    for (int u = 0; u < tSize; u++) {
        uint8_t s = tableSym[u];
        tbl->stateTable[tc[s]++] = (uint16_t)(tSize + u);
    }

    // Build symbolTT — two-pass (matching zstd reference)
    for (int s = 0; s <= maxSym; s++) { tbl->deltaNbBits[s] = 0; tbl->deltaFindState[s] = 0; }
    uint32_t total = 1;
    for (int s = 1; s <= maxSym; s++) if (nc[s] == -1) {
        tbl->deltaNbBits[s] = ((uint32_t)aLog << 16) - (1u << aLog);
        tbl->deltaFindState[s] = (int32_t)total - 1;
        total++;
    }
    for (int s = 0; s <= maxSym; s++) {
        if (nc[s] <= 0) continue;
        if (nc[s] == 1) {
            tbl->deltaNbBits[s] = ((uint32_t)aLog << 16) - (1u << aLog);
            tbl->deltaFindState[s] = (int32_t)total - 1;
            total++;
        } else {
            uint32_t p = (uint32_t)nc[s];
            uint32_t mb = aLog - highbit32(p - 1);
            uint32_t ms = p << mb;
            tbl->deltaNbBits[s] = (mb << 16) - ms;
            tbl->deltaFindState[s] = (int32_t)total - (int32_t)p;
            total += p;
        }
    }
}

// Write FSE table description into output buffer (RFC 8878 §4.1.1)
// Returns bytes written (rounded to byte boundary)
__device__ uint32_t fse_write_table_desc(uint8_t* dst, int16_t* nc, int maxSym, int aLog) {
    BitStream tbs;
    bs_init(&tbs, dst, 0);

    bs_addBits(&tbs, aLog - 5, 4);

    int remaining = (1 << aLog);
    for (int s = 0; s <= maxSym && remaining > 0; s++) {
        int16_t count = nc[s];
        int prob = (count == -1) ? 1 : count;
        int value = count + 1;
        int maxVal = remaining + 1;
        int nbBits = highbit32(maxVal) + 1;
        int threshold = (1 << nbBits) - 1 - maxVal;

        if (value < threshold) {
            bs_addBits(&tbs, value, nbBits - 1);
        } else {
            bs_addBits(&tbs, value + threshold, nbBits);
        }
        remaining -= prob;

        if (count == 0) {
            int run = 0;
            while (s + 1 + run <= maxSym && nc[s + 1 + run] == 0) run++;
            while (run >= 3) { bs_addBits(&tbs, 3, 2); run -= 3; s += 3; }
            bs_addBits(&tbs, run, 2);
            s += run;
        }
    }

    bs_flush(&tbs);
    uint32_t sz = tbs.pos;
    if (tbs.nbBits > 0) { dst[sz++] = (uint8_t)tbs.bits; }
    return sz;
}

// Write FSE table description into an existing bitstream
__device__ void fse_write_table_desc_bs(BitStream* bs, int16_t* nc, int maxSym, int aLog) {
    bs_addBits(bs, aLog - 5, 4);

    int remaining = (1 << aLog);
    for (int s = 0; s <= maxSym && remaining > 0; s++) {
        int16_t count = nc[s];
        int prob = (count == -1) ? 1 : count;
        int value = count + 1;
        int maxVal = remaining + 1;
        int nbBits = highbit32(maxVal) + 1;
        int threshold = (1 << nbBits) - 1 - maxVal;

        if (value < threshold) {
            bs_addBits(bs, value, nbBits - 1);
        } else {
            bs_addBits(bs, value + threshold, nbBits);
        }
        remaining -= prob;
        bs_flush(bs);

        if (count == 0) {
            int run = 0;
            while (s + 1 + run <= maxSym && nc[s + 1 + run] == 0) run++;
            while (run >= 3) { bs_addBits(bs, 3, 2); run -= 3; s += 3; }
            bs_addBits(bs, run, 2);
            s += run;
            bs_flush(bs);
        }
    }
}

// Finalize table descriptions bitstream: flush remaining bits to byte boundary
__device__ uint32_t fse_finish_table_descs(BitStream* bs) {
    bs_flush(bs);
    if (bs->nbBits > 0) {
        bs->dst[bs->pos++] = (uint8_t)(bs->bits);
        bs->bits = 0;
        bs->nbBits = 0;
    }
    return bs->pos;
}

// FSE encode symbol
__device__ __forceinline__ void fse_encode_symbol(
    BitStream* bs, uint32_t* state, uint8_t symbol, const DynFSE* tbl
) {
    uint32_t nbBitsOut = (*state + tbl->deltaNbBits[symbol]) >> 16;
    bs_addBits(bs, *state & ((1u << nbBitsOut) - 1), nbBitsOut);
    *state = tbl->stateTable[(*state >> nbBitsOut) + tbl->deltaFindState[symbol]];
}

// Initialize FSE state for a symbol (no bits output)
__device__ uint32_t fse_init_state(uint8_t symbol, const DynFSE* tbl) {
    uint32_t dnb = tbl->deltaNbBits[symbol];
    uint32_t nbBitsOut = (dnb + (1u << 15)) >> 16;
    uint32_t value = (nbBitsOut << 16) - dnb;
    int32_t idx = (int32_t)(value >> nbBitsOut) + tbl->deltaFindState[symbol];
    if (idx < 0) idx = 0;
    return tbl->stateTable[idx];
}

// Normalize histogram to sum = 1<<aLog
// hist[0..maxSym] = raw counts. Output: nc[0..maxSym] = normalized counts.
__device__ void fse_normalize(uint32_t* hist, int maxSym, int aLog, int16_t* nc) {
    uint32_t total = 0;
    for (int s = 0; s <= maxSym; s++) total += hist[s];
    if (total == 0) { nc[0] = (1 << aLog); return; }

    int tableSize = 1 << aLog;
    int distributed = 0;

    for (int s = 0; s <= maxSym; s++) {
        if (hist[s] == 0) { nc[s] = 0; continue; }
        // Proportional scaling
        uint32_t scaled = (uint32_t)(((uint64_t)hist[s] * tableSize + total/2) / total);
        if (scaled == 0) scaled = 1; // ensure non-zero symbols get at least 1
        nc[s] = (int16_t)scaled;
        distributed += scaled;
    }

    // Adjust to make sum exactly tableSize
    // Find the symbol with the largest count and adjust
    while (distributed != tableSize) {
        int bestSym = -1;
        uint32_t bestCount = 0;
        for (int s = 0; s <= maxSym; s++) {
            if (nc[s] > 1 && hist[s] > bestCount) {
                bestCount = hist[s]; bestSym = s;
            }
        }
        if (bestSym < 0) break;
        if (distributed > tableSize) { nc[bestSym]--; distributed--; }
        else { nc[bestSym]++; distributed++; }
    }
}

// ============================================================================
// Kernel: zstd_encode_block
// Encodes match-finder output into a Zstd compressed frame.
// Supports FSE_Compressed mode for non-uniform code distributions.
// Falls back to raw block if compressed size >= uncompressed.
//
// Reference: ZSTD_encodeSequences_body() in facebook/zstd
// Bit order (per RFC 8878):
//   Written forward, read backward by decoder.
//   First sequence encoded = LAST sequence (backward iteration).
//   Per sequence: FSE state bits (LL, ML, OF), then extra bits (LL, ML, OF).
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

    // Safety limit: stop writing if we approach max_frame_size
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

    // === Build frequency histograms (codes computed on-the-fly, no arrays) ===
    uint32_t llHist[36]; for (int i = 0; i < 36; i++) llHist[i] = 0;
    uint32_t mlHist[53]; for (int i = 0; i < 53; i++) mlHist[i] = 0;
    uint32_t ofHist[32]; for (int i = 0; i < 32; i++) ofHist[i] = 0;

    uint8_t llMax = 0, mlMax = 0, ofMax = 0;
    uint8_t llMin = 255, mlMin = 255, ofMin = 255;

    for (uint32_t i = 0; i < n_seqs; i++) {
        uint8_t lc = ll_code(my_seqs[i].literal_length);
        uint8_t mc = ml_code(my_seqs[i].match_length);
        uint8_t oc = of_code(my_seqs[i].offset);

        llHist[lc]++; mlHist[mc]++; ofHist[oc]++;
        if (lc > llMax) llMax = lc; if (lc < llMin) llMin = lc;
        if (mc > mlMax) mlMax = mc; if (mc < mlMin) mlMin = mc;
        if (oc > ofMax) ofMax = oc; if (oc < ofMin) ofMin = oc;
    }

    // Per-dimension mode
    bool llRLE = (llMin == llMax);
    bool ofRLE = (ofMin == ofMax);
    bool mlRLE = (mlMin == mlMax);

    // Accuracy logs (minimum 5 per RFC 8878 §4.1.1)
    int llALog = 6, mlALog = 6, ofALog = 5;

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


    // === Build compressed block ===
    uint32_t block_header_pos = pos;
    pos += 3; // reserve for block header

    // Early bailout: if literals alone >= chunk size, raw block is guaranteed smaller.
    // Writing n_lits bytes into the output before the safe_limit check would overflow
    // max_frame_size and corrupt adjacent sub-chunks.
    if (n_lits + 16 >= this_chunk_size) {
        pos = block_header_pos;
        const uint8_t* src_early = input + chunk_start;
        uint32_t bh_early = 1u | (ZSTD_BLOCK_TYPE_RAW << 1) | (this_chunk_size << 3);
        dst[pos++] = (uint8_t)(bh_early);
        dst[pos++] = (uint8_t)(bh_early >> 8);
        dst[pos++] = (uint8_t)(bh_early >> 16);
        for (uint32_t i = 0; i < this_chunk_size; i++) dst[pos++] = src_early[i];
        output_sizes[chunk_idx] = pos;
        return;
    }

    // --- Literals Section (Raw) ---
    if (n_lits < 32) {
        dst[pos++] = (uint8_t)(n_lits << 3);
    } else if (n_lits < 4096) {
        uint16_t hdr = (uint16_t)((n_lits << 4) | 0x04);
        write_le16(dst + pos, hdr); pos += 2;
    } else {
        uint32_t hdr = (n_lits << 4) | 0x0C;
        dst[pos++] = (uint8_t)hdr; dst[pos++] = (uint8_t)(hdr>>8); dst[pos++] = (uint8_t)(hdr>>16);
    }
    for (uint32_t i = 0; i < n_lits; i++) dst[pos++] = my_lits[i];

    // --- Sequences Section Header ---
    if (n_seqs < 128) {
        dst[pos++] = (uint8_t)n_seqs;
    } else if (n_seqs < 0x7F00) {
        dst[pos++] = (uint8_t)((n_seqs >> 8) + 128);
        dst[pos++] = (uint8_t)(n_seqs & 0xFF);
    } else {
        dst[pos++] = 0xFF;
        write_le16(dst + pos, (uint16_t)(n_seqs - 0x7F00)); pos += 2;
    }

    // --- Mode byte: LL, OF, ML (1=RLE, 2=FSE_Compressed) ---
    uint8_t llMode = llRLE ? 1 : 2;
    uint8_t ofMode = ofRLE ? 1 : 2;
    uint8_t mlMode = mlRLE ? 1 : 2;
    dst[pos++] = (llMode << 6) | (ofMode << 4) | (mlMode << 2);

    // --- Table descriptions / RLE symbols (order: LL, OF, ML) ---
    // Mode 0 (Predefined) needs no table desc. Mode 1 (RLE) needs 1 byte. Mode 2 (FSE) needs table.
    if (llMode == 1) { dst[pos++] = llMin; }
    else if (llMode == 2) {
        uint8_t tmp[128];
        uint32_t tblBytes = fse_write_table_desc(tmp, llNC, (int)llMax, llALog);
        for (uint32_t i = 0; i < tblBytes; i++) dst[pos++] = tmp[i];
    }
    // mode 0: no table description needed

    if (ofMode == 1) { dst[pos++] = ofMin; }
    else if (ofMode == 2) {
        uint8_t tmp[128];
        uint32_t tblBytes = fse_write_table_desc(tmp, ofNC, (int)ofMax, ofALog);
        for (uint32_t i = 0; i < tblBytes; i++) dst[pos++] = tmp[i];
    }

    if (mlMode == 1) { dst[pos++] = mlMin; }
    else if (mlMode == 2) {
        uint8_t tmp[128];
        uint32_t tblBytes = fse_write_table_desc(tmp, mlNC, (int)mlMax, mlALog);
        for (uint32_t i = 0; i < tblBytes; i++) dst[pos++] = tmp[i];
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

        // Initialize FSE states (no bits output)
        uint32_t stateLL = 0, stateOF = 0, stateML = 0;
        if (!llRLE) stateLL = fse_init_state(lastLL, &llTbl);
        if (!ofRLE) stateOF = fse_init_state(lastOF, &ofTbl);
        if (!mlRLE) stateML = fse_init_state(lastML, &mlTbl);

        // First sequence (last in array): extra bits only, no FSE state bits
        {
            uint8_t llBits = LL_bits[lastLL];
            if (llBits > 0)
                bs_addBits(&bs, my_seqs[last].literal_length - LL_baseline[lastLL], llBits);

            uint8_t mlBits = ML_bits[lastML];
            if (mlBits > 0)
                bs_addBits(&bs, (my_seqs[last].match_length + 3) - ML_baseline[lastML], mlBits);

            uint8_t ofBits = lastOF;
            if (ofBits > 0)
                bs_addBits(&bs, (my_seqs[last].offset + 3) - (1u << ofBits), ofBits);

            bs_flush(&bs);

        }

        // Remaining sequences backward: FSE state bits then extra bits
        for (int32_t n = (int32_t)n_seqs - 2; n >= 0; n--) {
            if (bs.pos >= safe_limit) goto fallback_raw;

            uint8_t lc = ll_code(my_seqs[n].literal_length);
            uint8_t mc = ml_code(my_seqs[n].match_length);
            uint8_t oc = of_code(my_seqs[n].offset);

            // FSE state bits: OF, ML, LL order (written forward, decoder reads backward as LL, ML, OF)
            if (!ofRLE) fse_encode_symbol(&bs, &stateOF, oc, &ofTbl);
            if (!mlRLE) fse_encode_symbol(&bs, &stateML, mc, &mlTbl);
            if (!llRLE) fse_encode_symbol(&bs, &stateLL, lc, &llTbl);

            bs_flush(&bs);

            // Extra bits: LL, ML, OF order
            uint8_t llBits = LL_bits[lc];
            if (llBits > 0)
                bs_addBits(&bs, my_seqs[n].literal_length - LL_baseline[lc], llBits);

            uint8_t mlBits = ML_bits[mc];
            if (mlBits > 0)
                bs_addBits(&bs, (my_seqs[n].match_length + 3) - ML_baseline[mc], mlBits);

            uint8_t ofBits = oc;
            if (ofBits > 0)
                bs_addBits(&bs, (my_seqs[n].offset + 3) - (1u << ofBits), ofBits);

            bs_flush(&bs);
        }

        // Flush final FSE states: ML, OF, LL order (per reference encoder)
        // States are in [tableSize, 2*tableSize-1]; mask to accuracyLog bits
        // to produce the decoder's initial state value in [0, tableSize-1].
        if (!mlRLE) bs_addBits(&bs, stateML & ((1u << mlTbl.accuracyLog) - 1), mlTbl.accuracyLog);
        if (!ofRLE) bs_addBits(&bs, stateOF & ((1u << ofTbl.accuracyLog) - 1), ofTbl.accuracyLog);
        if (!llRLE) bs_addBits(&bs, stateLL & ((1u << llTbl.accuracyLog) - 1), llTbl.accuracyLog);

        uint32_t seqDataEnd = bs_close(&bs);

        if (seqDataEnd >= max_frame_size) goto fallback_raw;

        uint32_t block_content_size = seqDataEnd - (block_header_pos + 3);

        if (block_content_size >= this_chunk_size) goto fallback_raw;

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
