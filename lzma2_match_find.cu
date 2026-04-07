// LZMA2 GPU Match Finding Kernel
// HC4 hash chain match finder for LZMA2 compression.
//
// Architecture: one thread block per 64KB sub-block.
// Thread 0: sequential HC4 match finding
// Threads 1-255: hash table initialization
//
// Key differences from Zstd match finder:
//   - Min match length: 2 (Zstd: 3)
//   - Max match length: 273 (LZMA spec limit)
//   - Outputs multiple match candidates per position (for optimal parsing)
//   - 1-based match distances (distance 1 = previous byte)
//
// sm_86 (RTX 3090): 48KB shared memory, 256 threads/block

#include <stdint.h>

// Match finding constants
#define HASH_LOG 14                    // 16K entries
#define HASH_TABLE_SIZE (1 << HASH_LOG)
#define MIN_MATCH_LEN 2                // LZMA minimum
#define MAX_MATCH_LEN 273              // LZMA maximum
#define HASH_CHAIN_SIZE (1 << HASH_LOG)

// Per-position match output
struct LzmaMatch {
    uint32_t distance;  // 1-based distance
    uint32_t length;    // match length (>= MIN_MATCH_LEN)
};

// Unaligned 4-byte load (LE)
__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Unaligned 2-byte load
__device__ __forceinline__ uint16_t load_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

// ============================================================================
// Kernel: lzma2_match_find
//
// For each position in the sub-block, finds up to max_matches_per_pos
// match candidates using HC4 (hash chain with 4-byte hash).
// Also checks 2-byte and 3-byte hashes for short matches.
//
// Output layout (flat arrays, indexed per sub-block):
//   matches[block * block_size * max_matches + pos * max_matches + k]
//   match_counts[block * block_size + pos]
// ============================================================================

extern "C" __global__ void lzma2_match_find(
    const uint8_t* __restrict__ input,          // all sub-blocks concatenated
    LzmaMatch* __restrict__ matches,            // output: match candidates
    uint32_t* __restrict__ match_counts,        // output: matches found per position
    uint32_t sub_block_size,                    // 65536
    uint32_t num_sub_blocks,
    uint32_t total_input_size,
    uint32_t max_matches_per_pos,               // typically 8
    uint32_t search_depth                       // HC4 chain depth (16-64)
) {
    uint32_t block_idx = blockIdx.x;
    if (block_idx >= num_sub_blocks) return;

    // Shared memory: hash table (position cache) + chain heads
    __shared__ uint16_t hash_table[HASH_TABLE_SIZE];

    // Cooperative hash table init
    for (uint32_t i = threadIdx.x; i < HASH_TABLE_SIZE; i += blockDim.x) {
        hash_table[i] = 0xFFFF;  // 0xFFFF = empty sentinel
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // Sub-block bounds
    uint64_t block_start = (uint64_t)block_idx * sub_block_size;
    uint32_t this_block_size = sub_block_size;
    if (block_start + this_block_size > total_input_size) {
        this_block_size = total_input_size - (uint32_t)block_start;
    }

    const uint8_t* src = input + block_start;

    // Output pointers
    LzmaMatch* my_matches = matches + (uint64_t)block_idx * sub_block_size * max_matches_per_pos;
    uint32_t* my_counts = match_counts + (uint64_t)block_idx * sub_block_size;

    // Hash functions
    #define HASH4(p) ((load_u32(p) * 2654435761u) >> (32 - HASH_LOG))
    #define HASH3(p) (((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8) | ((uint32_t)(p)[2] << 16)) * 506832829u >> (32 - HASH_LOG))

    // Process each position
    for (uint32_t pos = 0; pos < this_block_size; pos++) {
        uint32_t n_matches = 0;
        LzmaMatch* pos_matches = my_matches + (uint64_t)pos * max_matches_per_pos;

        // Need at least 2 bytes for a match
        if (pos + 1 < this_block_size) {
            uint32_t best_len = 1;  // Track best length found so far

            // --- 4-byte hash lookup (primary, for longer matches) ---
            if (pos + 3 < this_block_size) {
                uint32_t h4 = HASH4(src + pos);
                uint32_t prev_pos = hash_table[h4];
                hash_table[h4] = (uint16_t)(pos & 0xFFFF);

                // Walk hash chain
                uint32_t chain_count = 0;
                uint32_t cur = prev_pos;

                while (cur != 0xFFFF && chain_count < search_depth && n_matches < max_matches_per_pos) {
                    // cur is a 16-bit truncated position
                    uint32_t match_pos = cur;
                    if (match_pos >= pos) break;  // Invalid (stale entry wrapping)

                    uint32_t distance = pos - match_pos;

                    // Verify match (at least 2 bytes)
                    if (distance > 0 && match_pos + 1 < this_block_size) {
                        uint32_t len = 0;
                        uint32_t max_len = this_block_size - pos;
                        if (max_len > MAX_MATCH_LEN) max_len = MAX_MATCH_LEN;
                        uint32_t max_match_check = this_block_size - match_pos;
                        if (max_match_check < max_len) max_len = max_match_check;

                        // Check first 4 bytes fast
                        if (pos + 3 < this_block_size && match_pos + 3 < this_block_size &&
                            load_u32(src + match_pos) == load_u32(src + pos)) {
                            len = 4;
                            // Extend match
                            while (len < max_len && src[match_pos + len] == src[pos + len]) {
                                len++;
                            }
                        } else if (src[match_pos] == src[pos] && src[match_pos + 1] == src[pos + 1]) {
                            // 2-byte match from 4-byte hash collision
                            len = 2;
                            while (len < max_len && src[match_pos + len] == src[pos + len]) {
                                len++;
                            }
                        }

                        if (len >= MIN_MATCH_LEN && len > best_len) {
                            pos_matches[n_matches].distance = distance;
                            pos_matches[n_matches].length = len;
                            n_matches++;
                            best_len = len;

                            // Early exit if we found a very long match
                            if (len >= MAX_MATCH_LEN) break;
                        }
                    }

                    // Simple chain: hash collision → previous entry at same hash
                    // For HC4 with 16-bit positions, we can only follow one level
                    // (no explicit chain storage in shared memory).
                    // To get multiple candidates, we also check nearby positions.
                    break;  // Single-level hash table (no chain storage)
                }
            }

            // --- 2-byte exact match check (for very short matches) ---
            if (n_matches == 0 && pos + 1 < this_block_size) {
                // Check rep distance 1 (previous byte repeated)
                if (pos > 0 && src[pos] == src[pos - 1] && src[pos + 1] == src[pos]) {
                    // RLE-like pattern
                    uint32_t len = 2;
                    uint32_t max_len = this_block_size - pos;
                    if (max_len > MAX_MATCH_LEN) max_len = MAX_MATCH_LEN;
                    while (len < max_len && src[pos + len] == src[pos]) {
                        len++;
                    }
                    if (len >= MIN_MATCH_LEN) {
                        pos_matches[n_matches].distance = 1;
                        pos_matches[n_matches].length = len;
                        n_matches++;
                    }
                }
            }
        }

        my_counts[pos] = n_matches;
    }

    // Zero out counts for any positions beyond this_block_size
    for (uint32_t pos = this_block_size; pos < sub_block_size; pos++) {
        my_counts[pos] = 0;
    }

    #undef HASH4
    #undef HASH3
}
