// BLAKE3 GPU Implementation - Maximum Parallelism
// Each GPU thread processes 1KB (1 BLAKE3 chunk) for maximum GPU utilization
// RTX 3090: 10,496 CUDA cores - need lots of threads!
#include <stdint.h>
#include <cuda_runtime.h>

// BLAKE3 constants
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024  // BLAKE3 spec: 1KB chunks
#define BLAKE3_OUT_LEN 32

// IV for BLAKE3
__constant__ uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Message permutation for BLAKE3
__constant__ uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}
};

// Flags
#define CHUNK_START 1
#define CHUNK_END 2
#define PARENT 4
#define ROOT 8

// Right rotation
__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// G mixing function
__device__ __forceinline__ void g(uint32_t *state, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                   uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

// Round function
__device__ __forceinline__ void round_fn(uint32_t state[16], uint32_t *m, uint32_t round) {
    // Mix columns
    g(state, 0, 4, 8, 12, m[MSG_SCHEDULE[round][0]], m[MSG_SCHEDULE[round][1]]);
    g(state, 1, 5, 9, 13, m[MSG_SCHEDULE[round][2]], m[MSG_SCHEDULE[round][3]]);
    g(state, 2, 6, 10, 14, m[MSG_SCHEDULE[round][4]], m[MSG_SCHEDULE[round][5]]);
    g(state, 3, 7, 11, 15, m[MSG_SCHEDULE[round][6]], m[MSG_SCHEDULE[round][7]]);

    // Mix diagonals
    g(state, 0, 5, 10, 15, m[MSG_SCHEDULE[round][8]], m[MSG_SCHEDULE[round][9]]);
    g(state, 1, 6, 11, 12, m[MSG_SCHEDULE[round][10]], m[MSG_SCHEDULE[round][11]]);
    g(state, 2, 7, 8, 13, m[MSG_SCHEDULE[round][12]], m[MSG_SCHEDULE[round][13]]);
    g(state, 3, 4, 9, 14, m[MSG_SCHEDULE[round][14]], m[MSG_SCHEDULE[round][15]]);
}

// Compress a single block
__device__ void compress(uint32_t chaining_value[8], const uint8_t *block,
                        uint8_t block_len, uint64_t counter, uint8_t flags,
                        uint32_t out[16]) {
    uint32_t state[16];
    uint32_t block_words[16];

    // Load block as words (little-endian)
    for (int i = 0; i < 16; i++) {
        block_words[i] = ((uint32_t)block[i * 4 + 0]) |
                        ((uint32_t)block[i * 4 + 1] << 8) |
                        ((uint32_t)block[i * 4 + 2] << 16) |
                        ((uint32_t)block[i * 4 + 3] << 24);
    }

    // Initialize state
    for (int i = 0; i < 8; i++) {
        state[i] = chaining_value[i];
        state[i + 8] = IV[i];
    }
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = (uint32_t)block_len;
    state[15] = (uint32_t)flags;

    // 7 rounds
    #pragma unroll
    for (int round = 0; round < 7; round++) {
        round_fn(state, block_words, round);
    }

    // Output
    for (int i = 0; i < 8; i++) {
        out[i] = state[i] ^ state[i + 8];
        out[i + 8] = state[i + 8] ^ chaining_value[i];
    }
}

// Hash a single 1024-byte chunk
__device__ void hash_chunk(const uint8_t *chunk, uint64_t chunk_counter,
                          uint32_t key[8], uint8_t flags, uint32_t out[8],
                          uint32_t legacy_mode) {
    uint32_t chaining_value[8];

    // Initialize with key
    for (int i = 0; i < 8; i++) {
        chaining_value[i] = key[i];
    }

    // Process 16 blocks of 64 bytes each
    uint32_t block_output[16];
    for (int block_idx = 0; block_idx < 16; block_idx++) {
        const uint8_t *block = chunk + (block_idx * BLAKE3_BLOCK_LEN);
        uint8_t block_flags = flags;
        if (legacy_mode) {
            block_flags |= CHUNK_START;
        } else {
            if (block_idx == 0) block_flags |= CHUNK_START;
        }

        if (block_idx == 15) {
            block_flags |= CHUNK_END;
        }

        compress(chaining_value, block, BLAKE3_BLOCK_LEN, chunk_counter,
                block_flags, block_output);

        // Update chaining value with first 8 words
        for (int i = 0; i < 8; i++) {
            chaining_value[i] = block_output[i];
        }
    }

    // Final output
    for (int i = 0; i < 8; i++) {
        out[i] = chaining_value[i];
    }
}

// Parent hash for tree
__device__ void hash_parent(uint32_t left_child[8], uint32_t right_child[8],
                           uint32_t key[8], uint8_t flags, uint32_t out[8]) {
    uint8_t block[64];

    // Pack both children into one block (8 words each = 32 bytes each)
    for (int i = 0; i < 8; i++) {
        block[i * 4 + 0] = (uint8_t)(left_child[i]);
        block[i * 4 + 1] = (uint8_t)(left_child[i] >> 8);
        block[i * 4 + 2] = (uint8_t)(left_child[i] >> 16);
        block[i * 4 + 3] = (uint8_t)(left_child[i] >> 24);

        block[32 + i * 4 + 0] = (uint8_t)(right_child[i]);
        block[32 + i * 4 + 1] = (uint8_t)(right_child[i] >> 8);
        block[32 + i * 4 + 2] = (uint8_t)(right_child[i] >> 16);
        block[32 + i * 4 + 3] = (uint8_t)(right_child[i] >> 24);
    }

    uint32_t output[16];
    compress(key, block, 64, 0, flags | PARENT, output);

    for (int i = 0; i < 8; i++) {
        out[i] = output[i];
    }
}

// Maximum parallelism chunk hashing kernel
// Each thread processes exactly 1KB (1 chunk) for maximum GPU utilization
extern "C" __global__ void blake3_hash_chunks(
    const uint8_t* __restrict__ data,
    uint64_t file_len,
    uint32_t* __restrict__ chunk_outputs,  // Output for each chunk (8 u32s per chunk)
    uint64_t num_chunks,
    uint32_t legacy_mode
) {
    uint64_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (chunk_idx >= num_chunks) return;

    // Key = IV for file hashing
    uint32_t key[8];
    for (int i = 0; i < 8; i++) {
        key[i] = IV[i];
    }

    // Calculate this chunk's position
    uint64_t chunk_start = chunk_idx * BLAKE3_CHUNK_LEN;
    uint64_t chunk_len = BLAKE3_CHUNK_LEN;

    // Handle last chunk (may be partial)
    if (chunk_start + chunk_len > file_len) {
        chunk_len = file_len - chunk_start;
    }

    uint32_t chunk_hash[8];
    uint8_t flags = 0;

    // For full chunks, use optimized path
    if (chunk_len == BLAKE3_CHUNK_LEN) {
        hash_chunk(data + chunk_start, chunk_idx, key, flags, chunk_hash, legacy_mode);
    } else {
        // Partial chunk - process block by block
        uint32_t chaining_value[8];
        for (int j = 0; j < 8; j++) {
            chaining_value[j] = key[j];
        }

        uint64_t blocks_in_chunk = (chunk_len + BLAKE3_BLOCK_LEN - 1) / BLAKE3_BLOCK_LEN;

        for (uint64_t block_idx = 0; block_idx < blocks_in_chunk; block_idx++) {
            uint8_t block[64] = {0};  // Zero-padded
            uint64_t block_start = chunk_start + (block_idx * BLAKE3_BLOCK_LEN);
            uint64_t block_len = BLAKE3_BLOCK_LEN;

            if (block_start + block_len > chunk_start + chunk_len) {
                block_len = (chunk_start + chunk_len) - block_start;
            }

            // Copy block data
            for (uint64_t j = 0; j < block_len; j++) {
                block[j] = data[block_start + j];
            }

            uint8_t block_flags = flags;
            if (legacy_mode) {
                block_flags |= CHUNK_START;
            } else {
                if (block_idx == 0) block_flags |= CHUNK_START;
            }
            if (block_idx == blocks_in_chunk - 1) {
                block_flags |= CHUNK_END;
            }

            uint32_t block_output[16];
            compress(chaining_value, block, (uint8_t)block_len, chunk_idx,
                    block_flags, block_output);

            for (int j = 0; j < 8; j++) {
                chaining_value[j] = block_output[j];
            }
        }

        for (int j = 0; j < 8; j++) {
            chunk_hash[j] = chaining_value[j];
        }
    }

    // Write chunk output
    uint32_t *output = chunk_outputs + (chunk_idx * 8);
    for (int i = 0; i < 8; i++) {
        output[i] = chunk_hash[i];
    }
}

// Tree reduction kernel - combines chunk hashes into final hash
extern "C" __global__ void blake3_reduce_tree(
    uint32_t* __restrict__ hashes,
    uint64_t num_hashes,
    uint32_t* __restrict__ output,
    uint64_t level
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t pairs_in_level = num_hashes / 2;

    if (idx >= pairs_in_level) return;

    uint32_t key[8];
    for (int i = 0; i < 8; i++) {
        key[i] = IV[i];
    }

    uint32_t left[8], right[8];
    for (int i = 0; i < 8; i++) {
        left[i] = hashes[idx * 2 * 8 + i];
        right[i] = hashes[(idx * 2 + 1) * 8 + i];
    }

    uint32_t parent[8];
    uint8_t flags = 0;

    // Check if this is the root
    if (pairs_in_level == 1 && num_hashes % 2 == 0) {
        flags |= ROOT;
    }

    hash_parent(left, right, key, flags, parent);

    // Write back
    for (int i = 0; i < 8; i++) {
        output[idx * 8 + i] = parent[i];
    }
}
