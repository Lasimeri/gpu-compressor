#![allow(dead_code, unused_assignments)]
use anyhow::Result;
use cudarc::driver::*;
use std::sync::Arc;

use crate::constants::{
    LZMA2_CUSTOM_CHUNK_SIZE, LZMA2_HC4_SEARCH_DEPTH, LZMA2_MAX_MATCHES_PER_POS,
    PTX_LZMA2_MATCH_FIND,
};

// ============================================================================
// Range Coder — LZMA arithmetic encoder
// Reference: LZMA SDK LzmaEnc.c (Igor Pavlov)
// ============================================================================

struct RangeCoder {
    range: u32,
    cache: u8,
    cache_size: u64,
    low: u64,
    output: Vec<u8>,
}

impl RangeCoder {
    fn new() -> Self {
        Self {
            range: 0xFFFFFFFF,
            cache: 0,
            cache_size: 1,
            low: 0,
            output: Vec::with_capacity(65536),
        }
    }

    fn shift_low(&mut self) {
        if self.low < 0xFF000000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            self.output.push(self.cache.wrapping_add(carry));
            let pending = self.cache_size - 1;
            self.cache_size = 0;
            for _ in 0..pending {
                self.output.push((0xFFu8).wrapping_add(carry));
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low << 8) & 0xFFFFFFFF;
    }

    fn encode_bit(&mut self, prob: &mut u16, bit: u32) {
        let bound = (self.range >> 11) * (*prob as u32);
        if bit == 0 {
            self.range = bound;
            *prob += (2048 - *prob) >> 5;
        } else {
            self.low += bound as u64;
            self.range -= bound;
            *prob -= *prob >> 5;
        }
        if self.range < (1 << 24) {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn encode_bit_tree(&mut self, probs: &mut [u16], num_bits: u32, value: u32) {
        let mut m: u32 = 1;
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1;
            self.encode_bit(&mut probs[m as usize], bit);
            m = (m << 1) | bit;
        }
    }

    fn encode_bit_tree_reverse(&mut self, probs: &mut [u16], num_bits: u32, value: u32) {
        let mut m: u32 = 1;
        for i in 0..num_bits {
            let bit = (value >> i) & 1;
            self.encode_bit(&mut probs[m as usize], bit);
            m = (m << 1) | bit;
        }
    }

    fn encode_direct_bits(&mut self, value: u32, num_bits: u32) {
        for i in (0..num_bits).rev() {
            self.range >>= 1;
            if ((value >> i) & 1) != 0 {
                self.low += self.range as u64;
            }
            if self.range < (1 << 24) {
                self.range <<= 8;
                self.shift_low();
            }
        }
    }

    fn flush(&mut self) {
        for _ in 0..5 {
            self.shift_low();
        }
    }
}

// ============================================================================
// LZMA Probability Model — 12-state Markov chain
// ============================================================================

const NUM_STATES: usize = 12;
const NUM_POS_STATES: usize = 4; // 1 << pb (pb=2)
const NUM_LIT_STATES: usize = 7;
const LC: u32 = 3;
const LP: u32 = 0;
const PB: u32 = 2;
const POS_STATE_MASK: u32 = (1 << PB) - 1;

// State transitions
const STATE_AFTER_LIT: [u8; 12] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 4, 5];
const STATE_AFTER_MATCH: [u8; 12] = [7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 10];
const STATE_AFTER_REP: [u8; 12] = [8, 8, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11];
const STATE_AFTER_SHORT_REP: [u8; 12] = [9, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11];

fn is_lit_state(state: u8) -> bool {
    state < NUM_LIT_STATES as u8
}

struct LzmaModel {
    is_match: [[u16; NUM_POS_STATES]; NUM_STATES],
    is_rep: [u16; NUM_STATES],
    is_rep_g0: [u16; NUM_STATES],
    is_rep_g1: [u16; NUM_STATES],
    is_rep_g2: [u16; NUM_STATES],
    is_rep0_long: [[u16; NUM_POS_STATES]; NUM_STATES],

    literal_probs: Vec<u16>, // (1 << (lc + lp)) * 0x300

    match_len_choice: [u16; 2],
    match_len_low: [[u16; 8]; NUM_POS_STATES],  // 3 bits
    match_len_mid: [[u16; 8]; NUM_POS_STATES],  // 3 bits
    match_len_high: [u16; 256],                  // 8 bits

    rep_len_choice: [u16; 2],
    rep_len_low: [[u16; 8]; NUM_POS_STATES],
    rep_len_mid: [[u16; 8]; NUM_POS_STATES],
    rep_len_high: [u16; 256],

    dist_slot: [[u16; 64]; 4], // 6 bits, 4 len states
    dist_special: [u16; 128],  // for distances 4-127
    dist_align: [u16; 16],     // 4 bits

    reps: [u32; 4],
    state: u8,
}

impl LzmaModel {
    fn new() -> Self {
        let lit_size = (1usize << (LC + LP)) * 0x300;
        Self {
            is_match: [[1024; NUM_POS_STATES]; NUM_STATES],
            is_rep: [1024; NUM_STATES],
            is_rep_g0: [1024; NUM_STATES],
            is_rep_g1: [1024; NUM_STATES],
            is_rep_g2: [1024; NUM_STATES],
            is_rep0_long: [[1024; NUM_POS_STATES]; NUM_STATES],
            literal_probs: vec![1024; lit_size],
            match_len_choice: [1024; 2],
            match_len_low: [[1024; 8]; NUM_POS_STATES],
            match_len_mid: [[1024; 8]; NUM_POS_STATES],
            match_len_high: [1024; 256],
            rep_len_choice: [1024; 2],
            rep_len_low: [[1024; 8]; NUM_POS_STATES],
            rep_len_mid: [[1024; 8]; NUM_POS_STATES],
            rep_len_high: [1024; 256],
            dist_slot: [[1024; 64]; 4],
            dist_special: [1024; 128],
            dist_align: [1024; 16],
            reps: [1, 1, 1, 1],
            state: 0,
        }
    }

    fn encode_literal(&mut self, rc: &mut RangeCoder, byte: u8, pos: u32, prev_byte: u8) {
        let pos_state = pos & POS_STATE_MASK;

        // is_match = 0 (literal)
        rc.encode_bit(&mut self.is_match[self.state as usize][pos_state as usize], 0);

        // Literal sub-model selection
        let lit_state = ((pos & ((1 << LP) - 1)) << LC) | ((prev_byte as u32) >> (8 - LC));
        let probs = &mut self.literal_probs[(lit_state as usize) * 0x300..];

        if is_lit_state(self.state) {
            // Normal literal encoding
            let mut symbol = 1u32;
            for i in (0..8).rev() {
                let bit = ((byte as u32) >> i) & 1;
                rc.encode_bit(&mut probs[symbol as usize], bit);
                symbol = (symbol << 1) | bit;
            }
        } else {
            // After match: encode with match-byte context
            let match_byte = prev_byte; // Simplified: should be the byte at rep0 distance
            let mut symbol = 1u32;
            let mut match_bit;
            #[allow(unused_assignments)]
            let mut offset = 0x100u32;
            for i in (0..8).rev() {
                match_bit = ((match_byte as u32) >> i) & 1;
                let bit = ((byte as u32) >> i) & 1;
                rc.encode_bit(&mut probs[(offset + (match_bit << 8) + symbol) as usize], bit);
                symbol = (symbol << 1) | bit;
                if match_bit != bit {
                    offset = 0;
                    // Remaining bits: encode normally
                    for j in (0..i).rev() {
                        let b = ((byte as u32) >> j) & 1;
                        rc.encode_bit(&mut probs[symbol as usize], b);
                        symbol = (symbol << 1) | b;
                    }
                    break;
                }
            }
        }

        self.state = STATE_AFTER_LIT[self.state as usize];
    }

    fn encode_match(&mut self, rc: &mut RangeCoder, distance: u32, length: u32, pos: u32) {
        let pos_state = (pos & POS_STATE_MASK) as usize;

        // is_match = 1
        rc.encode_bit(&mut self.is_match[self.state as usize][pos_state], 1);
        // is_rep = 0 (new match)
        rc.encode_bit(&mut self.is_rep[self.state as usize], 0);

        // Encode length (length >= 2)
        self.encode_length(rc, length - 2, pos_state, false);

        // Encode distance
        let len_state = std::cmp::min(length - 2, 3) as usize;
        let dist = distance - 1; // Convert to 0-based for slot encoding

        // Distance slot (6 bits)
        let slot = self.get_dist_slot(dist);
        rc.encode_bit_tree(&mut self.dist_slot[len_state], 6, slot);

        if slot >= 4 {
            let footer_bits = (slot >> 1) - 1;
            let base = (2 | (slot & 1)) << footer_bits;
            let dist_reduced = dist - base;

            if slot < 14 {
                // Fixed-probability bits via reverse bit tree
                let base_idx = (base - slot) as usize;
                rc.encode_bit_tree_reverse(
                    &mut self.dist_special[base_idx..],
                    footer_bits,
                    dist_reduced,
                );
            } else {
                // Direct bits (middle) + alignment bits (4 bits)
                rc.encode_direct_bits(dist_reduced >> 4, footer_bits - 4);
                rc.encode_bit_tree_reverse(&mut self.dist_align, 4, dist_reduced & 0xF);
            }
        }

        // Update reps
        self.reps[3] = self.reps[2];
        self.reps[2] = self.reps[1];
        self.reps[1] = self.reps[0];
        self.reps[0] = distance;
        self.state = STATE_AFTER_MATCH[self.state as usize];
    }

    fn encode_length(
        &mut self,
        rc: &mut RangeCoder,
        len_minus2: u32,
        pos_state: usize,
        is_rep: bool,
    ) {
        let (choice, low, mid, high) = if is_rep {
            (
                &mut self.rep_len_choice,
                &mut self.rep_len_low,
                &mut self.rep_len_mid,
                &mut self.rep_len_high,
            )
        } else {
            (
                &mut self.match_len_choice,
                &mut self.match_len_low,
                &mut self.match_len_mid,
                &mut self.match_len_high,
            )
        };

        if len_minus2 < 8 {
            rc.encode_bit(&mut choice[0], 0);
            rc.encode_bit_tree(&mut low[pos_state], 3, len_minus2);
        } else if len_minus2 < 8 + 8 {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 0);
            rc.encode_bit_tree(&mut mid[pos_state], 3, len_minus2 - 8);
        } else {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 1);
            rc.encode_bit_tree(high, 8, len_minus2 - 16);
        }
    }

    fn get_dist_slot(&self, dist: u32) -> u32 {
        if dist < 4 {
            return dist;
        }
        let bits = 32 - dist.leading_zeros() - 1; // floor(log2(dist))
        ((bits << 1) | ((dist >> (bits - 1)) & 1)) as u32
    }
}

// ============================================================================
// LZMA2 Raw Block Helper
// ============================================================================

/// Emit raw (uncompressed) LZMA2 chunks, splitting at 64KB boundaries.
fn emit_raw_lzma2_chunks(out: &mut Vec<u8>, data: &[u8]) {
    let mut offset = 0;
    let mut first = true;
    while offset < data.len() {
        let chunk_len = (data.len() - offset).min(65536);
        out.push(if first { 0x01 } else { 0x02 });
        let us = (chunk_len - 1) as u16;
        out.push((us >> 8) as u8);
        out.push(us as u8);
        out.extend_from_slice(&data[offset..offset + chunk_len]);
        offset += chunk_len;
        first = false;
    }
}

// ============================================================================
// LZMA2 Block Assembly
// ============================================================================

fn encode_lzma2_block(
    data: &[u8],
    match_distances: &[u32],
    match_lengths: &[u32],
    match_counts: &[u32],
    max_matches: usize,
    _sub_block_size: usize,
) -> Vec<u8> {
    let mut rc = RangeCoder::new();
    let mut model = LzmaModel::new();

    // LZMA2 max uncompressed per chunk is 2MB. For 64KB sub-blocks,
    // ensure we don't hit edge cases at exact power-of-2 boundaries.
    let actual_size = data.len();

    // Greedy encoding using GPU-found matches
    let mut pos = 0u32;
    let mut prev_byte = 0u8;

    while (pos as usize) < actual_size {
        let p = pos as usize;
        let n_matches = if p < match_counts.len() {
            match_counts[p] as usize
        } else {
            0
        };

        // Find best match at this position
        let mut best_len = 0u32;
        let mut best_dist = 0u32;

        for k in 0..n_matches.min(max_matches) {
            let idx = p * max_matches + k;
            if idx >= match_distances.len() {
                break;
            }
            let d = match_distances[idx];
            let l = match_lengths[idx];
            if l > best_len {
                best_len = l;
                best_dist = d;
            }
        }

        if best_len >= 2 && best_dist > 0 {
            // Validate match: ensure the referenced data actually matches
            let match_start = (pos as usize).wrapping_sub(best_dist as usize);
            let match_valid = match_start < actual_size
                && (pos as usize) + (best_len as usize) <= actual_size
                && match_start + (best_len as usize) <= actual_size
                && data[match_start..match_start + best_len as usize]
                    == data[pos as usize..(pos as usize) + best_len as usize];

            if !match_valid {
                let byte = data[p];
                model.encode_literal(&mut rc, byte, pos, prev_byte);
                prev_byte = byte;
                pos += 1;
                continue;
            }

            // Clamp match length to not exceed block boundary
            best_len = best_len.min((actual_size - p) as u32);
            if best_len < 2 {
                let byte = data[p];
                model.encode_literal(&mut rc, byte, pos, prev_byte);
                prev_byte = byte;
                pos += 1;
                continue;
            }

            model.encode_match(&mut rc, best_dist, best_len, pos);
            for _ in 0..best_len {
                if (pos as usize) < actual_size {
                    prev_byte = data[pos as usize];
                }
                pos += 1;
            }
        } else {
            // Literal
            let byte = data[p];
            model.encode_literal(&mut rc, byte, pos, prev_byte);
            prev_byte = byte;
            pos += 1;
        }
    }

    rc.flush();
    let compressed_data = rc.output;

    let uncompressed_size = actual_size;
    let compressed_size = compressed_data.len();

    // LZMA2 uncompressed chunk limit: 64KB (2-byte size field).
    // Compressed chunks: 2MB uncompressed / 64KB compressed per chunk.
    if compressed_size >= uncompressed_size || compressed_size > 65536 {
        let mut block = Vec::with_capacity(uncompressed_size + uncompressed_size / 65536 * 4 + 4);
        emit_raw_lzma2_chunks(&mut block, data);
        return block;
    }

    // Compressed data fits in one LZMA2 chunk (both sizes < 65536)
    let mut block = Vec::with_capacity(compressed_size + 10);

    let control = 0xE0u8 | (((uncompressed_size - 1) >> 16) as u8 & 0x1F);
    block.push(control);

    let us = (uncompressed_size - 1) as u16;
    block.push((us >> 8) as u8);
    block.push(us as u8);

    let cs = (compressed_size - 1) as u16;
    block.push((cs >> 8) as u8);
    block.push(cs as u8);

    let props = LC + LP * 9 + PB * 45;
    block.push(props as u8);

    block.extend_from_slice(&compressed_data);

    block
}

// ============================================================================
// GPU Match Finding + Custom LZMA2 Encoding
// ============================================================================

/// Match data from GPU — shared between GPU thread and CPU encoder threads.
pub(crate) struct GpuMatchData {
    pub match_distances: Vec<u32>,
    pub match_lengths: Vec<u32>,
    pub match_counts: Vec<u32>,
    pub num_sub_blocks: usize,
    pub sub_block_size: usize,
    pub max_matches: usize,
}

/// Phase 1: GPU match finding only. Fast (milliseconds).
/// Returns match data that can be encoded on CPU threads.
pub(crate) fn gpu_find_matches(
    chunk_data: &[u8],
    device_id: i32,
) -> Result<GpuMatchData> {
    let sub_block_size = LZMA2_CUSTOM_CHUNK_SIZE;
    let total_size = chunk_data.len();
    let num_sub_blocks = total_size.div_ceil(sub_block_size);
    let max_matches = LZMA2_MAX_MATCHES_PER_POS;

    unsafe {
        let result = cuda_runtime_sys::cudaSetDevice(device_id);
        if result != cuda_runtime_sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }
    }

    let device = CudaDevice::new(device_id as usize)?;
    let dev = Arc::new(device);

    dev.load_ptx(
        PTX_LZMA2_MATCH_FIND.into(),
        "lzma2_module",
        &["lzma2_match_find"],
    )?;

    let d_input = dev.htod_sync_copy(chunk_data)?;

    let match_buf_size = num_sub_blocks * sub_block_size * max_matches * 2;
    let d_matches = dev.alloc_zeros::<u32>(match_buf_size)?;
    let d_match_counts = dev.alloc_zeros::<u32>(num_sub_blocks * sub_block_size)?;

    let func = dev
        .get_func("lzma2_module", "lzma2_match_find")
        .ok_or_else(|| anyhow::anyhow!("Failed to load lzma2_match_find kernel"))?;

    unsafe {
        func.launch(
            LaunchConfig {
                grid_dim: (num_sub_blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &d_input,
                &d_matches,
                &d_match_counts,
                sub_block_size as u32,
                num_sub_blocks as u32,
                total_size as u32,
                max_matches as u32,
                LZMA2_HC4_SEARCH_DEPTH,
            ),
        )?;
    }
    dev.synchronize()?;

    let h_matches = dev.dtoh_sync_copy(&d_matches)?;
    let h_match_counts = dev.dtoh_sync_copy(&d_match_counts)?;

    let total_match_entries = num_sub_blocks * sub_block_size * max_matches;
    let mut match_distances = vec![0u32; total_match_entries];
    let mut match_lengths = vec![0u32; total_match_entries];

    for i in 0..total_match_entries {
        match_distances[i] = h_matches[i * 2];
        match_lengths[i] = h_matches[i * 2 + 1];
    }

    Ok(GpuMatchData {
        match_distances,
        match_lengths,
        match_counts: h_match_counts,
        num_sub_blocks,
        sub_block_size,
        max_matches,
    })
}

/// Encode a single sub-block. GPU triage + liblzma encoding.
///
/// The GPU match finder classifies each sub-block:
/// - No matches found → incompressible → emit raw LZMA2 block (GPU speed, no CPU work)
/// - Matches found → compressible → encode with liblzma preset 9 (L0-quality ratio)
///
/// This gives the best of both worlds: GPU-accelerated incompressibility detection
/// skips the expensive liblzma encoder for random/pre-compressed data, while
/// compressible blocks get full BT4 match finding + optimal parsing from liblzma.
pub(crate) fn encode_single_sub_block(
    sub_data: &[u8],
    _match_distances: &[u32],
    _match_lengths: &[u32],
    match_counts: &[u32],
    _max_matches: usize,
) -> Vec<u8> {
    let has_matches = match_counts.iter().any(|&c| c > 0);

    if !has_matches {
        // GPU says: no matches → incompressible → raw block (zero CPU cost)
        let mut block = Vec::with_capacity(sub_data.len() + sub_data.len() / 65536 * 4 + 4);
        emit_raw_lzma2_chunks(&mut block, sub_data);
        block.push(0x00);
        return block;
    }

    // GPU says: matches exist → compressible → use liblzma preset 9 for best ratio
    match crate::compress_lzma2::compress_chunk_lzma2(sub_data, 64 * 1024 * 1024, 9) {
        Ok((chunks, _)) => chunks.into_iter().next().unwrap_or_default(),
        Err(_) => {
            let mut block = Vec::with_capacity(sub_data.len() + sub_data.len() / 65536 * 4 + 4);
            emit_raw_lzma2_chunks(&mut block, sub_data);
            block.push(0x00);
            block
        }
    }
}

/// Phase 2: CPU encoding from GPU match data. Parallelizes across sub-blocks.
/// Returns one LZMA2 stream per sub-block (each independently decodable with end marker).
pub(crate) fn cpu_encode_from_matches(
    chunk_data: &[u8],
    matches: &GpuMatchData,
) -> (Vec<Vec<u8>>, Vec<usize>) {
    let total_size = chunk_data.len();

    let encoded_blocks: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..matches.num_sub_blocks)
            .map(|i| {
                let sub_start = i * matches.sub_block_size;
                let sub_end = (sub_start + matches.sub_block_size).min(total_size);
                let sub_data = &chunk_data[sub_start..sub_end];
                let mc_start = i * matches.sub_block_size;
                let mc_end = mc_start + (sub_end - sub_start);
                let md_start = i * matches.sub_block_size * matches.max_matches;
                let md_end = md_start + (sub_end - sub_start) * matches.max_matches;
                let md = &matches.match_distances[md_start..md_end];
                let ml = &matches.match_lengths[md_start..md_end];
                let mc = &matches.match_counts[mc_start..mc_end];
                let max_matches = matches.max_matches;
                let sub_block_size = matches.sub_block_size;

                s.spawn(move || {
                    // Check if any matches were found in this sub-block
                    let has_matches = mc.iter().any(|&c| c > 0);

                    if !has_matches {
                        // No matches — emit raw LZMA2 chunks (64KB each)
                        let mut block = Vec::with_capacity(sub_data.len() + sub_data.len() / 65536 * 4 + 4);
                        emit_raw_lzma2_chunks(&mut block, sub_data);
                        block.push(0x00);
                        return block;
                    }

                    let mut block =
                        encode_lzma2_block(sub_data, md, ml, mc, max_matches, sub_block_size);
                    block.push(0x00);

                    // Verify blocks with matches (match encoding has edge cases).
                    if let Err(_) = crate::compress_lzma2::decompress_chunk_lzma2(
                        &block,
                        sub_data.len(),
                    ) {
                        block.clear();
                        emit_raw_lzma2_chunks(&mut block, sub_data);
                        block.push(0x00);
                    }
                    block
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut compressed_chunks = Vec::with_capacity(matches.num_sub_blocks);
    let mut sizes = Vec::with_capacity(matches.num_sub_blocks);
    for block in encoded_blocks {
        sizes.push(block.len());
        compressed_chunks.push(block);
    }

    (compressed_chunks, sizes)
}

/// Combined single-call API (for backward compatibility).
pub(crate) fn compress_chunk_lzma2_custom(
    chunk_data: &[u8],
    device_id: i32,
    _level: u32,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let matches = gpu_find_matches(chunk_data, device_id)?;
    Ok(cpu_encode_from_matches(chunk_data, &matches))
}
