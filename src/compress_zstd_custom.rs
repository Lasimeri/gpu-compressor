use anyhow::Result;
use cudarc::driver::*;
use std::sync::Arc;

use crate::constants::{CUSTOM_ZSTD_CHUNK_SIZE, CUSTOM_ZSTD_SEARCH_DEPTH_LAZY, PTX_ZSTD_COMPRESS};

/// Compress a pipeline chunk (typically 4MB) using the custom GPU Zstd kernel.
/// Returns compressed sub-chunks and their sizes, compatible with the existing pipeline.
///
/// The kernel splits the input into 64KB sub-chunks, finds LZ77 matches on GPU,
/// and assembles spec-compliant Zstd frames. Each sub-chunk becomes one Zstd frame.
pub(crate) fn compress_chunk_zstd_custom(
    chunk_data: &[u8],
    device_id: i32,
    level: u32,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let chunk_size = CUSTOM_ZSTD_CHUNK_SIZE; // 64KB sub-chunks
    let total_size = chunk_data.len();
    let num_sub_chunks = total_size.div_ceil(chunk_size);

    // Max output per sub-chunk: frame header (10B) + raw data (chunk_size)
    // For compressed blocks, output should be smaller, but we allocate for worst case
    let max_frame_size = 10 + chunk_size;

    // Max sequences per sub-chunk (worst case: every 3 bytes is a match)
    let max_sequences_per_chunk = chunk_size / 3;

    let search_depth = match level {
        1 => CUSTOM_ZSTD_SEARCH_DEPTH_LAZY as u32,
        2 => 64u32, // optimal
        _ => 1u32,  // greedy
    };

    unsafe {
        // Initialize CUDA device
        let result = cuda_runtime_sys::cudaSetDevice(device_id);
        if result != cuda_runtime_sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }
    }

    // Initialize cudarc device
    let device = CudaDevice::new(device_id as usize)?;
    let dev = Arc::new(device);

    // Load custom Zstd kernels
    dev.load_ptx(
        PTX_ZSTD_COMPRESS.into(),
        "zstd_module",
        &[
            "zstd_compress_raw",
            "zstd_match_find",
            "zstd_encode_block",
        ],
    )?;

    // Upload input data to GPU
    let d_input = dev.htod_sync_copy(chunk_data)?;

    // For level 0 or as initial implementation: use raw block wrapping
    // For level >= 1: use match finding + encoding
    if level == 0 {
        return compress_raw_blocks(&dev, &d_input, total_size, num_sub_chunks, chunk_size, max_frame_size);
    }

    // === Level 1+: Match finding + encoding pipeline ===

    // Allocate GPU buffers for match finder output
    // Sequence is 3x u32 = 12 bytes; allocate as u32 for proper alignment
    let d_sequences = dev.alloc_zeros::<u32>(num_sub_chunks * max_sequences_per_chunk * 3)?;
    // Each sub-chunk can produce up to chunk_size literals; allocate full capacity
    let d_literals = dev.alloc_zeros::<u8>(num_sub_chunks * chunk_size)?;
    let d_seq_counts = dev.alloc_zeros::<u32>(num_sub_chunks)?;
    let d_lit_counts = dev.alloc_zeros::<u32>(num_sub_chunks)?;

    // Launch match finding kernel
    let func_match = dev
        .get_func("zstd_module", "zstd_match_find")
        .ok_or_else(|| anyhow::anyhow!("Failed to load zstd_match_find kernel"))?;

    let match_cfg = LaunchConfig {
        grid_dim: (num_sub_chunks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0, // hash table uses static __shared__ memory in kernel
    };

    unsafe {
        func_match.launch(
            match_cfg,
            (
                &d_input,
                &d_sequences,
                &d_literals,
                &d_seq_counts,
                &d_lit_counts,
                chunk_size as u32,
                num_sub_chunks as u32,
                total_size as u32,
                search_depth,
                max_sequences_per_chunk as u32,
            ),
        )?;
    }
    dev.synchronize()?;

    // Allocate output buffer for encoded frames
    let d_output = dev.alloc_zeros::<u8>(num_sub_chunks * max_frame_size)?;
    let d_output_sizes = dev.alloc_zeros::<u32>(num_sub_chunks)?;

    // Launch block encoding kernel
    let func_encode = dev
        .get_func("zstd_module", "zstd_encode_block")
        .ok_or_else(|| anyhow::anyhow!("Failed to load zstd_encode_block kernel"))?;

    let encode_cfg = LaunchConfig {
        grid_dim: (num_sub_chunks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func_encode.launch(
            encode_cfg,
            (
                &d_input,
                &d_sequences,
                &d_literals,
                &d_seq_counts,
                &d_lit_counts,
                &d_output,
                &d_output_sizes,
                chunk_size as u32,
                num_sub_chunks as u32,
                total_size as u32,
                max_sequences_per_chunk as u32,
                max_frame_size as u32,
            ),
        )?;
    }
    dev.synchronize()?;

    // Download output sizes
    let h_output_sizes = dev.dtoh_sync_copy(&d_output_sizes)?;

    // Download compressed frames
    let h_output = dev.dtoh_sync_copy(&d_output)?;

    // Extract individual frames
    let mut compressed_chunks = Vec::with_capacity(num_sub_chunks);
    let mut sizes = Vec::with_capacity(num_sub_chunks);

    for i in 0..num_sub_chunks {
        let frame_size = h_output_sizes[i] as usize;
        let frame_start = i * max_frame_size;
        let frame_data = h_output[frame_start..frame_start + frame_size].to_vec();

        sizes.push(frame_size);
        compressed_chunks.push(frame_data);
    }

    Ok((compressed_chunks, sizes))
}

/// Fallback: wrap each sub-chunk as a raw (uncompressed) Zstd frame.
fn compress_raw_blocks(
    dev: &Arc<CudaDevice>,
    d_input: &CudaSlice<u8>,
    total_size: usize,
    num_sub_chunks: usize,
    chunk_size: usize,
    max_frame_size: usize,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let func_raw = dev
        .get_func("zstd_module", "zstd_compress_raw")
        .ok_or_else(|| anyhow::anyhow!("Failed to load zstd_compress_raw kernel"))?;

    // Allocate output
    let d_output = dev.alloc_zeros::<u8>(num_sub_chunks * max_frame_size)?;
    let d_output_sizes = dev.alloc_zeros::<u32>(num_sub_chunks)?;

    let threads_per_block = 256u32;
    let num_blocks = (num_sub_chunks as u32).div_ceil(threads_per_block);

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func_raw.launch(
            cfg,
            (
                d_input,
                &d_output,
                &d_output_sizes,
                chunk_size as u32,
                num_sub_chunks as u32,
                total_size as u32,
            ),
        )?;
    }
    dev.synchronize()?;

    // Download results
    let h_output_sizes = dev.dtoh_sync_copy(&d_output_sizes)?;
    let h_output = dev.dtoh_sync_copy(&d_output)?;

    let mut compressed_chunks = Vec::with_capacity(num_sub_chunks);
    let mut sizes = Vec::with_capacity(num_sub_chunks);

    for i in 0..num_sub_chunks {
        let frame_size = h_output_sizes[i] as usize;
        let frame_start = i * max_frame_size;
        let frame_data = h_output[frame_start..frame_start + frame_size].to_vec();

        sizes.push(frame_size);
        compressed_chunks.push(frame_data);
    }

    Ok((compressed_chunks, sizes))
}
