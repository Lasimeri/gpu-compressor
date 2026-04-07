use anyhow::Result;
use cudarc::driver::*;
use std::sync::Arc;

use crate::constants::{CUSTOM_ZSTD_CHUNK_SIZE, CUSTOM_ZSTD_SEARCH_DEPTH_LAZY, PTX_ZSTD_COMPRESS};

// FFI to libzstd for CPU-side FSE encoding of GPU-found sequences
#[repr(C)]
struct ZstdSequence {
    offset: u32,
    lit_length: u32,
    match_length: u32,
    rep: u32,
}

extern "C" {
    fn ZSTD_createCCtx() -> *mut std::ffi::c_void;
    fn ZSTD_freeCCtx(cctx: *mut std::ffi::c_void) -> usize;
    fn ZSTD_CCtx_reset(cctx: *mut std::ffi::c_void, reset: u32) -> usize;
    fn ZSTD_CCtx_setParameter(cctx: *mut std::ffi::c_void, param: i32, value: i32) -> usize;
    fn ZSTD_compressSequences(
        cctx: *mut std::ffi::c_void,
        dst: *mut u8,
        dst_capacity: usize,
        in_seqs: *const ZstdSequence,
        in_seqs_size: usize,
        src: *const u8,
        src_size: usize,
    ) -> usize;
    fn ZSTD_isError(code: usize) -> u32;
    fn ZSTD_getErrorName(code: usize) -> *const std::ffi::c_char;
    fn ZSTD_compressBound(src_size: usize) -> usize;
    fn ZSTD_compress(
        dst: *mut u8,
        dst_capacity: usize,
        src: *const u8,
        src_size: usize,
        compression_level: i32,
    ) -> usize;
}

/// GPU LZ77 match finding + CPU FSE encoding via libzstd.
///
/// The GPU kernel (zstd_match_find) runs LZ77 match finding with shared-memory hash tables
/// across all sub-chunks in parallel. The resulting sequences are downloaded and encoded
/// into Zstd frames by libzstd's ZSTD_compressSequences on CPU.
pub(crate) fn compress_chunk_zstd_custom(
    chunk_data: &[u8],
    device_id: i32,
    level: u32,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let chunk_size = CUSTOM_ZSTD_CHUNK_SIZE;
    let total_size = chunk_data.len();
    let num_sub_chunks = total_size.div_ceil(chunk_size);
    let max_sequences_per_chunk = chunk_size / 3;

    let search_depth = match level {
        1 => CUSTOM_ZSTD_SEARCH_DEPTH_LAZY as u32,
        2 => 64u32,
        _ => 1u32,
    };

    unsafe {
        let result = cuda_runtime_sys::cudaSetDevice(device_id);
        if result != cuda_runtime_sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }
    }

    let device = CudaDevice::new(device_id as usize)?;
    let dev = Arc::new(device);

    dev.load_ptx(
        PTX_ZSTD_COMPRESS.into(),
        "zstd_module",
        &["zstd_match_find"],
    )?;

    let d_input = dev.htod_sync_copy(chunk_data)?;

    // === GPU: Parallel LZ77 match finding ===
    let d_sequences = dev.alloc_zeros::<u32>(num_sub_chunks * max_sequences_per_chunk * 3)?;
    let d_literals = dev.alloc_zeros::<u8>(num_sub_chunks * chunk_size)?;
    let d_seq_counts = dev.alloc_zeros::<u32>(num_sub_chunks)?;
    let d_lit_counts = dev.alloc_zeros::<u32>(num_sub_chunks)?;

    let func_match = dev
        .get_func("zstd_module", "zstd_match_find")
        .ok_or_else(|| anyhow::anyhow!("Failed to load zstd_match_find kernel"))?;

    unsafe {
        func_match.launch(
            LaunchConfig {
                grid_dim: (num_sub_chunks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
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

    // === Download GPU results ===
    let h_sequences = dev.dtoh_sync_copy(&d_sequences)?;
    let h_seq_counts = dev.dtoh_sync_copy(&d_seq_counts)?;

    // === CPU: FSE encoding via libzstd ZSTD_compressSequences ===
    let mut compressed_chunks = Vec::with_capacity(num_sub_chunks);
    let mut sizes = Vec::with_capacity(num_sub_chunks);

    unsafe {
        let cctx = ZSTD_createCCtx();
        if cctx.is_null() {
            return Err(anyhow::anyhow!("Failed to create ZSTD_CCtx"));
        }

        ZSTD_CCtx_setParameter(cctx, 1001, 0); // blockDelimiters = 0

        for i in 0..num_sub_chunks {
            let sub_start = i * chunk_size;
            let sub_end = std::cmp::min(sub_start + chunk_size, total_size);
            let sub_data = &chunk_data[sub_start..sub_end];
            let sub_size = sub_end - sub_start;
            let n_seqs = h_seq_counts[i] as usize;

            let seq_base = i * max_sequences_per_chunk * 3;
            let mut zstd_seqs: Vec<ZstdSequence> = Vec::with_capacity(n_seqs);

            for j in 0..n_seqs {
                let idx = seq_base + j * 3;
                let lit_len = h_sequences[idx];
                let match_len_minus3 = h_sequences[idx + 1];
                let offset = h_sequences[idx + 2];
                let actual_match_len = match_len_minus3 + 3;

                zstd_seqs.push(ZstdSequence {
                    offset,
                    lit_length: lit_len,
                    match_length: actual_match_len,
                    rep: 0,
                });
            }

            // Trailing literals are implicit — ZSTD_compressSequences computes them
            // from srcSize - total_covered. Do NOT pass an explicit trailing sequence.

            let out_bound = ZSTD_compressBound(sub_size);
            let mut out_buf = vec![0u8; out_bound];

            ZSTD_CCtx_reset(cctx, 1); // ZSTD_reset_session_only

            let ret = ZSTD_compressSequences(
                cctx,
                out_buf.as_mut_ptr(),
                out_bound,
                zstd_seqs.as_ptr(),
                zstd_seqs.len(),
                sub_data.as_ptr(),
                sub_size,
            );

            if ZSTD_isError(ret) != 0 {
                let fallback = ZSTD_compress(
                    out_buf.as_mut_ptr(),
                    out_bound,
                    sub_data.as_ptr(),
                    sub_size,
                    1,
                );
                if ZSTD_isError(fallback) != 0 {
                    ZSTD_freeCCtx(cctx);
                    let err = std::ffi::CStr::from_ptr(ZSTD_getErrorName(ret));
                    return Err(anyhow::anyhow!(
                        "Sub-chunk {} compression failed: {}",
                        i,
                        err.to_str().unwrap_or("unknown")
                    ));
                }
                out_buf.truncate(fallback);
            } else {
                out_buf.truncate(ret);
            }

            sizes.push(out_buf.len());
            compressed_chunks.push(out_buf);
        }

        ZSTD_freeCCtx(cctx);
    }

    Ok((compressed_chunks, sizes))
}
