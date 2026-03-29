use crate::cli::Algorithm;
use crate::compress_gdeflate::compress_buffer_gdeflate;
use crate::nvcomp_bindings as nvcomp;
use anyhow::Result;
use cuda_runtime_sys::*;

// Helper function to compress a buffer and return compressed data
// Processes 64KB chunks in 128MB batches for good performance with manageable VRAM
pub(crate) fn compress_buffer(
    input_data: &[u8],
    algorithm: Algorithm,
    device_id: i32,
    _quiet: bool,
) -> Result<Vec<u8>> {
    match algorithm {
        Algorithm::Gdeflate => compress_buffer_gdeflate(input_data, device_id),
        Algorithm::Zstd => Err(anyhow::anyhow!(
            "Zstd uses streaming pipeline; use compress_file_impl() instead"
        )),
    }
}

// Streaming: Compress a single chunk immediately (no batching)
pub(crate) fn compress_chunk_zstd(
    chunk_data: &[u8],
    device_id: i32,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let chunk_size = chunk_data.len();

    unsafe {
        // Initialize CUDA device
        let result = cudaSetDevice(device_id);
        if result != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }

        // Zstd compression options
        let compress_opts = nvcomp::nvcompBatchedZstdCompressOpts_t { reserved: [0; 64] };

        // Single chunk - upload to GPU
        let mut d_uncompressed_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        cudaMalloc(&mut d_uncompressed_ptr, chunk_size);
        cudaMemcpy(
            d_uncompressed_ptr,
            chunk_data.as_ptr() as *const std::ffi::c_void,
            chunk_size,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        // Device arrays for single chunk
        let mut d_uncompressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_chunk_size_dev: *mut usize = std::ptr::null_mut();
        let mut d_compressed_ptr_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_compressed_size_dev: *mut usize = std::ptr::null_mut();
        let mut d_status_dev: *mut nvcomp::nvcompStatus_t = std::ptr::null_mut();

        cudaMalloc(
            &mut d_uncompressed_ptrs_dev,
            std::mem::size_of::<*mut std::ffi::c_void>(),
        );
        cudaMalloc(
            &mut d_chunk_size_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
            std::mem::size_of::<usize>(),
        );
        cudaMalloc(
            &mut d_compressed_ptr_dev,
            std::mem::size_of::<*mut std::ffi::c_void>(),
        );
        cudaMalloc(
            &mut d_compressed_size_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
            std::mem::size_of::<usize>(),
        );
        cudaMalloc(
            &mut d_status_dev as *mut *mut nvcomp::nvcompStatus_t as *mut *mut std::ffi::c_void,
            std::mem::size_of::<nvcomp::nvcompStatus_t>(),
        );

        cudaMemcpy(
            d_uncompressed_ptrs_dev,
            &d_uncompressed_ptr as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<*mut std::ffi::c_void>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        cudaMemcpy(
            d_chunk_size_dev as *mut std::ffi::c_void,
            &chunk_size as *const usize as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        // Get max compressed size
        let mut max_compressed_size: usize = 0;
        nvcomp::nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            chunk_size,
            compress_opts,
            &mut max_compressed_size as *mut usize,
        );

        // Get temp buffer size
        let mut temp_bytes: usize = 0;
        nvcomp::nvcompBatchedZstdCompressGetTempSizeSync(
            d_uncompressed_ptrs_dev as *const *const std::ffi::c_void,
            d_chunk_size_dev as *const usize,
            1,          // Single chunk
            chunk_size, // max_uncompressed_chunk_bytes
            compress_opts,
            &mut temp_bytes,
            chunk_size,           // max_total_uncompressed_bytes
            std::ptr::null_mut(), // stream
        );

        let mut d_temp: *mut std::ffi::c_void = std::ptr::null_mut();
        cudaMalloc(&mut d_temp, temp_bytes);

        // Allocate compressed output
        let mut d_compressed_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        cudaMalloc(&mut d_compressed_ptr, max_compressed_size);
        cudaMemcpy(
            d_compressed_ptr_dev,
            &d_compressed_ptr as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<*mut std::ffi::c_void>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        // Compress (single chunk, no batch loop)
        nvcomp::nvcompBatchedZstdCompressAsync(
            d_uncompressed_ptrs_dev as *const *const std::ffi::c_void,
            d_chunk_size_dev as *const usize,
            chunk_size,
            1, // Single chunk
            d_temp,
            temp_bytes,
            d_compressed_ptr_dev as *const *mut std::ffi::c_void,
            d_compressed_size_dev,
            compress_opts,
            d_status_dev,
            std::ptr::null_mut(), // stream
        );

        let err = cudaDeviceSynchronize();
        if err != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("cudaDeviceSynchronize failed: {:?}", err));
        }

        // Download compressed size
        let mut compressed_size: usize = 0;
        cudaMemcpy(
            &mut compressed_size as *mut usize as *mut std::ffi::c_void,
            d_compressed_size_dev as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );

        // Download compressed data
        let mut compressed_data = vec![0u8; compressed_size];
        cudaMemcpy(
            compressed_data.as_mut_ptr() as *mut std::ffi::c_void,
            d_compressed_ptr,
            compressed_size,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );

        // Cleanup
        cudaFree(d_uncompressed_ptr);
        cudaFree(d_compressed_ptr);
        cudaFree(d_temp);
        cudaFree(d_uncompressed_ptrs_dev);
        cudaFree(d_chunk_size_dev as *mut std::ffi::c_void);
        cudaFree(d_compressed_ptr_dev);
        cudaFree(d_compressed_size_dev as *mut std::ffi::c_void);
        cudaFree(d_status_dev as *mut std::ffi::c_void);

        // Return as single-element vectors for compatibility
        Ok((vec![compressed_data], vec![compressed_size]))
    }
}
