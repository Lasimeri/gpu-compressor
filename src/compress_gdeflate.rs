use crate::constants::{BATCH_SIZE, CHUNK_SIZE, GDEFLATE_MAX_COMPRESSION};
use crate::nvcomp_bindings as nvcomp;
use anyhow::Result;
use cuda_runtime_sys::*;

pub(crate) fn compress_buffer_gdeflate(input_data: &[u8], device_id: i32) -> Result<Vec<u8>> {
    let file_size = input_data.len();

    unsafe {
        // Initialize CUDA device
        let result = cudaSetDevice(device_id);
        if result != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }

        // Calculate total number of 64KB chunks
        let total_chunks = file_size.div_ceil(CHUNK_SIZE);
        let chunks_per_batch = BATCH_SIZE.div_ceil(CHUNK_SIZE);

        let compress_opts = nvcomp::nvcompBatchedGdeflateCompressOpts_t {
            algorithm: GDEFLATE_MAX_COMPRESSION,
            reserved: [0; 60],
        };

        let mut max_compressed_chunk_size: usize = 0;
        nvcomp::nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
            CHUNK_SIZE,
            compress_opts,
            &mut max_compressed_chunk_size as *mut usize,
        );

        let mut all_compressed_sizes = Vec::new();
        let mut all_compressed_chunks = Vec::new();

        // Process chunks in batches of 128MB
        let num_batches = total_chunks.div_ceil(chunks_per_batch);

        for batch_idx in 0..num_batches {
            let batch_start_chunk = batch_idx * chunks_per_batch;
            let batch_end_chunk = std::cmp::min(batch_start_chunk + chunks_per_batch, total_chunks);
            let batch_num_chunks = batch_end_chunk - batch_start_chunk;

            // Allocate GPU memory for this batch
            let mut d_uncompressed_ptrs =
                vec![std::ptr::null_mut::<std::ffi::c_void>(); batch_num_chunks];
            let mut batch_chunk_sizes = Vec::new();

            // Upload chunks for this batch
            for (i, d_uncompressed_ptr) in d_uncompressed_ptrs
                .iter_mut()
                .enumerate()
                .take(batch_num_chunks)
            {
                let global_chunk_idx = batch_start_chunk + i;
                let chunk_start = global_chunk_idx * CHUNK_SIZE;
                let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, file_size);
                let this_chunk_size = chunk_end - chunk_start;
                batch_chunk_sizes.push(this_chunk_size);

                let chunk_data = &input_data[chunk_start..chunk_end];

                cudaMalloc(d_uncompressed_ptr, this_chunk_size);
                cudaMemcpy(
                    *d_uncompressed_ptr,
                    chunk_data.as_ptr() as *const std::ffi::c_void,
                    this_chunk_size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                );
            }

            // Allocate device arrays
            let mut d_uncompressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_chunk_sizes_dev: *mut usize = std::ptr::null_mut();
            let mut d_compressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_compressed_sizes_dev: *mut usize = std::ptr::null_mut();
            let mut d_statuses_dev: *mut nvcomp::nvcompStatus_t = std::ptr::null_mut();

            cudaMalloc(
                &mut d_uncompressed_ptrs_dev,
                batch_num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
            );
            cudaMalloc(
                &mut d_chunk_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<usize>(),
            );
            cudaMalloc(
                &mut d_compressed_ptrs_dev,
                batch_num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
            );
            cudaMalloc(
                &mut d_compressed_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<usize>(),
            );
            cudaMalloc(
                &mut d_statuses_dev as *mut *mut nvcomp::nvcompStatus_t
                    as *mut *mut std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<nvcomp::nvcompStatus_t>(),
            );

            cudaMemcpy(
                d_uncompressed_ptrs_dev,
                d_uncompressed_ptrs.as_ptr() as *const std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                d_chunk_sizes_dev as *mut std::ffi::c_void,
                batch_chunk_sizes.as_ptr() as *const std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<usize>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            // Get temp buffer size
            let mut temp_bytes: usize = 0;
            nvcomp::nvcompBatchedGdeflateCompressGetTempSizeSync(
                d_uncompressed_ptrs_dev as *const *const std::ffi::c_void,
                d_chunk_sizes_dev as *const usize,
                batch_num_chunks,
                CHUNK_SIZE,
                compress_opts,
                &mut temp_bytes,
                batch_num_chunks * CHUNK_SIZE,
                std::ptr::null_mut(),
            );

            let mut d_temp: *mut std::ffi::c_void = std::ptr::null_mut();
            cudaMalloc(&mut d_temp, temp_bytes);

            // Allocate compressed buffers
            let mut d_compressed_ptrs =
                vec![std::ptr::null_mut::<std::ffi::c_void>(); batch_num_chunks];
            for (_i, d_compressed_ptr) in d_compressed_ptrs
                .iter_mut()
                .enumerate()
                .take(batch_num_chunks)
            {
                cudaMalloc(d_compressed_ptr, max_compressed_chunk_size);
            }

            cudaMemcpy(
                d_compressed_ptrs_dev,
                d_compressed_ptrs.as_ptr() as *const std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            // Compress this batch
            nvcomp::nvcompBatchedGdeflateCompressAsync(
                d_uncompressed_ptrs_dev as *const *const std::ffi::c_void,
                d_chunk_sizes_dev as *const usize,
                CHUNK_SIZE,
                batch_num_chunks,
                d_temp,
                temp_bytes,
                d_compressed_ptrs_dev as *const *mut std::ffi::c_void,
                d_compressed_sizes_dev,
                compress_opts,
                d_statuses_dev,
                std::ptr::null_mut(),
            );

            let err = cudaDeviceSynchronize();
            if err != cudaError::cudaSuccess {
                return Err(anyhow::anyhow!("cudaDeviceSynchronize failed: {:?}", err));
            }

            // Download compressed sizes
            let mut batch_compressed_sizes = vec![0usize; batch_num_chunks];
            cudaMemcpy(
                batch_compressed_sizes.as_mut_ptr() as *mut std::ffi::c_void,
                d_compressed_sizes_dev as *const std::ffi::c_void,
                batch_num_chunks * std::mem::size_of::<usize>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );

            // Download compressed data
            for i in 0..batch_num_chunks {
                let mut chunk_data = vec![0u8; batch_compressed_sizes[i]];
                cudaMemcpy(
                    chunk_data.as_mut_ptr() as *mut std::ffi::c_void,
                    d_compressed_ptrs[i],
                    batch_compressed_sizes[i],
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                all_compressed_sizes.push(batch_compressed_sizes[i]);
                all_compressed_chunks.push(chunk_data);
            }

            // Cleanup this batch
            for ptr in &d_uncompressed_ptrs {
                cudaFree(*ptr);
            }
            for ptr in &d_compressed_ptrs {
                cudaFree(*ptr);
            }
            cudaFree(d_temp);
            cudaFree(d_uncompressed_ptrs_dev);
            cudaFree(d_chunk_sizes_dev as *mut std::ffi::c_void);
            cudaFree(d_compressed_ptrs_dev);
            cudaFree(d_compressed_sizes_dev as *mut std::ffi::c_void);
            cudaFree(d_statuses_dev as *mut std::ffi::c_void);
        }

        // Build NVGD format output
        let mut output = Vec::new();

        // Write header
        output.extend_from_slice(b"NVGD");
        output.extend_from_slice(&(file_size as u64).to_le_bytes());
        output.extend_from_slice(&(CHUNK_SIZE as u64).to_le_bytes());
        output.extend_from_slice(&(total_chunks as u64).to_le_bytes());

        // Write compressed chunk sizes
        for size in &all_compressed_sizes {
            output.extend_from_slice(&(*size as u64).to_le_bytes());
        }

        // Write compressed chunks
        for chunk in &all_compressed_chunks {
            output.extend_from_slice(chunk);
        }

        Ok(output)
    }
}
