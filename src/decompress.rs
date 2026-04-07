use crate::compress_lzma2::decompress_chunk_lzma2;
use crate::nvcomp_bindings as nvcomp;
use anyhow::Result;
use cuda_runtime_sys::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub(crate) fn decompress_file(
    input_path: &PathBuf,
    output_path: &PathBuf,
    device_id: i32,
) -> Result<()> {
    let mut input_file = File::open(input_path)?;
    let mut magic = [0u8; 4];
    input_file.read_exact(&mut magic)?;

    if &magic == b"NVZS" {
        println!("gpu-compressor: zstd decompression");
        println!("  input:  {}", input_path.display());
        println!("  output: {}", output_path.display());
        decompress_file_zstd_streaming(input_path, output_path, device_id)
    } else if &magic == b"NVLZ" {
        println!("gpu-compressor: lzma2 decompression");
        println!("  input:  {}", input_path.display());
        println!("  output: {}", output_path.display());
        decompress_file_lzma2_streaming(input_path, output_path)
    } else {
        Err(anyhow::anyhow!(
            "Invalid file format (expected NVZS or NVLZ)"
        ))
    }
}

fn decompress_file_lzma2_streaming(
    input_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let mut input_file = File::open(input_path)?;

    // Read header: magic(4) + original_size(8) + chunk_size(8) + num_chunks(8) = 28 bytes
    let mut header_buf = [0u8; 4 + 8 + 8 + 8];
    input_file.read_exact(&mut header_buf)?;

    let original_size = u64::from_le_bytes(header_buf[4..12].try_into().unwrap()) as usize;
    let chunk_size = u64::from_le_bytes(header_buf[12..20].try_into().unwrap()) as usize;
    let num_chunks = u64::from_le_bytes(header_buf[20..28].try_into().unwrap()) as usize;

    println!(
        "  original: {} bytes ({:.2} GB)",
        original_size,
        original_size as f64 / 1_000_000_000.0
    );
    println!(
        "  chunks:   {} x {}MB",
        num_chunks,
        chunk_size / (1024 * 1024)
    );
    println!("  decompressing...");

    // Read compressed chunk sizes
    let mut compressed_sizes = vec![0usize; num_chunks];
    for compressed_size in compressed_sizes.iter_mut().take(num_chunks) {
        let mut size_buf = [0u8; 8];
        input_file.read_exact(&mut size_buf)?;
        *compressed_size = u64::from_le_bytes(size_buf) as usize;
    }

    // Calculate decompressed sizes for all chunks
    let mut decompressed_sizes = vec![chunk_size; num_chunks];
    if num_chunks > 0 {
        decompressed_sizes[num_chunks - 1] = original_size - (num_chunks - 1) * chunk_size;
    }

    let mut output_file = File::create(output_path)?;
    let start = std::time::Instant::now();

    for chunk_idx in 0..num_chunks {
        eprintln!("  chunk {}/{}", chunk_idx + 1, num_chunks);
        let mut compressed = vec![0u8; compressed_sizes[chunk_idx]];
        input_file.read_exact(&mut compressed)?;
        let decompressed = decompress_chunk_lzma2(&compressed, decompressed_sizes[chunk_idx])?;
        output_file.write_all(&decompressed)?;
    }

    let elapsed = start.elapsed();
    let throughput = (original_size as f64 / elapsed.as_secs_f64()) / 1_000_000_000.0;
    println!(
        "  time:       {:.3}s @ {:.2} GB/s",
        elapsed.as_secs_f64(),
        throughput
    );
    println!("  done.");

    Ok(())
}

fn decompress_file_zstd_streaming(
    input_path: &Path,
    output_path: &Path,
    device_id: i32,
) -> Result<()> {
    // Initialize CUDA device
    unsafe {
        let result = cudaSetDevice(device_id);
        if result != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }
    }

    unsafe {
        // Open input file for streaming
        let mut input_file = File::open(input_path)?;

        // Read header: magic + original_size + chunk_size + num_chunks
        let mut header_buf = [0u8; 4 + 8 + 8 + 8]; // Total: 28 bytes
        input_file.read_exact(&mut header_buf)?;

        let original_size = u64::from_le_bytes(header_buf[4..12].try_into().unwrap()) as usize;
        let chunk_size = u64::from_le_bytes(header_buf[12..20].try_into().unwrap()) as usize;
        let num_chunks = u64::from_le_bytes(header_buf[20..28].try_into().unwrap()) as usize;

        println!(
            "  original: {} bytes ({:.2} GB)",
            original_size,
            original_size as f64 / 1_000_000_000.0
        );

        // Read compressed chunk sizes (only metadata, not the actual chunks)
        let mut compressed_sizes = vec![0usize; num_chunks];
        for compressed_size in compressed_sizes.iter_mut().take(num_chunks) {
            let mut size_buf = [0u8; 8];
            input_file.read_exact(&mut size_buf)?;
            *compressed_size = u64::from_le_bytes(size_buf) as usize;
        }

        // Now input_file is positioned at the start of chunk data

        // Calculate decompressed sizes for all chunks
        let mut decompressed_sizes = vec![chunk_size; num_chunks];
        if num_chunks > 0 {
            decompressed_sizes[num_chunks - 1] = original_size - (num_chunks - 1) * chunk_size;
        }

        // Use micro-batches of 1 chunk for maximum decompression speed
        let micro_batch_size = 1usize;
        let num_batches = num_chunks.div_ceil(micro_batch_size);

        println!(
            "  chunks:   {} x {}MB (micro-batch: {})",
            num_chunks,
            chunk_size / (1024 * 1024),
            micro_batch_size
        );
        println!(
            "  max ram:  ~{:.2} GB",
            (chunk_size as f64 * micro_batch_size as f64 * 2.0) / 1_000_000_000.0
        );
        println!("  decompressing...");

        // Write decompressed data directly to output file
        let mut output_file = File::create(output_path)?;

        let gpu_start = std::time::Instant::now();

        // Zstd decompression options
        let decompress_opts = nvcomp::nvcompBatchedZstdDecompressOpts_t {
            backend: nvcomp::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
            reserved: [0; 60],
        };

        // Process chunks in micro-batches
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * micro_batch_size;
            let batch_end = std::cmp::min(batch_start + micro_batch_size, num_chunks);
            let batch_size = batch_end - batch_start;
            eprintln!("  batch {}/{}", batch_idx + 1, num_batches);

            // Read compressed chunks for this micro-batch
            let mut compressed_chunks = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let chunk_idx = batch_start + i;
                let mut chunk = vec![0u8; compressed_sizes[chunk_idx]];
                input_file.read_exact(&mut chunk)?;
                compressed_chunks.push(chunk);
            }

            // Allocate GPU memory for this micro-batch
            let mut d_compressed_ptrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); batch_size];
            let mut d_decompressed_ptrs =
                vec![std::ptr::null_mut::<std::ffi::c_void>(); batch_size];

            for i in 0..batch_size {
                let chunk_idx = batch_start + i;
                cudaMalloc(&mut d_compressed_ptrs[i], compressed_sizes[chunk_idx]);
                cudaMalloc(&mut d_decompressed_ptrs[i], decompressed_sizes[chunk_idx]);

                // Upload compressed chunk
                cudaMemcpy(
                    d_compressed_ptrs[i],
                    compressed_chunks[i].as_ptr() as *const std::ffi::c_void,
                    compressed_sizes[chunk_idx],
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                );
            }

            // Compressed chunks can be dropped now (will happen at end of scope)

            // Setup device arrays for micro-batch
            let mut d_compressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_compressed_sizes_dev: *mut usize = std::ptr::null_mut();
            let mut d_decompressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_decompressed_sizes_dev: *mut usize = std::ptr::null_mut();
            let mut d_actual_decompressed_sizes_dev: *mut usize = std::ptr::null_mut();
            let mut d_statuses_dev: *mut nvcomp::nvcompStatus_t = std::ptr::null_mut();

            cudaMalloc(
                &mut d_compressed_ptrs_dev,
                batch_size * std::mem::size_of::<*mut std::ffi::c_void>(),
            );
            cudaMalloc(
                &mut d_compressed_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
                batch_size * std::mem::size_of::<usize>(),
            );
            cudaMalloc(
                &mut d_decompressed_ptrs_dev,
                batch_size * std::mem::size_of::<*mut std::ffi::c_void>(),
            );
            cudaMalloc(
                &mut d_decompressed_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
                batch_size * std::mem::size_of::<usize>(),
            );
            cudaMalloc(
                &mut d_actual_decompressed_sizes_dev as *mut *mut usize
                    as *mut *mut std::ffi::c_void,
                batch_size * std::mem::size_of::<usize>(),
            );
            cudaMalloc(
                &mut d_statuses_dev as *mut *mut nvcomp::nvcompStatus_t
                    as *mut *mut std::ffi::c_void,
                batch_size * std::mem::size_of::<nvcomp::nvcompStatus_t>(),
            );

            // Copy metadata to device
            let batch_compressed_sizes: Vec<usize> = (batch_start..batch_end)
                .map(|i| compressed_sizes[i])
                .collect();
            let batch_decompressed_sizes: Vec<usize> = (batch_start..batch_end)
                .map(|i| decompressed_sizes[i])
                .collect();

            cudaMemcpy(
                d_compressed_ptrs_dev,
                d_compressed_ptrs.as_ptr() as *const std::ffi::c_void,
                batch_size * std::mem::size_of::<*mut std::ffi::c_void>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                d_compressed_sizes_dev as *mut std::ffi::c_void,
                batch_compressed_sizes.as_ptr() as *const std::ffi::c_void,
                batch_size * std::mem::size_of::<usize>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                d_decompressed_ptrs_dev,
                d_decompressed_ptrs.as_ptr() as *const std::ffi::c_void,
                batch_size * std::mem::size_of::<*mut std::ffi::c_void>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                d_decompressed_sizes_dev as *mut std::ffi::c_void,
                batch_decompressed_sizes.as_ptr() as *const std::ffi::c_void,
                batch_size * std::mem::size_of::<usize>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            // Get temp size for micro-batch
            let batch_total_size: usize = batch_decompressed_sizes.iter().sum();
            let mut temp_bytes: usize = 0;
            nvcomp::nvcompBatchedZstdDecompressGetTempSizeSync(
                d_compressed_ptrs_dev as *const *const std::ffi::c_void,
                d_compressed_sizes_dev as *const usize,
                batch_size,
                chunk_size,
                &mut temp_bytes,
                batch_total_size,
                decompress_opts,
                d_statuses_dev,
                std::ptr::null_mut(),
            );

            let mut d_temp: *mut std::ffi::c_void = std::ptr::null_mut();
            cudaMalloc(&mut d_temp, temp_bytes);

            // Decompress this micro-batch
            let status = nvcomp::nvcompBatchedZstdDecompressAsync(
                d_compressed_ptrs_dev as *const *const std::ffi::c_void,
                d_compressed_sizes_dev as *const usize,
                d_decompressed_sizes_dev as *const usize,
                d_actual_decompressed_sizes_dev,
                batch_size,
                d_temp,
                temp_bytes,
                d_decompressed_ptrs_dev as *const *mut std::ffi::c_void,
                decompress_opts,
                d_statuses_dev,
                std::ptr::null_mut(),
            );

            if status != nvcomp::nvcompStatus_t_nvcompSuccess {
                for ptr in &d_compressed_ptrs {
                    cudaFree(*ptr);
                }
                for ptr in &d_decompressed_ptrs {
                    cudaFree(*ptr);
                }
                cudaFree(d_temp);
                cudaFree(d_compressed_ptrs_dev);
                cudaFree(d_compressed_sizes_dev as *mut std::ffi::c_void);
                cudaFree(d_decompressed_ptrs_dev);
                cudaFree(d_decompressed_sizes_dev as *mut std::ffi::c_void);
                cudaFree(d_actual_decompressed_sizes_dev as *mut std::ffi::c_void);
                cudaFree(d_statuses_dev as *mut std::ffi::c_void);
                return Err(anyhow::anyhow!(
                    "Micro-batch {} decompression failed: {:?}",
                    batch_idx,
                    status
                ));
            }

            let err = cudaDeviceSynchronize();
            if err != cudaError::cudaSuccess {
                return Err(anyhow::anyhow!("cudaDeviceSynchronize failed: {:?}", err));
            }

            // Download and write decompressed chunks to output file
            for (i, d_decompressed_ptr) in d_decompressed_ptrs.iter().enumerate().take(batch_size) {
                let chunk_idx = batch_start + i;
                let mut decompressed_chunk = vec![0u8; decompressed_sizes[chunk_idx]];
                cudaMemcpy(
                    decompressed_chunk.as_mut_ptr() as *mut std::ffi::c_void,
                    *d_decompressed_ptr,
                    decompressed_sizes[chunk_idx],
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                output_file.write_all(&decompressed_chunk)?;
            }

            // Free all memory for this micro-batch
            for ptr in &d_compressed_ptrs {
                cudaFree(*ptr);
            }
            for ptr in &d_decompressed_ptrs {
                cudaFree(*ptr);
            }
            cudaFree(d_temp);
            cudaFree(d_compressed_ptrs_dev);
            cudaFree(d_compressed_sizes_dev as *mut std::ffi::c_void);
            cudaFree(d_decompressed_ptrs_dev);
            cudaFree(d_decompressed_sizes_dev as *mut std::ffi::c_void);
            cudaFree(d_actual_decompressed_sizes_dev as *mut std::ffi::c_void);
            cudaFree(d_statuses_dev as *mut std::ffi::c_void);

            // All buffers dropped here, freeing RAM
        }

        let gpu_time = gpu_start.elapsed();
        let gpu_throughput = (original_size as f64 / gpu_time.as_secs_f64()) / 1_000_000_000.0;
        println!(
            "  time:       {:.3}s @ {:.2} GB/s",
            gpu_time.as_secs_f64(),
            gpu_throughput
        );

        println!("  done.");
    }

    Ok(())
}

