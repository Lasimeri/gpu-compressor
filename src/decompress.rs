use crate::blake3::blake3_hash_file;
use crate::nvcomp_bindings as nvcomp;
use anyhow::Result;
use cuda_runtime_sys::*;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub(crate) fn decompress_file(
    input_path: &PathBuf,
    output_path: &PathBuf,
    device_id: i32,
) -> Result<()> {
    // Read only the magic header (4 bytes) to determine format
    let mut input_file = File::open(input_path)?;
    let mut magic = [0u8; 4];
    input_file.read_exact(&mut magic)?;

    if &magic == b"NVMC" {
        // Multi-chunk format - decompress each chunk separately
        decompress_multi_chunk_file(input_path, output_path, device_id)
    } else if &magic == b"NVGD" {
        // For gdeflate, we still need to load the whole file (for now)
        println!("gpu-compressor: gdeflate decompression");
        println!("  input:  {}", input_path.display());
        println!("  output: {}", output_path.display());
        let mut file_data = magic.to_vec();
        input_file.read_to_end(&mut file_data)?;
        decompress_file_gdeflate(&file_data, output_path, device_id)
    } else if &magic == b"NVZS" {
        println!("gpu-compressor: zstd decompression");
        println!("  input:  {}", input_path.display());
        println!("  output: {}", output_path.display());
        // Zstd uses streaming decompression
        decompress_file_zstd_streaming(input_path, output_path, device_id)
    } else {
        Err(anyhow::anyhow!(
            "Invalid file format (expected NVGD, NVZS, or NVMC)"
        ))
    }
}

fn decompress_file_gdeflate(file_data: &[u8], output_path: &Path, device_id: i32) -> Result<()> {
    // Initialize CUDA device
    unsafe {
        let result = cudaSetDevice(device_id);
        if result != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }
    }

    unsafe {
        // Parse header for single-file NVGD format
        let mut offset = 4; // Skip magic (already checked)

        let original_size =
            u64::from_le_bytes(file_data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let chunk_size =
            u64::from_le_bytes(file_data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let num_chunks =
            u64::from_le_bytes(file_data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        println!(
            "  original: {} bytes ({:.2} MB)",
            original_size,
            original_size as f64 / 1_000_000.0
        );
        println!("  chunks:   {} x {}KB", num_chunks, chunk_size / 1024);

        // Read compressed chunk sizes
        let mut compressed_sizes = vec![0usize; num_chunks];
        for compressed_size in compressed_sizes.iter_mut().take(num_chunks) {
            *compressed_size =
                u64::from_le_bytes(file_data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
        }

        // Read compressed chunks
        let mut compressed_chunks = Vec::new();
        for size in &compressed_sizes {
            compressed_chunks.push(file_data[offset..offset + *size].to_vec());
            offset += *size;
        }

        println!("  decompressing...");
        let gpu_start = std::time::Instant::now();

        // Upload compressed chunks to GPU
        let mut d_compressed_ptrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); num_chunks];
        for i in 0..num_chunks {
            cudaMalloc(
                &mut d_compressed_ptrs[i] as *mut *mut std::ffi::c_void,
                compressed_sizes[i],
            );
            cudaMemcpy(
                d_compressed_ptrs[i],
                compressed_chunks[i].as_ptr() as *const std::ffi::c_void,
                compressed_sizes[i],
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
        }

        // Allocate decompressed output
        let mut d_decompressed_ptrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); num_chunks];
        let mut decompressed_sizes = vec![chunk_size; num_chunks];
        if num_chunks > 0 {
            decompressed_sizes[num_chunks - 1] = original_size - (num_chunks - 1) * chunk_size;
        }

        for i in 0..num_chunks {
            cudaMalloc(
                &mut d_decompressed_ptrs[i] as *mut *mut std::ffi::c_void,
                decompressed_sizes[i],
            );
        }

        // Setup device arrays
        let mut d_compressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_compressed_sizes_dev: *mut usize = std::ptr::null_mut();
        let mut d_decompressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_decompressed_sizes_dev: *mut usize = std::ptr::null_mut();
        let mut d_statuses_dev: *mut nvcomp::nvcompStatus_t = std::ptr::null_mut();

        cudaMalloc(
            &mut d_compressed_ptrs_dev,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
        );
        cudaMalloc(
            &mut d_compressed_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
        );
        cudaMalloc(
            &mut d_decompressed_ptrs_dev,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
        );
        cudaMalloc(
            &mut d_decompressed_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
        );
        cudaMalloc(
            &mut d_statuses_dev as *mut *mut nvcomp::nvcompStatus_t as *mut *mut std::ffi::c_void,
            num_chunks * std::mem::size_of::<nvcomp::nvcompStatus_t>(),
        );

        cudaMemcpy(
            d_compressed_ptrs_dev,
            d_compressed_ptrs.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        cudaMemcpy(
            d_compressed_sizes_dev as *mut std::ffi::c_void,
            compressed_sizes.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        cudaMemcpy(
            d_decompressed_ptrs_dev,
            d_decompressed_ptrs.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        cudaMemcpy(
            d_decompressed_sizes_dev as *mut std::ffi::c_void,
            decompressed_sizes.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        // Get temp size
        let decompress_opts = nvcomp::nvcompBatchedGdeflateDecompressOpts_t {
            backend: nvcomp::nvcompDecompressBackend_t_NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
            reserved: [0; 60],
        };
        let mut temp_bytes: usize = 0;
        let status = nvcomp::nvcompBatchedGdeflateDecompressGetTempSizeSync(
            d_compressed_ptrs_dev as *const *const std::ffi::c_void,
            d_compressed_sizes_dev as *const usize,
            num_chunks,
            chunk_size,
            &mut temp_bytes,
            original_size,
            decompress_opts,
            d_statuses_dev,
            std::ptr::null_mut(),
        );
        if status != nvcomp::nvcompStatus_t_nvcompSuccess {
            return Err(anyhow::anyhow!(
                "Failed to get decompress temp size: {:?}",
                status
            ));
        }

        let mut d_temp: *mut std::ffi::c_void = std::ptr::null_mut();
        cudaMalloc(&mut d_temp, temp_bytes);

        // Decompress!
        let status = nvcomp::nvcompBatchedGdeflateDecompressAsync(
            d_compressed_ptrs_dev as *const *const std::ffi::c_void,
            d_compressed_sizes_dev as *const usize,
            d_decompressed_sizes_dev as *const usize,
            d_decompressed_sizes_dev,
            num_chunks,
            d_temp,
            temp_bytes,
            d_decompressed_ptrs_dev as *const *mut std::ffi::c_void,
            decompress_opts,
            d_statuses_dev,
            std::ptr::null_mut(),
        );

        if status != nvcomp::nvcompStatus_t_nvcompSuccess {
            return Err(anyhow::anyhow!("Decompression failed: {:?}", status));
        }

        let err = cudaDeviceSynchronize();
        if err != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("cudaDeviceSynchronize failed: {:?}", err));
        }
        let gpu_time = gpu_start.elapsed();
        let gpu_throughput = (original_size as f64 / gpu_time.as_secs_f64()) / 1_000_000_000.0;
        println!(
            "  time:       {:.3}s @ {:.2} GB/s",
            gpu_time.as_secs_f64(),
            gpu_throughput
        );

        // Download decompressed data
        let mut output_data = vec![0u8; original_size];
        let mut offset = 0;
        for i in 0..num_chunks {
            cudaMemcpy(
                output_data[offset..].as_mut_ptr() as *mut std::ffi::c_void,
                d_decompressed_ptrs[i],
                decompressed_sizes[i],
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            offset += decompressed_sizes[i];
        }

        // Cleanup
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
        cudaFree(d_statuses_dev as *mut std::ffi::c_void);

        // Write output
        let mut output_file = File::create(output_path)?;
        output_file.write_all(&output_data)?;

        println!("  done.");
    }

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

        // Read header: magic + tar_size + chunk_size + num_chunks
        let mut header_buf = [0u8; 4 + 8 + 8 + 8]; // Total: 28 bytes
        input_file.read_exact(&mut header_buf)?;

        let tar_size = u64::from_le_bytes(header_buf[4..12].try_into().unwrap()) as usize;
        let chunk_size = u64::from_le_bytes(header_buf[12..20].try_into().unwrap()) as usize;
        let num_chunks = u64::from_le_bytes(header_buf[20..28].try_into().unwrap()) as usize;

        println!(
            "  original: {} bytes ({:.2} GB)",
            tar_size,
            tar_size as f64 / 1_000_000_000.0
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
            decompressed_sizes[num_chunks - 1] = tar_size - (num_chunks - 1) * chunk_size;
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

        // Create temporary tar file
        let tar_path = format!("{}.temp.tar", output_path.display());
        let mut tar_file = File::create(&tar_path)?;

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

            // Download and write decompressed chunks to tar file
            for (i, d_decompressed_ptr) in d_decompressed_ptrs.iter().enumerate().take(batch_size) {
                let chunk_idx = batch_start + i;
                let mut decompressed_chunk = vec![0u8; decompressed_sizes[chunk_idx]];
                cudaMemcpy(
                    decompressed_chunk.as_mut_ptr() as *mut std::ffi::c_void,
                    *d_decompressed_ptr,
                    decompressed_sizes[chunk_idx],
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                tar_file.write_all(&decompressed_chunk)?;
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
        let gpu_throughput = (tar_size as f64 / gpu_time.as_secs_f64()) / 1_000_000_000.0;
        println!(
            "  time:       {:.3}s @ {:.2} GB/s",
            gpu_time.as_secs_f64(),
            gpu_throughput
        );

        println!("  done.");

        // Close tar file
        drop(tar_file);

        // Extract tar archive to get original file and .blake3 hash file
        println!("  extracting archive...");
        let tar_file_read = File::open(&tar_path)?;
        let mut tar_archive = tar::Archive::new(tar_file_read);

        // Get the directory where output should go
        let output_dir = output_path.parent().unwrap_or_else(|| Path::new("."));

        // Extract all files from tar
        tar_archive.unpack(output_dir)?;

        // Find the extracted .blake3 file
        let blake3_path = format!("{}.blake3", output_path.display());

        // Read hash from .blake3 file for verification
        println!("  verifying integrity...");
        let mut blake3_file = File::open(&blake3_path)?;
        let mut blake3_content = String::new();
        blake3_file.read_to_string(&mut blake3_content)?;
        let expected_hash = blake3_content.trim();

        // Hash the extracted original file
        print!("  hashing... ");
        std::io::stdout().flush()?;
        // Always use GPU 0 for hashing
        let decompressed_hash = blake3_hash_file(output_path, 0)?;
        println!("ok");

        if decompressed_hash == expected_hash {
            println!("  integrity: ok");
        } else {
            println!("  integrity: FAILED");
            println!("   Expected: {}", expected_hash);
            println!("   Got:      {}", decompressed_hash);
            return Err(anyhow::anyhow!(
                "Decompressed file hash does not match original"
            ));
        }

        // Clean up temporary tar file
        fs::remove_file(&tar_path)?;
    }

    Ok(())
}

fn decompress_multi_chunk_file(
    input_path: &PathBuf,
    output_path: &PathBuf,
    device_id: i32,
) -> Result<()> {
    println!("gpu-compressor: multi-chunk decompression");

    // Read the multi-chunk file header
    let mut input_file = File::open(input_path)?;
    let mut header = [0u8; 20]; // NVMC(4) + original_size(8) + num_chunks(8)
    input_file.read_exact(&mut header)?;

    let original_size = u64::from_le_bytes(header[4..12].try_into().unwrap()) as usize;
    let num_chunks = u64::from_le_bytes(header[12..20].try_into().unwrap()) as usize;

    println!(
        "  original: {:.2} GB ({} chunks)",
        original_size as f64 / 1_000_000_000.0,
        num_chunks
    );

    // Read remaining compressed data
    let mut compressed_data = Vec::new();
    input_file.read_to_end(&mut compressed_data)?;

    // Create output file
    let mut output_file = File::create(output_path)?;
    let mut decompressed_total = 0usize;

    // Process each compressed chunk
    let mut offset = 0;
    for chunk_idx in 0..num_chunks {
        println!("  Decompressing chunk {}/{}...", chunk_idx + 1, num_chunks);

        // Each chunk is a complete NVGD compressed file
        // Extract this chunk's data (starts with NVGD magic)
        if offset >= compressed_data.len() {
            return Err(anyhow::anyhow!(
                "Unexpected end of multi-chunk file at chunk {}",
                chunk_idx
            ));
        }

        // Find the NVGD magic header
        if &compressed_data[offset..offset + 4] != b"NVGD" {
            return Err(anyhow::anyhow!(
                "Expected NVGD magic at chunk {}",
                chunk_idx
            ));
        }

        // Parse this chunk's header to find its size
        let _chunk_original_size =
            u64::from_le_bytes(compressed_data[offset + 4..offset + 12].try_into().unwrap())
                as usize;
        let _chunk_chunk_size = u64::from_le_bytes(
            compressed_data[offset + 12..offset + 20]
                .try_into()
                .unwrap(),
        ) as usize;
        let chunk_num_chunks = u64::from_le_bytes(
            compressed_data[offset + 20..offset + 28]
                .try_into()
                .unwrap(),
        ) as usize;

        // Calculate total size of this compressed chunk
        let chunk_header_size = 28 + chunk_num_chunks * 8; // Header + size array
        let mut chunk_compressed_size = 0usize;

        for i in 0..chunk_num_chunks {
            let size_offset = offset + 28 + i * 8;
            let comp_size = u64::from_le_bytes(
                compressed_data[size_offset..size_offset + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            chunk_compressed_size += comp_size;
        }

        let total_chunk_size = chunk_header_size + chunk_compressed_size;

        // Write this chunk to a temp file
        let temp_compressed = output_path.with_extension(format!("tmp_c{}", chunk_idx));
        let mut temp_file = File::create(&temp_compressed)?;
        temp_file.write_all(&compressed_data[offset..offset + total_chunk_size])?;
        drop(temp_file);

        // Decompress this chunk
        let temp_decompressed = output_path.with_extension(format!("tmp_d{}", chunk_idx));

        // Call decompress on the temp file (will use NVGD path)
        decompress_file(&temp_compressed, &temp_decompressed, device_id)?;

        // Append decompressed data to output
        let mut chunk_data = Vec::new();
        File::open(&temp_decompressed)?.read_to_end(&mut chunk_data)?;
        output_file.write_all(&chunk_data)?;
        decompressed_total += chunk_data.len();

        // Clean up temp files
        fs::remove_file(temp_compressed)?;
        fs::remove_file(temp_decompressed)?;

        offset += total_chunk_size;
    }

    println!(
        "  done. ({:.2} GB)",
        decompressed_total as f64 / 1_000_000_000.0
    );

    Ok(())
}

// Multi-file parallel decompression - for simplicity, decompress files sequentially for now
// TODO: Implement truly parallel decompression with interleaved chunks
pub(crate) fn decompress_multi_files(
    inputs: &[PathBuf],
    outputs: &[PathBuf],
    device_id: i32,
) -> Result<()> {
    println!("gpu-compressor: multi-file decompression");
    println!("  files: {}", inputs.len());

    let start_time = Instant::now();

    for (input, output) in inputs.iter().zip(outputs.iter()) {
        println!("  {}", input.file_name().unwrap().to_str().unwrap());
        decompress_file(input, output, device_id)?;
    }

    let elapsed = start_time.elapsed();
    println!("  done.");
    println!("  time: {:.1}s", elapsed.as_secs_f64());

    Ok(())
}
