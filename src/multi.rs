use crate::blake3::blake3_hash_file;
use crate::cli::Algorithm;
use crate::constants::ZSTD_CHUNK_SIZE;
use crate::nvcomp_bindings as nvcomp;
use anyhow::Result;
use cuda_runtime_sys::*;
use std::fs::{self};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

// Compress multiple pre-split chunks in a single GPU batch (for multi-file processing)
pub(crate) fn compress_buffer_zstd_multi(
    chunks: &[Vec<u8>],
    device_id: i32,
) -> Result<Vec<Vec<u8>>> {
    unsafe {
        // Initialize CUDA device
        let result = cudaSetDevice(device_id);
        if result != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }

        let num_chunks = chunks.len();

        // Zstd compression options
        let compress_opts = nvcomp::nvcompBatchedZstdCompressOpts_t {
            reserved: [0; 64],
        };

        // Allocate GPU memory for input chunks
        let mut d_uncompressed_ptrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); num_chunks];
        let mut chunk_sizes = Vec::new();

        // Upload all chunks to GPU
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_size = chunk.len();
            chunk_sizes.push(chunk_size);

            cudaMalloc(&mut d_uncompressed_ptrs[i], chunk_size);
            cudaMemcpy(
                d_uncompressed_ptrs[i],
                chunk.as_ptr() as *const std::ffi::c_void,
                chunk_size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
        }

        // Allocate device arrays for metadata
        let mut d_uncompressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_chunk_sizes_dev: *mut usize = std::ptr::null_mut();
        let mut d_compressed_ptrs_dev: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut d_compressed_sizes_dev: *mut usize = std::ptr::null_mut();
        let mut d_statuses_dev: *mut nvcomp::nvcompStatus_t = std::ptr::null_mut();

        cudaMalloc(
            &mut d_uncompressed_ptrs_dev,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
        );
        cudaMalloc(
            &mut d_chunk_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
        );
        cudaMalloc(
            &mut d_compressed_ptrs_dev,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
        );
        cudaMalloc(
            &mut d_compressed_sizes_dev as *mut *mut usize as *mut *mut std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
        );
        cudaMalloc(
            &mut d_statuses_dev as *mut *mut nvcomp::nvcompStatus_t as *mut *mut std::ffi::c_void,
            num_chunks * std::mem::size_of::<nvcomp::nvcompStatus_t>(),
        );

        cudaMemcpy(
            d_uncompressed_ptrs_dev,
            d_uncompressed_ptrs.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        cudaMemcpy(
            d_chunk_sizes_dev as *mut std::ffi::c_void,
            chunk_sizes.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        // Get max compressed size
        let mut max_compressed_chunk_size: usize = 0;
        nvcomp::nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            ZSTD_CHUNK_SIZE,
            compress_opts,
            &mut max_compressed_chunk_size as *mut usize,
        );

        // Allocate output buffers
        let mut d_compressed_ptrs = vec![std::ptr::null_mut::<std::ffi::c_void>(); num_chunks];
        for (_i, d_compressed_ptr) in d_compressed_ptrs.iter_mut().enumerate().take(num_chunks) {
            cudaMalloc(d_compressed_ptr, max_compressed_chunk_size);
        }
        cudaMemcpy(
            d_compressed_ptrs_dev,
            d_compressed_ptrs.as_ptr() as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<*mut std::ffi::c_void>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );

        // Get temp space size using sync version
        let mut temp_bytes: usize = 0;
        nvcomp::nvcompBatchedZstdCompressGetTempSizeSync(
            d_uncompressed_ptrs_dev as *const *const std::ffi::c_void,
            d_chunk_sizes_dev as *const usize,
            num_chunks,
            ZSTD_CHUNK_SIZE,
            compress_opts,
            &mut temp_bytes,
            num_chunks * ZSTD_CHUNK_SIZE, // max_total_uncompressed_bytes
            std::ptr::null_mut(),         // stream
        );

        let mut d_temp: *mut std::ffi::c_void = std::ptr::null_mut();
        cudaMalloc(&mut d_temp, temp_bytes);

        // Compress
        nvcomp::nvcompBatchedZstdCompressAsync(
            d_uncompressed_ptrs_dev as *const *const std::ffi::c_void,
            d_chunk_sizes_dev as *const usize,
            ZSTD_CHUNK_SIZE,
            num_chunks,
            d_temp,
            temp_bytes,
            d_compressed_ptrs_dev as *const *mut std::ffi::c_void,
            d_compressed_sizes_dev,
            compress_opts,
            d_statuses_dev,
            std::ptr::null_mut(), // stream
        );

        let err = cudaDeviceSynchronize();
        if err != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!(
                "cudaDeviceSynchronize failed after compression: {:?}",
                err
            ));
        }

        // Download compressed sizes
        let mut compressed_sizes = vec![0usize; num_chunks];
        cudaMemcpy(
            compressed_sizes.as_mut_ptr() as *mut std::ffi::c_void,
            d_compressed_sizes_dev as *const std::ffi::c_void,
            num_chunks * std::mem::size_of::<usize>(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );

        // Download compressed data
        let mut compressed_chunks = Vec::new();
        for i in 0..num_chunks {
            let mut compressed_chunk = vec![0u8; compressed_sizes[i]];
            cudaMemcpy(
                compressed_chunk.as_mut_ptr() as *mut std::ffi::c_void,
                d_compressed_ptrs[i],
                compressed_sizes[i],
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            compressed_chunks.push(compressed_chunk);
        }

        // Free GPU memory
        for ptr in d_uncompressed_ptrs {
            cudaFree(ptr);
        }
        for ptr in d_compressed_ptrs {
            cudaFree(ptr);
        }
        cudaFree(d_temp);
        cudaFree(d_uncompressed_ptrs_dev);
        cudaFree(d_chunk_sizes_dev as *mut std::ffi::c_void);
        cudaFree(d_compressed_ptrs_dev);
        cudaFree(d_compressed_sizes_dev as *mut std::ffi::c_void);
        cudaFree(d_statuses_dev as *mut std::ffi::c_void);

        Ok(compressed_chunks)
    }
}

// Multi-file parallel compression - interleaves chunks from multiple files for maximum GPU utilization
// Async multi-file compression with true parallelism - all files read/compress/write simultaneously
pub(crate) async fn compress_multi_files_async(
    inputs: &[PathBuf],
    outputs: &[PathBuf],
    algorithm: Algorithm,
    device_id: i32,
    chunk_size: usize,
) -> Result<()> {
    use futures::future::join_all;
    use tokio::fs::File as TokioFile;
    use tokio::io::AsyncReadExt;
    use tokio::sync::mpsc;

    if algorithm != Algorithm::Zstd {
        return Err(anyhow::anyhow!(
            "Async multi-file processing currently only supports Zstd algorithm"
        ));
    }

    println!("gpu-compressor: zstd multi-file compression");
    println!(
        "  files: {} (2 concurrent)",
        inputs.len()
    );

    // Initialize CUDA device
    unsafe {
        let result = cudaSetDevice(device_id);
        if result != cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device {}", device_id));
        }
    }

    // Get file sizes and calculate total chunks for all files
    let mut all_file_info = Vec::new();
    let mut grand_total_size = 0u64;
    for (input, output) in inputs.iter().zip(outputs.iter()) {
        let metadata = tokio::fs::metadata(input).await?;
        let file_size = metadata.len();
        grand_total_size += file_size;
        let total_chunks = (file_size as usize).div_ceil(chunk_size);

        println!(
            "    {} ({:.2} GB, {} chunks)",
            input.file_name().unwrap().to_str().unwrap(),
            file_size as f64 / 1_000_000_000.0,
            total_chunks
        );

        all_file_info.push((input.clone(), output.clone(), file_size, total_chunks));
    }

    println!(
        "  total: {:.2} GB",
        grand_total_size as f64 / 1_000_000_000.0
    );

    // Hash all input files before compression
    println!("  hashing inputs...");
    let mut input_hashes = Vec::new();
    for (input_path, _, _, _) in &all_file_info {
        print!(
            "    {}... ",
            input_path.file_name().unwrap().to_str().unwrap()
        );
        std::io::stdout().flush()?;
        // Always use GPU 0 for hashing
        let hash = blake3_hash_file(input_path, 0)?;
        println!("ok");
        input_hashes.push(hash);
    }
    println!("  hashing complete.\n");

    let overall_start_time = Instant::now();

    // Process files in batches of 2
    const MAX_CONCURRENT_FILES: usize = 2;
    for batch_idx in (0..all_file_info.len()).step_by(MAX_CONCURRENT_FILES) {
        let batch_end = std::cmp::min(batch_idx + MAX_CONCURRENT_FILES, all_file_info.len());
        let file_info: Vec<_> = all_file_info[batch_idx..batch_end].to_vec();
        let batch_hashes: Vec<_> = input_hashes[batch_idx..batch_end].to_vec();

        println!(
            "  batch {}-{}/{}",
            batch_idx + 1,
            batch_end,
            all_file_info.len()
        );

        let batch_total_size: u64 = file_info.iter().map(|(_, _, size, _)| size).sum();
        let total_chunks: usize = file_info.iter().map(|(_, _, _, chunks)| chunks).sum();

        // Channels for pipeline
        // Reader tasks -> Compression queue
        let (read_tx, mut read_rx) = mpsc::channel::<(usize, Vec<u8>, usize)>(32); // (file_idx, chunk_data, chunk_idx)

        // Compressor -> Writer tasks
        let mut write_txs = Vec::new();
        let mut write_rxs = Vec::new();
        for _ in 0..file_info.len() {
            let (tx, rx) = mpsc::channel::<(Vec<u8>, usize)>(32); // (compressed_chunk, chunk_idx)
            write_txs.push(tx);
            write_rxs.push(rx);
        }

        let batch_start_time = Instant::now();

        // Spawn reader tasks (one per file in this batch)
        let mut reader_handles = Vec::new();
        for (file_idx, (input_path, _, file_size, total_chunks)) in file_info.iter().enumerate() {
            let input_path = input_path.clone();
            let file_size = *file_size;
            let total_chunks = *total_chunks;
            let read_tx = read_tx.clone();

            let reader_handle = tokio::spawn(async move {
                let mut file = TokioFile::open(&input_path).await?;
                let mut bytes_read = 0u64;

                for chunk_idx in 0..total_chunks {
                    let remaining = file_size - bytes_read;
                    let current_chunk_size = std::cmp::min(chunk_size as u64, remaining) as usize;
                    let mut chunk = vec![0u8; current_chunk_size];

                    file.read_exact(&mut chunk).await?;
                    bytes_read += current_chunk_size as u64;

                    // Send chunk to compression queue
                    read_tx
                        .send((file_idx, chunk, chunk_idx))
                        .await
                        .map_err(|e| anyhow::anyhow!("Reader channel error: {}", e))?;
                }

                Ok::<(), anyhow::Error>(())
            });

            reader_handles.push(reader_handle);
        }
        drop(read_tx); // Close sender so compressor knows when done

        // Spawn GPU compressor task
        let write_txs_clone = write_txs.clone();
        let compressor_handle = tokio::task::spawn_blocking(move || {
            let target_batch_size = 9; // 9 chunks per batch
            let mut batch_chunks = Vec::new();
            let mut batch_metadata = Vec::new(); // (file_idx, chunk_idx)
            let mut chunks_processed = 0;

            // Timing metrics
            let mut total_read_wait_time = 0.0;
            let mut total_gpu_time = 0.0;
            let mut total_write_send_time = 0.0;

            let mut batch_start = Instant::now();
            while let Some((file_idx, chunk_data, chunk_idx)) = read_rx.blocking_recv() {
                let read_wait_time = batch_start.elapsed().as_secs_f64();

                batch_chunks.push(chunk_data);
                batch_metadata.push((file_idx, chunk_idx));

                if batch_chunks.len() >= target_batch_size {
                    total_read_wait_time += read_wait_time;

                    // Calculate batch size
                    let batch_size_bytes: usize = batch_chunks.iter().map(|c| c.len()).sum();
                    let batch_size_mb = batch_size_bytes as f64 / 1_000_000.0;

                    // GPU compression timing
                    let gpu_start = Instant::now();
                    match compress_buffer_zstd_multi(&batch_chunks, device_id) {
                        Ok(compressed_batch) => {
                            let gpu_time = gpu_start.elapsed().as_secs_f64();
                            total_gpu_time += gpu_time;

                            let compressed_size: usize =
                                compressed_batch.iter().map(|c| c.len()).sum();
                            let compressed_mb = compressed_size as f64 / 1_000_000.0;
                            let gpu_throughput = batch_size_mb / gpu_time;

                            // Write send timing
                            let write_start = Instant::now();
                            for (compressed_chunk, (file_idx, chunk_idx)) in
                                compressed_batch.iter().zip(batch_metadata.iter())
                            {
                                let _ = write_txs_clone[*file_idx]
                                    .blocking_send((compressed_chunk.clone(), *chunk_idx));
                            }
                            let write_send_time = write_start.elapsed().as_secs_f64();
                            total_write_send_time += write_send_time;

                            // Dynamic update: use \r to overwrite same line
                            print!("\r  {}-{}/{} ({} chunks, {:.1}MB) | gpu: {:.3}s ({:.1}MB/s) | out: {:.1}MB ({:.3}s)",
                                chunks_processed + 1,
                                chunks_processed + batch_chunks.len(),
                                total_chunks,
                                batch_chunks.len(),
                                batch_size_mb,
                                gpu_time,
                                gpu_throughput,
                                compressed_mb,
                                write_send_time);
                            std::io::stdout().flush().unwrap();

                            chunks_processed += batch_chunks.len();
                            batch_chunks.clear();
                            batch_metadata.clear();
                        }
                        Err(e) => {
                            eprintln!("  error: compression failed: {}", e);
                            return Err(e);
                        }
                    }

                    // Reset batch start for next iteration
                    batch_start = Instant::now();
                }
            }

            // Process remaining chunks
            if !batch_chunks.is_empty() {
                let batch_size_bytes: usize = batch_chunks.iter().map(|c| c.len()).sum();
                let batch_size_mb = batch_size_bytes as f64 / 1_000_000.0;

                let gpu_start = Instant::now();
                let compressed_batch = compress_buffer_zstd_multi(&batch_chunks, device_id)?;
                let gpu_time = gpu_start.elapsed().as_secs_f64();
                total_gpu_time += gpu_time;

                let compressed_size: usize = compressed_batch.iter().map(|c| c.len()).sum();
                let compressed_mb = compressed_size as f64 / 1_000_000.0;
                let gpu_throughput = batch_size_mb / gpu_time;

                let write_start = Instant::now();
                for (compressed_chunk, (file_idx, chunk_idx)) in
                    compressed_batch.iter().zip(batch_metadata.iter())
                {
                    let _ = write_txs_clone[*file_idx]
                        .blocking_send((compressed_chunk.clone(), *chunk_idx));
                }
                let write_send_time = write_start.elapsed().as_secs_f64();
                total_write_send_time += write_send_time;

                // Final batch: print with newline
                println!("\r  {}-{}/{} ({} chunks, {:.1}MB) | gpu: {:.3}s ({:.1}MB/s) | out: {:.1}MB ({:.3}s)",
                    chunks_processed + 1,
                    chunks_processed + batch_chunks.len(),
                    total_chunks,
                    batch_chunks.len(),
                    batch_size_mb,
                    gpu_time,
                    gpu_throughput,
                    compressed_mb,
                    write_send_time);
            } else {
                // No final batch, just add newline after last dynamic update
                println!();
            }

            // Print timing summary
            println!("\n  metrics:");
            println!("    read wait:  {:.2}s", total_read_wait_time);
            println!("    gpu time:   {:.2}s", total_gpu_time);
            println!(
                "    write time: {:.2}s",
                total_write_send_time
            );

            Ok::<(), anyhow::Error>(())
        });

        // Spawn writer tasks (one per file) - write directly to final files with placeholders
        let mut writer_handles = Vec::new();
        for (file_idx, mut write_rx) in write_rxs.into_iter().enumerate() {
            let (_input_path, output_path, file_size, total_chunks) = file_info[file_idx].clone();
            let _input_hash = batch_hashes[file_idx].clone();

            let writer_handle = tokio::spawn(async move {
                use tokio::io::{AsyncSeekExt, AsyncWriteExt};

                let mut output_file = TokioFile::create(&output_path).await?;
                let mut chunk_sizes_with_idx = Vec::new();

                // Write standard NVZS header (28 bytes + N*8 chunk sizes)
                output_file.write_all(b"NVZS").await?; // Magic (4 bytes)
                output_file.write_all(&file_size.to_le_bytes()).await?; // Original size (8 bytes)
                output_file
                    .write_all(&(chunk_size as u64).to_le_bytes())
                    .await?; // Chunk size (8 bytes)
                output_file
                    .write_all(&(total_chunks as u64).to_le_bytes())
                    .await?; // Num chunks (8 bytes)

                let chunk_sizes_offset = 28u64; // 4 + 8 + 8 + 8 = 28

                // Write placeholder chunk sizes
                for _ in 0..total_chunks {
                    output_file.write_all(&0u64.to_le_bytes()).await?;
                }

                // Receive and write compressed chunks
                while let Some((compressed_chunk, chunk_idx)) = write_rx.recv().await {
                    output_file.write_all(&compressed_chunk).await?;
                    chunk_sizes_with_idx.push((chunk_idx, compressed_chunk.len()));
                }

                // Flush writes
                output_file.flush().await?;

                // Sort chunk sizes by index to ensure correct order
                chunk_sizes_with_idx.sort_by_key(|(idx, _)| *idx);
                let chunk_sizes: Vec<usize> = chunk_sizes_with_idx
                    .into_iter()
                    .map(|(_, size)| size)
                    .collect();

                // Seek back to chunk sizes position
                output_file
                    .seek(std::io::SeekFrom::Start(chunk_sizes_offset))
                    .await?;

                // Write actual chunk sizes
                for &size in &chunk_sizes {
                    output_file.write_all(&(size as u64).to_le_bytes()).await?;
                }

                // Final flush
                output_file.flush().await?;

                Ok::<Vec<usize>, anyhow::Error>(chunk_sizes)
            });

            writer_handles.push(writer_handle);
        }

        // Wait for all readers to complete
        for handle in reader_handles {
            handle.await??;
        }

        // Wait for compressor to complete
        compressor_handle.await??;

        // Close all writer channels
        drop(write_txs);

        // Wait for all writers and collect results
        let writer_results = join_all(writer_handles).await;

        // Print batch results
        println!("  completed:");
        for (file_idx, result) in writer_results.into_iter().enumerate() {
            let chunk_sizes = result??;
            let (input_path, output_path, file_size, _total_chunks) = &file_info[file_idx];

            let compressed_size: usize = chunk_sizes.iter().sum();
            let ratio = (compressed_size as f64 / *file_size as f64) * 100.0;
            println!(
                "    {} -> {} ({:.2}%)",
                input_path.file_name().unwrap().to_str().unwrap(),
                output_path.file_name().unwrap().to_str().unwrap(),
                ratio
            );
        }

        // Reset GPU state after dual-GPU compression before hashing
        unsafe {
            cuda_runtime_sys::cudaSetDevice(0);
            cuda_runtime_sys::cudaDeviceSynchronize();
            cuda_runtime_sys::cudaDeviceReset();
            cuda_runtime_sys::cudaSetDevice(1);
            cuda_runtime_sys::cudaDeviceSynchronize();
            cuda_runtime_sys::cudaDeviceReset();
            cuda_runtime_sys::cudaSetDevice(0);
        }

        // Hash compressed output files and write hash to .blake3 files
        println!("  hashing outputs...");
        for (_input_path, output_path, _, _) in &file_info {
            print!(
                "    {}... ",
                output_path.file_name().unwrap().to_str().unwrap()
            );
            std::io::stdout().flush()?;
            // Always use GPU 0 for hashing (compression may use both GPUs)
            let output_hash = blake3_hash_file(output_path, 0)?;

            // Write hash to .blake3 file
            let hash_blake3_path = format!("{}.blake3", output_path.display());
            let mut hash_file = fs::File::create(&hash_blake3_path)?;
            writeln!(hash_file, "{}", output_hash)?;

            println!(
                "ok -> {}",
                PathBuf::from(&hash_blake3_path)
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
            );
        }

        let batch_elapsed = batch_start_time.elapsed();
        let batch_throughput =
            batch_total_size as f64 / batch_elapsed.as_secs_f64() / 1_000_000_000.0;
        println!("    time:       {:.1}s", batch_elapsed.as_secs_f64());
        println!("    throughput: {:.2} GB/s", batch_throughput);
    } // End of batch loop

    // Print overall summary
    let overall_elapsed = overall_start_time.elapsed();
    let overall_throughput =
        grand_total_size as f64 / overall_elapsed.as_secs_f64() / 1_000_000_000.0;

    println!("\n  done.");
    println!("  time:       {:.1}s", overall_elapsed.as_secs_f64());
    println!("  throughput: {:.2} GB/s", overall_throughput);

    Ok(())
}
