use anyhow::Result;
use cudarc::driver::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::constants::{BLAKE3_CHUNK_SIZE, PTX_BLAKE3};

/// GPU-accelerated BLAKE3 file hashing with streaming chunks
pub(crate) fn blake3_hash_file(file_path: &Path, device_id: usize) -> Result<String> {
    println!("blake3: gpu streaming hash");
    println!("  input: {}", file_path.display());

    let mut file = File::open(file_path)?;
    let file_size = file.metadata()?.len();

    println!("  size:  {:.2} GB", file_size as f64 / 1_000_000_000.0);

    // Initialize GPU
    let device = CudaDevice::new(device_id)?;
    let dev = Arc::new(device);

    println!("  gpu:   {} ({})", device_id, dev.name()?);

    // Load BLAKE3 kernels
    dev.load_ptx(
        PTX_BLAKE3.into(),
        "blake3_module",
        &["blake3_hash_chunks", "blake3_reduce_tree"],
    )?;
    let func_chunks = dev
        .get_func("blake3_module", "blake3_hash_chunks")
        .ok_or_else(|| anyhow::anyhow!("Failed to load BLAKE3 chunks kernel"))?;
    let func_reduce = dev
        .get_func("blake3_module", "blake3_reduce_tree")
        .ok_or_else(|| anyhow::anyhow!("Failed to load BLAKE3 tree reduction kernel"))?;

    // Stream file in 4MB batches (each batch processes ~4096 x 1KB BLAKE3 chunks in parallel)
    const BATCH_SIZE: usize = 4 * 1024 * 1024; // 4MB batches keep GPU continuously fed
    let num_batches = file_size.div_ceil(BATCH_SIZE as u64) as usize;
    let total_blake3_chunks = file_size.div_ceil(BLAKE3_CHUNK_SIZE as u64) as usize;

    println!("  batches: {} x 4MB ({} chunks)", num_batches, total_blake3_chunks);

    if file_size == 0 {
        return Err(anyhow::anyhow!("Empty file"));
    }

    let total_start = Instant::now();

    // Collect all BLAKE3 chunk hashes (8 u32s per chunk = 32 bytes)
    let mut all_chunk_hashes = Vec::with_capacity(total_blake3_chunks * 8);
    let mut bytes_read = 0u64;

    // Process file in 4MB batches
    for batch_idx in 0..num_batches {
        // Read 4MB batch from disk
        let batch_bytes = std::cmp::min(BATCH_SIZE as u64, file_size - bytes_read) as usize;
        let mut batch_buffer = vec![0u8; batch_bytes];
        file.read_exact(&mut batch_buffer)?;

        // Calculate how many 1KB BLAKE3 chunks in this batch
        let blake3_chunks_in_batch = batch_bytes.div_ceil(BLAKE3_CHUNK_SIZE);

        // Upload entire batch to GPU
        let d_data = dev.htod_sync_copy(&batch_buffer)?;

        // Allocate output for BLAKE3 chunk hashes
        let d_chunk_outputs = dev.alloc_zeros::<u32>(blake3_chunks_in_batch * 8)?;

        // Launch one thread per 1KB BLAKE3 chunk
        let threads_per_block = 256;
        let num_blocks = blake3_chunks_in_batch.div_ceil(threads_per_block) as u32;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Hash all 1KB chunks within this batch
        unsafe {
            func_chunks.clone().launch(
                cfg,
                (
                    &d_data,
                    batch_bytes as u64,
                    &d_chunk_outputs,
                    blake3_chunks_in_batch as u64,
                ),
            )?;
        }

        dev.synchronize()?;

        // Download the chunk hashes
        let h_chunk_hashes = dev.dtoh_sync_copy(&d_chunk_outputs)?;
        all_chunk_hashes.extend_from_slice(&h_chunk_hashes);

        bytes_read += batch_bytes as u64;

        // Show progress every 100 batches or on last batch
        if batch_idx % 100 == 0 || batch_idx == num_batches - 1 {
            let progress = bytes_read as f64 / file_size as f64 * 100.0;
            eprint!(
                "\r  progress: {}/{} ({:.1}%)",
                batch_idx + 1,
                num_batches,
                progress
            );
        }
    }
    eprintln!();

    let hash_time = total_start.elapsed();
    let hash_gbs = (file_size as f64 / hash_time.as_secs_f64()) / 1_000_000_000.0;
    println!(
        "  hashed: {:.2} GB in {:.3}s ({:.2} GB/s)",
        file_size as f64 / 1_000_000_000.0,
        hash_time.as_secs_f64(),
        hash_gbs
    );

    // If single chunk, we're done
    if total_blake3_chunks == 1 {
        let hash_bytes: Vec<u8> = all_chunk_hashes
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        println!("  time: {:.3}s", hash_time.as_secs_f64());
        return Ok(hex::encode(&hash_bytes[0..32]));
    }

    // Tree reduction on GPU: Combine all chunk hashes
    let tree_start = Instant::now();

    // Upload all chunk hashes to GPU for final reduction
    let d_all_hashes = dev.htod_sync_copy(&all_chunk_hashes)?;
    let mut current_hashes = d_all_hashes;
    let mut current_count = total_blake3_chunks;
    let mut level = 0;

    while current_count > 1 {
        let next_count = current_count.div_ceil(2);
        let d_next_hashes = dev.alloc_zeros::<u32>(next_count * 8)?;

        let pairs = current_count / 2;
        let threads = 256;
        let blocks = pairs.div_ceil(threads) as u32;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_reduce.clone().launch(
                cfg,
                (&current_hashes, current_count as u64, &d_next_hashes, 0u64),
            )?;
        }

        dev.synchronize()?;

        current_hashes = d_next_hashes;
        current_count = if current_count % 2 == 1 {
            pairs + 1
        } else {
            pairs
        };
        level += 1;
    }

    let tree_time = tree_start.elapsed();
    println!(
        "  tree reduction: {} levels, {:.3}s",
        level,
        tree_time.as_secs_f64()
    );

    // Get final hash
    let h_output = dev.dtoh_sync_copy(&current_hashes)?;
    let hash_bytes: Vec<u8> = h_output.iter().flat_map(|&x| x.to_le_bytes()).collect();

    let total_time = total_start.elapsed();
    println!("  time: {:.3}s", total_time.as_secs_f64());
    println!(
        "  throughput: {:.2} GB/s",
        file_size as f64 / total_time.as_secs_f64() / 1_000_000_000.0
    );

    Ok(hex::encode(&hash_bytes[0..32]))
}
