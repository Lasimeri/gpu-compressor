use anyhow::Result;
use crossbeam_channel::bounded;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use crate::blake3::blake3_hash_file;
use crate::compress_zstd::compress_chunk_zstd;
use crate::constants::ZSTD_CHUNK_SIZE;
use crate::format::{CompressedChunk, PipelineMsg};

/// Dual GPU async pipeline: Process alternating chunks on two GPUs simultaneously
pub(crate) fn compress_file_streaming_dual_gpu(
    input_path: &Path,
    output_path: &Path,
    gpu0: i32,
    gpu1: i32,
    _quiet: bool,
) -> Result<()> {
    compress_file_streaming_dual_gpu_with_hash(input_path, output_path, None, gpu0, gpu1, _quiet)
}

/// Dual GPU async pipeline with optional pre-computed hash
pub(crate) fn compress_file_streaming_dual_gpu_with_hash(
    input_path: &Path,
    output_path: &Path,
    precomputed_hash: Option<&str>,
    gpu0: i32,
    gpu1: i32,
    _quiet: bool,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len();

    eprintln!("gpu-compressor: zstd compression (dual gpu)");
    eprintln!("  input:  {}", input_path.display());
    eprintln!("  output: {}", output_path.display());
    eprintln!("  size:   {:.2} GB", file_size as f64 / 1e9);
    eprintln!("  gpu 0:  device {}", gpu0);
    eprintln!("  gpu 1:  device {}", gpu1);

    // Use pre-computed hash or compute it now
    let input_hash = if let Some(hash) = precomputed_hash {
        eprintln!("  hash: (cached)");
        hash.to_string()
    } else {
        eprintln!("  hashing input...");
        // Always use GPU 0 for hashing
        let h = blake3_hash_file(input_path, 0)?;
        eprintln!("  hash: {}", h);
        h
    };

    // Prepare .blake3 content
    let blake3_content = format!("{}\n", input_hash);
    let blake3_filename = format!(
        "{}.blake3",
        input_path.file_name().unwrap().to_str().unwrap()
    );

    // Calculate tar size
    let original_file_size = file_size;
    let blake3_size = blake3_content.len() as u64;
    let tar_size =
        512 + original_file_size.div_ceil(512) * 512 + 512 + blake3_size.div_ceil(512) * 512 + 1024;

    eprintln!("  pipeline: read 4MB -> gpu {} (even) | gpu {} (odd) -> write", gpu0, gpu1);

    // 8MB buffering: 2 chunks per GPU (4MB each) for smooth I/O
    const PIPELINE_QUEUE_SIZE: usize = 2;

    // Channels: Read -> GPU threads (separate for each GPU)
    let (read_tx0, read_rx0) = bounded::<PipelineMsg>(PIPELINE_QUEUE_SIZE);
    let (read_tx1, read_rx1) = bounded::<PipelineMsg>(PIPELINE_QUEUE_SIZE);

    // Channels: GPU threads -> Writer (single merged channel)
    let (compress_tx, compress_rx) = bounded::<CompressedChunk>(PIPELINE_QUEUE_SIZE * 2);

    // Atomic counters
    let bytes_read = Arc::new(AtomicU64::new(0));
    let bytes_compressed = Arc::new(AtomicU64::new(0));
    let bytes_written = Arc::new(AtomicU64::new(0));
    let _ = bytes_written; // Used in writer thread

    // No spinner infrastructure -- plain eprintln status messages

    // Read thread: Distribute chunks alternately to GPU 0 and GPU 1
    let input_path_clone = input_path.to_path_buf();
    let blake3_content_clone = blake3_content.clone();
    let blake3_filename_clone = blake3_filename.clone();
    let bytes_read_reader = bytes_read.clone();

    let reader_thread = thread::spawn(move || -> Result<()> {
        use crossbeam_channel::Sender;
        use std::io::Write;

        eprintln!("  tar: starting");

        // Custom writer that buffers and distributes to alternating GPUs
        struct DualChannelWriter {
            buffer: Vec<u8>,
            batch_size: usize,
            batch_index: usize,
            sender0: Sender<PipelineMsg>,
            sender1: Sender<PipelineMsg>,
            bytes_sent: Arc<AtomicU64>,
            gpu0_id: i32,
            gpu1_id: i32,
        }

        impl Write for DualChannelWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let mut offset = 0;
                while offset < buf.len() {
                    let space_left = self.batch_size - self.buffer.len();
                    let to_copy = std::cmp::min(space_left, buf.len() - offset);

                    self.buffer
                        .extend_from_slice(&buf[offset..offset + to_copy]);
                    offset += to_copy;

                    // Send batch when full
                    if self.buffer.len() == self.batch_size {
                        self.flush_batch()?;
                    }
                }
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                if !self.buffer.is_empty() {
                    self.flush_batch()?;
                }
                Ok(())
            }
        }

        impl DualChannelWriter {
            fn flush_batch(&mut self) -> std::io::Result<()> {
                let batch_data =
                    std::mem::replace(&mut self.buffer, Vec::with_capacity(self.batch_size));
                let batch_len = batch_data.len();

                // Alternate between GPU 0 (even) and GPU 1 (odd)
                let sender = if self.batch_index.is_multiple_of(2) {
                    &self.sender0
                } else {
                    &self.sender1
                };

                sender
                    .send(PipelineMsg::Chunk {
                        chunk_index: self.batch_index,
                        data: batch_data,
                    })
                    .map_err(std::io::Error::other)?;

                self.bytes_sent
                    .fetch_add(batch_len as u64, Ordering::Relaxed);
                let gpu_id = if self.batch_index.is_multiple_of(2) {
                    self.gpu0_id
                } else {
                    self.gpu1_id
                };
                eprintln!(
                    "  tar: chunk #{} -> gpu {} ({:.2} MB)",
                    self.batch_index,
                    gpu_id,
                    batch_len as f64 / 1_000_000.0
                );
                self.batch_index += 1;
                Ok(())
            }
        }

        let tar_chunk_size = ZSTD_CHUNK_SIZE; // 4MB chunks

        let writer = DualChannelWriter {
            buffer: Vec::with_capacity(tar_chunk_size),
            batch_size: tar_chunk_size,
            batch_index: 0,
            sender0: read_tx0.clone(),
            sender1: read_tx1.clone(),
            bytes_sent: bytes_read_reader.clone(),
            gpu0_id: gpu0,
            gpu1_id: gpu1,
        };

        let mut tar_builder = tar::Builder::new(writer);

        // Add original file
        eprintln!("  tar: reading file");
        let original_file = File::open(&input_path_clone)?;
        let file_metadata = original_file.metadata()?;
        // 8MB read buffer for smooth disk I/O
        let mut buffered_file = std::io::BufReader::with_capacity(8 * 1024 * 1024, original_file);

        let mut header = tar::Header::new_gnu();
        header.set_metadata(&file_metadata);
        header.set_cksum();
        tar_builder.append_data(
            &mut header,
            input_path_clone.file_name().unwrap(),
            &mut buffered_file,
        )?;

        // Add .blake3 hash
        eprintln!("  tar: adding hash");
        let mut blake3_header = tar::Header::new_gnu();
        blake3_header.set_size(blake3_content_clone.len() as u64);
        blake3_header.set_mode(0o644);
        blake3_header.set_cksum();
        tar_builder.append_data(
            &mut blake3_header,
            &blake3_filename_clone,
            blake3_content_clone.as_bytes(),
        )?;

        // Finish tar and flush remaining buffered data
        // NOTE: tar::Builder::finish() does NOT call flush() on the inner writer
        eprintln!("  tar: finalizing");
        let mut writer = tar_builder.into_inner()?;
        writer.flush()?;

        // Send Done signals to both GPUs
        eprintln!("  tar: done");
        read_tx0
            .send(PipelineMsg::Done)
            .map_err(|_| anyhow::anyhow!("GPU 0 channel closed"))?;
        read_tx1
            .send(PipelineMsg::Done)
            .map_err(|_| anyhow::anyhow!("GPU 1 channel closed"))?;
        Ok(())
    });

    // GPU 0 thread: Compress even chunks (0, 2, 4, ...)
    let bytes_compressed_gpu0 = bytes_compressed.clone();
    let compress_tx0 = compress_tx.clone();
    let gpu0_thread = thread::spawn(move || -> Result<()> {
        eprintln!("  gpu {}: ready", gpu0);
        loop {
            match read_rx0.recv() {
                Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                    eprintln!(
                        "  gpu {}: chunk #{} ({:.2} MB)",
                        gpu0,
                        chunk_index,
                        data.len() as f64 / 1_000_000.0
                    );

                    let (compressed_chunks, sizes) =
                        compress_chunk_zstd(&data, gpu0).map_err(|e| {
                            anyhow::anyhow!(
                                "GPU {} compression failed for chunk {}: {}",
                                gpu0,
                                chunk_index,
                                e
                            )
                        })?;

                    let compressed_size: u64 =
                        compressed_chunks.iter().map(|c| c.len() as u64).sum();
                    eprintln!(
                        "  gpu {}: chunk #{} ({:.2} MB -> {:.2} MB, {:.1}%)",
                        gpu0,
                        chunk_index,
                        data.len() as f64 / 1_000_000.0,
                        compressed_size as f64 / 1_000_000.0,
                        (compressed_size as f64 / data.len() as f64) * 100.0
                    );

                    bytes_compressed_gpu0.fetch_add(compressed_size, Ordering::Relaxed);

                    compress_tx0
                        .send(CompressedChunk {
                            chunks: compressed_chunks,
                            chunk_index,
                            compressed_sizes: sizes.iter().map(|&s| s as u64).collect(),
                        })
                        .map_err(|_| anyhow::anyhow!("Compressor channel closed"))?;
                }
                Ok(PipelineMsg::Done) => {
                    eprintln!("  gpu {}: done", gpu0);
                    break;
                }
                Err(_) => {
                    return Err(anyhow::anyhow!("GPU {} reader channel error", gpu0));
                }
            }
        }
        Ok(())
    });

    // GPU 1 thread: Compress odd chunks (1, 3, 5, ...)
    let bytes_compressed_gpu1 = bytes_compressed.clone();
    let compress_tx1 = compress_tx.clone();
    let gpu1_thread = thread::spawn(move || -> Result<()> {
        eprintln!("  gpu {}: ready", gpu1);
        loop {
            match read_rx1.recv() {
                Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                    eprintln!(
                        "  gpu {}: chunk #{} ({:.2} MB)",
                        gpu1,
                        chunk_index,
                        data.len() as f64 / 1_000_000.0
                    );

                    let (compressed_chunks, sizes) =
                        compress_chunk_zstd(&data, gpu1).map_err(|e| {
                            anyhow::anyhow!(
                                "GPU {} compression failed for chunk {}: {}",
                                gpu1,
                                chunk_index,
                                e
                            )
                        })?;

                    let compressed_size: u64 =
                        compressed_chunks.iter().map(|c| c.len() as u64).sum();
                    eprintln!(
                        "  gpu {}: chunk #{} ({:.2} MB -> {:.2} MB, {:.1}%)",
                        gpu1,
                        chunk_index,
                        data.len() as f64 / 1_000_000.0,
                        compressed_size as f64 / 1_000_000.0,
                        (compressed_size as f64 / data.len() as f64) * 100.0
                    );

                    bytes_compressed_gpu1.fetch_add(compressed_size, Ordering::Relaxed);

                    compress_tx1
                        .send(CompressedChunk {
                            chunks: compressed_chunks,
                            chunk_index,
                            compressed_sizes: sizes.iter().map(|&s| s as u64).collect(),
                        })
                        .map_err(|_| anyhow::anyhow!("Compressor channel closed"))?;
                }
                Ok(PipelineMsg::Done) => {
                    eprintln!("  gpu {}: done", gpu1);
                    break;
                }
                Err(_) => {
                    return Err(anyhow::anyhow!("GPU {} reader channel error", gpu1));
                }
            }
        }
        Ok(())
    });

    // Drop the original compress_tx so writer knows when both GPUs are done
    drop(compress_tx);

    // Writer thread: Collect from both GPUs and write in order
    let output_path_clone = output_path.to_path_buf();
    let bytes_written_writer = bytes_written.clone();
    let writer_thread = thread::spawn(move || -> Result<(Vec<Vec<u64>>, u64)> {
        eprintln!("  write: creating file");
        let file = File::create(&output_path_clone)
            .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;
        // 8MB write buffer for smooth disk I/O
        let mut output_file = BufWriter::with_capacity(8 * 1024 * 1024, file);

        // Write header
        eprintln!("  write: header");
        output_file.write_all(b"NVZS")?;
        output_file.write_all(&tar_size.to_le_bytes())?;
        output_file.write_all(&(ZSTD_CHUNK_SIZE as u64).to_le_bytes())?;

        let total_chunks = (tar_size as usize).div_ceil(ZSTD_CHUNK_SIZE);
        output_file.write_all(&(total_chunks as u64).to_le_bytes())?;

        let sizes_offset = output_file.stream_position()?;
        let sizes_placeholder = vec![0u8; total_chunks * 8];
        output_file.write_all(&sizes_placeholder)?;
        eprintln!("  write: waiting");

        // Use BTreeMap to order chunks from both GPUs
        let mut pending_chunks: BTreeMap<usize, CompressedChunk> = BTreeMap::new();
        let mut next_chunk_to_write = 0;
        let mut all_batch_sizes = Vec::new();
        let mut total_compressed = 0u64;

        // Collect all chunks from both GPUs
        while let Ok(chunk) = compress_rx.recv() {
            pending_chunks.insert(chunk.chunk_index, chunk);

            // Write all sequential chunks that are ready
            while let Some(chunk) = pending_chunks.remove(&next_chunk_to_write) {
                let total_batch_size: u64 = chunk.chunks.iter().map(|c| c.len() as u64).sum();
                eprintln!(
                    "  write: chunk #{} ({:.2} MB, {} parts)",
                    chunk.chunk_index,
                    total_batch_size as f64 / 1_000_000.0,
                    chunk.chunks.len()
                );

                // Write each compressed chunk individually
                for individual_chunk in &chunk.chunks {
                    output_file.write_all(individual_chunk).map_err(|e| {
                        anyhow::anyhow!("Failed to write chunk {}: {}", chunk.chunk_index, e)
                    })?;
                }

                total_compressed += total_batch_size;
                bytes_written_writer.fetch_add(total_batch_size, Ordering::Relaxed);
                all_batch_sizes.push(chunk.compressed_sizes);
                next_chunk_to_write += 1;
            }
        }

        // Write any remaining chunks in order
        for (_, chunk) in pending_chunks {
            let total_batch_size: u64 = chunk.chunks.iter().map(|c| c.len() as u64).sum();
            for individual_chunk in &chunk.chunks {
                output_file.write_all(individual_chunk)?;
            }
            total_compressed += total_batch_size;
            bytes_written_writer.fetch_add(total_batch_size, Ordering::Relaxed);
            all_batch_sizes.push(chunk.compressed_sizes);
        }

        eprintln!("  write: flushing");
        output_file
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush before seek: {}", e))?;

        // Write sizes
        eprintln!("  write: metadata");
        output_file
            .seek(SeekFrom::Start(sizes_offset))
            .map_err(|e| anyhow::anyhow!("Failed to seek: {}", e))?;
        for batch_sizes in &all_batch_sizes {
            for size in batch_sizes {
                output_file.write_all(&size.to_le_bytes())?;
            }
        }

        output_file
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to final flush: {}", e))?;

        eprintln!("  write: done ({:.2} GB)", total_compressed as f64 / 1e9);

        Ok((all_batch_sizes, total_compressed))
    });

    // Wait for threads
    reader_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Reader thread panicked"))??;

    gpu0_thread
        .join()
        .map_err(|_| anyhow::anyhow!("GPU 0 thread panicked"))??;

    gpu1_thread
        .join()
        .map_err(|_| anyhow::anyhow!("GPU 1 thread panicked"))??;

    let (all_batch_sizes, total_compressed) = writer_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

    let ratio = (total_compressed as f64 / tar_size as f64) * 100.0;
    eprintln!("");
    eprintln!("  original:   {:.2} GB", tar_size as f64 / 1e9);
    eprintln!("  compressed: {:.2} GB ({:.2}%)", total_compressed as f64 / 1e9, ratio);
    eprintln!("  chunks:     {}", all_batch_sizes.len());

    Ok(())
}
