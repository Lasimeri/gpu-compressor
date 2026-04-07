#![allow(dead_code)]
use anyhow::Result;
use crossbeam_channel::bounded;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::compress_lzma2::compress_chunk_lzma2;
use crate::constants::LZMA2_CHUNK_SIZE;
use crate::format::{CompressedChunk, PipelineMsg};
use crate::tui::TuiState;

/// Dual GPU async pipeline: Process alternating chunks on two GPUs simultaneously (LZMA2)
pub(crate) fn compress_file_streaming_lzma2_dual_gpu(
    input_path: &Path,
    output_path: &Path,
    gpu0: i32,
    gpu1: i32,
    _quiet: bool,
    level: u32,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len();

    let mode_label = "lzma2 compression (dual gpu)".to_string();

    // 8MB buffering: 2 chunks per GPU for smooth I/O
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
    let chunk_gpu0 = Arc::new(AtomicU64::new(0));
    let chunk_gpu1 = Arc::new(AtomicU64::new(0));

    // TUI state
    let mut tui = TuiState::new(
        file_size,
        bytes_read.clone(),
        bytes_compressed.clone(),
        bytes_written.clone(),
        chunk_gpu0.clone(),
        Some(chunk_gpu1.clone()),
        vec![gpu0, gpu1],
        mode_label.clone(),
        input_path,
        output_path,
        _quiet,
    );

    // Draw initial TUI
    if !_quiet {
        tui.draw();
    }

    // Stats monitor thread
    let stats_done = Arc::new(AtomicU64::new(0)); // 1 = stop
    {
        let bytes_read_s = bytes_read.clone();
        let bytes_compressed_s = bytes_compressed.clone();
        let bytes_written_s = bytes_written.clone();
        let chunk_gpu0_s = chunk_gpu0.clone();
        let chunk_gpu1_s = chunk_gpu1.clone();
        let stats_done_s = stats_done.clone();
        let quiet = _quiet;
        let input_path_s = input_path.to_path_buf();
        let output_path_s = output_path.to_path_buf();
        let mode_label_s = mode_label.clone();
        thread::spawn(move || {
            let mut tui_state = TuiState::new(
                file_size,
                bytes_read_s,
                bytes_compressed_s,
                bytes_written_s,
                chunk_gpu0_s,
                Some(chunk_gpu1_s),
                vec![gpu0, gpu1],
                mode_label_s,
                input_path_s,
                output_path_s,
                quiet,
            );
            loop {
                thread::sleep(Duration::from_millis(200));
                if stats_done_s.load(Ordering::Relaxed) != 0 {
                    break;
                }
                tui_state.draw();
            }
        });
    }

    // Read thread: Read raw file in 8MB chunks, distribute alternately to GPU 0 and GPU 1
    let input_path_clone = input_path.to_path_buf();
    let bytes_read_reader = bytes_read.clone();

    let reader_thread = thread::spawn(move || -> Result<()> {
        let mut file = BufReader::with_capacity(16 * 1024 * 1024, File::open(&input_path_clone)?);
        let mut chunk_index = 0usize;
        let mut remaining = file_size;

        while remaining > 0 {
            let to_read = std::cmp::min(LZMA2_CHUNK_SIZE as u64, remaining) as usize;
            let mut buf = vec![0u8; to_read];
            file.read_exact(&mut buf)?;
            remaining -= to_read as u64;

            bytes_read_reader.fetch_add(to_read as u64, Ordering::Relaxed);

            // Alternate between GPU 0 (even) and GPU 1 (odd)
            let sender = if chunk_index % 2 == 0 {
                &read_tx0
            } else {
                &read_tx1
            };

            sender
                .send(PipelineMsg::Chunk {
                    chunk_index,
                    data: buf,
                })
                .map_err(|_| anyhow::anyhow!("GPU channel closed"))?;

            chunk_index += 1;
        }

        // Send Done signals to both GPUs
        read_tx0
            .send(PipelineMsg::Done)
            .map_err(|_| anyhow::anyhow!("GPU 0 channel closed"))?;
        read_tx1
            .send(PipelineMsg::Done)
            .map_err(|_| anyhow::anyhow!("GPU 1 channel closed"))?;
        Ok(())
    });

    let preset = level.min(9);

    // GPU 0 thread: Compress even chunks (0, 2, 4, ...)
    let bytes_compressed_gpu0 = bytes_compressed.clone();
    let compress_tx0 = compress_tx.clone();
    let chunk_gpu0_gpu = chunk_gpu0.clone();
    let gpu0_thread = thread::spawn(move || -> Result<()> {
        loop {
            match read_rx0.recv() {
                Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                    chunk_gpu0_gpu.store(chunk_index as u64, Ordering::Relaxed);

                    let (compressed_chunks, sizes) =
                        compress_chunk_lzma2(&data, 16 * 1024 * 1024, preset).map_err(|e| {
                            anyhow::anyhow!(
                                "GPU {} compression failed for chunk {}: {}",
                                gpu0,
                                chunk_index,
                                e
                            )
                        })?;

                    let compressed_size: u64 =
                        compressed_chunks.iter().map(|c| c.len() as u64).sum();

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
    let chunk_gpu1_gpu = chunk_gpu1.clone();
    let gpu1_thread = thread::spawn(move || -> Result<()> {
        loop {
            match read_rx1.recv() {
                Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                    chunk_gpu1_gpu.store(chunk_index as u64, Ordering::Relaxed);

                    let (compressed_chunks, sizes) =
                        compress_chunk_lzma2(&data, 16 * 1024 * 1024, preset).map_err(|e| {
                            anyhow::anyhow!(
                                "GPU {} compression failed for chunk {}: {}",
                                gpu1,
                                chunk_index,
                                e
                            )
                        })?;

                    let compressed_size: u64 =
                        compressed_chunks.iter().map(|c| c.len() as u64).sum();

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
        let file = File::create(&output_path_clone)
            .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;
        let mut output_file = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write header using raw file size
        let total_chunks = (file_size as usize).div_ceil(LZMA2_CHUNK_SIZE);

        output_file.write_all(b"NVLZ")?;
        output_file.write_all(&file_size.to_le_bytes())?;
        output_file.write_all(&(LZMA2_CHUNK_SIZE as u64).to_le_bytes())?;
        output_file.write_all(&(total_chunks as u64).to_le_bytes())?;

        let sizes_offset = output_file.stream_position()?;
        let sizes_placeholder = vec![0u8; total_chunks * 8];
        output_file.write_all(&sizes_placeholder)?;

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

        output_file
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush before seek: {}", e))?;

        // Write sizes
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

    let (_all_batch_sizes, total_compressed) = writer_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

    // Stop stats thread
    stats_done.store(1, Ordering::Relaxed);

    // Print summary
    tui.print_summary(total_compressed, true);

    Ok(())
}
