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

use crate::compress_zstd::compress_chunk_zstd;
use crate::compress_zstd_custom::compress_chunk_zstd_custom;
use crate::constants::{CUSTOM_ZSTD_CHUNK_SIZE, ZSTD_CHUNK_SIZE};
use crate::format::{CompressedChunk, PipelineMsg};
use crate::tui::TuiState;

/// Single GPU streaming pipeline: read -> compress -> write
pub(crate) fn compress_file_streaming_zstd(
    input_path: &Path,
    output_path: &Path,
    device_id: i32,
    _quiet: bool,
    level: u32,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len();

    let mode_label = if level > 0 {
        format!("custom zstd level {} (single gpu)", level)
    } else {
        "zstd compression (single gpu)".to_string()
    };

    const PIPELINE_QUEUE_SIZE: usize = 2;

    // Channels: Read -> GPU -> Writer
    let (read_tx, read_rx) = bounded::<PipelineMsg>(PIPELINE_QUEUE_SIZE);
    let (compress_tx, compress_rx) = bounded::<CompressedChunk>(PIPELINE_QUEUE_SIZE * 2);

    // Atomic counters
    let bytes_read = Arc::new(AtomicU64::new(0));
    let bytes_compressed = Arc::new(AtomicU64::new(0));
    let bytes_written = Arc::new(AtomicU64::new(0));
    let chunk_gpu0 = Arc::new(AtomicU64::new(0));

    // TUI state
    let mut tui = TuiState::new(
        file_size,
        bytes_read.clone(),
        bytes_compressed.clone(),
        bytes_written.clone(),
        chunk_gpu0.clone(),
        None,
        vec![device_id],
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
                None,
                vec![device_id],
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

    // Read thread: Read raw file in streaming chunks
    let input_path_clone = input_path.to_path_buf();
    let bytes_read_reader = bytes_read.clone();

    let reader_thread = thread::spawn(move || -> Result<()> {
        let mut file = BufReader::with_capacity(16 * 1024 * 1024, File::open(&input_path_clone)?);
        let mut chunk_index = 0usize;
        let mut remaining = file_size;

        while remaining > 0 {
            let to_read = std::cmp::min(ZSTD_CHUNK_SIZE as u64, remaining) as usize;
            let mut buf = vec![0u8; to_read];
            file.read_exact(&mut buf)?;
            remaining -= to_read as u64;

            bytes_read_reader.fetch_add(to_read as u64, Ordering::Relaxed);

            read_tx
                .send(PipelineMsg::Chunk {
                    chunk_index,
                    data: buf,
                })
                .map_err(|_| anyhow::anyhow!("GPU channel closed"))?;

            chunk_index += 1;
        }

        read_tx
            .send(PipelineMsg::Done)
            .map_err(|_| anyhow::anyhow!("GPU channel closed"))?;
        Ok(())
    });

    // GPU thread: Compress chunks with Zstd
    let bytes_compressed_gpu = bytes_compressed.clone();
    let chunk_gpu0_gpu = chunk_gpu0.clone();
    let gpu_thread = thread::spawn(move || -> Result<()> {
        loop {
            match read_rx.recv() {
                Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                    chunk_gpu0_gpu.store(chunk_index as u64, Ordering::Relaxed);

                    let (compressed_chunks, sizes) = if level > 0 {
                        compress_chunk_zstd_custom(&data, device_id, level)
                    } else {
                        compress_chunk_zstd(&data, device_id)
                    }
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "GPU {} compression failed for chunk {}: {}",
                            device_id,
                            chunk_index,
                            e
                        )
                    })?;

                    let compressed_size: u64 =
                        compressed_chunks.iter().map(|c| c.len() as u64).sum();

                    bytes_compressed_gpu.fetch_add(compressed_size, Ordering::Relaxed);

                    compress_tx
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
                    return Err(anyhow::anyhow!("GPU {} reader channel error", device_id));
                }
            }
        }
        Ok(())
    });

    // compress_tx is owned by gpu_thread closure — writer sees EOF when GPU thread drops it

    // Writer thread: Collect and write in order (BTreeMap reordering, same as dual GPU)
    let output_path_clone = output_path.to_path_buf();
    let bytes_written_writer = bytes_written.clone();
    let writer_thread = thread::spawn(move || -> Result<(Vec<Vec<u64>>, u64)> {
        let file = File::create(&output_path_clone)
            .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;
        let mut output_file = BufWriter::with_capacity(16 * 1024 * 1024, file);

        // Write header
        let chunk_size = if level > 0 { CUSTOM_ZSTD_CHUNK_SIZE } else { ZSTD_CHUNK_SIZE };
        let total_chunks = (file_size as usize).div_ceil(chunk_size);

        output_file.write_all(b"NVZS")?;
        output_file.write_all(&file_size.to_le_bytes())?;
        output_file.write_all(&(chunk_size as u64).to_le_bytes())?;
        output_file.write_all(&(total_chunks as u64).to_le_bytes())?;

        let sizes_offset = output_file.stream_position()?;
        let sizes_placeholder = vec![0u8; total_chunks * 8];
        output_file.write_all(&sizes_placeholder)?;

        // BTreeMap reordering (same pattern as dual GPU writer)
        let mut pending_chunks: BTreeMap<usize, CompressedChunk> = BTreeMap::new();
        let mut next_chunk_to_write = 0;
        let mut all_batch_sizes = Vec::new();
        let mut total_compressed = 0u64;

        while let Ok(chunk) = compress_rx.recv() {
            pending_chunks.insert(chunk.chunk_index, chunk);

            while let Some(chunk) = pending_chunks.remove(&next_chunk_to_write) {
                let total_batch_size: u64 = chunk.chunks.iter().map(|c| c.len() as u64).sum();

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

    // Wait for threads (same order as dual GPU)
    reader_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Reader thread panicked"))??;

    gpu_thread
        .join()
        .map_err(|_| anyhow::anyhow!("GPU thread panicked"))??;

    let (_all_batch_sizes, total_compressed) = writer_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

    // Stop stats thread
    stats_done.store(1, Ordering::Relaxed);

    // Print summary
    tui.print_summary(total_compressed, false);

    Ok(())
}
