use anyhow::Result;
use crossbeam_channel::bounded;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::blake3::blake3_hash_file;
use crate::compress_zstd::compress_chunk_zstd;
use crate::constants::{PIPELINE_QUEUE_SIZE, ZSTD_CHUNK_SIZE};
use crate::format::{CompressedChunk, PipelineMsg};

/// Stream-compress a file with parallel read/compress/write pipeline using Zstd
pub(crate) fn compress_file_streaming_zstd(
    input_path: &Path,
    output_path: &Path,
    device_id: i32,
    quiet: bool,
) -> Result<()> {
    compress_file_streaming_zstd_with_hash(input_path, output_path, None, device_id, quiet)
}

pub(crate) fn compress_file_streaming_zstd_with_hash(
    input_path: &Path,
    output_path: &Path,
    precomputed_hash: Option<&str>,
    device_id: i32,
    quiet: bool,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len();

    eprintln!("gpu-compressor: zstd compression (single gpu)");
    eprintln!("  input:  {}", input_path.display());
    eprintln!("  output: {}", output_path.display());
    eprintln!("  size:   {:.2} GB", file_size as f64 / 1_000_000_000.0);

    // Use pre-computed hash or compute it now
    let input_hash = if let Some(hash) = precomputed_hash {
        eprintln!("  hash: (cached)");
        hash.to_string()
    } else {
        eprintln!("  hashing input...");
        // Always use GPU 0 for hashing
        blake3_hash_file(input_path, 0)?
    };
    eprintln!("  hash: {}", input_hash);

    // Prepare .blake3 content in memory (never write to disk)
    let blake3_content = format!("{}\n", input_hash);
    let blake3_filename = format!(
        "{}.blake3",
        input_path.file_name().unwrap().to_str().unwrap()
    );

    // Calculate tar size ahead of time
    // Tar format: each file has 512-byte header + data rounded to 512 bytes + 1024 bytes end marker
    let original_file_size = file_size;
    let blake3_size = blake3_content.len() as u64;

    let tar_size = 512 + original_file_size.div_ceil(512) * 512 +  // Original file
        512 + blake3_size.div_ceil(512) * 512 +           // .blake3 file
        1024; // End marker

    eprintln!(
        "  pipeline: read {}MB -> gpu zstd -> write",
        ZSTD_CHUNK_SIZE / (1024 * 1024)
    );

    // Channels for pipeline coordination
    let (read_tx, read_rx) = bounded::<PipelineMsg>(PIPELINE_QUEUE_SIZE);
    let (compress_tx, compress_rx) = bounded::<CompressedChunk>(PIPELINE_QUEUE_SIZE);

    // Atomic counters for comprehensive pipeline stats (decoupled from display)
    let bytes_read = Arc::new(AtomicU64::new(0)); // Disk read progress
    let bytes_compressed = Arc::new(AtomicU64::new(0)); // GPU compression output
    let bytes_written = Arc::new(AtomicU64::new(0)); // Disk write progress

    // Multi-progress for clean, compact display
    let mp = MultiProgress::new();

    // Main progress bar
    let pb = if !quiet {
        let pb = mp.add(ProgressBar::new(tar_size));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%) | {msg}")
                .unwrap()
                .progress_chars("##-. "),
        );
        Arc::new(pb)
    } else {
        Arc::new(ProgressBar::hidden())
    };

    // Status spinners for each thread
    let tar_spinner = if !quiet {
        let s = mp.add(ProgressBar::new_spinner());
        s.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        Arc::new(s)
    } else {
        Arc::new(ProgressBar::hidden())
    };

    let compress_spinner = if !quiet {
        let s = mp.add(ProgressBar::new_spinner());
        s.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} {msg}")
                .unwrap(),
        );
        Arc::new(s)
    } else {
        Arc::new(ProgressBar::hidden())
    };

    let writer_spinner = if !quiet {
        let s = mp.add(ProgressBar::new_spinner());
        s.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.yellow} {msg}")
                .unwrap(),
        );
        Arc::new(s)
    } else {
        Arc::new(ProgressBar::hidden())
    };

    // Dedicated stats monitoring thread (completely decoupled, runs at fixed 200ms intervals)
    let pb_stats = pb.clone();
    let bytes_read_stats = bytes_read.clone();
    let bytes_compressed_stats = bytes_compressed.clone();
    let bytes_written_stats = bytes_written.clone();
    let tar_spinner_tick = tar_spinner.clone();
    let compress_spinner_tick = compress_spinner.clone();
    let writer_spinner_tick = writer_spinner.clone();
    let stats_thread = thread::spawn(move || {
        let mut last_read = 0u64;
        let mut last_compressed = 0u64;
        let mut last_written = 0u64;
        let mut last_time = Instant::now();

        loop {
            thread::sleep(Duration::from_millis(200)); // Update every 200ms

            // Tick spinners to keep them animated
            tar_spinner_tick.tick();
            compress_spinner_tick.tick();
            writer_spinner_tick.tick();

            let now = Instant::now();
            let delta_time = now.duration_since(last_time).as_secs_f64();

            let curr_read = bytes_read_stats.load(Ordering::Relaxed);
            let curr_compressed = bytes_compressed_stats.load(Ordering::Relaxed);
            let curr_written = bytes_written_stats.load(Ordering::Relaxed);

            // Calculate instantaneous speeds (GB/s)
            let read_speed = if delta_time > 0.0 {
                (curr_read - last_read) as f64 / delta_time / 1_000_000_000.0
            } else {
                0.0
            };

            let compress_speed = if delta_time > 0.0 {
                (curr_compressed - last_compressed) as f64 / delta_time / 1_000_000_000.0
            } else {
                0.0
            };

            let write_speed = if delta_time > 0.0 {
                (curr_written - last_written) as f64 / delta_time / 1_000_000_000.0
            } else {
                0.0
            };

            // Compression ratio
            let ratio = if curr_read > 0 {
                (curr_compressed as f64 / curr_read as f64) * 100.0
            } else {
                0.0
            };

            // Update progress bar position based on bytes read
            pb_stats.set_position(curr_read);

            // Format: Read | GPU | Write | Ratio
            let msg = format!(
                "Read: {:.2} GB/s | GPU: {:.2} GB/s | Write: {:.2} GB/s | Ratio: {:.1}%",
                read_speed, compress_speed, write_speed, ratio
            );
            pb_stats.set_message(msg);

            // Update for next iteration
            last_read = curr_read;
            last_compressed = curr_compressed;
            last_written = curr_written;
            last_time = now;

            // Exit when progress bar is finished
            if pb_stats.is_finished() {
                break;
            }
        }
    });

    // Tar generator thread: Create tar in memory and stream to compression
    let input_path_clone = input_path.to_path_buf();
    let bytes_read_reader = bytes_read.clone();
    let blake3_content_clone = blake3_content.clone();
    let blake3_filename_clone = blake3_filename.clone();
    let tar_spinner_clone = tar_spinner.clone();
    let tar_generator_thread = thread::spawn(move || -> Result<()> {
        use crossbeam_channel::Sender;
        use std::io::Write;

        tar_spinner_clone.set_message("tar: starting");

        // Custom writer that buffers and sends batches
        struct ChannelWriter {
            buffer: Vec<u8>,
            batch_size: usize,
            batch_index: usize,
            sender: Sender<PipelineMsg>,
            bytes_sent: Arc<AtomicU64>,
            spinner: Arc<ProgressBar>,
        }

        impl Write for ChannelWriter {
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

        impl ChannelWriter {
            fn flush_batch(&mut self) -> std::io::Result<()> {
                let batch_data =
                    std::mem::replace(&mut self.buffer, Vec::with_capacity(self.batch_size));
                let batch_len = batch_data.len();

                self.sender
                    .send(PipelineMsg::Chunk {
                        chunk_index: self.batch_index,
                        data: batch_data,
                    })
                    .map_err(std::io::Error::other)?;

                self.bytes_sent
                    .fetch_add(batch_len as u64, Ordering::Relaxed);
                self.spinner.set_message(format!(
                    "tar: batch #{} ({:.2} MB)",
                    self.batch_index,
                    batch_len as f64 / 1_000_000.0
                ));
                self.batch_index += 1;
                Ok(())
            }
        }

        // Use 4MB chunks for tar streaming to match Zstd compression chunk size
        let tar_chunk_size = ZSTD_CHUNK_SIZE; // 4MB for streaming (keeps GPU fed)

        // Create channel writer with 4MB chunks
        let writer = ChannelWriter {
            buffer: Vec::with_capacity(tar_chunk_size),
            batch_size: tar_chunk_size,
            batch_index: 0,
            sender: read_tx.clone(),
            bytes_sent: bytes_read_reader.clone(),
            spinner: tar_spinner_clone.clone(),
        };

        // Create tar builder with channel writer
        let mut tar_builder = tar::Builder::new(writer);

        // Add original file to tar with large buffer for maximum SSD throughput
        tar_spinner_clone.set_message("tar: reading file");
        let original_file = File::open(&input_path_clone)?;
        let file_metadata = original_file.metadata()?;
        // 8MB read buffer for smooth disk I/O (matches dual GPU pipeline)
        let mut buffered_file = std::io::BufReader::with_capacity(8 * 1024 * 1024, original_file);

        let mut header = tar::Header::new_gnu();
        header.set_metadata(&file_metadata);
        header.set_cksum();
        tar_builder.append_data(
            &mut header,
            input_path_clone.file_name().unwrap(),
            &mut buffered_file,
        )?;

        // Add .blake3 content to tar (in memory)
        tar_spinner_clone.set_message("tar: adding hash");
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
        // NOTE: tar::Builder::finish() does NOT call flush() on the inner writer,
        // so we must explicitly retrieve and flush to send the last partial chunk
        tar_spinner_clone.set_message("tar: finalizing");
        let mut writer = tar_builder.into_inner()?;
        writer.flush()?;

        // Send Done signal
        tar_spinner_clone.finish_with_message("tar: done");
        read_tx
            .send(PipelineMsg::Done)
            .map_err(|_| anyhow::anyhow!("Tar generator channel closed"))?;
        Ok(())
    });

    // Compressor thread: GPU compression
    let bytes_compressed_gpu = bytes_compressed.clone();
    let compress_spinner_clone = compress_spinner.clone();
    let compress_thread = thread::spawn(move || -> Result<()> {
        compress_spinner_clone.set_message("gpu: ready");
        loop {
            match read_rx.recv() {
                Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                    compress_spinner_clone.set_message(format!(
                        "gpu: chunk #{} ({:.2} MB)",
                        chunk_index,
                        data.len() as f64 / 1_000_000.0
                    ));

                    // Streaming compression: Process single 4MB chunk immediately (no batching)
                    let (compressed_chunks, sizes) = compress_chunk_zstd(&data, device_id)
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "GPU compression failed for chunk {}: {}",
                                chunk_index,
                                e
                            )
                        })?;

                    // Calculate total compressed size for stats (but keep chunks separate!)
                    let compressed_size: u64 =
                        compressed_chunks.iter().map(|c| c.len() as u64).sum();
                    compress_spinner_clone.set_message(format!(
                        "gpu: chunk #{} ({:.2} MB -> {:.2} MB, {:.1}%)",
                        chunk_index,
                        data.len() as f64 / 1_000_000.0,
                        compressed_size as f64 / 1_000_000.0,
                        (compressed_size as f64 / data.len() as f64) * 100.0
                    ));

                    // Update compressed bytes counter
                    bytes_compressed_gpu.fetch_add(compressed_size, Ordering::Relaxed);

                    compress_tx
                        .send(CompressedChunk {
                            chunks: compressed_chunks, // Keep individual chunks separate!
                            chunk_index,
                            compressed_sizes: sizes.iter().map(|&s| s as u64).collect(),
                        })
                        .map_err(|_| anyhow::anyhow!("Compressor channel closed"))?;
                }
                Ok(PipelineMsg::Done) => {
                    compress_spinner_clone.finish_with_message("gpu: done");
                    drop(compress_tx);
                    break;
                }
                Err(_) => {
                    return Err(anyhow::anyhow!("Reader channel error"));
                }
            }
        }
        Ok(())
    });

    // Writer thread: Write compressed data to disk
    let output_path_clone = output_path.to_path_buf();
    let bytes_written_writer = bytes_written.clone();
    let tar_size_writer = tar_size;
    let writer_spinner_clone = writer_spinner.clone();
    let writer_thread = thread::spawn(move || -> Result<(Vec<Vec<u64>>, u64)> {
        writer_spinner_clone.set_message("write: creating file");
        let file = File::create(&output_path_clone)
            .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;
        // 8MB write buffer for smooth disk I/O (matches dual GPU pipeline)
        let mut output_file = BufWriter::with_capacity(8 * 1024 * 1024, file);

        // Write header placeholder
        writer_spinner_clone.set_message("write: header");
        output_file.write_all(b"NVZS")?;
        output_file.write_all(&tar_size_writer.to_le_bytes())?;
        output_file.write_all(&(ZSTD_CHUNK_SIZE as u64).to_le_bytes())?;

        let total_chunks = (tar_size_writer as usize).div_ceil(ZSTD_CHUNK_SIZE);
        output_file.write_all(&(total_chunks as u64).to_le_bytes())?;

        let sizes_offset = output_file.stream_position()?;
        let sizes_placeholder = vec![0u8; total_chunks * 8];
        output_file.write_all(&sizes_placeholder)?;
        writer_spinner_clone.set_message("write: waiting");

        let mut all_batch_sizes = Vec::new();
        let mut total_compressed = 0u64;
        let mut _chunk_count = 0;

        // Write compressed chunks in order
        while let Ok(chunk) = compress_rx.recv() {
            let total_batch_size: u64 = chunk.chunks.iter().map(|c| c.len() as u64).sum();
            writer_spinner_clone.set_message(format!(
                "write: chunk #{} ({:.2} MB)",
                chunk.chunk_index,
                total_batch_size as f64 / 1_000_000.0
            ));

            // Write each compressed chunk individually (important for decompression!)
            for individual_chunk in &chunk.chunks {
                output_file.write_all(individual_chunk).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to write chunk in batch {}: {}",
                        chunk.chunk_index,
                        e
                    )
                })?;
            }

            total_compressed += total_batch_size;
            _chunk_count += 1;

            // Update atomic counter for live stats
            bytes_written_writer.fetch_add(total_batch_size, Ordering::Relaxed);

            all_batch_sizes.push(chunk.compressed_sizes);
        }

        writer_spinner_clone.set_message("write: flushing");

        // Flush buffer before seeking
        output_file
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush before seek: {}", e))?;

        // Go back and write sizes
        writer_spinner_clone.set_message("write: metadata");
        output_file
            .seek(SeekFrom::Start(sizes_offset))
            .map_err(|e| anyhow::anyhow!("Failed to seek: {}", e))?;
        for batch_sizes in &all_batch_sizes {
            for size in batch_sizes {
                output_file.write_all(&size.to_le_bytes())?;
            }
        }

        // Final flush
        output_file
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to final flush: {}", e))?;

        writer_spinner_clone.finish_with_message(format!(
            "write: done ({:.2} GB)",
            total_compressed as f64 / 1_000_000_000.0
        ));

        Ok((all_batch_sizes, total_compressed))
    });

    // Wait for all threads and propagate errors
    match tar_generator_thread.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(anyhow::anyhow!("tar thread panicked")),
    }

    match compress_thread.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(anyhow::anyhow!("compressor thread panicked")),
    }

    let total_compressed = match writer_thread.join() {
        Ok(Ok((_, total))) => total,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(anyhow::anyhow!("writer thread panicked")),
    };

    pb.finish_with_message("done");

    let _ = stats_thread.join();

    let ratio = (total_compressed as f64 / tar_size as f64) * 100.0;
    eprintln!("");
    eprintln!("  compressed: {:.2} GB ({:.2}%)", total_compressed as f64 / 1_000_000_000.0, ratio);

    // Hash compressed output file and write hash to .blake3 file
    eprintln!("  hashing output...");
    // Always use GPU 0 for hashing
    let output_hash = blake3_hash_file(output_path, 0)?;

    // Write hash to .blake3 file
    let hash_blake3_path = format!("{}.blake3", output_path.display());
    let mut hash_file = fs::File::create(&hash_blake3_path)?;
    writeln!(hash_file, "{}", output_hash)?;

    eprintln!(
        "  -> {}",
        PathBuf::from(&hash_blake3_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
    );

    Ok(())
}
