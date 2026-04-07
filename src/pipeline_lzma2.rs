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
use crate::constants::LZMA2_CUSTOM_CHUNK_SIZE;
use crate::format::{CompressedChunk, PipelineMsg};
use crate::tui::TuiState;

/// LZMA2 streaming pipeline with multi-threaded compression.
///
/// L0: Reader → CPU compressor threads → Writer
/// L1+: Reader → GPU match finder → CPU encoder pool (all cores) → Writer
///
/// The key optimization: GPU match finding is fast (ms) while CPU range coding
/// is slow (seconds). The GPU thread runs ahead, queuing match data for a pool
/// of CPU encoder threads that work in parallel across chunks AND sub-blocks.
pub(crate) fn compress_file_streaming_lzma2(
    input_path: &Path,
    output_path: &Path,
    gpu_ids: &[i32],
    _quiet: bool,
    level: u32,
    dict_size_mb: u32,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len();
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let device_id = gpu_ids[0]; // primary GPU for TUI display
    let dict_size_bytes = (dict_size_mb as u64 * 1024 * 1024) as u32;
    // Pipeline chunk size = dict size (each chunk gets the full dictionary window)
    let pipeline_chunk_size = dict_size_mb as usize * 1024 * 1024;

    let mode_label = if level > 0 {
        format!("lzma2 L{} ({} gpu + {} cpu, {}MB dict)", level, gpu_ids.len(), num_cpus, dict_size_mb)
    } else {
        format!("lzma2 preset {} ({} cpu, {}MB dict)", level.min(9), num_cpus, dict_size_mb)
    };

    // Channels
    let (read_tx, read_rx) = bounded::<PipelineMsg>(4);
    let (compress_tx, compress_rx) = bounded::<CompressedChunk>(8);

    // Atomic counters
    let bytes_read = Arc::new(AtomicU64::new(0));
    let bytes_compressed = Arc::new(AtomicU64::new(0));
    let bytes_written = Arc::new(AtomicU64::new(0));
    let chunk_gpu0 = Arc::new(AtomicU64::new(0));

    // TUI
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

    if !_quiet {
        tui.draw();
    }

    // Stats thread
    let stats_done = Arc::new(AtomicU64::new(0));
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

    // Reader thread
    let input_path_clone = input_path.to_path_buf();
    let bytes_read_reader = bytes_read.clone();
    let reader_thread = thread::spawn(move || -> Result<()> {
        let mut file = BufReader::with_capacity(16 * 1024 * 1024, File::open(&input_path_clone)?);
        let mut chunk_index = 0usize;
        let mut remaining = file_size;

        while remaining > 0 {
            let to_read = std::cmp::min(pipeline_chunk_size as u64, remaining) as usize;
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

    // Compression thread(s)
    let bytes_compressed_enc = bytes_compressed.clone();
    let chunk_gpu0_enc = chunk_gpu0.clone();
    let preset = level.min(9);

    let encoder_thread = if level > 0 {
        // L1+: Fully async GPU→CPU pipeline
        // GPU thread unpacks each chunk into individual sub-block encoding jobs.
        // N CPU threads each grab one sub-block at a time from a shared queue.
        // A collector reassembles completed sub-blocks into chunks for the writer.
        use crate::compress_lzma2_custom::{gpu_find_matches, encode_single_sub_block};

        // Sub-block job: one 64KB encoding task
        struct SubBlockJob {
            sub_data: Vec<u8>,
            match_distances: Vec<u32>,
            match_lengths: Vec<u32>,
            match_counts: Vec<u32>,
            max_matches: usize,
            chunk_index: usize,
            sub_index: usize,
            total_subs: usize,
        }

        let (job_tx, job_rx) = bounded::<SubBlockJob>(num_cpus * 4);
        let job_rx = std::sync::Arc::new(job_rx);

        // Collector: reassembles sub-blocks into chunks
        // Key: (chunk_index, sub_index) → encoded block
        let (result_tx, result_rx) = bounded::<(usize, usize, usize, Vec<u8>)>(num_cpus * 4);

        // GPU thread(s): one per GPU, all sharing read_rx + job_tx
        // With multiple GPUs, chunks are distributed round-robin automatically
        // by crossbeam's MPMC channel.
        let _num_gpus = gpu_ids.len();
        let mut gpu_handles = Vec::new();
        for &gid in gpu_ids {
            let rx = read_rx.clone();
            let jtx = job_tx.clone();
            let chunk_c_gpu = chunk_gpu0_enc.clone();
            gpu_handles.push(thread::spawn(move || -> Result<()> {
                loop {
                    match rx.recv() {
                        Ok(PipelineMsg::Chunk { data, chunk_index }) => {
                            chunk_c_gpu.store(chunk_index as u64, Ordering::Relaxed);
                            let matches = gpu_find_matches(&data, gid)?;
                        let sub_block_size = matches.sub_block_size;
                        let max_matches = matches.max_matches;
                        let num_subs = matches.num_sub_blocks;

                        // Distribute sub-block jobs
                        for i in 0..num_subs {
                            let sub_start = i * sub_block_size;
                            let sub_end = (sub_start + sub_block_size).min(data.len());
                            let mc_start = i * sub_block_size;
                            let mc_end = mc_start + (sub_end - sub_start);
                            let md_start = i * sub_block_size * max_matches;
                            let md_end = md_start + (sub_end - sub_start) * max_matches;

                            jtx.send(SubBlockJob {
                                sub_data: data[sub_start..sub_end].to_vec(),
                                match_distances: matches.match_distances[md_start..md_end].to_vec(),
                                match_lengths: matches.match_lengths[md_start..md_end].to_vec(),
                                match_counts: matches.match_counts[mc_start..mc_end].to_vec(),
                                max_matches,
                                chunk_index,
                                sub_index: i,
                                total_subs: num_subs,
                            }).map_err(|_| anyhow::anyhow!("Job queue closed"))?;
                        }
                    }
                    Ok(PipelineMsg::Done) | Err(_) => break,
                }
            }
            Ok(())
        }));
        }
        drop(read_rx); // all GPU threads have clones
        drop(job_tx);  // GPU threads have clones

        // N CPU encoder threads: each grabs one sub-block at a time
        let num_encoders = num_cpus.min(8);
        let mut cpu_handles = Vec::new();
        for _ in 0..num_encoders {
            let jrx = std::sync::Arc::clone(&job_rx);
            let rtx = result_tx.clone();
            cpu_handles.push(thread::spawn(move || -> Result<()> {
                while let Ok(job) = jrx.recv() {
                    let block = encode_single_sub_block(
                        &job.sub_data,
                        &job.match_distances,
                        &job.match_lengths,
                        &job.match_counts,
                        job.max_matches,
                    );
                    rtx.send((job.chunk_index, job.sub_index, job.total_subs, block))
                        .map_err(|_| anyhow::anyhow!("Result channel closed"))?;
                }
                Ok(())
            }));
        }
        drop(result_tx);

        // Collector thread: reassembles sub-blocks into CompressedChunks
        let bytes_c = bytes_compressed_enc.clone();
        let collector_handle = thread::spawn(move || -> Result<()> {
            use std::collections::HashMap;
            let mut pending: HashMap<usize, Vec<(usize, Vec<u8>)>> = HashMap::new();
            let mut chunk_sizes: HashMap<usize, usize> = HashMap::new();

            while let Ok((chunk_idx, sub_idx, total_subs, block)) = result_rx.recv() {
                chunk_sizes.insert(chunk_idx, total_subs);
                pending.entry(chunk_idx).or_default().push((sub_idx, block));

                // Check if this chunk is complete
                if let Some(subs) = pending.get(&chunk_idx) {
                    if subs.len() == total_subs {
                        let mut subs = pending.remove(&chunk_idx).unwrap();
                        subs.sort_by_key(|(idx, _)| *idx);

                        let chunks: Vec<Vec<u8>> = subs.into_iter().map(|(_, b)| b).collect();
                        let sizes: Vec<u64> = chunks.iter().map(|c| c.len() as u64).collect();
                        let compressed_size: u64 = sizes.iter().sum();
                        bytes_c.fetch_add(compressed_size, Ordering::Relaxed);

                        compress_tx.send(CompressedChunk {
                            chunks,
                            chunk_index: chunk_idx,
                            compressed_sizes: sizes,
                        }).map_err(|_| anyhow::anyhow!("Compress channel closed"))?;
                    }
                }
            }
            Ok(())
        });

        thread::spawn(move || -> Result<()> {
            for h in gpu_handles {
                h.join().map_err(|_| anyhow::anyhow!("GPU thread panicked"))??;
            }
            // All GPUs done → job_tx dropped → CPU encoders drain and exit
            for h in cpu_handles {
                h.join().map_err(|_| anyhow::anyhow!("CPU encoder panicked"))??;
            }
            // CPU encoders done → result_tx dropped → collector drains and exits
            collector_handle.join().map_err(|_| anyhow::anyhow!("Collector panicked"))??;
            Ok(())
        })
    } else {
        // L0: N CPU workers with liblzma pulling from shared read channel
        let num_workers = num_cpus.min(8);
        let mut handles = Vec::new();
        for _ in 0..num_workers {
            let rx = read_rx.clone();
            let tx = compress_tx.clone();
            let bytes_c = bytes_compressed_enc.clone();
            let chunk_c = chunk_gpu0_enc.clone();
            handles.push(thread::spawn(move || -> Result<()> {
                while let Ok(msg) = rx.recv() {
                    match msg {
                        PipelineMsg::Chunk { data, chunk_index } => {
                            chunk_c.store(chunk_index as u64, Ordering::Relaxed);
                            let (compressed_chunks, sizes) =
                                compress_chunk_lzma2(&data, dict_size_bytes, preset)?;
                            let compressed_size: u64 =
                                compressed_chunks.iter().map(|c| c.len() as u64).sum();
                            bytes_c.fetch_add(compressed_size, Ordering::Relaxed);
                            tx.send(CompressedChunk {
                                chunks: compressed_chunks,
                                chunk_index,
                                compressed_sizes: sizes.iter().map(|&s| s as u64).collect(),
                            }).map_err(|_| anyhow::anyhow!("Compress channel closed"))?;
                        }
                        PipelineMsg::Done => break,
                    }
                }
                Ok(())
            }));
        }
        drop(read_rx);
        drop(compress_tx);

        thread::spawn(move || -> Result<()> {
            for h in handles {
                h.join().map_err(|_| anyhow::anyhow!("Worker panicked"))??;
            }
            Ok(())
        })
    };


    // Writer thread
    let output_path_clone = output_path.to_path_buf();
    let bytes_written_writer = bytes_written.clone();
    let writer_thread = thread::spawn(move || -> Result<(Vec<Vec<u64>>, u64)> {
        let file = File::create(&output_path_clone)
            .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;
        let mut output_file = BufWriter::with_capacity(16 * 1024 * 1024, file);

        let nvlz_chunk_size = if level > 0 { LZMA2_CUSTOM_CHUNK_SIZE } else { pipeline_chunk_size };
        let total_chunks = (file_size as usize).div_ceil(nvlz_chunk_size);

        output_file.write_all(b"NVLZ")?;
        output_file.write_all(&file_size.to_le_bytes())?;
        output_file.write_all(&(nvlz_chunk_size as u64).to_le_bytes())?;
        output_file.write_all(&(total_chunks as u64).to_le_bytes())?;

        let sizes_offset = output_file.stream_position()?;
        let sizes_placeholder = vec![0u8; total_chunks * 8];
        output_file.write_all(&sizes_placeholder)?;

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

        for (_, chunk) in pending_chunks {
            let total_batch_size: u64 = chunk.chunks.iter().map(|c| c.len() as u64).sum();
            for individual_chunk in &chunk.chunks {
                output_file.write_all(individual_chunk)?;
            }
            total_compressed += total_batch_size;
            bytes_written_writer.fetch_add(total_batch_size, Ordering::Relaxed);
            all_batch_sizes.push(chunk.compressed_sizes);
        }

        output_file.flush()?;
        output_file.seek(SeekFrom::Start(sizes_offset))?;
        for batch_sizes in &all_batch_sizes {
            for size in batch_sizes {
                output_file.write_all(&size.to_le_bytes())?;
            }
        }
        output_file.flush()?;

        Ok((all_batch_sizes, total_compressed))
    });

    // Join
    reader_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Reader thread panicked"))??;

    encoder_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Encoder thread panicked"))??;

    let (_all_batch_sizes, total_compressed) = writer_thread
        .join()
        .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

    stats_done.store(1, Ordering::Relaxed);
    tui.print_summary(total_compressed, false);

    Ok(())
}
