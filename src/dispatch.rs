use anyhow::Result;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::blake3::blake3_hash_file;
use crate::cli::Algorithm;
use crate::compress_zstd::compress_buffer;
use crate::compress_zstd_custom::compress_chunk_zstd_custom;
use crate::constants::{CUSTOM_ZSTD_CHUNK_SIZE, GDEFLATE_MAX_SIZE, MAX_FILE_CHUNK_SIZE};
use crate::cuda::detect_gpus;
use crate::pipeline::compress_file_streaming_zstd;
use crate::pipeline_dual::compress_file_streaming_dual_gpu;

/// Helper function to expand directories into file lists
pub(crate) fn expand_directory_inputs(
    inputs: &[PathBuf],
    outputs: &[PathBuf],
    extension: &str,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut expanded_inputs = Vec::new();
    let mut expanded_outputs = Vec::new();

    // If inputs contains a single directory and outputs is empty or a single directory
    if inputs.len() == 1 && inputs[0].is_dir() {
        let input_dir = &inputs[0];
        let output_dir = if outputs.is_empty() {
            input_dir.clone()
        } else {
            outputs[0].clone()
        };

        // Create output directory if it doesn't exist
        if !output_dir.exists() {
            fs::create_dir_all(&output_dir)?;
        }

        println!("  scanning: {}", input_dir.display());

        // Find all regular files recursively, preserving directory structure
        for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                // Get relative path from input_dir to preserve directory structure
                let relative_path = path.strip_prefix(input_dir).unwrap_or(path);
                let output_path =
                    output_dir.join(format!("{}{}", relative_path.display(), extension));

                // Create parent directories in output if needed
                if let Some(parent) = output_path.parent() {
                    if !parent.exists() {
                        fs::create_dir_all(parent)?;
                    }
                }

                expanded_inputs.push(path.to_path_buf());
                expanded_outputs.push(output_path);
            }
        }

        println!(
            "   Found {} files to compress (recursive)",
            expanded_inputs.len()
        );
    } else {
        // No directory expansion needed
        expanded_inputs = inputs.to_vec();
        expanded_outputs = outputs.to_vec();
    }

    Ok((expanded_inputs, expanded_outputs))
}

pub(crate) fn compress_file(
    input_path: &Path,
    output_path: &Path,
    algorithm: Algorithm,
    device_id: i32,
    level: u32,
) -> Result<()> {
    compress_file_impl(input_path, output_path, algorithm, device_id, false, level)
}

pub(crate) fn compress_large_file_in_chunks(
    input_path: &Path,
    output_path: &Path,
    algorithm: Algorithm,
    device_id: i32,
    quiet: bool,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len() as usize;

    if !quiet {
        println!(
            "  large file ({:.2} GB), chunking to 128MB",
            file_size as f64 / 1_000_000_000.0
        );
    }

    // Calculate number of chunks needed
    let num_file_chunks = file_size.div_ceil(MAX_FILE_CHUNK_SIZE);

    // Create output file with multi-chunk header
    let mut output_file = File::create(output_path)?;

    // Write multi-chunk header: NVMC (nvCOMP Multi-Chunk) + original size + num chunks
    output_file.write_all(b"NVMC")?;
    output_file.write_all(&(file_size as u64).to_le_bytes())?;
    output_file.write_all(&(num_file_chunks as u64).to_le_bytes())?;

    // Open input file once and stream chunks
    let mut input_file = File::open(input_path)?;

    // Compress each 1GB chunk by streaming
    for chunk_idx in 0..num_file_chunks {
        let chunk_start = chunk_idx * MAX_FILE_CHUNK_SIZE;
        let chunk_end = std::cmp::min(chunk_start + MAX_FILE_CHUNK_SIZE, file_size);
        let chunk_size = chunk_end - chunk_start;

        if !quiet {
            println!(
                "  Chunk {}/{}: {:.2} MB",
                chunk_idx + 1,
                num_file_chunks,
                chunk_size as f64 / 1_000_000.0
            );
        }

        // Read only this chunk into memory
        let mut chunk_data = vec![0u8; chunk_size];
        input_file.read_exact(&mut chunk_data)?;

        // Compress this chunk directly to a buffer using the same GPU logic
        let compressed_chunk = compress_buffer(&chunk_data, algorithm, device_id, quiet)?;

        // Append compressed chunk directly to output
        output_file.write_all(&compressed_chunk)?;
    }

    if !quiet {
        println!("  done. ({} chunks)", num_file_chunks);
    }

    Ok(())
}

pub(crate) fn compress_file_impl(
    input_path: &Path,
    output_path: &Path,
    algorithm: Algorithm,
    device_id: i32,
    quiet: bool,
    level: u32,
) -> Result<()> {
    let file_size = fs::metadata(input_path)?.len() as usize;

    // Auto-generate filename if output is a directory
    let final_output_path = if output_path.is_dir() {
        let input_filename = input_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid input filename"))?;
        let extension = match algorithm {
            Algorithm::Gdeflate => "nvgd",
            Algorithm::Zstd => "nvzs",
        };
        output_path.join(input_filename).with_extension(extension)
    } else {
        output_path.to_path_buf()
    };

    // Route based on algorithm
    match algorithm {
        Algorithm::Zstd => {
            // Auto-detect GPUs and choose pipeline
            let available_gpus = detect_gpus()?;

            if level > 0 {
                eprintln!("  mode: custom zstd (level {})", level);
                return compress_file_custom_zstd(
                    input_path,
                    &final_output_path,
                    device_id,
                    level,
                    quiet,
                );
            }

            if available_gpus.len() >= 2 {
                // Dual GPU async pipeline
                eprintln!(
                    "  mode: dual gpu pipeline ({} devices)",
                    available_gpus.len()
                );
                compress_file_streaming_dual_gpu(
                    input_path,
                    &final_output_path,
                    available_gpus[0],
                    available_gpus[1],
                    quiet,
                )
            } else {
                // Single GPU streaming
                eprintln!("  mode: single gpu streaming");
                compress_file_streaming_zstd(input_path, &final_output_path, device_id, quiet)
            }
        }
        Algorithm::Gdeflate => {
            // Gdeflate: check size limit and route accordingly
            if file_size > GDEFLATE_MAX_SIZE {
                return compress_large_file_in_chunks(
                    input_path,
                    &final_output_path,
                    algorithm,
                    device_id,
                    quiet,
                );
            }

            // Small gdeflate files: use buffered compression
            if !quiet {
                println!("gpu-compressor: gdeflate compression");
                println!("  input:  {}", input_path.display());
                println!("  output: {}", final_output_path.display());
            }

            let mut input_file = File::open(input_path)?;
            let mut input_data = vec![0u8; file_size];
            input_file.read_exact(&mut input_data)?;

            if !quiet {
                println!(
                    "  size:   {} bytes ({:.2} MB)",
                    file_size,
                    file_size as f64 / 1_000_000.0
                );
            }

            let compressed_data = compress_buffer(&input_data, algorithm, device_id, quiet)?;

            let mut output_file = File::create(&final_output_path)?;
            output_file.write_all(&compressed_data)?;

            if !quiet {
                println!(
                    "  result: {} bytes ({:.2} MB)",
                    compressed_data.len(),
                    compressed_data.len() as f64 / 1_000_000.0
                );
                println!(
                    "  ratio:  {:.2}%",
                    (compressed_data.len() as f64 / file_size as f64) * 100.0
                );
                println!("  done.");
            }

            Ok(())
        }
    }
}

/// Custom Zstd compression: reads file in 4MB streaming chunks, compresses via
/// custom GPU kernel, writes NVZS container with sub-chunk Zstd frames.
#[inline(never)]
fn compress_file_custom_zstd(
    input_path: &Path,
    output_path: &Path,
    device_id: i32,
    level: u32,
    quiet: bool,
) -> Result<()> {
    use std::io::BufWriter;

    let file_size = fs::metadata(input_path)?.len();

    if !quiet {
        eprintln!("gpu-compressor: custom zstd compression (level {})", level);
        eprintln!("  input:  {}", input_path.display());
        eprintln!("  output: {}", output_path.display());
        eprintln!("  size:   {:.2} GB", file_size as f64 / 1_000_000_000.0);
    }

    // Hash input file
    if !quiet {
        eprintln!("  hashing input...");
    }
    let input_hash = blake3_hash_file(input_path, 0)?;
    if !quiet {
        eprintln!("  hash: {}", input_hash);
    }

    // Prepare tar with hash (matching single-file pipeline behavior)
    let blake3_content = format!("{}\n", input_hash);
    let blake3_filename = format!(
        "{}.blake3",
        input_path.file_name().unwrap().to_str().unwrap()
    );

    // Build tar in memory, then compress in sub-chunks
    // For simplicity, build tar to a temp buffer, then compress in 4MB chunks
    if !quiet {
        eprintln!("  building tar archive...");
    }

    let mut tar_data = Vec::new();
    {
        let mut tar_builder = tar::Builder::new(&mut tar_data);

        let original_file = File::open(input_path)?;
        let file_metadata = original_file.metadata()?;
        let mut buffered_file = std::io::BufReader::with_capacity(8 * 1024 * 1024, original_file);

        let mut header = tar::Header::new_gnu();
        header.set_metadata(&file_metadata);
        header.set_cksum();
        tar_builder.append_data(
            &mut header,
            input_path.file_name().unwrap(),
            &mut buffered_file,
        )?;

        let mut blake3_header = tar::Header::new_gnu();
        blake3_header.set_size(blake3_content.len() as u64);
        blake3_header.set_mode(0o644);
        blake3_header.set_cksum();
        tar_builder.append_data(
            &mut blake3_header,
            &blake3_filename,
            blake3_content.as_bytes(),
        )?;

        tar_builder.finish()?;
    }

    let tar_total_size = tar_data.len();
    let sub_chunk_size = CUSTOM_ZSTD_CHUNK_SIZE; // 64KB sub-chunks

    if !quiet {
        eprintln!("  tar size: {:.2} MB", tar_total_size as f64 / 1_000_000.0);
        eprintln!("  compressing...");
    }

    // Compress entire tar in one GPU batch, producing individual 64KB Zstd frames
    let (compressed_sub_chunks, _sub_sizes) =
        compress_chunk_zstd_custom(&tar_data, device_id, level)?;

    let total_nvzs_chunks = compressed_sub_chunks.len();
    let total_compressed: u64 = compressed_sub_chunks.iter().map(|c| c.len() as u64).sum();

    if !quiet {
        let ratio = (total_compressed as f64 / tar_total_size as f64) * 100.0;
        eprintln!(
            "  {} sub-chunks: {:.2} MB -> {:.2} MB ({:.1}%)",
            total_nvzs_chunks,
            tar_total_size as f64 / 1_000_000.0,
            total_compressed as f64 / 1_000_000.0,
            ratio
        );
    }

    // Write NVZS container — each sub-chunk Zstd frame is a separate NVZS chunk
    let file = File::create(output_path)?;
    let mut output_file = BufWriter::with_capacity(8 * 1024 * 1024, file);

    // Header: use 64KB chunk size so decompressor knows each chunk decompresses to 64KB
    output_file.write_all(b"NVZS")?;
    output_file.write_all(&(tar_total_size as u64).to_le_bytes())?;
    output_file.write_all(&(sub_chunk_size as u64).to_le_bytes())?;
    output_file.write_all(&(total_nvzs_chunks as u64).to_le_bytes())?;

    // Chunk sizes
    for chunk in &compressed_sub_chunks {
        output_file.write_all(&(chunk.len() as u64).to_le_bytes())?;
    }

    // Compressed data
    for chunk in &compressed_sub_chunks {
        output_file.write_all(chunk)?;
    }

    output_file.flush()?;

    let ratio = (total_compressed as f64 / tar_total_size as f64) * 100.0;
    if !quiet {
        eprintln!("");
        eprintln!("  compressed: {:.2} GB ({:.2}%)", total_compressed as f64 / 1_000_000_000.0, ratio);
    }

    // Hash compressed output
    if !quiet {
        eprintln!("  hashing output...");
    }
    let output_hash = blake3_hash_file(output_path, 0)?;
    let hash_path = format!("{}.blake3", output_path.display());
    let mut hash_file = fs::File::create(&hash_path)?;
    writeln!(hash_file, "{}", output_hash)?;
    if !quiet {
        eprintln!(
            "  -> {}",
            PathBuf::from(&hash_path).file_name().unwrap().to_str().unwrap()
        );
    }

    Ok(())
}

pub(crate) fn compress_directory(
    input_dir: &Path,
    output_dir: &Path,
    algorithm: Algorithm,
    device_id: i32,
    level: u32,
) -> Result<()> {
    let algo_name = match algorithm {
        Algorithm::Gdeflate => "Gdeflate",
        Algorithm::Zstd => "Zstd",
    };
    println!("gpu-compressor: {} directory compression", algo_name.to_lowercase());
    println!("  input:  {}", input_dir.display());
    println!("  output: {}", output_dir.display());
    println!("  phase 1: hash -> phase 2: compress\n");

    // Create output directory
    fs::create_dir_all(output_dir)?;
    let start_time = std::time::Instant::now();

    // ===== PHASE 1: DISCOVER AND HASH ALL FILES =====
    println!("  discovering files...");

    use std::collections::HashMap;
    let mut file_list: Vec<(PathBuf, u64, PathBuf)> = Vec::new(); // (input, size, output)

    // Discover all files first
    for entry_result in WalkDir::new(input_dir) {
        let entry = match entry_result {
            Ok(e) => e,
            Err(err) => {
                eprintln!("  warning: skipped: {}", err);
                continue;
            }
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let input_path = entry.path();
        let file_size = match fs::metadata(input_path) {
            Ok(m) => m.len(),
            Err(err) => {
                eprintln!(
                    "  warning: {}: {}",
                    input_path.display(),
                    err
                );
                continue;
            }
        };

        let relative_path = match input_path.strip_prefix(input_dir) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("  warning: bad path: {}", input_path.display());
                continue;
            }
        };

        let extension = match algorithm {
            Algorithm::Gdeflate => "nvgd",
            Algorithm::Zstd => "nvzs",
        };
        let output_path = output_dir.join(relative_path).with_extension(extension);

        if let Some(parent) = output_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "  warning: mkdir failed: {} ({})",
                    relative_path.display(),
                    e
                );
                continue;
            }
        }

        file_list.push((input_path.to_path_buf(), file_size, output_path));
    }

    println!("   Found {} files\n", file_list.len());

    // Hash all files
    // Always use GPU 0 for hashing (compression may use both GPUs)
    let hash_device_id = 0;
    let mut file_hashes: HashMap<PathBuf, String> = HashMap::new();

    for (input_path, file_size, _) in &file_list {
        let filename = input_path.file_name().unwrap().to_string_lossy();
        eprint!("  hashing: {}...", filename);

        let hash = if *file_size == 0 {
            eprintln!(" empty file");
            "0000000000000000000000000000000000000000000000000000000000000000".to_string()
        } else {
            match blake3_hash_file(input_path, hash_device_id) {
                Ok(h) => {
                    eprintln!(" ok");
                    h
                }
                Err(e) => {
                    eprintln!(" FAILED: {}", e);
                    continue;
                }
            }
        };

        file_hashes.insert(input_path.clone(), hash);
    }

    println!("  hashed {} files.\n", file_hashes.len());

    // ===== PHASE 2: COMPRESS ALL FILES =====
    println!("  compressing...");

    let mut total_input_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    let mut successful_files = 0usize;

    for (input_path, file_size, output_path) in &file_list {
        let filename = input_path.file_name().unwrap().to_string_lossy();
        eprintln!("  {}", filename);

        total_input_bytes += file_size;

        // Use standard single-file compression (streams, creates tar with .blake3, dual GPU)
        if let Err(e) = compress_file_impl(input_path, output_path, algorithm, device_id, false, level) {
            eprintln!("    FAILED: {}", e);
            continue;
        }

        if let Ok(out_meta) = fs::metadata(output_path) {
            total_output_bytes += out_meta.len();
        }

        successful_files += 1;
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let ratio = if total_input_bytes > 0 {
        (total_output_bytes as f64 / total_input_bytes as f64) * 100.0
    } else {
        0.0
    };
    let throughput = (total_input_bytes as f64 / elapsed) / 1_000_000_000.0;

    println!("");
    println!("  summary:");
    println!("   Files:      {} compressed", successful_files);
    println!(
        "   Input:      {:.2} GB",
        total_input_bytes as f64 / 1_000_000_000.0
    );
    println!(
        "   Output:     {:.2} GB",
        total_output_bytes as f64 / 1_000_000_000.0
    );
    println!("   Ratio:      {:.2}%", ratio);
    println!("   Time:       {:.2}s", elapsed);
    println!("   Throughput: {:.2} GB/s", throughput);

    Ok(())
}
