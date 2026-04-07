use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::cuda::detect_gpus;
use crate::pipeline::compress_file_streaming_zstd;
use crate::pipeline_dual::compress_file_streaming_dual_gpu;
use crate::pipeline_lzma2::compress_file_streaming_lzma2;
// LZMA2 dual GPU pipeline superseded by multi-threaded single pipeline

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
    device_id: i32,
    level: u32,
) -> Result<()> {
    compress_file_impl(input_path, output_path, device_id, false, level)
}

pub(crate) fn compress_file_impl(
    input_path: &Path,
    output_path: &Path,
    device_id: i32,
    quiet: bool,
    level: u32,
) -> Result<()> {
    // Auto-generate filename if output is a directory
    let final_output_path = if output_path.is_dir() {
        let input_filename = input_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid input filename"))?;
        output_path.join(input_filename).with_extension("nvzs")
    } else {
        output_path.to_path_buf()
    };

    // Auto-detect GPUs and choose pipeline
    let available_gpus = detect_gpus()?;

    if available_gpus.len() >= 2 && level == 0 {
        eprintln!("  mode: dual gpu pipeline ({} devices)", available_gpus.len());
        compress_file_streaming_dual_gpu(
            input_path,
            &final_output_path,
            available_gpus[0],
            available_gpus[1],
            quiet,
        )
    } else {
        if level > 0 && available_gpus.len() >= 2 {
            eprintln!("  note: custom levels use single GPU pipeline");
        }
        compress_file_streaming_zstd(input_path, &final_output_path, device_id, quiet, level)
    }
}

pub(crate) fn compress_directory(
    input_dir: &Path,
    output_dir: &Path,
    device_id: i32,
    level: u32,
) -> Result<()> {
    println!("gpu-compressor: zstd directory compression");
    println!("  input:  {}", input_dir.display());
    println!("  output: {}", output_dir.display());

    // Create output directory
    fs::create_dir_all(output_dir)?;
    let start_time = std::time::Instant::now();

    // Discover all files
    println!("  discovering files...");

    let mut file_list: Vec<(PathBuf, u64, PathBuf)> = Vec::new(); // (input, size, output)

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

        let output_path = output_dir.join(relative_path).with_extension("nvzs");

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

    // Compress all files
    println!("  compressing...");

    let mut total_input_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    let mut successful_files = 0usize;

    for (input_path, file_size, output_path) in &file_list {
        let filename = input_path.file_name().unwrap().to_string_lossy();
        eprintln!("  {}", filename);

        total_input_bytes += file_size;

        // Use standard single-file compression (streams raw data, dual GPU if available)
        if let Err(e) = compress_file_impl(input_path, output_path, device_id, false, level) {
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

pub(crate) fn compress_file_lzma2(
    input_path: &Path, output_path: &Path, _device_id: i32, level: u32, dict_size_mb: u32,
) -> Result<()> {
    let final_output_path = if output_path.is_dir() {
        let input_filename = input_path.file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid input filename"))?;
        output_path.join(input_filename).with_extension("nvlz")
    } else { output_path.to_path_buf() };
    let available_gpus = detect_gpus()?;
    compress_file_streaming_lzma2(input_path, &final_output_path, &available_gpus, false, level, dict_size_mb)
}
