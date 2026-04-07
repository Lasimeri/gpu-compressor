#![allow(clashing_extern_declarations)]

mod cli;
mod compress_lzma2;
mod compress_lzma2_custom;
mod compress_zstd;
mod compress_zstd_custom;
mod constants;
mod cuda;
mod decompress;
mod dispatch;
mod format;
mod multi;
mod nvcomp_bindings;
mod pipeline;
mod pipeline_dual;
mod pipeline_lzma2;
mod pipeline_lzma2_dual;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use cli::{Args, Commands};
use decompress::decompress_file;
use dispatch::{compress_directory, compress_file, compress_file_lzma2, expand_directory_inputs};
use multi::compress_multi_files_async;

fn auto_compress_output_zstd(input: &PathBuf) -> PathBuf {
    input.with_extension("nvzs")
}

fn auto_compress_output_lzma2(input: &PathBuf) -> PathBuf {
    input.with_extension("nvlz")
}

/// Auto-generate a decompress output path for `input` by stripping compression extension.
fn auto_decompress_output(input: &PathBuf) -> PathBuf {
    match input.extension().and_then(|e| e.to_str()) {
        Some("nvzs") | Some("nvlz") => input.with_extension(""),
        _ => {
            // Unknown extension: append .out
            let mut p = input.clone();
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            p.set_file_name(format!("{}.out", name));
            p
        }
    }
}

/// Resolve output path for a single input, given an optional base output spec.
/// If `output_spec` is Some(dir) → place output inside that dir with auto-generated filename.
/// If `output_spec` is Some(file) and there's exactly one input → use it directly.
/// If `output_spec` is None → auto-generate alongside input.
fn resolve_output(
    input: &PathBuf,
    output_spec: &Option<PathBuf>,
    auto_fn: impl Fn(&PathBuf) -> PathBuf,
    single_input: bool,
) -> PathBuf {
    match output_spec {
        None => auto_fn(input),
        Some(p) if p.is_dir() => {
            // Place auto-named file inside the given directory
            let auto = auto_fn(input);
            p.join(auto.file_name().unwrap_or_default())
        }
        Some(p) if single_input => p.clone(),
        Some(_) => {
            // Multiple inputs with a non-directory output spec: fall back to auto
            auto_fn(input)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("gpu-compressor v0.1.0");

    let args = Args::parse();

    match args.command {
        Commands::Compress {
            input,
            output,
            algorithm,
            device,
            level,
            chunk_size,
            dict_size,
        } => {
            let is_lzma2 = algorithm == "lzma2";
            let ext = if is_lzma2 { ".nvlz" } else { ".nvzs" };
            let auto_fn: fn(&PathBuf) -> PathBuf = if is_lzma2 {
                auto_compress_output_lzma2
            } else {
                auto_compress_output_zstd
            };

            let single = input.len() == 1;

            if single && input[0].is_dir() {
                let out = match &output {
                    Some(p) => p.clone(),
                    None => input[0].clone(),
                };
                if is_lzma2 {
                    return Err(anyhow::anyhow!(
                        "Directory compression not yet supported for LZMA2"
                    ));
                }
                compress_directory(&input[0], &out, device, level)?;
            } else {
                let outputs_hint: Vec<PathBuf> = match &output {
                    Some(p) if p.is_dir() => vec![p.clone()],
                    Some(p) if single => vec![p.clone()],
                    _ => vec![],
                };
                let (expanded_inputs, raw_outputs) =
                    expand_directory_inputs(&input, &outputs_hint, ext)?;

                if expanded_inputs.is_empty() {
                    return Err(anyhow::anyhow!("No files found to compress"));
                }

                let expanded_single = expanded_inputs.len() == 1;
                let expanded_outputs: Vec<PathBuf> = if raw_outputs.len() == expanded_inputs.len() {
                    raw_outputs
                } else {
                    expanded_inputs
                        .iter()
                        .map(|inp| resolve_output(inp, &output, auto_fn, expanded_single))
                        .collect()
                };

                let start = Instant::now();

                if is_lzma2 {
                    for (inp, out) in expanded_inputs.iter().zip(expanded_outputs.iter()) {
                        compress_file_lzma2(inp, out, device, level, dict_size)?;
                    }
                } else if expanded_inputs.len() >= 2 && level == 0 {
                    compress_multi_files_async(
                        &expanded_inputs,
                        &expanded_outputs,
                        device,
                        chunk_size,
                    )
                    .await?;
                } else {
                    for (inp, out) in expanded_inputs.iter().zip(expanded_outputs.iter()) {
                        compress_file(inp, out, device, level)?;
                    }
                }

                let elapsed = start.elapsed().as_secs_f64();
                if expanded_inputs.len() > 1 {
                    eprintln!("  total time: {:.2}s", elapsed);
                }
            }
        }

        Commands::Decompress {
            input,
            output,
            device,
        } => {
            if input.is_empty() {
                return Err(anyhow::anyhow!("No input files specified"));
            }

            let single = input.len() == 1;
            let start = Instant::now();

            for inp in &input {
                let out = resolve_output(inp, &output, auto_decompress_output, single);
                decompress_file(inp, &out, device)?;
            }

            let elapsed = start.elapsed().as_secs_f64();
            if input.len() > 1 {
                eprintln!("  total time: {:.2}s", elapsed);
            }
        }
    }

    Ok(())
}
