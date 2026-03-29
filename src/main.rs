mod blake3;
mod cli;
mod compress_gdeflate;
mod compress_zstd;
mod constants;
mod cuda;
mod decompress;
mod dispatch;
mod format;
mod multi;
mod nvcomp_bindings;
mod pipeline;
mod pipeline_dual;

use anyhow::Result;
use clap::Parser;

use blake3::blake3_hash_file;
use cli::{Algorithm, Args, Commands};
use decompress::{decompress_file, decompress_multi_files};
use dispatch::{compress_directory, compress_file, expand_directory_inputs};
use multi::compress_multi_files_async;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Compress {
            input,
            output,
            algorithm,
            device,
        } => {
            if input.is_dir() {
                compress_directory(&input, &output, algorithm, device)?;
            } else {
                compress_file(&input, &output, algorithm, device)?;
            }
        }
        Commands::CompressMulti {
            inputs,
            outputs,
            algorithm,
            device,
            chunk_size,
        } => {
            // Expand directories to file lists
            let extension = match algorithm {
                Algorithm::Gdeflate => ".nvgd",
                Algorithm::Zstd => ".nvzs",
            };
            let (expanded_inputs, expanded_outputs) =
                expand_directory_inputs(&inputs, &outputs, extension)?;

            if expanded_inputs.is_empty() {
                return Err(anyhow::anyhow!("No files found to compress"));
            }

            if expanded_inputs.len() != expanded_outputs.len() {
                return Err(anyhow::anyhow!(
                    "Number of inputs ({}) must match number of outputs ({})",
                    expanded_inputs.len(),
                    expanded_outputs.len()
                ));
            }

            // Use async version for true parallel processing
            compress_multi_files_async(
                &expanded_inputs,
                &expanded_outputs,
                algorithm,
                device,
                chunk_size,
            )
            .await?;
        }
        Commands::Decompress {
            input,
            output,
            device,
        } => {
            decompress_file(&input, &output, device)?;
        }
        Commands::DecompressMulti {
            inputs,
            outputs,
            device,
        } => {
            if inputs.len() != outputs.len() {
                return Err(anyhow::anyhow!(
                    "Number of inputs ({}) must match number of outputs ({})",
                    inputs.len(),
                    outputs.len()
                ));
            }
            decompress_multi_files(&inputs, &outputs, device)?;
        }
        Commands::Hash { input, device } => {
            let hash = blake3_hash_file(&input, device as usize)?;
            println!("\n  blake3: {}", hash);
        }
    }

    // Skip normal cleanup to avoid CUDA driver segfault during tokio runtime teardown.
    // CUDA contexts allocated across multiple GPU threads have non-deterministic drop
    // ordering that conflicts with the async runtime's thread pool shutdown.
    std::process::exit(0);
}
