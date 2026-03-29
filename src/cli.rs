use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Algorithm {
    Gdeflate,
    Zstd,
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gdeflate" | "deflate" => Ok(Algorithm::Gdeflate),
            "zstd" | "zstandard" => Ok(Algorithm::Zstd),
            _ => Err(format!(
                "Unknown algorithm: {}. Use 'gdeflate' or 'zstd'",
                s
            )),
        }
    }
}

#[derive(Parser)]
#[command(name = "gpu-compressor")]
#[command(about = "GPU-accelerated file compression using nvCOMP")]
pub(crate) struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Compress a file using GPU
    Compress {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "zstd")]
        algorithm: Algorithm,
        #[arg(short, long, default_value = "0")]
        device: i32,
    },
    /// Compress multiple files in parallel using GPU (maximizes GPU utilization)
    /// If inputs contains a directory, all files in that directory will be compressed
    CompressMulti {
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<PathBuf>,
        #[arg(short, long, num_args = 0..)]
        outputs: Vec<PathBuf>,
        #[arg(short, long, default_value = "zstd")]
        algorithm: Algorithm,
        #[arg(short, long, default_value = "0")]
        device: i32,
        #[arg(long, default_value = "134217728")]
        chunk_size: usize, // Default 128MB in bytes
    },
    /// Decompress a file using GPU
    Decompress {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "0")]
        device: i32,
    },
    /// Decompress multiple files in parallel using GPU (maximizes GPU utilization)
    DecompressMulti {
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<PathBuf>,
        #[arg(short, long, num_args = 1..)]
        outputs: Vec<PathBuf>,
        #[arg(short, long, default_value = "0")]
        device: i32,
    },
    /// GPU-accelerated BLAKE3 file hashing for verification
    Hash {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "0")]
        device: i32,
    },
}
