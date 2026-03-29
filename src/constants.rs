pub(crate) const CHUNK_SIZE: usize = 65536; // 64KB chunks for gdeflate GPU compression (legacy)
pub(crate) const ZSTD_CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4MB chunks for Zstd GPU compression (streamed, no batching)
pub(crate) const GDEFLATE_MAX_COMPRESSION: i32 = 5; // Highest compression ratio
pub(crate) const BLAKE3_CHUNK_SIZE: usize = 1024; // BLAKE3 chunk: 1KB per thread for max GPU parallelism
pub(crate) const PTX_BLAKE3: &str = include_str!("../blake3.ptx");
pub(crate) const PTX_ZSTD_COMPRESS: &str = include_str!("../zstd_compress.ptx");
pub(crate) const BATCH_SIZE: usize = 128 * 1024 * 1024; // 128MB - process gdeflate data at once on GPU
pub(crate) const MAX_FILE_CHUNK_SIZE: usize = 128 * 1024 * 1024; // 128MB - stream in chunks
pub(crate) const PIPELINE_QUEUE_SIZE: usize = 16; // Channel queue depth for pipeline (deeper queue for better I/O overlap)
pub(crate) const GDEFLATE_MAX_SIZE: usize = 2 * 1024 * 1024 * 1024 - 1; // ~2GB limit for gdeflate

// Custom Zstd kernel constants
pub(crate) const CUSTOM_ZSTD_CHUNK_SIZE: usize = 65536; // 64KB sub-chunks for match finding
pub(crate) const CUSTOM_ZSTD_SEARCH_DEPTH_LAZY: usize = 16;
