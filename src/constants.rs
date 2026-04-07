pub(crate) const ZSTD_CHUNK_SIZE: usize = 8 * 1024 * 1024; // 8MB chunks for Zstd GPU compression (matches optimal NVMe block size)
pub(crate) const PTX_ZSTD_COMPRESS: &str = include_str!("../zstd_compress.ptx");

// Custom Zstd kernel constants
pub(crate) const CUSTOM_ZSTD_CHUNK_SIZE: usize = 65536; // 64KB sub-chunks for match finding
pub(crate) const CUSTOM_ZSTD_SEARCH_DEPTH_LAZY: usize = 16;

// LZMA2 constants
pub(crate) const LZMA2_CHUNK_SIZE: usize = 8 * 1024 * 1024; // 8MB pipeline chunks
pub(crate) const LZMA2_CUSTOM_CHUNK_SIZE: usize = 65536; // to avoid LZMA2 boundary edge case at exact 2^16
pub(crate) const PTX_LZMA2_MATCH_FIND: &str = include_str!("../lzma2_match_find.ptx");
pub(crate) const LZMA2_MAX_MATCHES_PER_POS: usize = 8;
pub(crate) const LZMA2_HC4_SEARCH_DEPTH: u32 = 32;
#[allow(dead_code)]
pub(crate) const LZMA2_DICT_SIZE: u32 = 16 * 1024 * 1024; // 16MB default dictionary (Phase 2)
#[allow(dead_code)]
pub(crate) const LZMA2_DEFAULT_PRESET: u32 = 6;
