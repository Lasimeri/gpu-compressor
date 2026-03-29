/// Message types for pipeline communication
pub(crate) enum PipelineMsg {
    Chunk { data: Vec<u8>, chunk_index: usize },
    Done,
}

/// Compressed chunk ready for writing
pub(crate) struct CompressedChunk {
    pub chunks: Vec<Vec<u8>>, // Individual compressed chunks, not concatenated
    #[allow(dead_code)]
    pub chunk_index: usize,
    pub compressed_sizes: Vec<u64>,
}
