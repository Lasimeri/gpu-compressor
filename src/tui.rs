use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// TUI state tracker — holds all dynamic data for rendering
pub struct TuiState {
    /// File size in bytes
    pub file_size: u64,
    /// Bytes read from input
    pub bytes_read: Arc<AtomicU64>,
    /// Bytes compressed
    pub bytes_compressed: Arc<AtomicU64>,
    /// Bytes written to output
    pub bytes_written: Arc<AtomicU64>,
    /// Current chunk index on GPU 0 (for single or first GPU)
    pub chunk_gpu0: Arc<AtomicU64>,
    /// Current chunk index on GPU 1 (for dual GPU only)
    pub chunk_gpu1: Option<Arc<AtomicU64>>,
    /// GPU device IDs
    pub gpu_devices: Vec<i32>,
    /// Mode label (e.g., "zstd (single gpu)" or "zstd (dual gpu)")
    pub mode_label: String,
    /// Input path for display
    pub input_path: String,
    /// Output path for display
    pub output_path: String,
    /// Start time for calculating speeds
    pub start_time: Instant,
    /// Whether to suppress output
    pub quiet: bool,
    /// Number of lines printed (for cursor movement)
    num_lines: usize,
    /// True until the first draw completes — suppresses cursor-up on initial render
    first_draw: bool,
}

impl TuiState {
    pub fn new(
        file_size: u64,
        bytes_read: Arc<AtomicU64>,
        bytes_compressed: Arc<AtomicU64>,
        bytes_written: Arc<AtomicU64>,
        chunk_gpu0: Arc<AtomicU64>,
        chunk_gpu1: Option<Arc<AtomicU64>>,
        gpu_devices: Vec<i32>,
        mode_label: String,
        input_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        quiet: bool,
    ) -> Self {
        Self {
            file_size,
            bytes_read,
            bytes_compressed,
            bytes_written,
            chunk_gpu0,
            chunk_gpu1,
            gpu_devices,
            mode_label,
            input_path: input_path.as_ref().display().to_string(),
            output_path: output_path.as_ref().display().to_string(),
            start_time: Instant::now(),
            quiet,
            num_lines: 0,
            first_draw: true,
        }
    }

    /// Format file size in human-readable form
    fn format_size(bytes: u64) -> String {
        if bytes >= 1_000_000_000 {
            format!("{:.2} GB", bytes as f64 / 1e9)
        } else if bytes >= 1_000_000 {
            format!("{:.2} MB", bytes as f64 / 1e6)
        } else {
            format!("{:.2} KB", bytes as f64 / 1e3)
        }
    }

    /// Format bandwidth in GB/s
    fn format_bandwidth(bytes: u64, secs: f64) -> String {
        let gbs = (bytes as f64 / 1e9) / secs;
        format!("{:.2} GB/s", gbs)
    }

    /// Draw the TUI (or redraw if already drawn)
    pub fn draw(&mut self) {
        if self.quiet {
            return;
        }

        let bytes_read = self.bytes_read.load(Ordering::Relaxed);
        let bytes_compressed = self.bytes_compressed.load(Ordering::Relaxed);
        let bytes_written = self.bytes_written.load(Ordering::Relaxed);
        let chunk_gpu0 = self.chunk_gpu0.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        // Calculate throughput
        let read_bw = if elapsed > 0.0 {
            Self::format_bandwidth(bytes_read, elapsed)
        } else {
            "0.00 GB/s".to_string()
        };

        let _gpu_bw = if elapsed > 0.0 {
            Self::format_bandwidth(bytes_compressed, elapsed)
        } else {
            "0.00 GB/s".to_string()
        };

        let write_bw = if elapsed > 0.0 {
            Self::format_bandwidth(bytes_written, elapsed)
        } else {
            "0.00 GB/s".to_string()
        };

        let ratio = if bytes_read > 0 {
            (bytes_compressed as f64 / bytes_read as f64) * 100.0
        } else {
            0.0
        };

        let progress_pct = if self.file_size > 0 {
            (bytes_read as f64 / self.file_size as f64) * 100.0
        } else {
            0.0
        };

        // Build progress bar (20 chars)
        let progress_filled = ((progress_pct / 100.0) * 20.0) as usize;
        let progress_empty = 20 - progress_filled;
        let progress_bar = format!(
            "[{}{}] {:.0}%",
            "█".repeat(progress_filled),
            "░".repeat(progress_empty),
            progress_pct
        );

        // Move cursor up if already drawn (skip on first render to avoid duplicate)
        if self.first_draw {
            self.first_draw = false;
        } else if self.num_lines > 0 {
            eprint!("\x1b[{}A", self.num_lines);
        }

        // Clear and redraw all lines
        let mut lines = vec![];
        lines.push(format!("gpu-compressor — {}", self.mode_label));
        lines.push("──────────────────────────────────────".to_string());
        lines.push(format!("  Input:    {} ({})", self.input_path, Self::format_size(self.file_size)));
        lines.push(format!("  Output:   {}", self.output_path));
        lines.push(format!("  Progress: {} ", progress_bar));
        lines.push(format!("  Read:     {} @ {}", Self::format_size(bytes_read), read_bw));

        // GPU line(s)
        if let Some(ref chunk_gpu1) = self.chunk_gpu1 {
            // Dual GPU: show both
            let chunk_gpu1_val = chunk_gpu1.load(Ordering::Relaxed);
            lines.push(format!("  GPU {}:    chunk #{} (processing)", self.gpu_devices[0], chunk_gpu0));
            lines.push(format!("  GPU {}:    chunk #{} (processing)", self.gpu_devices[1], chunk_gpu1_val));
        } else {
            // Single GPU
            lines.push(format!("  GPU 0:    chunk #{} (processing)", chunk_gpu0));
        }

        lines.push(format!("  Write:    {} @ {}", Self::format_size(bytes_written), write_bw));
        lines.push(format!("  Ratio:    {:.1}%", ratio));
        lines.push("──────────────────────────────────────".to_string());

        for line in &lines {
            eprint!("\x1b[2K\r{}\n", line);
        }

        self.num_lines = lines.len();
    }

    /// Final summary after completion
    pub fn print_summary(&self, total_compressed: u64, is_dual: bool) {
        if self.quiet {
            return;
        }

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let ratio = (total_compressed as f64 / self.file_size as f64) * 100.0;
        let throughput = Self::format_bandwidth(self.file_size, elapsed);

        eprintln!("");
        eprintln!("gpu-compressor — complete");
        eprintln!(
            "  {} → {} ({:.1}%)",
            Self::format_size(self.file_size),
            Self::format_size(total_compressed),
            ratio
        );
        eprintln!("  Time: {:.1}s @ {}", elapsed, throughput);
        eprintln!("  Pipeline: {} GPU{}", if is_dual { "dual" } else { "single" }, if is_dual { "s" } else { "" });
    }

    /// Clear the TUI display
    #[allow(dead_code)]
    pub fn clear(&self) {
        if self.quiet {
            return;
        }
        if self.num_lines > 0 {
            eprint!("\x1b[{}A\x1b[2J", self.num_lines);
        }
    }
}
