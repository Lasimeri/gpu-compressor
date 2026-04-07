use anyhow::Result;
use std::ffi::c_void;

// liblzma FFI via dlopen — avoids link-time dependency that crashes with CUDA runtime.
// Struct layouts verified against liblzma 5.8.3 on x86_64 via offsetof().

const LZMA_FILTER_LZMA2: u64 = 0x21;
const LZMA_VLI_UNKNOWN: u64 = u64::MAX;
const LZMA_FINISH: u32 = 3;
const LZMA_OK: u32 = 0;
const LZMA_STREAM_END: u32 = 1;

#[repr(C)]
struct LzmaStream {
    next_in: *const u8,
    avail_in: usize,
    total_in: u64,
    next_out: *mut u8,
    avail_out: usize,
    total_out: u64,
    _rest: [u8; 88],
}

#[repr(C)]
struct LzmaOptionsLzma {
    dict_size: u32,
    _pad0: u32,
    preset_dict: *const u8,
    preset_dict_size: u32,
    lc: u32,
    lp: u32,
    pb: u32,
    mode: u32,
    nice_len: u32,
    mf: u32,
    depth: u32,
    _reserved: [u8; 64],
}

#[repr(C)]
struct LzmaFilter {
    id: u64,
    options: *mut LzmaOptionsLzma,
}

// Function pointer types
type FnLzmaRawEncoder = unsafe extern "C" fn(*mut LzmaStream, *const LzmaFilter) -> u32;
type FnLzmaRawDecoder = unsafe extern "C" fn(*mut LzmaStream, *const LzmaFilter) -> u32;
type FnLzmaCode = unsafe extern "C" fn(*mut LzmaStream, u32) -> u32;
type FnLzmaEnd = unsafe extern "C" fn(*mut LzmaStream);
type FnLzmaPreset = unsafe extern "C" fn(*mut LzmaOptionsLzma, u32) -> u32;

struct Lzma {
    raw_encoder: FnLzmaRawEncoder,
    raw_decoder: FnLzmaRawDecoder,
    code: FnLzmaCode,
    end: FnLzmaEnd,
    preset: FnLzmaPreset,
    _lib: *mut c_void,
}

// dlopen/dlsym FFI
extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

impl Lzma {
    fn load() -> Result<Self> {
        unsafe {
            let lib = dlopen(b"liblzma.so.5\0".as_ptr(), RTLD_NOW);
            if lib.is_null() {
                return Err(anyhow::anyhow!("Failed to load liblzma.so.5"));
            }
            let get = |name: &[u8]| -> *mut c_void {
                dlsym(lib, name.as_ptr())
            };
            let raw_encoder = std::mem::transmute(get(b"lzma_raw_encoder\0"));
            let raw_decoder = std::mem::transmute(get(b"lzma_raw_decoder\0"));
            let code = std::mem::transmute(get(b"lzma_code\0"));
            let end = std::mem::transmute(get(b"lzma_end\0"));
            let preset = std::mem::transmute(get(b"lzma_lzma_preset\0"));
            Ok(Self { raw_encoder, raw_decoder, code, end, preset, _lib: lib })
        }
    }
}

pub(crate) fn compress_chunk_lzma2(
    chunk_data: &[u8],
    _dict_size: u32,
    preset: u32,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let lz = Lzma::load()?;
    unsafe {
        let mut opts: LzmaOptionsLzma = std::mem::zeroed();
        if (lz.preset)(&mut opts, preset.min(9)) != 0 {
            return Err(anyhow::anyhow!("lzma_lzma_preset failed"));
        }

        let filters = [
            LzmaFilter { id: LZMA_FILTER_LZMA2, options: &mut opts },
            LzmaFilter { id: LZMA_VLI_UNKNOWN, options: std::ptr::null_mut() },
        ];

        let mut strm: LzmaStream = std::mem::zeroed();
        let ret = (lz.raw_encoder)(&mut strm, filters.as_ptr());
        if ret != LZMA_OK {
            return Err(anyhow::anyhow!("lzma_raw_encoder init failed: {}", ret));
        }

        let out_capacity = chunk_data.len() + chunk_data.len() / 8 + 1024;
        let mut out_buf = vec![0u8; out_capacity];

        strm.next_in = chunk_data.as_ptr();
        strm.avail_in = chunk_data.len();
        strm.next_out = out_buf.as_mut_ptr();
        strm.avail_out = out_capacity;

        let ret = (lz.code)(&mut strm, LZMA_FINISH);
        if ret != LZMA_STREAM_END {
            (lz.end)(&mut strm);
            return Err(anyhow::anyhow!("lzma_code failed: {}", ret));
        }

        let compressed_size = strm.total_out as usize;
        (lz.end)(&mut strm);

        out_buf.truncate(compressed_size);
        let size = out_buf.len();
        Ok((vec![out_buf], vec![size]))
    }
}

pub(crate) fn decompress_chunk_lzma2(
    compressed: &[u8],
    decompressed_size: usize,
) -> Result<Vec<u8>> {
    let lz = Lzma::load()?;
    unsafe {
        let mut opts: LzmaOptionsLzma = std::mem::zeroed();
        opts.dict_size = 64 * 1024 * 1024;

        let filters = [
            LzmaFilter { id: LZMA_FILTER_LZMA2, options: &mut opts },
            LzmaFilter { id: LZMA_VLI_UNKNOWN, options: std::ptr::null_mut() },
        ];

        let mut strm: LzmaStream = std::mem::zeroed();
        let ret = (lz.raw_decoder)(&mut strm, filters.as_ptr());
        if ret != LZMA_OK {
            return Err(anyhow::anyhow!("lzma_raw_decoder init failed: {}", ret));
        }

        let mut out_buf = vec![0u8; decompressed_size];
        strm.next_in = compressed.as_ptr();
        strm.avail_in = compressed.len();
        strm.next_out = out_buf.as_mut_ptr();
        strm.avail_out = decompressed_size;

        let ret = (lz.code)(&mut strm, LZMA_FINISH);
        if ret != LZMA_STREAM_END && ret != LZMA_OK {
            (lz.end)(&mut strm);
            return Err(anyhow::anyhow!("lzma2 decompression failed: {}", ret));
        }

        let actual = decompressed_size - strm.avail_out;
        (lz.end)(&mut strm);
        out_buf.truncate(actual);
        Ok(out_buf)
    }
}
