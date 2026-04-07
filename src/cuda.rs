use anyhow::Result;

// Use nvcomp_bindings for cudaDeviceProp which is generated from current CUDA headers
// (1008 bytes on CUDA 13.x). The cuda-runtime-sys crate has an outdated 712-byte definition
// that causes a 296-byte buffer overflow on cudaGetDeviceProperties.
use crate::nvcomp_bindings::cudaDeviceProp;

extern "C" {
    #[allow(clashing_extern_declarations)]
    fn cudaGetDeviceCount(count: *mut i32) -> cuda_runtime_sys::cudaError;
    #[allow(clashing_extern_declarations)]
    fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> cuda_runtime_sys::cudaError;
}

/// Detect available NVIDIA GPUs
pub(crate) fn detect_gpus() -> Result<Vec<i32>> {
    let mut device_count: i32 = 0;

    let result = unsafe { cudaGetDeviceCount(&mut device_count) };
    if result != cuda_runtime_sys::cudaError::cudaSuccess {
        return Err(anyhow::anyhow!("cudaGetDeviceCount failed: {:?}", result));
    }

    if device_count == 0 {
        return Err(anyhow::anyhow!("No NVIDIA GPUs detected"));
    }

    let gpus: Vec<i32> = (0..device_count).collect();
    eprintln!("gpu: detected {} device(s)", device_count);

    for &gpu_id in &gpus {
        let mut props: Box<cudaDeviceProp> = Box::new(unsafe { std::mem::zeroed() });

        let result = unsafe { cudaGetDeviceProperties(&mut *props, gpu_id) };
        if result != cuda_runtime_sys::cudaError::cudaSuccess {
            eprintln!("  [{}] <failed to query: {:?}>", gpu_id, result);
            continue;
        }

        let name = unsafe { std::ffi::CStr::from_ptr(props.name.as_ptr()) }.to_string_lossy();
        eprintln!("  [{}] {}", gpu_id, name);
    }

    Ok(gpus)
}
