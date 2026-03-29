use anyhow::Result;

/// Detect available NVIDIA GPUs with proper error checking on all FFI calls
pub(crate) fn detect_gpus() -> Result<Vec<i32>> {
    let mut device_count: i32 = 0;

    // SAFETY: cudaGetDeviceCount writes to a valid i32 pointer
    let result = unsafe { cuda_runtime_sys::cudaGetDeviceCount(&mut device_count) };
    if result != cuda_runtime_sys::cudaError::cudaSuccess {
        return Err(anyhow::anyhow!(
            "cudaGetDeviceCount failed: {:?}",
            result
        ));
    }

    if device_count == 0 {
        return Err(anyhow::anyhow!("No NVIDIA GPUs detected"));
    }

    let gpus: Vec<i32> = (0..device_count).collect();
    eprintln!("gpu: detected {} device(s)", device_count);

    for &gpu_id in &gpus {
        // SAFETY: cudaDeviceProp is a POD struct, zeroing is valid initialization.
        // cudaGetDeviceProperties writes into the provided struct.
        let mut props: cuda_runtime_sys::cudaDeviceProp =
            unsafe { std::mem::zeroed() };

        let result =
            unsafe { cuda_runtime_sys::cudaGetDeviceProperties(&mut props, gpu_id) };

        if result != cuda_runtime_sys::cudaError::cudaSuccess {
            eprintln!("  [{}] <failed to query: {:?}>", gpu_id, result);
            continue;
        }

        // SAFETY: cudaGetDeviceProperties guarantees null-terminated name on success
        let name =
            unsafe { std::ffi::CStr::from_ptr(props.name.as_ptr()) }.to_string_lossy();
        eprintln!("  [{}] {}", gpu_id, name);
    }

    Ok(gpus)
}
