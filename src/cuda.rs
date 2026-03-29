use anyhow::Result;

/// Detect available NVIDIA GPUs
pub(crate) fn detect_gpus() -> Result<Vec<i32>> {
    unsafe {
        let mut device_count: i32 = 0;
        let result = cuda_runtime_sys::cudaGetDeviceCount(&mut device_count);

        if result != cuda_runtime_sys::cudaError::cudaSuccess {
            return Err(anyhow::anyhow!("Failed to get GPU count"));
        }

        if device_count == 0 {
            return Err(anyhow::anyhow!("No NVIDIA GPUs detected"));
        }

        let gpus: Vec<i32> = (0..device_count).collect();
        eprintln!("gpu: detected {} device(s)", device_count);

        // Print GPU info
        for &gpu_id in &gpus {
            let mut props: cuda_runtime_sys::cudaDeviceProp = std::mem::zeroed();
            cuda_runtime_sys::cudaGetDeviceProperties(&mut props, gpu_id);
            let name = std::ffi::CStr::from_ptr(props.name.as_ptr()).to_string_lossy();
            eprintln!("  [{}] {}", gpu_id, name);
        }

        Ok(gpus)
    }
}
