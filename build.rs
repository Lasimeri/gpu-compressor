use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=blake3.cu");

    // Link nvCOMP library
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-lib=dylib=nvcomp");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Generate bindings for nvCOMP
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I/opt/cuda/include")
        .clang_arg("-I/usr/local/cuda/include")
        .clang_arg("-I/usr/local/include")
        .clang_arg("-I/usr/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate nvCOMP bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("nvcomp_bindings.rs"))
        .expect("Couldn't write bindings!");

    // Compile BLAKE3 CUDA kernel to PTX
    let output_blake3 = Command::new("/opt/cuda/bin/nvcc")
        .args([
            "blake3.cu",
            "-ptx",
            "-o",
            "blake3.ptx",
            "--gpu-architecture=sm_86",
            "-O3",
            "--use_fast_math",
            "-Xptxas=-v",
        ])
        .output()
        .expect("Failed to compile blake3.cu");

    if !output_blake3.status.success() {
        panic!(
            "CUDA compilation failed for blake3.cu:\n{}",
            String::from_utf8_lossy(&output_blake3.stderr)
        );
    }
}
