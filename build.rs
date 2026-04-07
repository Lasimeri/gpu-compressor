use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=blake3.cu");
    println!("cargo:rerun-if-changed=zstd_compress.cu");

    // Link nvCOMP library
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-lib=dylib=nvcomp");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=zstd");
    // liblzma loaded dynamically via dlopen in compress_lzma2.rs

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

    // Compile custom Zstd compression kernel to PTX
    let output_zstd = Command::new("/opt/cuda/bin/nvcc")
        .args([
            "zstd_compress.cu",
            "-ptx",
            "-o",
            "zstd_compress.ptx",
            "--gpu-architecture=sm_86",
            "-O3",
            "--use_fast_math",
            "--maxrregcount=64",
            "-Xptxas=-v",
        ])
        .output()
        .expect("Failed to compile zstd_compress.cu");

    if !output_zstd.status.success() {
        panic!(
            "CUDA compilation failed for zstd_compress.cu:\n{}",
            String::from_utf8_lossy(&output_zstd.stderr)
        );
    }

    // Compile LZMA2 match finding kernel to PTX
    println!("cargo:rerun-if-changed=lzma2_match_find.cu");

    let output_lzma2 = Command::new("/opt/cuda/bin/nvcc")
        .args([
            "lzma2_match_find.cu",
            "-ptx",
            "-o",
            "lzma2_match_find.ptx",
            "--gpu-architecture=sm_86",
            "-O3",
            "--use_fast_math",
            "--maxrregcount=64",
            "-Xptxas=-v",
        ])
        .output()
        .expect("Failed to compile lzma2_match_find.cu");

    if !output_lzma2.status.success() {
        panic!(
            "CUDA compilation failed for lzma2_match_find.cu:\n{}",
            String::from_utf8_lossy(&output_lzma2.stderr)
        );
    }
}
