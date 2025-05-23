// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

/// Recursively reads the filenames in a directory and stores them in a Vec.
fn _read_dir_recursively(dir_path: &PathBuf, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            _read_dir_recursively(&path, paths)?;
        } else {
            paths.push(path);
        }
    }

    Ok(())
}

/// Recursively reads the filenames in a directory and stores them in a Vec.
fn read_dir_recursively(dir_path: &PathBuf) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    _read_dir_recursively(dir_path, &mut paths)?;
    Ok(paths)
}

fn main() -> Result<()> {
    let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
        |_| num_cpus::get_physical(),
        |s| usize::from_str(&s).unwrap(),
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .unwrap();

    // Remove unecessary directories from cutlass, important for packaging.
    // Done here instead of in submodule for easier rebasing.
    let root = PathBuf::from_str(".").unwrap();
    let root = root.join("cutlass");
    for name in [
        "docs", "media", "test", "examples", "python", "tools", ".github", "cmake",
    ] {
        let dir = root.join(name);
        std::fs::remove_dir_all(dir).ok();
    }

    println!("cargo:rerun-if-changed=build.rs");

    let paths = read_dir_recursively(&PathBuf::from_str("kernels")?)?;
    for file in paths.iter() {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_V1_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            let current_dir = std::env::current_dir()?;
            path.canonicalize().unwrap_or_else(|_| {
                panic!(
                    "Directory doesn't exists: {} (the current directory is {})",
                    &path.display(),
                    current_dir.display()
                )
            })
        }
    };
    set_cuda_include_dir()?;

    let ccbin_env = std::env::var("CANDLE_NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");

    let compute_cap = compute_cap()?;

    let out_file = build_dir.join("libflashattentionv1.a");

    let kernel_dir = PathBuf::from("kernels");
    let kernels: Vec<_> = paths
        .iter()
        .filter(|f| f.extension().map(|ext| ext == "cu").unwrap_or_default())
        .collect();
    let cu_files: Vec<_> = kernels
        .iter()
        .map(|f| {
            let mut obj_file = out_dir.join(f);
            fs::create_dir_all(obj_file.parent().unwrap()).unwrap();
            obj_file.set_extension("o");
            (f, obj_file)
        })
        .collect();
    let out_modified: Result<_, _> = out_file.metadata().and_then(|m| m.modified());
    let should_compile = if out_file.exists() {
        kernel_dir
            .read_dir()
            .expect("kernels folder should exist")
            .any(|entry| {
                if let (Ok(entry), Ok(out_modified)) = (entry, &out_modified) {
                    let in_modified = entry.metadata().unwrap().modified().unwrap();
                    in_modified.duration_since(*out_modified).is_ok()
                } else {
                    true
                }
            })
    } else {
        true
    };
    if should_compile {
        cu_files
            .par_iter()
            .map(|(cu_file, obj_file)| {
                let mut command = std::process::Command::new("nvcc");
                command
                    .arg("-O3")
                    .arg("-std=c++17")
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("-c")
                    .args(["-o", obj_file.to_str().unwrap()])
                    .args(["--default-stream", "per-thread"])
                    .arg("-Icutlass/include")
                    .arg("-U__CUDA_NO_HALF_OPERATORS__")
                    .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                    .arg("-U__CUDA_NO_HALF2_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                    .arg("--expt-relaxed-constexpr")
                    .arg("--expt-extended-lambda")
                    .arg("--use_fast_math")
                    .arg("--ptxas-options=-v")
                    .arg("--verbose");
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = command
                    .spawn()
                    .context("failed spawning nvcc")?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "nvcc error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<()>>()?;
        let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
        let mut command = std::process::Command::new("nvcc");
        command
            .arg("--lib")
            .args(["-o", out_file.to_str().unwrap()])
            .args(obj_files);
        let output = command
            .spawn()
            .context("failed spawning nvcc")?
            .wait_with_output()?;
        if !output.status.success() {
            anyhow::bail!(
                "nvcc error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                &command,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        }
    }
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattentionv1");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn set_cuda_include_dir() -> Result<()> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::<PathBuf>::into);
    let root = env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .context("cannot find include/cuda.h")?;
    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

#[allow(unused)]
fn compute_cap() -> Result<usize> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let mut compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .context("Could not parse code")?
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .context("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.")?;
        let out = std::str::from_utf8(&out.stdout).context("stdout is not a utf8 string")?;
        let mut lines = out.lines();
        assert_eq!(
            lines.next().context("missing line in stdout")?,
            "compute_cap"
        );
        let cap = lines
            .next()
            .context("missing line in stdout")?
            .replace('.', "");
        let cap = cap
            .parse::<usize>()
            .with_context(|| format!("cannot parse as int {cap}"))?;
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
            .arg("--list-gpu-code")
            .output()
            .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).unwrap();

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().unwrap();
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute cap
    if !supported_nvcc_codes.contains(&compute_cap) {
        anyhow::bail!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        anyhow::bail!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
