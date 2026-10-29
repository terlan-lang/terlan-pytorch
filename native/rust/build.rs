fn main() {
    let root = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let mut build = cxx_build::bridge(root.join("src/lib.rs"));
    build.include(&root);
    build.file(root.join("cpp/libtorch_metadata.cc"));
    build.file(root.join("cpp/terlan_tensor_packet.cc"));
    build.file(root.join("cpp/cuda_stream.cc"));
    build.file(root.join("cpp/optimizer.cc"));
    build.file(root.join("cpp/tensor_compat.cc"));
    build.file(root.join("cpp/tensor_ops.cc"));
    build.file(root.join("cpp/tensor_advanced.cc"));
    build.file(root.join("cpp/torchscript_compat.cc"));
    build.file(root.join("cpp/terlan_enum_adapters.cc"));
    build.file(root.join("cpp/terlan_string_adapters.cc"));
    build.file(root.join("cpp/terlan_collection_adapters.cc"));
    build.file(root.join("cpp/terlan_exception_adapters.cc"));
    build.file(root.join("cpp/terlan_owned_value_adapters.cc"));
    build.file(root.join("cpp/terlan_mutation_adapters.cc"));
    build.include(root.join("include"));
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let external_root_0 = std::path::PathBuf::from(std::env::var_os("LIBTORCH").expect("LIBTORCH must point at the external C++ SDK root"));
    build.include(external_root_0.join("include"));
    build.include(external_root_0.join("include/torch/csrc/api/include"));
    println!("cargo:rustc-link-search=native={}", external_root_0.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    println!("cargo:rustc-link-lib=dylib=shm");
    let cuda_library = external_root_0.join("lib/libtorch_cuda.so").exists();
    let cuda_include_candidates = [
        external_root_0.join("include"),
        std::path::PathBuf::from("/usr/local/cuda/include"),
        std::path::PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include"),
    ];
    let cuda_include = cuda_include_candidates.into_iter().find(|path| {
        path.join("cuda_runtime_api.h").exists() && path.join("cuda.h").exists()
    });
    if cuda_library {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
        if let Some(include) = cuda_include {
            build.define("TERLAN_LIBTORCH_CUDA", None);
            build.include(include);
        } else {
            println!(
                "cargo:warning=CUDA LibTorch is present but CUDA developer headers are unavailable; compiling the explicit CPU-safe stream fallback"
            );
        }
    }
    // Keep the installed helper relocatable.  Development invocations should
    // provide LIBTORCH/lib through the bounded runner environment; embedding
    // the caller's absolute SDK path would make artifacts machine-specific.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../cuda-runtime/usr/lib/x86_64-linux-gnu/libcusparseLt/12");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../cuda-runtime/usr/lib/x86_64-linux-gnu/nvshmem/12");
    println!("cargo:rerun-if-changed={}", external_root_0.join("build-version").display());
    println!("cargo:rerun-if-changed={}", external_root_0.join("build-hash").display());
    build.std("c++17").compile("terlan_native_boundary_cxx");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=cpp");
}
