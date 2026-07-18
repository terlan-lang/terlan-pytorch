use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let root = PathBuf::from(std::env::var_os("LIBTORCH").expect("LIBTORCH must point at the external C distribution"));
    let mut c_build = cc::Build::new();
    c_build.file("c/terlan_pytorch_tensor_from_f64.c");
    c_build.file("c/terlan_pytorch_tensor_from_i64.c");
    c_build.file("c/terlan_pytorch_conv2d.c");
    c_build.include(root.join("include"));
    c_build.include(root.join("include/torch/csrc/api/include"));
    c_build.include(".").warnings_into_errors(true).flag_if_supported("-std=c11").compile("terlan_native_boundary_c_abi");
    let mut cpp_build = cc::Build::new();
    cpp_build.cpp(true);
    cpp_build.file("c/terlan_pytorch_torchscript.cpp");
    cpp_build.include(root.join("include"));
    cpp_build.include(root.join("include/torch/csrc/api/include"));
    cpp_build.include(".").warnings_into_errors(true).flag_if_supported("-std=c++20").compile("terlan_native_boundary_cpp_abi");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=c");
    println!("cargo:rustc-link-search=native={}", root.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(std::ffi::OsStr::new("linux")) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", root.join("lib").display());
    }
}
