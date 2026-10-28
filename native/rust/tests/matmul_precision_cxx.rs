use terlan_pytorch_cxx::ffi;

#[test]
fn float32_matmul_precision_matches_libtorch_global_context() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let original = ffi::float32_matmul_precision_exact().expect("CXX exception");
    let original = original.as_ref().expect("original precision").to_str().expect("UTF-8 precision");
    let original_code = match original {
        "highest" => 0,
        "high" => 1,
        _ => 2,
    };
    ffi::set_float32_matmul_precision_code_exact(0).expect("CXX exception");
    assert_eq!(ffi::float32_matmul_precision_exact().expect("CXX exception").as_ref().expect("precision result").to_str().expect("UTF-8 precision"), "highest");
    ffi::set_float32_matmul_precision_code_exact(1).expect("CXX exception");
    assert_eq!(ffi::float32_matmul_precision_exact().expect("CXX exception").as_ref().expect("precision result").to_str().expect("UTF-8 precision"), "high");
    ffi::set_float32_matmul_precision_code_exact(2).expect("CXX exception");
    assert_eq!(ffi::float32_matmul_precision_exact().expect("CXX exception").as_ref().expect("precision result").to_str().expect("UTF-8 precision"), "medium");
    ffi::set_float32_matmul_precision_code_exact(original_code).expect("CXX exception");
}
