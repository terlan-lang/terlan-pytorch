use terlan_pytorch_cxx::ffi;

#[test]
fn default_dtype_state_matches_libtorch() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let original = ffi::default_dtype_exact().expect("CXX exception");
    ffi::set_default_dtype_exact(6).expect("CXX exception");
    assert_eq!(ffi::default_dtype_exact().expect("CXX exception"), 6);
    ffi::set_default_dtype_exact(original).expect("CXX exception");
    assert_eq!(ffi::default_dtype_exact().expect("CXX exception"), original);
}
