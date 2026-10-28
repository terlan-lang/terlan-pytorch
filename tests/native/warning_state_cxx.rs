use terlan_pytorch_cxx::ffi;

#[test]
fn warning_state_matches_libtorch_global_context() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let original = ffi::warn_always_enabled_exact().expect("CXX exception");
    ffi::set_warn_always_exact(true).expect("CXX exception");
    assert!(ffi::warn_always_enabled_exact().expect("CXX exception"));
    ffi::set_warn_always_exact(false).expect("CXX exception");
    assert!(!ffi::warn_always_enabled_exact().expect("CXX exception"));
    ffi::set_warn_always_exact(original).expect("CXX exception");
}
