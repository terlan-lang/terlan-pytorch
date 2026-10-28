use terlan_pytorch_cxx::ffi;

#[test]
fn inference_mode_query_matches_libtorch() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    assert!(!ffi::inference_mode_enabled_exact().expect("CXX exception"));
    assert!(ffi::inference_mode_transition_probe_exact().expect("CXX exception"));
    assert!(!ffi::inference_mode_enabled_exact().expect("CXX exception"));
}
