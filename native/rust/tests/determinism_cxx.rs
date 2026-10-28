use terlan_pytorch_cxx::ffi;

#[test]
fn deterministic_configuration_matches_libtorch_global_context() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let original_enabled = ffi::deterministic_algorithms_enabled_exact().expect("CXX exception");
    let original_warn_only = ffi::deterministic_algorithms_warn_only_exact().expect("CXX exception");

    ffi::set_deterministic_algorithms_exact(true, false).expect("CXX exception");
    assert!(ffi::deterministic_algorithms_enabled_exact().expect("CXX exception"));
    assert!(!ffi::deterministic_algorithms_warn_only_exact().expect("CXX exception"));
    assert_eq!(ffi::deterministic_debug_mode_exact().expect("CXX exception"), 2);

    ffi::set_deterministic_algorithms_exact(true, true).expect("CXX exception");
    assert!(ffi::deterministic_algorithms_enabled_exact().expect("CXX exception"));
    assert!(ffi::deterministic_algorithms_warn_only_exact().expect("CXX exception"));
    assert_eq!(ffi::deterministic_debug_mode_exact().expect("CXX exception"), 1);

    ffi::set_deterministic_debug_mode_exact(0).expect("CXX exception");
    assert!(!ffi::deterministic_algorithms_enabled_exact().expect("CXX exception"));
    assert!(!ffi::deterministic_algorithms_warn_only_exact().expect("CXX exception"));
    assert_eq!(ffi::deterministic_debug_mode_exact().expect("CXX exception"), 0);

    ffi::set_deterministic_debug_mode_exact(1).expect("CXX exception");
    assert_eq!(ffi::deterministic_debug_mode_exact().expect("CXX exception"), 1);
    ffi::set_deterministic_debug_mode_exact(2).expect("CXX exception");
    assert_eq!(ffi::deterministic_debug_mode_exact().expect("CXX exception"), 2);

    ffi::set_deterministic_algorithms_exact(original_enabled, original_warn_only).expect("CXX exception");
}
