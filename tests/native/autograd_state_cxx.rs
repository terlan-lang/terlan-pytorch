use terlan_pytorch_cxx::ffi;

#[test]
fn autograd_state_matches_libtorch_global_context() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let original_grad = ffi::grad_mode_enabled_exact().expect("CXX exception");
    let original_anomaly = ffi::anomaly_mode_enabled_exact().expect("CXX exception");
    let original_check_nan = ffi::anomaly_check_nan_enabled_exact().expect("CXX exception");

    ffi::set_grad_mode_enabled_exact(false).expect("CXX exception");
    assert!(!ffi::grad_mode_enabled_exact().expect("CXX exception"));
    ffi::set_grad_mode_enabled_exact(true).expect("CXX exception");
    assert!(ffi::grad_mode_enabled_exact().expect("CXX exception"));

    ffi::set_anomaly_mode_exact(true, false).expect("CXX exception");
    assert!(ffi::anomaly_mode_enabled_exact().expect("CXX exception"));
    assert!(!ffi::anomaly_check_nan_enabled_exact().expect("CXX exception"));
    ffi::set_anomaly_mode_exact(true, true).expect("CXX exception");
    assert!(ffi::anomaly_mode_enabled_exact().expect("CXX exception"));
    assert!(ffi::anomaly_check_nan_enabled_exact().expect("CXX exception"));
    ffi::set_anomaly_mode_exact(false, original_check_nan).expect("CXX exception");
    assert!(!ffi::anomaly_mode_enabled_exact().expect("CXX exception"));

    ffi::set_grad_mode_enabled_exact(original_grad).expect("CXX exception");
    ffi::set_anomaly_mode_exact(original_anomaly, original_check_nan).expect("CXX exception");
}
