use terlan_pytorch_cxx::ffi;

#[test]
fn autocast_state_matches_libtorch() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let original_enabled = ffi::autocast_cpu_enabled_exact().expect("CXX exception");
    let original_cache = ffi::autocast_cache_enabled_exact().expect("CXX exception");
    let original_dtype = ffi::autocast_cpu_dtype_exact().expect("CXX exception");
    ffi::set_autocast_cpu_enabled_exact(true).expect("CXX exception");
    assert!(ffi::autocast_cpu_enabled_exact().expect("CXX exception"));
    ffi::set_autocast_cpu_enabled_exact(false).expect("CXX exception");
    assert!(!ffi::autocast_cpu_enabled_exact().expect("CXX exception"));
    ffi::set_autocast_cache_enabled_exact(!original_cache).expect("CXX exception");
    assert_eq!(ffi::autocast_cache_enabled_exact().expect("CXX exception"), !original_cache);
    ffi::set_autocast_cpu_dtype_exact(5).expect("CXX exception");
    assert_eq!(ffi::autocast_cpu_dtype_exact().expect("CXX exception"), 5);
    ffi::set_autocast_cpu_dtype_exact(original_dtype).expect("CXX exception");
    ffi::set_autocast_cache_enabled_exact(original_cache).expect("CXX exception");
    ffi::set_autocast_cpu_enabled_exact(original_enabled).expect("CXX exception");
}
