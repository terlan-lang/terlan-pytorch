use terlan_pytorch_cxx::ffi;

#[test]
fn threading_configuration_matches_libtorch() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    ffi::init_num_threads_exact().expect("CXX exception");
    let threads = ffi::get_num_threads_exact().expect("CXX exception");
    let interop = ffi::get_num_interop_threads_exact().expect("CXX exception");
    assert!(threads > 0);
    assert!(interop > 0);
    ffi::set_num_threads_exact(threads).expect("CXX exception");
    ffi::set_num_interop_threads_exact(interop).expect("CXX exception");
}
