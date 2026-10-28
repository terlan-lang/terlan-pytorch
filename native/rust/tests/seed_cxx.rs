use terlan_pytorch_cxx::ffi;

#[test]
fn generated_seed_matches_libtorch_state() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let reset = ffi::manual_seed_exact(0).expect("CXX exception");
    drop(reset);
    let generated = ffi::seed_exact().expect("CXX exception");
    assert_eq!(generated, ffi::initial_seed_exact().expect("CXX exception"));
    let reset = ffi::manual_seed_exact(0).expect("CXX exception");
    drop(reset);
    assert_eq!(ffi::initial_seed_exact().expect("CXX exception"), 0);
}
