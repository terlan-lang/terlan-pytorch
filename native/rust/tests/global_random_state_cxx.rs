use terlan_pytorch_cxx::ffi;

#[test]
fn global_random_state_matches_libtorch() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let generator = ffi::manual_seed_exact(123456).expect("CXX exception");
    assert_eq!(ffi::initial_seed_exact().expect("CXX exception"), 123456);
    drop(generator);
    let restored = ffi::manual_seed_exact(0).expect("CXX exception");
    drop(restored);
    assert_eq!(ffi::initial_seed_exact().expect("CXX exception"), 0);
}
