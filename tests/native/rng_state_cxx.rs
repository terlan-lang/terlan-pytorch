use terlan_pytorch_cxx::ffi;

#[test]
fn rng_state_roundtrips_through_libtorch() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let state = ffi::terlan_owned_value_pytorch_randomcompletion_get_rng_state();
    ffi::set_rng_state_exact(state.as_ref().expect("RNG state tensor")).expect("CXX exception");
}
