use terlan_pytorch_cxx::ffi;

#[test]
fn libtorch_exceptions_cross_the_cxx_boundary_as_errors() {
    let result = ffi::tensor_load_torchscript(
        "/terlan-pytorch-tests/this-torchscript-model-does-not-exist.pt",
    );

    match result {
        Ok(_) => panic!("loading a missing TorchScript model unexpectedly succeeded"),
        Err(error) => assert!(
            !error.what().is_empty(),
            "the translated LibTorch exception must retain a diagnostic",
        ),
    }
}
