use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn tensor(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let cpu = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(!cpu.is_null(), "tensor construction returned null");
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        ffi::terlan_owned_value_pytorch_tensor_to_device(
            cpu.as_ref().expect("constructed tensor"),
            1,
            0,
        )
    } else {
        cpu
    }
}

fn assert_allclose(label: &str, actual: &ffi::Tensor, expected: &ffi::Tensor) {
    let result = ffi::terlan_exception_pytorch_typerelationsnative_allclose_native(
        actual, expected, 1e-6, 1e-6, false,
    );
    let result = result.as_ref().expect("contained allclose result");
    assert!(result.is_ok(), "allclose failed: {}", result.message());
    assert!(result.bool_value(), "{label}: Tensor values differ");
}

fn gaussian(
    input: &ffi::Tensor,
    target: &ffi::Tensor,
    var_is_tensor: bool,
    variance: &ffi::Tensor,
    scalar: f64,
    full: bool,
    reduction: &str,
) -> UniquePtr<ffi::Tensor> {
    ffi::terlan_owned_value_pytorch_functionalcompletion_gaussian_nll_loss_configured(
        input,
        target,
        var_is_tensor,
        variance,
        scalar,
        full,
        1e-6,
        reduction,
    )
}

#[test]
fn gaussian_nll_loss_matches_tensor_scalar_and_broadcast_variance() {
    let input = tensor(&[1.0, 2.0, 4.0, 8.0], &[4]);
    let target = tensor(&[0.0, 1.0, 5.0, 7.0], &[4]);
    let variance = tensor(&[1.0, 2.0, 4.0, 8.0], &[4]);
    let tensor_result = gaussian(
        input.as_ref().expect("input"),
        target.as_ref().expect("target"),
        true,
        variance.as_ref().expect("variance"),
        0.0,
        false,
        "none",
    );
    assert_allclose(
        "tensor variance",
        tensor_result.as_ref().expect("tensor result"),
        tensor(&[0.5, 0.5965735912322998, 0.8181471824645996, 1.1022207736968994], &[4])
            .as_ref()
            .expect("tensor expected"),
    );

    let scalar_result = gaussian(
        input.as_ref().expect("input"),
        target.as_ref().expect("target"),
        false,
        input.as_ref().expect("scalar placeholder"),
        2.0,
        true,
        "mean",
    );
    assert_allclose(
        "scalar variance/full mean",
        scalar_result.as_ref().expect("scalar result"),
        tensor(&[1.5155121088027954], &[1])
            .as_ref()
            .expect("scalar expected"),
    );

    let broadcast_input = tensor(&[1.0, 2.0, 4.0, 2.0, 4.0, 8.0], &[2, 3]);
    let broadcast_target = tensor(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);
    let broadcast_variance = tensor(&[1.0, 2.0], &[2]);
    let broadcast = gaussian(
        broadcast_input.as_ref().expect("broadcast input"),
        broadcast_target.as_ref().expect("broadcast target"),
        true,
        broadcast_variance.as_ref().expect("broadcast variance"),
        0.0,
        false,
        "none",
    );
    assert_allclose(
        "one-short broadcast variance",
        broadcast.as_ref().expect("broadcast result"),
        tensor(&[0.5, 2.0, 8.0, 1.3465735912322998, 4.3465735912323, 16.346572875976562], &[2, 3])
            .as_ref()
            .expect("broadcast expected"),
    );
}
