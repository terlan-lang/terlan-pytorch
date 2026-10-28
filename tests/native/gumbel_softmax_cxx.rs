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

#[test]
fn gumbel_softmax_soft_and_hard_outputs_are_normalized() {
    let logits = tensor(&[1.0, 2.0, 3.0, 0.5, -1.0, 2.5], &[2, 3]);
    let expected = tensor(&[1.0, 1.0], &[2]);
    let soft = ffi::terlan_owned_value_pytorch_functionalcompletion_gumbel_softmax_configured(
        logits.as_ref().expect("logits"),
        1.0,
        false,
        1e-10,
        -1,
    );
    let hard = ffi::terlan_owned_value_pytorch_functionalcompletion_gumbel_softmax_configured(
        logits.as_ref().expect("logits"),
        1.0,
        true,
        1e-10,
        -1,
    );
    let soft_rows = ffi::terlan_owned_value_pytorch_tensor_sum_dimensions(
        soft.as_ref().expect("soft output"),
        &[1],
        false,
    );
    let hard_rows = ffi::terlan_owned_value_pytorch_tensor_sum_dimensions(
        hard.as_ref().expect("hard output"),
        &[1],
        false,
    );
    assert_allclose(
        "soft gumbel rows",
        soft_rows.as_ref().expect("soft row sums"),
        expected.as_ref().expect("expected row sums"),
    );
    assert_allclose(
        "hard gumbel rows",
        hard_rows.as_ref().expect("hard row sums"),
        expected.as_ref().expect("expected row sums"),
    );
}
