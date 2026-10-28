use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn tensor(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let cpu_tensor = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(
        !cpu_tensor.is_null(),
        "LibTorch tensor construction returned null"
    );
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        ffi::terlan_owned_value_pytorch_tensor_to_device(
            cpu_tensor.as_ref().expect("constructed CPU tensor"),
            1,
            0,
        )
    } else {
        cpu_tensor
    }
}

fn shape(tensor: &ffi::Tensor) -> Vec<i64> {
    ffi::terlan_collection_pytorch_tensor_shape(tensor)
        .as_ref()
        .expect("tensor shape projection")
        .iter()
        .copied()
        .collect()
}

fn assert_allclose(label: &str, actual: &ffi::Tensor, values: &[f64], dimensions: &[i64]) {
    let expected = tensor(values, dimensions);
    let result = ffi::terlan_exception_pytorch_typerelationsnative_allclose_native(
        actual,
        expected.as_ref().expect("expected tensor"),
        1e-6,
        1e-6,
        false,
    );
    let result = result.as_ref().expect("contained allclose result");
    assert!(result.is_ok(), "allclose failed: {}", result.message());
    assert!(result.bool_value(), "{label}: Tensor values differ");
}

fn response_norm(
    input: &ffi::Tensor,
    size: i64,
    alpha: f64,
    beta: f64,
    k: f64,
) -> UniquePtr<ffi::Tensor> {
    ffi::terlan_owned_value_pytorch_functionalcompletion_local_response_norm_configured(
        input, size, alpha, beta, k,
    )
}

#[test]
fn local_response_norm_matches_channel_window_and_empty_semantics() {
    let input4d = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
    let output4d = response_norm(input4d.as_ref().expect("rank-four input"), 3, 1.0, 1.0, 1.0);
    let output4d = output4d.as_ref().expect("rank-four result");
    assert_eq!(shape(output4d), [1, 3, 1, 1]);
    assert_allclose(
        "configured rank-four",
        output4d,
        &[0.375, 6.0 / 17.0, 0.5625],
        &[1, 3, 1, 1],
    );

    let default_output = response_norm(
        input4d.as_ref().expect("rank-four input"),
        3,
        0.0001,
        0.75,
        1.0,
    );
    let default_output = default_output.as_ref().expect("default result");
    assert_allclose(
        "default rank-four",
        default_output,
        &[0.999875018226382, 1.999300285711114, 2.999025369540712],
        &[1, 3, 1, 1],
    );

    let input3d = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1]);
    let output3d = response_norm(
        input3d.as_ref().expect("rank-three input"),
        3,
        1.0,
        1.0,
        1.0,
    );
    let output3d = output3d.as_ref().expect("rank-three result");
    assert_eq!(shape(output3d), [1, 3, 1]);
    assert_allclose(
        "configured rank-three",
        output3d,
        &[0.375, 6.0 / 17.0, 0.5625],
        &[1, 3, 1],
    );

    let empty_input = tensor(&[], &[1, 3, 1, 0]);
    let empty_output = response_norm(
        empty_input.as_ref().expect("empty input"),
        3,
        0.0001,
        0.75,
        1.0,
    );
    assert!(!empty_output.is_null(), "empty input returned null");
    assert_eq!(
        shape(empty_output.as_ref().expect("empty result")),
        [1, 3, 1, 0]
    );
}
