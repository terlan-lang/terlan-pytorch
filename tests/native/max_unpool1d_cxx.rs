use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn floats(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let cpu_tensor = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(!cpu_tensor.is_null(), "float tensor construction returned null");
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

fn ints(values: &[i64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let cpu_tensor = ffi::terlan_owned_value_pytorch_tensor_from_ints(values, shape);
    assert!(!cpu_tensor.is_null(), "integer tensor construction returned null");
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
    let expected = floats(values, dimensions);
    let result = ffi::terlan_exception_pytorch_typerelationsnative_allclose_native(
        actual,
        expected.as_ref().expect("expected tensor"),
        0.0,
        0.0,
        false,
    );
    let result = result.as_ref().expect("contained allclose result");
    assert!(result.is_ok(), "allclose failed: {}", result.message());
    assert!(result.bool_value(), "{label}: Tensor values differ");
}

fn max_unpool(
    input: &ffi::Tensor,
    indices: &ffi::Tensor,
    kernel_size: i64,
    has_stride: bool,
    stride: i64,
    padding: i64,
    has_output_size: bool,
    output_size: &[i64],
) -> UniquePtr<ffi::Tensor> {
    ffi::terlan_owned_value_pytorch_functionalcompletion_max_unpool1d_configured(
        input,
        indices,
        kernel_size,
        has_stride,
        stride,
        padding,
        has_output_size,
        output_size,
    )
}

#[test]
fn max_unpool1d_matches_default_stride_padding_and_target_size() {
    let input = floats(&[10.0, 20.0], &[1, 1, 2]);
    let default_indices = ints(&[0, 5], &[1, 1, 2]);
    let default = max_unpool(
        input.as_ref().expect("input"),
        default_indices.as_ref().expect("default indices"),
        3,
        false,
        3,
        0,
        false,
        &[],
    );
    let default = default.as_ref().expect("default-stride result");
    assert_eq!(shape(default), [1, 1, 6]);
    assert_allclose(
        "default stride",
        default,
        &[10.0, 0.0, 0.0, 0.0, 0.0, 20.0],
        &[1, 1, 6],
    );

    let padded_indices = ints(&[0, 2], &[1, 1, 2]);
    let padded = max_unpool(
        input.as_ref().expect("input"),
        padded_indices.as_ref().expect("padded indices"),
        3,
        true,
        2,
        1,
        false,
        &[],
    );
    let padded = padded.as_ref().expect("padded result");
    assert_eq!(shape(padded), [1, 1, 3]);
    assert_allclose("padding", padded, &[10.0, 0.0, 20.0], &[1, 1, 3]);

    let sized_indices = ints(&[0, 3], &[1, 1, 2]);
    let sized = max_unpool(
        input.as_ref().expect("input"),
        sized_indices.as_ref().expect("sized indices"),
        3,
        true,
        2,
        0,
        true,
        &[1, 1, 4],
    );
    let sized = sized.as_ref().expect("target-size result");
    assert_eq!(shape(sized), [1, 1, 4]);
    assert_allclose("target output size", sized, &[10.0, 0.0, 0.0, 20.0], &[1, 1, 4]);
}
