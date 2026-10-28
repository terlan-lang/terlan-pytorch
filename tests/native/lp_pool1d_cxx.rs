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

fn lp_pool(
    input: &ffi::Tensor,
    norm_type_is_int: bool,
    norm_type_int: i64,
    norm_type_float: f64,
    kernel_size: i64,
    has_stride: bool,
    stride: i64,
    ceil_mode: bool,
) -> UniquePtr<ffi::Tensor> {
    ffi::terlan_owned_value_pytorch_functionalcompletion_lp_pool1d_configured(
        input,
        norm_type_is_int,
        norm_type_int,
        norm_type_float,
        kernel_size,
        has_stride,
        stride,
        ceil_mode,
    )
}

#[test]
fn lp_pool1d_matches_default_stride_ceil_and_infinite_norm_semantics() {
    let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let default = lp_pool(
        input.as_ref().expect("default input"),
        true,
        2,
        0.0,
        2,
        false,
        2,
        false,
    );
    let default = default.as_ref().expect("default stride result");
    assert_eq!(shape(default), [1, 1, 3]);
    assert_allclose(
        "integer norm/default stride",
        default,
        &[2.23606797749979, 5.0, 7.810249675906654],
        &[1, 1, 3],
    );

    let stride_one = lp_pool(
        input.as_ref().expect("stride-one input"),
        false,
        0,
        2.0,
        2,
        true,
        1,
        false,
    );
    assert_allclose(
        "explicit stride",
        stride_one.as_ref().expect("stride-one result"),
        &[
            2.23606797749979,
            3.605551275463989,
            5.0,
            6.4031242374328485,
            7.810249675906654,
        ],
        &[1, 1, 5],
    );

    let ceil_input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    let ceil = lp_pool(
        ceil_input.as_ref().expect("ceil input"),
        false,
        0,
        2.0,
        3,
        true,
        2,
        true,
    );
    assert_allclose(
        "ceil_mode",
        ceil.as_ref().expect("ceil result"),
        &[3.7416573867739413, 7.0710678118654755],
        &[1, 1, 2],
    );

    let infinity_input = tensor(&[1.0, -4.0, 3.0, -2.0, 5.0, -6.0], &[1, 1, 6]);
    let positive_infinity = lp_pool(
        infinity_input.as_ref().expect("positive infinity input"),
        false,
        0,
        f64::INFINITY,
        2,
        false,
        2,
        false,
    );
    assert_allclose(
        "positive infinite norm",
        positive_infinity
            .as_ref()
            .expect("positive infinity result"),
        &[4.0, 3.0, 6.0],
        &[1, 1, 3],
    );

    let negative_infinity = lp_pool(
        infinity_input.as_ref().expect("negative infinity input"),
        false,
        0,
        f64::NEG_INFINITY,
        2,
        false,
        2,
        false,
    );
    assert_allclose(
        "negative infinite norm",
        negative_infinity
            .as_ref()
            .expect("negative infinity result"),
        &[1.0, 2.0, 5.0],
        &[1, 1, 3],
    );
}
