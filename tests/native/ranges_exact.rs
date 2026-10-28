use std::ptr;

use terlan_pytorch_cxx::{
    arange_exact_native, eye_exact_native, ffi, range_exact_native, spacing_scalar_exact_native,
    spacing_scalar_tensor_exact_native, spacing_tensor_scalar_exact_native,
    spacing_tensor_tensor_exact_native, CAbiError, Tensor,
};

/// Returns the native optional-argument suffix with every value absent.
fn inferred_options() -> (i64, bool, i64, bool, i64, i64, bool, bool, bool) {
    (6, false, 0, false, 0, -1, false, false, false)
}

/// Creates one exact arange call with concise integer scalar inputs.
fn integer_arange(form: i64, start: i64, end: i64, step: i64) -> Tensor {
    let (dtype, has_dtype, layout, has_layout, device_type, device_index, has_device, pin, has_pin) =
        inferred_options();
    arange_exact_native(
        form,
        true,
        start,
        0.0,
        true,
        end,
        0.0,
        true,
        step,
        0.0,
        dtype,
        has_dtype,
        layout,
        has_layout,
        device_type,
        device_index,
        has_device,
        pin,
        has_pin,
    )
    .expect("construct integer arange")
}

/// Creates one scalar-endpoint spacing tensor with explicit float64 options.
fn scalar_spacing(operation: i64, start: f64, end: f64, steps: i64, base: f64) -> Tensor {
    spacing_scalar_exact_native(
        operation, false, 0, start, false, 0, end, steps, base, 7, true, 0, true, 0, -1, true,
        false, true,
    )
    .expect("construct scalar spacing")
}

/// Builds an independently owned float64 expected tensor.
fn expected_f64(values: &[f64]) -> Tensor {
    Tensor::from_floats(values, &[values.len() as i64]).expect("construct expected tensor")
}

/// Asserts exact shape, dtype, and values against an independent tensor.
fn assert_exact(actual: &Tensor, values: &[f64], dtype: i64) {
    let expected = expected_f64(values)
        .to_dtype(dtype)
        .expect("convert expected dtype");
    assert_eq!(actual.shape().unwrap(), vec![values.len() as i64]);
    assert_eq!(actual.dtype().unwrap(), dtype);
    assert!(actual
        .equal_native(&expected)
        .expect("compare exact values"));
}

/// Extracts an expected native error without accepting accidental success.
fn native_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}: unexpectedly succeeded"),
        Err(error) => error,
    }
}

#[test]
fn arange_forms_preserve_scalar_inference_and_retained_output() {
    let end_only = integer_arange(0, 99, 4, 99);
    let bounded = integer_arange(1, 1, 5, 99);
    let stepped = integer_arange(2, 1, 7, 2);
    assert_exact(&end_only, &[0.0, 1.0, 2.0, 3.0], 4);
    assert_exact(&bounded, &[1.0, 2.0, 3.0, 4.0], 4);
    assert_exact(&stepped, &[1.0, 3.0, 5.0], 4);

    let mut output = Tensor::from_ints(&[], &[0]).expect("construct empty output");
    output
        .arange_out_exact_native(2, true, 5, 0.0, true, 0, 0.0, true, -2, 0.0)
        .expect("write descending arange");
    assert_exact(&output, &[5.0, 3.0, 1.0], 4);

    let floating = arange_exact_native(
        2, false, 0, 0.5, false, 0, 2.0, false, 0, 0.5, 6, false, 0, false, 0, -1, false, false,
        false,
    )
    .expect("construct inferred floating arange");
    assert_exact(&floating, &[0.5, 1.0, 1.5], 6);
}

#[test]
fn eye_forms_preserve_shape_options_and_retained_output() {
    let square = eye_exact_native(3, 99, false, 6, false, 0, false, 0, -1, false, false, false)
        .expect("construct square eye");
    assert_eq!(square.shape().unwrap(), vec![3, 3]);
    assert_eq!(square.dtype().unwrap(), 6);

    let rectangular = eye_exact_native(2, 3, true, 7, true, 0, true, 0, -1, true, false, true)
        .expect("construct rectangular eye");
    let expected = Tensor::from_floats(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3])
        .expect("construct expected identity");
    assert!(rectangular.equal_native(&expected).unwrap());

    let mut output = Tensor::from_floats(&[], &[0]).expect("construct empty output");
    output
        .eye_out_exact_native(2, 3, true)
        .expect("write rectangular eye");
    assert!(output.equal_native(&expected).unwrap());
}

#[test]
fn spacing_endpoint_overloads_match_scalar_reference_and_outputs() {
    let scalar_linear = scalar_spacing(0, 0.0, 1.0, 5, 10.0);
    assert_exact(&scalar_linear, &[0.0, 0.25, 0.5, 0.75, 1.0], 7);
    let scalar_log = scalar_spacing(1, 0.0, 3.0, 4, 2.0);
    assert_exact(&scalar_log, &[1.0, 2.0, 4.0, 8.0], 7);

    let start = Tensor::from_float(0.0).expect("construct tensor start");
    let end = Tensor::from_float(1.0).expect("construct tensor end");
    let tensor_tensor = spacing_tensor_tensor_exact_native(
        0, &start, &end, 5, 10.0, 7, true, 0, true, 0, -1, true, false, true,
    )
    .expect("construct tensor-tensor spacing");
    let tensor_scalar = spacing_tensor_scalar_exact_native(
        0, &start, false, 0, 1.0, 5, 10.0, 7, true, 0, true, 0, -1, true, false, true,
    )
    .expect("construct tensor-scalar spacing");
    let scalar_tensor = spacing_scalar_tensor_exact_native(
        0, false, 0, 0.0, &end, 5, 10.0, 7, true, 0, true, 0, -1, true, false, true,
    )
    .expect("construct scalar-tensor spacing");
    for actual in [&tensor_tensor, &tensor_scalar, &scalar_tensor] {
        assert!(actual.equal_native(&scalar_linear).unwrap());
    }

    let mut scalar_output = Tensor::from_floats(&[], &[0]).expect("construct scalar output");
    scalar_output
        .spacing_scalar_out_exact_native(0, false, 0, 0.0, false, 0, 1.0, 5, 10.0)
        .expect("write scalar spacing");
    let mut tensor_output = Tensor::from_floats(&[], &[0]).expect("construct tensor output");
    tensor_output
        .spacing_tensor_tensor_out_exact_native(0, &start, &end, 5, 10.0)
        .expect("write tensor spacing");
    let mut mixed_output = Tensor::from_floats(&[], &[0]).expect("construct mixed output");
    mixed_output
        .spacing_tensor_scalar_out_exact_native(0, &start, false, 0, 1.0, 5, 10.0)
        .expect("write tensor-scalar spacing");
    let mut reverse_output = Tensor::from_floats(&[], &[0]).expect("construct reverse output");
    reverse_output
        .spacing_scalar_tensor_out_exact_native(0, false, 0, 0.0, &end, 5, 10.0)
        .expect("write scalar-tensor spacing");
    for actual in [
        &scalar_output,
        &tensor_output,
        &mixed_output,
        &reverse_output,
    ] {
        assert!(actual.equal_native(&scalar_linear).unwrap());
    }
}

#[test]
fn inclusive_range_forms_preserve_endpoints_and_output_dtype() {
    let unit = range_exact_native(
        false, true, 1, 0.0, true, 4, 0.0, true, 99, 0.0, 6, false, 0, false, 0, -1, false, false,
        false,
    )
    .expect("construct inclusive unit range");
    let stepped = range_exact_native(
        true, true, 5, 0.0, true, 1, 0.0, true, -2, 0.0, 6, false, 0, false, 0, -1, false, false,
        false,
    )
    .expect("construct inclusive stepped range");
    assert_exact(&unit, &[1.0, 2.0, 3.0, 4.0], 6);
    assert_exact(&stepped, &[5.0, 3.0, 1.0], 6);

    let mut output = Tensor::from_ints(&[], &[0]).expect("construct range output");
    output
        .range_out_exact_native(true, true, 2, 0.0, true, 6, 0.0, true, 2, 0.0)
        .expect("write inclusive range");
    assert_exact(&output, &[2.0, 4.0, 6.0], 4);
}

#[test]
fn malformed_exact_range_requests_fail_without_publishing_outputs() {
    let bad_form = arange_exact_native(
        3, true, 0, 0.0, true, 4, 0.0, true, 1, 0.0, 6, false, 0, false, 0, -1, false, false, false,
    );
    assert_eq!(native_error(bad_form, "invalid arange form").status, 7090);
    let bad_layout = eye_exact_native(2, 2, true, 6, true, 1, true, 0, -1, true, false, true);
    assert_eq!(native_error(bad_layout, "invalid layout").status, 7090);
    let bad_spacing = spacing_scalar_exact_native(
        2, true, 0, 0.0, true, 1, 0.0, 2, 10.0, 6, false, 0, false, 0, -1, false, false, false,
    );
    assert_eq!(
        native_error(bad_spacing, "invalid spacing operation").status,
        7090
    );
    let negative_steps = spacing_scalar_exact_native(
        0, true, 0, 0.0, true, 1, 0.0, -1, 10.0, 6, false, 0, false, 0, -1, false, false, false,
    );
    assert_eq!(native_error(negative_steps, "negative steps").status, 7092);

    let statuses = unsafe {
        [
            ffi::terlan_pytorch_eye_exact(
                2,
                2,
                true,
                6,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
                ptr::null_mut(),
            ),
            ffi::terlan_pytorch_eye_out_exact(ptr::null_mut(), 2, 2, true),
        ]
    };
    assert_eq!(statuses, [7090; 2]);
}

#[test]
fn exact_range_families_execute_on_cuda_when_required() {
    let progression = arange_exact_native(
        2, true, 0, 0.0, true, 4, 0.0, true, 1, 0.0, 6, true, 0, true, 1, 0, true, false, true,
    );
    match progression {
        Ok(progression) => {
            assert_eq!(progression.device_type().unwrap(), 1);
            let identity = eye_exact_native(2, 2, true, 6, true, 0, true, 1, 0, true, false, true)
                .expect("construct CUDA identity");
            let linear = spacing_scalar_exact_native(
                0, false, 0, 0.0, false, 0, 1.0, 3, 10.0, 6, true, 0, true, 1, 0, true, false, true,
            )
            .expect("construct CUDA linear spacing");
            assert_eq!(identity.device_type().unwrap(), 1);
            assert_eq!(linear.device_type().unwrap(), 1);
            assert_exact(
                &progression.to_device(0, -1).unwrap(),
                &[0.0, 1.0, 2.0, 3.0],
                6,
            );
        }
        Err(error) => {
            if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
                panic!("required CUDA range execution failed: {error:?}");
            }
        }
    }
}
