use std::ptr;

use terlan_pytorch_cxx::{ffi, Tensor};

fn float_tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn int_tensor(values: &[i64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_ints(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn assert_float(actual: &Tensor, expected: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .equal_native(&float_tensor(expected, shape, device))
        .unwrap());
}

fn assert_same(actual: &Tensor, expected: &Tensor, device: i64) {
    let actual = if device == 0 {
        actual.clone().unwrap()
    } else {
        actual.to_device(0, -1).unwrap()
    };
    let expected = if device == 0 {
        expected.clone().unwrap()
    } else {
        expected.to_device(0, -1).unwrap()
    };
    assert_eq!(actual.to_packet().unwrap(), expected.to_packet().unwrap());
}

fn verify(device: i64) {
    let source = float_tensor(
        &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        &[4],
        device,
    );
    let source_before = source.to_device(0, -1).unwrap().to_packet().unwrap();
    let configurations = [
        (
            false,
            0.0,
            false,
            0.0,
            false,
            0.0,
            [0.0, f64::MAX, f64::MIN, 2.0],
        ),
        (
            true,
            -1.0,
            false,
            0.0,
            false,
            0.0,
            [-1.0, f64::MAX, f64::MIN, 2.0],
        ),
        (
            false,
            0.0,
            true,
            10.0,
            false,
            0.0,
            [0.0, 10.0, f64::MIN, 2.0],
        ),
        (
            false,
            0.0,
            false,
            0.0,
            true,
            -10.0,
            [0.0, f64::MAX, -10.0, 2.0],
        ),
        (
            true,
            -1.0,
            true,
            10.0,
            false,
            0.0,
            [-1.0, 10.0, f64::MIN, 2.0],
        ),
        (
            true,
            -1.0,
            false,
            0.0,
            true,
            -10.0,
            [-1.0, f64::MAX, -10.0, 2.0],
        ),
        (false, 0.0, true, 10.0, true, -10.0, [0.0, 10.0, -10.0, 2.0]),
        (
            true,
            -1.0,
            true,
            10.0,
            true,
            -10.0,
            [-1.0, 10.0, -10.0, 2.0],
        ),
    ];

    for (has_nan, nan, has_posinf, posinf, has_neginf, neginf, expected) in configurations {
        let functional = source
            .nan_to_num_configured(has_nan, nan, has_posinf, posinf, has_neginf, neginf)
            .unwrap();
        assert_float(&functional, &expected, &[4], device);

        let mut output = float_tensor(&[], &[0], device);
        output
            .nan_to_num_configured_out(
                &source, has_nan, nan, has_posinf, posinf, has_neginf, neginf,
            )
            .unwrap();
        assert_float(&output, &expected, &[4], device);

        let mut mutation = source.clone().unwrap();
        mutation
            .nan_to_num_configured_in_place(has_nan, nan, has_posinf, posinf, has_neginf, neginf)
            .unwrap();
        assert_float(&mutation, &expected, &[4], device);
    }

    assert_eq!(
        source.to_device(0, -1).unwrap().to_packet().unwrap(),
        source_before,
        "functional and output forms mutated source storage"
    );

    let matrix = float_tensor(
        &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        &[2, 2],
        device,
    );
    let transposed = matrix.transpose(0, 1).unwrap();
    assert_float(
        &transposed
            .nan_to_num_configured(true, -1.0, true, 10.0, true, -10.0)
            .unwrap(),
        &[-1.0, -10.0, 10.0, 2.0],
        &[2, 2],
        device,
    );

    let integers = int_tensor(&[-2, 0, 3], &[3], device);
    assert_same(
        &integers
            .nan_to_num_configured(false, 0.0, false, 0.0, false, 0.0)
            .unwrap(),
        &integers,
        device,
    );

    let real = float_tensor(
        &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        &[4],
        device,
    );
    let imaginary = float_tensor(
        &[f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 3.0],
        &[4],
        device,
    );
    let complex = real.complex(&imaginary).unwrap();
    let replaced_complex = complex
        .nan_to_num_configured(true, -1.0, true, 10.0, true, -10.0)
        .unwrap();
    assert_float(
        &replaced_complex.real().unwrap(),
        &[-1.0, 10.0, -10.0, 2.0],
        &[4],
        device,
    );
    assert_float(
        &replaced_complex.imag().unwrap(),
        &[10.0, -1.0, -10.0, 3.0],
        &[4],
        device,
    );

    let leaf = source.clone().unwrap().require_grad().unwrap();
    leaf.nan_to_num_configured(true, -1.0, true, 10.0, true, -10.0)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_float(
        &leaf.gradient().unwrap(),
        &[0.0, 0.0, 0.0, 1.0],
        &[4],
        device,
    );

    let mut wrong_dtype = int_tensor(&[42], &[1], device);
    assert!(wrong_dtype
        .nan_to_num_configured_out(&source, true, -1.0, true, 10.0, true, -10.0,)
        .is_err());
    assert_same(&wrong_dtype, &int_tensor(&[42], &[1], device), device);

    let trainable = source.clone().unwrap().require_grad().unwrap();
    let trainable_before = trainable.to_device(0, -1).unwrap().to_packet().unwrap();
    let mut trainable = trainable;
    assert!(trainable
        .nan_to_num_configured_in_place(true, -1.0, true, 10.0, true, -10.0)
        .is_err());
    assert_eq!(
        trainable.to_device(0, -1).unwrap().to_packet().unwrap(),
        trainable_before
    );
}

#[test]
fn cpu_nan_to_num_values_options_storage_dtypes_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_nan_to_num_values_options_storage_dtypes_errors_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}

#[test]
fn nan_to_num_rejects_null_boundary_handles() {
    let mut output = ptr::null_mut();
    let status = unsafe {
        ffi::terlan_pytorch_nan_to_num(
            ptr::null_mut(),
            false,
            0.0,
            false,
            0.0,
            false,
            0.0,
            &mut output,
        )
    };
    assert_eq!(status, 7260);
    assert!(output.is_null());
}
