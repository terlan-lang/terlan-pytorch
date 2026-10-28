use std::ptr;

use terlan_pytorch_cxx::{ffi, Tensor};

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn assert_tensor(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(&tensor(values, shape, device)).unwrap());
}

fn packet(value: &Tensor) -> Vec<u8> {
    value.to_device(0, -1).unwrap().to_packet().unwrap()
}

fn verify(device: i64) {
    let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3], device);
    let input_before = packet(&input);

    assert_tensor(
        &input
            .constant_pad_configured(&[1, 1, 1, 1], true, -2, 0.0)
            .unwrap(),
        &[
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 2.0, 3.0, -2.0, -2.0, 4.0, 5.0, 6.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0,
        ],
        &[1, 1, 4, 5],
        device,
    );
    assert_tensor(
        &input
            .constant_pad_configured(&[1, 1, 1, 1], false, 0, 0.5)
            .unwrap(),
        &[
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 0.5, 0.5, 4.0, 5.0, 6.0, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
        ],
        &[1, 1, 4, 5],
        device,
    );

    let mut output = tensor(&[], &[0], device);
    output
        .constant_pad_configured_out(&input, &[1, 1, 1, 1], false, 0, 0.5)
        .unwrap();
    assert_tensor(
        &output,
        &[
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 0.5, 0.5, 4.0, 5.0, 6.0, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
        ],
        &[1, 1, 4, 5],
        device,
    );

    assert_tensor(
        &input.pad_without_value(&[1, 1, 1, 1], "reflect").unwrap(),
        &[
            5.0, 4.0, 5.0, 6.0, 5.0, 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0, 2.0, 1.0,
            2.0, 3.0, 2.0,
        ],
        &[1, 1, 4, 5],
        device,
    );
    assert_tensor(
        &input.pad_without_value(&[1, 1, 1, 1], "replicate").unwrap(),
        &[
            1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 4.0, 4.0,
            5.0, 6.0, 6.0,
        ],
        &[1, 1, 4, 5],
        device,
    );
    assert_tensor(
        &input.pad_without_value(&[1, 1, 1, 1], "circular").unwrap(),
        &[
            6.0, 4.0, 5.0, 6.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.0, 6.0, 4.0, 5.0, 6.0, 4.0, 3.0, 1.0,
            2.0, 3.0, 1.0,
        ],
        &[1, 1, 4, 5],
        device,
    );
    assert_tensor(
        &input
            .pad_with_value(&[1, 1, 1, 1], "constant", 0.5)
            .unwrap(),
        &[
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 0.5, 0.5, 4.0, 5.0, 6.0, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
        ],
        &[1, 1, 4, 5],
        device,
    );

    let cropped_input = tensor(&[1.0, 2.0, 3.0, 4.0], &[4], device);
    assert_tensor(
        &cropped_input
            .constant_pad_configured(&[-1, -1], true, 0, 0.0)
            .unwrap(),
        &[2.0, 3.0],
        &[2],
        device,
    );
    let transposed = input.transpose(2, 3).unwrap();
    assert_tensor(
        &transposed
            .constant_pad_configured(&[1, 0], true, 0, 0.0)
            .unwrap(),
        &[0.0, 1.0, 4.0, 0.0, 2.0, 5.0, 0.0, 3.0, 6.0],
        &[1, 1, 3, 3],
        device,
    );

    let output_before = packet(&output);
    assert!(output
        .constant_pad_configured_out(&input, &[1], true, 0, 0.0)
        .is_err());
    assert_eq!(packet(&output), output_before);
    assert!(input.pad_without_value(&[1, 1], "unknown").is_err());

    let leaf = input.clone().unwrap().require_grad().unwrap();
    leaf.constant_pad_configured(&[1, 1, 1, 1], false, 0, 0.5)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_tensor(
        &leaf.gradient().unwrap(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[1, 1, 2, 3],
        device,
    );
    assert_eq!(packet(&input), input_before);
}

#[test]
fn cpu_padding_values_storage_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_padding_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}

#[test]
fn padding_rejects_null_boundary_handles() {
    let mut output = ptr::null_mut();
    let status = unsafe {
        ffi::terlan_pytorch_constant_pad(ptr::null_mut(), ptr::null(), 0, true, 0, 0.0, &mut output)
    };
    assert_eq!(status, 7280);
    assert!(output.is_null());
}
