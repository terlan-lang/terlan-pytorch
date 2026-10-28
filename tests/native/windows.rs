use std::ptr;

use terlan_pytorch_cxx::{ffi, window_native, CAbiError, Tensor};

/// Creates one window with explicit operation and placement parameters.
fn window(
    operation: i64,
    window_length: i64,
    periodic: bool,
    alpha: f64,
    beta: f64,
    dtype: i64,
    device_type: i64,
    device_index: i64,
) -> Tensor {
    window_native(
        operation,
        window_length,
        periodic,
        alpha,
        beta,
        dtype,
        device_type,
        device_index,
        false,
    )
    .expect("construct window")
}

/// Builds one independently owned expected tensor on the selected device.
fn expected(values: &[f64], device_type: i64, device_index: i64) -> Tensor {
    Tensor::from_floats(values, &[values.len() as i64])
        .expect("construct expected values")
        .to_dtype(6)
        .expect("convert expected dtype")
        .to_device(device_type, device_index)
        .expect("place expected values")
}

/// Asserts approximate equality through LibTorch's independent allclose schema.
fn assert_window(actual: &Tensor, wanted: &Tensor) {
    assert!(actual
        .allclose_native(wanted, 1e-5, 1e-6, false)
        .expect("compare window values"));
}

/// Extracts one expected native failure without accepting a successful tensor.
fn window_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}: unexpectedly succeeded"),
        Err(error) => error,
    }
}

#[test]
fn cpu_windows_match_pinned_libtorch_values_and_output_semantics() {
    let cases = [
        (0, 0.0, 0.0, vec![0.0, 0.5, 1.0, 0.5, 0.0]),
        (1, 0.0, 0.0, vec![0.0, 0.34, 1.0, 0.34, 0.0]),
        (2, 0.54, 0.46, vec![0.08, 0.54, 1.0, 0.54, 0.08]),
        (3, 0.0, 0.0, vec![0.0, 0.5, 1.0, 0.5, 0.0]),
        (4, 0.0, 0.0, vec![1.0, 1.0, 1.0, 1.0, 1.0]),
    ];
    for (operation, alpha, beta, values) in cases {
        let actual = window(operation, 5, false, alpha, beta, 6, 0, -1);
        let wanted = expected(&values, 0, -1);
        assert_window(&actual, &wanted);
        assert_eq!(actual.shape().unwrap(), vec![5]);
        assert_eq!(actual.dtype().unwrap(), 6);

        let mut output = Tensor::from_floats(&[], &[0])
            .expect("construct empty output")
            .to_dtype(6)
            .expect("convert output dtype");
        output
            .window_out_native(operation, 5, false, alpha, beta)
            .expect("write retained output");
        assert_window(&output, &wanted);
        assert_eq!(output.shape().unwrap(), vec![5]);
    }
}

#[test]
fn window_edge_lengths_and_custom_coefficients_match_libtorch() {
    for length in [0, 1] {
        for operation in 0..=4 {
            let actual = window(operation, length, true, 0.54, 12.0, 7, 0, -1);
            assert_eq!(actual.shape().unwrap(), vec![length]);
            assert_eq!(actual.dtype().unwrap(), 7);
        }
    }

    let hamming = window(2, 5, false, 1.0, 0.0, 6, 0, -1);
    let kaiser = window(4, 5, false, 0.0, 0.0, 6, 0, -1);
    let ones = expected(&[1.0; 5], 0, -1);
    assert_window(&hamming, &ones);
    assert_window(&kaiser, &ones);
}

#[test]
fn malformed_window_requests_fail_without_publishing_outputs() {
    let invalid = [
        window_native(-1, 5, true, 0.0, 0.0, 6, 0, -1, false),
        window_native(5, 5, true, 0.0, 0.0, 6, 0, -1, false),
        window_native(0, -1, true, 0.0, 0.0, 6, 0, -1, false),
        window_native(0, 5, true, f64::NAN, 0.0, 6, 0, -1, false),
        window_native(0, 5, true, 0.0, f64::INFINITY, 6, 0, -1, false),
        window_native(0, 5, true, 0.0, 0.0, -1, 0, -1, false),
        window_native(0, 5, true, 0.0, 0.0, 46, 0, -1, false),
        window_native(0, 5, true, 0.0, 0.0, 6, 0, 0, false),
    ];
    for result in invalid {
        assert_eq!(window_error(result, "invalid window").status, 7250);
    }

    let output = 1_usize as *mut ffi::AtenTensorOpaque;
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_window(0, 5, true, 0.0, 0.0, 6, 0, -1, false, ptr::null_mut()),
            ffi::terlan_pytorch_window_out(ptr::null_mut(), 0, 5, true, 0.0, 0.0),
            ffi::terlan_pytorch_window_out(output, -1, 5, true, 0.0, 0.0),
        ]
    };
    assert_eq!(statuses, [7250; 3]);
    assert!(!output.is_null());

    let integral = window_error(
        window_native(0, 5, true, 0.0, 0.0, 4, 0, -1, false),
        "integral window",
    );
    assert_eq!(integral.status, 7251);
}

#[test]
fn cuda_windows_execute_when_cuda_is_required() {
    let probe = window_native(0, 5, false, 0.0, 0.0, 6, 1, 0, false);
    match probe {
        Ok(first) => {
            assert_eq!(first.device_type().unwrap(), 1);
            for operation in 1..=4 {
                let actual = window(operation, 5, false, 0.54, 0.0, 6, 1, 0);
                assert_eq!(actual.device_type().unwrap(), 1);
                assert_eq!(actual.device_index().unwrap(), 0);
                assert_eq!(actual.shape().unwrap(), vec![5]);
            }
        }
        Err(error) => {
            if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
                panic!("required CUDA window failed: {error:?}");
            }
        }
    }
}
