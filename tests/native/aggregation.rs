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

fn assert_float(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .equal_native(&float_tensor(values, shape, device))
        .unwrap());
}

fn assert_int(actual: &Tensor, values: &[i64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 4);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .equal_native(&int_tensor(values, shape, device))
        .unwrap());
}

fn host_packet(tensor: &Tensor) -> Vec<u8> {
    tensor.to_device(0, -1).unwrap().to_packet().unwrap()
}

fn verify_bincount(device: i64) {
    let values = int_tensor(&[0, 1, 1, 3, 3, 3], &[6], device);
    let weights = float_tensor(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], &[6], device);
    let values_before = host_packet(&values);
    let weights_before = host_packet(&weights);

    assert_int(
        &values.bincount_configured(&values, false, 0).unwrap(),
        &[1, 2, 0, 3],
        &[4],
        device,
    );
    assert_int(
        &values.bincount_configured(&values, false, 6).unwrap(),
        &[1, 2, 0, 3, 0, 0],
        &[6],
        device,
    );
    assert_float(
        &values.bincount_configured(&weights, true, 0).unwrap(),
        &[0.5, 2.5, 0.0, 7.5],
        &[4],
        device,
    );
    assert_float(
        &values.bincount_configured(&weights, true, 6).unwrap(),
        &[0.5, 2.5, 0.0, 7.5, 0.0, 0.0],
        &[6],
        device,
    );

    let mut unweighted_output = int_tensor(&[], &[0], device);
    unweighted_output
        .bincount_configured_out(&values, &values, false, 6)
        .unwrap();
    assert_int(&unweighted_output, &[1, 2, 0, 3, 0, 0], &[6], device);
    let mut weighted_output = float_tensor(&[], &[0], device);
    weighted_output
        .bincount_configured_out(&values, &weights, true, 6)
        .unwrap();
    assert_float(
        &weighted_output,
        &[0.5, 2.5, 0.0, 7.5, 0.0, 0.0],
        &[6],
        device,
    );

    let empty = int_tensor(&[], &[0], device);
    assert_int(
        &empty.bincount_configured(&empty, false, 3).unwrap(),
        &[0, 0, 0],
        &[3],
        device,
    );

    let strided_storage = int_tensor(&[0, 99, 1, 99, 1, 99, 3], &[7], device);
    let strided = strided_storage.as_strided(&[4], &[2]).unwrap();
    assert_int(
        &strided.bincount_configured(&strided, false, 0).unwrap(),
        &[1, 2, 0, 1],
        &[4],
        device,
    );

    let floating = float_tensor(&[0.0, 1.0], &[2], device);
    assert!(floating.bincount_configured(&floating, false, 0).is_err());
    let short_weights = float_tensor(&[1.0], &[1], device);
    assert!(values.bincount_configured(&short_weights, true, 0).is_err());
    assert!(values.bincount_configured(&values, false, -1).is_err());

    // Negative CUDA values can raise an asynchronous device-side assertion and
    // poison the process before later ownership checks can synchronize.  Keep
    // the value-domain and retained-output atomicity proof on CPU; CUDA still
    // executes the dtype, shape, option, and autograd failure paths above.
    if device == 0 {
        let negative = int_tensor(&[0, -1], &[2], device);
        assert!(negative.bincount_configured(&negative, false, 0).is_err());
        let mut retained = int_tensor(&[42], &[1], device);
        let retained_before = host_packet(&retained);
        assert!(retained
            .bincount_configured_out(&negative, &negative, false, 0)
            .is_err());
        assert_eq!(host_packet(&retained), retained_before);
    }
    assert_eq!(host_packet(&values), values_before);
    assert_eq!(host_packet(&weights), weights_before);

    let differentiable_weights = weights.clone().unwrap().require_grad().unwrap();
    let weighted = values
        .bincount_configured(&differentiable_weights, true, 0)
        .unwrap();
    assert!(weighted.sum().unwrap().backward().is_err());
}

fn verify_histc(device: i64) {
    let input = float_tensor(&[-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], &[2, 3], device);
    let input_before = host_packet(&input);

    assert_float(
        &input.histc_configured(4, false, 0, 0, -1.0, 3.0).unwrap(),
        &[1.0, 2.0, 1.0, 2.0],
        &[4],
        device,
    );
    assert_float(
        &input.histc_configured(4, true, -1, 3, 0.0, 0.0).unwrap(),
        &[1.0, 2.0, 1.0, 2.0],
        &[4],
        device,
    );
    assert_float(
        &input.histc_configured(4, false, 0, 0, 0.0, 0.0).unwrap(),
        &[1.0, 2.0, 1.0, 2.0],
        &[4],
        device,
    );

    let mut output = float_tensor(&[], &[0], device);
    output
        .histc_configured_out(&input, 4, false, 0, 0, -1.0, 3.0)
        .unwrap();
    assert_float(&output, &[1.0, 2.0, 1.0, 2.0], &[4], device);
    output
        .histc_configured_out(&input, 4, true, -1, 3, 0.0, 0.0)
        .unwrap();
    assert_float(&output, &[1.0, 2.0, 1.0, 2.0], &[4], device);

    let strided_storage = float_tensor(&[-1.0, 99.0, 0.0, 99.0, 1.0, 99.0, 3.0], &[7], device);
    let strided = strided_storage.as_strided(&[4], &[2]).unwrap();
    assert_float(
        &strided.histc_configured(4, false, 0, 0, -1.0, 3.0).unwrap(),
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
        device,
    );

    let output_before = host_packet(&output);
    assert!(output
        .histc_configured_out(&input, 0, false, 0, 0, -1.0, 3.0)
        .is_err());
    assert_eq!(host_packet(&output), output_before);
    assert!(input.histc_configured(4, false, 0, 0, 3.0, -1.0).is_err());

    let trainable = input.clone().unwrap().require_grad().unwrap();
    let histogram = trainable
        .histc_configured(4, false, 0, 0, -1.0, 3.0)
        .unwrap();
    assert!(histogram.sum().unwrap().backward().is_err());
    assert_eq!(host_packet(&input), input_before);
}

fn index_expected(reduction: i64, include_self: bool) -> [f64; 4] {
    match (reduction, include_self) {
        (0, true) => [120.0, 420.0, 120.0, 200.0],
        (0, false) => [12.0, 21.0, 4.0, 5.0],
        (1, true) => [6.0, 10.0, 17.0, 22.5],
        (1, false) => [4.0, 5.0, 4.0, 5.0],
        (2, true) => [10.0, 20.0, 30.0, 40.0],
        (2, false) => [6.0, 7.0, 4.0, 5.0],
        (3, true) => [2.0, 3.0, 4.0, 5.0],
        (3, false) => [2.0, 3.0, 4.0, 5.0],
        _ => unreachable!(),
    }
}

fn verify_index_reduce(device: i64) {
    let input = float_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2], device);
    let index = int_tensor(&[0, 1, 0], &[3], device);
    let source = float_tensor(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[3, 2], device);
    let input_before = host_packet(&input);
    let source_before = host_packet(&source);
    let index_before = host_packet(&index);

    for reduction in 0..4 {
        for include_self in [true, false] {
            let expected = index_expected(reduction, include_self);
            let functional = input
                .index_reduce_configured(0, &index, &source, reduction, include_self)
                .unwrap();
            assert_float(&functional, &expected, &[2, 2], device);

            let mut output = float_tensor(&[], &[0], device);
            output
                .index_reduce_configured_out(&input, 0, &index, &source, reduction, include_self)
                .unwrap();
            assert_float(&output, &expected, &[2, 2], device);

            let mut mutation = input.clone().unwrap();
            mutation
                .index_reduce_configured_in_place(0, &index, &source, reduction, include_self)
                .unwrap();
            assert_float(&mutation, &expected, &[2, 2], device);
        }
    }

    let transposed_source = float_tensor(&[2.0, 4.0, 6.0, 3.0, 5.0, 7.0], &[2, 3], device)
        .transpose(0, 1)
        .unwrap();
    assert_float(
        &input
            .index_reduce_configured(-2, &index, &transposed_source, 3, false)
            .unwrap(),
        &[2.0, 3.0, 4.0, 5.0],
        &[2, 2],
        device,
    );

    let input_leaf = input.clone().unwrap().require_grad().unwrap();
    let source_leaf = source.clone().unwrap().require_grad().unwrap();
    input_leaf
        .index_reduce_configured(0, &index, &source_leaf, 1, true)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_float(
        &input_leaf.gradient().unwrap(),
        &[1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5],
        &[2, 2],
        device,
    );
    assert_float(
        &source_leaf.gradient().unwrap(),
        &[1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0],
        &[3, 2],
        device,
    );

    assert!(input
        .index_reduce_configured(0, &index, &source, 9, true)
        .is_err());
    let floating_index = float_tensor(&[0.0, 1.0, 0.0], &[3], device);
    assert!(input
        .index_reduce_configured(0, &floating_index, &source, 0, true)
        .is_err());
    assert!(input
        .index_reduce_configured(2, &index, &source, 0, true)
        .is_err());
    // An out-of-range CUDA index is likewise a device-side assertion.  The
    // CPU path proves both the error and transactional `out` behavior without
    // invalidating the CUDA context used by the remaining evidence.
    if device == 0 {
        let invalid_index = int_tensor(&[0, 2, 0], &[3], device);
        let mut output = float_tensor(&[9.0], &[1], device);
        let output_before = host_packet(&output);
        assert!(output
            .index_reduce_configured_out(&input, 0, &invalid_index, &source, 0, true)
            .is_err());
        assert_eq!(host_packet(&output), output_before);
    }

    let trainable = input.clone().unwrap().require_grad().unwrap();
    let trainable_before = host_packet(&trainable);
    let mut trainable = trainable;
    assert!(trainable
        .index_reduce_configured_in_place(0, &index, &source, 0, true)
        .is_err());
    assert_eq!(host_packet(&trainable), trainable_before);

    assert_eq!(host_packet(&input), input_before);
    assert_eq!(host_packet(&source), source_before);
    assert_eq!(host_packet(&index), index_before);
}

fn verify(device: i64) {
    verify_bincount(device);
    verify_histc(device);
    verify_index_reduce(device);
}

#[test]
fn cpu_counting_and_index_reduction_values_storage_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_counting_and_index_reduction_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}

#[test]
fn aggregation_rejects_null_boundary_handles() {
    let mut output = ptr::null_mut();
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_bincount(ptr::null_mut(), ptr::null_mut(), false, 0, &mut output),
            ffi::terlan_pytorch_histc(ptr::null_mut(), 4, false, 0, 0, -1.0, 3.0, &mut output),
            ffi::terlan_pytorch_index_reduce(
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                true,
                &mut output,
            ),
        ]
    };
    assert_eq!(statuses, [7270, 7270, 7270]);
    assert!(output.is_null());
}
