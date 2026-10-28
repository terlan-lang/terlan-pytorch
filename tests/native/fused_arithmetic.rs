use terlan_pytorch_cxx::{CAbiError, Tensor};

/// One fused arithmetic input and its expected configured result.
type FusedCase = (i64, Tensor, Tensor, Tensor, Tensor);

/// Places one floating-point tensor on the selected device.
fn tensor(values: &[f64], shape: &[i64], device_type: i64, device_index: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .expect("create fused arithmetic tensor")
        .to_device(device_type, device_index)
        .expect("place fused arithmetic tensor")
}

/// Places one floating-point tensor with an explicit scalar dtype on the selected device.
fn tensor_dtype(
    values: &[f64],
    shape: &[i64],
    dtype: i64,
    device_type: i64,
    device_index: i64,
) -> Tensor {
    Tensor::from_floats(values, shape)
        .expect("create typed fused arithmetic tensor")
        .to_dtype(dtype)
        .expect("convert fused arithmetic tensor dtype")
        .to_device(device_type, device_index)
        .expect("place typed fused arithmetic tensor")
}

/// Exports one device tensor through a normalized CPU packet.
fn packet(tensor: &Tensor) -> Vec<u8> {
    tensor
        .to_device(0, -1)
        .expect("normalize tensor to CPU")
        .to_packet()
        .expect("export tensor packet")
}

/// Compares all tensor metadata and values through the pointer-free packet ABI.
fn assert_tensor_eq(actual: &Tensor, expected: &Tensor, context: &str) {
    assert_eq!(packet(actual), packet(expected), "{context}");
}

/// Converts an expected tensor into visibly different retained output storage.
fn poisoned_output(expected: &Tensor, device_type: i64, device_index: i64) -> Tensor {
    let shape = expected.shape().expect("read expected shape");
    let length = expected.numel().expect("read expected length") as usize;
    tensor(&vec![-99.0; length], &shape, device_type, device_index)
}

/// Builds one configured mixed-scalar case for every fused operation.
fn fused_cases(device_type: i64, device_index: i64) -> Vec<FusedCase> {
    let matrix = [1.0, 2.0, 3.0, 4.0];
    let identity_batch = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let matrix_batch = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    vec![
        (
            0,
            tensor(&[1.0; 4], &[2, 2], device_type, device_index),
            tensor(&identity_batch, &[2, 2, 2], device_type, device_index),
            tensor(&matrix_batch, &[2, 2, 2], device_type, device_index),
            tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2], device_type, device_index),
        ),
        (
            1,
            tensor(&[10.0, 20.0], &[2], device_type, device_index),
            tensor(&[4.0, 8.0], &[2], device_type, device_index),
            tensor(&[2.0, 4.0], &[2], device_type, device_index),
            tensor(&[11.0, 21.0], &[2], device_type, device_index),
        ),
        (
            2,
            tensor(&[1.0, 2.0], &[2], device_type, device_index),
            tensor(&[2.0, 3.0], &[2], device_type, device_index),
            tensor(&[4.0, 5.0], &[2], device_type, device_index),
            tensor(&[5.0, 9.5], &[2], device_type, device_index),
        ),
        (
            3,
            tensor(&[1.0; 4], &[2, 2], device_type, device_index),
            tensor(&matrix, &[2, 2], device_type, device_index),
            tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2], device_type, device_index),
            tensor(&[2.5, 3.0, 3.5, 4.0], &[2, 2], device_type, device_index),
        ),
        (
            4,
            tensor(&[1.0, 1.0], &[2], device_type, device_index),
            tensor(&matrix, &[2, 2], device_type, device_index),
            tensor(&[1.0, 2.0], &[2], device_type, device_index),
            tensor(&[4.5, 7.5], &[2], device_type, device_index),
        ),
        (
            5,
            tensor(&[1.0; 4], &[2, 2], device_type, device_index),
            tensor(&[1.0, 2.0], &[2], device_type, device_index),
            tensor(&[3.0, 4.0], &[2], device_type, device_index),
            tensor(&[3.5, 4.0, 5.0, 6.0], &[2, 2], device_type, device_index),
        ),
        (
            6,
            tensor(&[1.0; 8], &[2, 2, 2], device_type, device_index),
            tensor(&identity_batch, &[2, 2, 2], device_type, device_index),
            tensor(&matrix_batch, &[2, 2, 2], device_type, device_index),
            tensor(
                &[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                &[2, 2, 2],
                device_type,
                device_index,
            ),
        ),
    ]
}

/// Verifies functional, retained-output, and in-place forms on one device.
fn verify_all_forms(device_type: i64, device_index: i64) {
    for (operation, accumulator, first, second, expected) in fused_cases(device_type, device_index)
    {
        let accumulator_before = packet(&accumulator);
        let first_before = packet(&first);
        let second_before = packet(&second);
        let functional = accumulator
            .fused_arithmetic(
                &first, &second, operation, true, 2, 0.0, false, 0, 0.5, false, 0,
            )
            .expect("execute fused functional form");
        assert_tensor_eq(&functional, &expected, "functional fused result");

        let mut output = poisoned_output(&expected, device_type, device_index);
        output
            .fused_arithmetic_out(
                &accumulator,
                &first,
                &second,
                operation,
                true,
                2,
                0.0,
                false,
                0,
                0.5,
                false,
                0,
            )
            .expect("execute fused retained-output form");
        assert_tensor_eq(&output, &expected, "retained fused result");

        let mut mutation = accumulator.clone().expect("clone fused accumulator");
        mutation
            .fused_arithmetic_in_place(&first, &second, operation, true, 2, 0.0, false, 0, 0.5)
            .expect("execute fused in-place form");
        assert_tensor_eq(&mutation, &expected, "in-place fused result");

        assert_eq!(packet(&accumulator), accumulator_before);
        assert_eq!(packet(&first), first_before);
        assert_eq!(packet(&second), second_before);
    }
}

/// Verifies exact dtype overloads for matrix and batched matrix addition.
fn verify_dtype_forms(device_type: i64, device_index: i64) {
    let matrix_accumulator = tensor_dtype(&[2.0], &[1, 1], 6, device_type, device_index);
    let matrix_left = tensor_dtype(&[1.0, 2.0], &[1, 2], 5, device_type, device_index);
    let matrix_right = tensor_dtype(&[3.0, 4.0], &[2, 1], 5, device_type, device_index);
    let matrix_expected = tensor_dtype(&[13.0], &[1, 1], 6, device_type, device_index);
    let matrix_result = matrix_accumulator
        .fused_arithmetic(
            &matrix_left,
            &matrix_right,
            3,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        )
        .expect("execute addmm dtype form");
    assert_eq!(matrix_result.dtype().expect("read addmm dtype"), 6);
    assert_tensor_eq(&matrix_result, &matrix_expected, "addmm dtype result");
    let mut matrix_output = tensor_dtype(&[0.0], &[1, 1], 6, device_type, device_index);
    matrix_output
        .fused_arithmetic_out(
            &matrix_accumulator,
            &matrix_left,
            &matrix_right,
            3,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        )
        .expect("execute addmm dtype out form");
    assert_tensor_eq(&matrix_output, &matrix_expected, "addmm dtype output");

    let batch_accumulator = tensor_dtype(&[2.0], &[1, 1, 1], 6, device_type, device_index);
    let batch_left = tensor_dtype(&[1.0, 2.0], &[1, 1, 2], 5, device_type, device_index);
    let batch_right = tensor_dtype(&[3.0, 4.0], &[1, 2, 1], 5, device_type, device_index);
    let batch_expected = tensor_dtype(&[13.0], &[1, 1, 1], 6, device_type, device_index);
    let batch_result = batch_accumulator
        .fused_arithmetic(
            &batch_left,
            &batch_right,
            6,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        )
        .expect("execute baddbmm dtype form");
    assert_eq!(batch_result.dtype().expect("read baddbmm dtype"), 6);
    assert_tensor_eq(&batch_result, &batch_expected, "baddbmm dtype result");
    let mut batch_output = tensor_dtype(&[0.0], &[1, 1, 1], 6, device_type, device_index);
    batch_output
        .fused_arithmetic_out(
            &batch_accumulator,
            &batch_left,
            &batch_right,
            6,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        )
        .expect("execute baddbmm dtype out form");
    assert_tensor_eq(&batch_output, &batch_expected, "baddbmm dtype output");
}

/// Verifies that CUDA-specific dtype overloads fail atomically on CPU.
fn verify_cpu_dtype_rejections() {
    let matrix_accumulator = tensor_dtype(&[2.0], &[1, 1], 6, 0, -1);
    let matrix_left = tensor_dtype(&[1.0, 2.0], &[1, 2], 5, 0, -1);
    let matrix_right = tensor_dtype(&[3.0, 4.0], &[2, 1], 5, 0, -1);
    let matrix_error = tensor_error(
        matrix_accumulator.fused_arithmetic(
            &matrix_left,
            &matrix_right,
            3,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        ),
        "CPU addmm dtype overload unexpectedly succeeded",
    );
    assert_eq!(matrix_error.status, 7231);

    let batch_accumulator = tensor_dtype(&[2.0], &[1, 1, 1], 6, 0, -1);
    let batch_left = tensor_dtype(&[1.0, 2.0], &[1, 1, 2], 5, 0, -1);
    let batch_right = tensor_dtype(&[3.0, 4.0], &[1, 2, 1], 5, 0, -1);
    let batch_error = tensor_error(
        batch_accumulator.fused_arithmetic(
            &batch_left,
            &batch_right,
            6,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        ),
        "CPU baddbmm dtype overload unexpectedly succeeded",
    );
    assert_eq!(batch_error.status, 7231);

    let mut output = tensor_dtype(&[91.0], &[1, 1], 6, 0, -1);
    let output_before = packet(&output);
    let output_error = output
        .fused_arithmetic_out(
            &matrix_accumulator,
            &matrix_left,
            &matrix_right,
            3,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            true,
            6,
        )
        .expect_err("CPU addmm dtype output unexpectedly succeeded");
    assert_eq!(output_error.status, 7231);
    assert_eq!(packet(&output), output_before);
}

/// Extracts the required native status from a tensor-producing failure.
fn tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn cpu_executes_every_fused_arithmetic_schema_shape() {
    verify_all_forms(0, -1);
    verify_cpu_dtype_rejections();
}

#[test]
fn malformed_requests_fail_without_mutating_retained_inputs() {
    let accumulator = tensor(&[1.0; 4], &[2, 2], 0, -1);
    let first = tensor(&[1.0; 4], &[2, 2], 0, -1);
    let second = tensor(&[1.0; 4], &[2, 2], 0, -1);
    let invalid_operation = tensor_error(
        accumulator.fused_arithmetic(&first, &second, 99, true, 1, 0.0, true, 1, 0.0, false, 0),
        "unknown operation unexpectedly succeeded",
    );
    let invalid_dtype_operation = tensor_error(
        accumulator.fused_arithmetic(&first, &second, 1, true, 1, 0.0, true, 1, 0.0, true, 7),
        "pointwise dtype operation unexpectedly succeeded",
    );
    let invalid_dtype = tensor_error(
        accumulator.fused_arithmetic(&first, &second, 3, true, 1, 0.0, true, 1, 0.0, true, 99),
        "invalid dtype unexpectedly succeeded",
    );
    assert_eq!(invalid_operation.status, 7230);
    assert_eq!(invalid_dtype_operation.status, 7230);
    assert_eq!(invalid_dtype.status, 7230);

    let bad_right = tensor(&[1.0; 6], &[3, 2], 0, -1);
    let mut output = tensor(&[7.0; 4], &[2, 2], 0, -1);
    let output_before = packet(&output);
    let output_error = output
        .fused_arithmetic_out(
            &accumulator,
            &first,
            &bad_right,
            3,
            true,
            1,
            0.0,
            true,
            1,
            0.0,
            false,
            0,
        )
        .expect_err("invalid output shape unexpectedly succeeded");
    assert_eq!(output_error.status, 7231);
    assert_eq!(packet(&output), output_before);

    let mut mutation = accumulator.clone().expect("clone mutation accumulator");
    let mutation_before = packet(&mutation);
    let mutation_error = mutation
        .fused_arithmetic_in_place(&first, &bad_right, 3, true, 1, 0.0, true, 1, 0.0)
        .expect_err("invalid mutation shape unexpectedly succeeded");
    assert_eq!(mutation_error.status, 7231);
    assert_eq!(packet(&mutation), mutation_before);
}

#[test]
fn cuda_executes_every_fused_arithmetic_schema_shape_when_required() {
    let probe = Tensor::from_float(1.0)
        .expect("create CUDA probe")
        .to_device(1, 0);
    match probe {
        Ok(_) => {
            verify_all_forms(1, 0);
            verify_dtype_forms(1, 0);
        }
        Err(error) => {
            if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
                panic!("required CUDA placement failed: {error:?}");
            }
        }
    }
}
