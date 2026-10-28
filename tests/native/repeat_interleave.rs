use terlan_pytorch_cxx::Tensor;

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

fn check_float(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .equal_native(&float_tensor(values, shape, device))
        .unwrap());
}

fn check_int(actual: &Tensor, values: &[i64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 4);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .equal_native(&int_tensor(values, shape, device))
        .unwrap());
}

fn verify(device: i64) {
    let repeats = int_tensor(&[1, 2, 0, 3], &[4], device);
    let indices = repeats.repeat_indices_native().unwrap();
    check_int(&indices, &[0, 1, 1, 3, 3, 3], &[6], device);
    check_int(
        &repeats.repeat_indices_output_size_native(6).unwrap(),
        &[0, 1, 1, 3, 3, 3],
        &[6],
        device,
    );
    let mut output = int_tensor(&[], &[0], device);
    output.repeat_indices_out_native(&repeats).unwrap();
    check_int(&output, &[0, 1, 1, 3, 3, 3], &[6], device);
    output.zero_in_place().unwrap();
    output
        .repeat_indices_out_output_size_native(&repeats, 6)
        .unwrap();
    check_int(&output, &[0, 1, 1, 3, 3, 3], &[6], device);
    check_int(&repeats, &[1, 2, 0, 3], &[4], device);

    let matrix = float_tensor(&[1., 2., 3., 4.], &[2, 2], device);
    let flat_repeats = int_tensor(&[1, 0, 2, 1], &[4], device);
    check_float(
        &matrix
            .repeat_tensor_flat_native(&flat_repeats, 0, 0)
            .unwrap(),
        &[1., 3., 3., 4.],
        &[4],
        device,
    );
    check_float(
        &matrix
            .repeat_tensor_flat_output_size_native(&flat_repeats, 0, 4)
            .unwrap(),
        &[1., 3., 3., 4.],
        &[4],
        device,
    );
    let column_repeats = int_tensor(&[1, 2], &[2], device);
    check_float(
        &matrix
            .repeat_tensor_dimension_native(&column_repeats, -1, 0)
            .unwrap(),
        &[1., 2., 2., 3., 4., 4.],
        &[2, 3],
        device,
    );
    check_float(
        &matrix
            .repeat_tensor_dimension_output_size_native(&column_repeats, 1, 3)
            .unwrap(),
        &[1., 2., 2., 3., 4., 4.],
        &[2, 3],
        device,
    );

    check_float(
        &matrix.repeat_scalar_flat_native(2, 0, 0).unwrap(),
        &[1., 1., 2., 2., 3., 3., 4., 4.],
        &[8],
        device,
    );
    check_float(
        &matrix
            .repeat_scalar_flat_output_size_native(2, 0, 8)
            .unwrap(),
        &[1., 1., 2., 2., 3., 3., 4., 4.],
        &[8],
        device,
    );
    check_float(
        &matrix.repeat_scalar_dimension_native(2, 0, 0).unwrap(),
        &[1., 2., 1., 2., 3., 4., 3., 4.],
        &[4, 2],
        device,
    );
    check_float(
        &matrix
            .repeat_scalar_dimension_output_size_native(2, 0, 4)
            .unwrap(),
        &[1., 2., 1., 2., 3., 4., 3., 4.],
        &[4, 2],
        device,
    );

    let transposed = matrix.transpose(0, 1).unwrap();
    check_float(
        &transposed.repeat_scalar_dimension_native(2, 1, 0).unwrap(),
        &[1., 1., 3., 3., 2., 2., 4., 4.],
        &[2, 4],
        device,
    );

    let leaf = float_tensor(&[1., 2., 3.], &[3], device)
        .require_grad()
        .unwrap();
    let gradient_repeats = int_tensor(&[1, 2, 3], &[3], device);
    leaf.repeat_tensor_flat_native(&gradient_repeats, 0, 0)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check_float(&leaf.gradient().unwrap(), &[1., 2., 3.], &[3], device);

    let empty_repeats = int_tensor(&[], &[0], device);
    check_int(
        &empty_repeats.repeat_indices_native().unwrap(),
        &[],
        &[0],
        device,
    );
    check_float(
        &matrix.repeat_scalar_flat_native(0, 0, 0).unwrap(),
        &[],
        &[0],
        device,
    );

    assert!(matrix
        .repeat_tensor_flat_native(&column_repeats, 0, 0)
        .is_err());
    assert!(matrix
        .repeat_tensor_dimension_native(&column_repeats, 3, 0)
        .is_err());
    assert!(matrix.repeat_scalar_flat_native(-1, 0, 0).is_err());
    let floating_repeats = float_tensor(&[1., 1., 1., 1.], &[4], device);
    assert!(floating_repeats.repeat_indices_native().is_err());

    if device == 0 {
        assert!(repeats.repeat_indices_output_size_native(5).is_err());
        assert!(matrix
            .repeat_tensor_flat_output_size_native(&flat_repeats, 0, 5)
            .is_err());
        let mut retained = int_tensor(&[42], &[1], device);
        assert!(retained
            .repeat_indices_out_output_size_native(&repeats, 5)
            .is_err());
        check_int(&retained, &[42], &[1], device);
    }
    check_float(&matrix, &[1., 2., 3., 4.], &[2, 2], device);
    check_int(&repeats, &[1, 2, 0, 3], &[4], device);
}

#[test]
fn cpu_repeat_interleave_values_options_outputs_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_repeat_interleave_values_options_outputs_errors_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}

#[test]
fn cuda_output_size_mismatch_isolated_parent() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none()
        || std::env::var_os("TERLAN_PYTORCH_REPEAT_CUDA_ASSERT_CHILD").is_some()
    {
        return;
    }
    let outcome = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("cuda_output_size_mismatch_isolated_child")
        .env("TERLAN_PYTORCH_REPEAT_CUDA_ASSERT_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        outcome.status.success(),
        "isolated CUDA mismatch child failed: {}",
        String::from_utf8_lossy(&outcome.stderr)
    );
}

#[test]
fn cuda_output_size_mismatch_isolated_child() {
    if std::env::var_os("TERLAN_PYTORCH_REPEAT_CUDA_ASSERT_CHILD").is_none() {
        return;
    }
    let repeats = int_tensor(&[1, 2, 0, 3], &[4], 1);
    // CUDA validates this hint in the asynchronous kernel rather than at the
    // call boundary. Copying the result to CPU forces synchronization and
    // must surface the pinned device-side assertion as a boundary error.
    let invalid = repeats.repeat_indices_output_size_native(5).unwrap();
    assert!(invalid.to_device(0, -1).is_err());
}
