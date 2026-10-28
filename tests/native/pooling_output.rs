use terlan_pytorch_cxx::Tensor;

/// Compares complete tensor packets after normalizing CUDA values to CPU.
fn assert_tensor_eq(actual: &Tensor, expected: &Tensor, cuda: bool, context: &str) {
    let actual = if cuda {
        actual.to_device(0, -1).expect("copy actual to CPU")
    } else {
        actual.clone().expect("clone actual")
    };
    let expected = if cuda {
        expected.to_device(0, -1).expect("copy expected to CPU")
    } else {
        expected.clone().expect("clone expected")
    };
    assert_eq!(
        actual.to_packet().expect(context),
        expected.to_packet().expect(context)
    );
}

/// Places a floating-point tensor on the selected test device.
fn tensor(values: &[f64], shape: &[i64], device_type: i64, device_index: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .expect("create pooling tensor")
        .to_device(device_type, device_index)
        .expect("place pooling tensor")
}

/// Proves adaptive and average-pool retained outputs and divisor variants.
fn verify_average_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let line = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], device_type, device_index);
    let expected_line = line
        .adaptive_avg_pool1d(&[2])
        .expect("compute adaptive 1D average pool");
    let mut line_output = expected_line.clone().expect("clone adaptive 1D output");
    line_output
        .adaptive_avg_pool1d_out(&line, &[2])
        .expect("write adaptive 1D average pool");
    assert_tensor_eq(&line_output, &expected_line, cuda, "adaptive 1D output");

    let plane = tensor(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        device_type,
        device_index,
    );
    let expected_plane = plane
        .adaptive_avg_pool2d(&[1, 1])
        .expect("compute adaptive 2D average pool");
    let mut plane_output = expected_plane.clone().expect("clone adaptive 2D output");
    plane_output
        .adaptive_avg_pool2d_out(&plane, &[1, 1])
        .expect("write adaptive 2D average pool");
    assert_tensor_eq(&plane_output, &expected_plane, cuda, "adaptive 2D output");

    let volume = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 2, 2],
        device_type,
        device_index,
    );
    let expected_volume = volume
        .adaptive_avg_pool3d(&[1, 1, 1])
        .expect("compute adaptive 3D average pool");
    let mut volume_output = expected_volume.clone().expect("clone adaptive 3D output");
    volume_output
        .adaptive_avg_pool3d_out(&volume, &[1, 1, 1])
        .expect("write adaptive 3D average pool");
    assert_tensor_eq(&volume_output, &expected_volume, cuda, "adaptive 3D output");

    let adaptive_gradient = tensor(&[8.0], &[1, 1, 1, 1, 1], device_type, device_index);
    let expected_adaptive_gradient = tensor(
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[1, 1, 2, 2, 2],
        device_type,
        device_index,
    );
    let mut adaptive_gradient_output = expected_adaptive_gradient
        .clone()
        .expect("clone adaptive gradient output");
    adaptive_gradient_output
        .adaptive_avg_pool3d_backward_out(&adaptive_gradient, &volume)
        .expect("write adaptive average-pool gradient");
    assert_tensor_eq(
        &adaptive_gradient_output,
        &expected_adaptive_gradient,
        cuda,
        "adaptive average-pool gradient",
    );

    let expected_pool1d = line
        .avg_pool1d(&[2], &[], &[0], false, true)
        .expect("compute 1D average pool");
    let mut pool1d_output = expected_pool1d.clone().expect("clone 1D average output");
    pool1d_output
        .avg_pool1d_out(&line, &[2], &[], &[0], false, true)
        .expect("write 1D average pool");
    assert_tensor_eq(&pool1d_output, &expected_pool1d, cuda, "average 1D output");

    let expected_pool2d = plane
        .avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, true)
        .expect("compute 2D average pool");
    let mut pool2d_output = expected_pool2d.clone().expect("clone 2D average output");
    pool2d_output
        .avg_pool2d_out(&plane, &[2, 2], &[2, 2], &[0, 0], false, true)
        .expect("write 2D average pool");
    assert_tensor_eq(&pool2d_output, &expected_pool2d, cuda, "average 2D output");

    let expected_pool3d = volume
        .avg_pool3d(&[2, 2, 2], &[2, 2, 2], &[0, 0, 0], false, true)
        .expect("compute 3D average pool");
    let mut pool3d_output = expected_pool3d.clone().expect("clone 3D average output");
    pool3d_output
        .avg_pool3d_out(&volume, &[2, 2, 2], &[2, 2, 2], &[0, 0, 0], false, true)
        .expect("write 3D average pool");
    assert_tensor_eq(&pool3d_output, &expected_pool3d, cuda, "average 3D output");

    let expected_divisor2d = plane
        .avg_pool2d_divisor(&[2, 2], &[2, 2], &[0, 0], false, true, 8)
        .expect("compute divided 2D average pool");
    let mut divisor2d_output = expected_divisor2d.clone().expect("clone divided 2D output");
    divisor2d_output
        .avg_pool2d_out_divisor(&plane, &[2, 2], &[2, 2], &[0, 0], false, true, 8)
        .expect("write divided 2D average pool");
    assert_tensor_eq(
        &divisor2d_output,
        &expected_divisor2d,
        cuda,
        "divided average 2D output",
    );

    let expected_divisor3d = volume
        .avg_pool3d_divisor(&[2, 2, 2], &[2, 2, 2], &[0, 0, 0], false, true, 16)
        .expect("compute divided 3D average pool");
    let mut divisor3d_output = expected_divisor3d.clone().expect("clone divided 3D output");
    divisor3d_output
        .avg_pool3d_out_divisor(&volume, &[2, 2, 2], &[2, 2, 2], &[0, 0, 0], false, true, 16)
        .expect("write divided 3D average pool");
    assert_tensor_eq(
        &divisor3d_output,
        &expected_divisor3d,
        cuda,
        "divided average 3D output",
    );

    let gradient2d = tensor(&[2.0], &[1, 1, 1, 1], device_type, device_index);
    let expected_default_gradient2d = gradient2d
        .avg_pool2d_backward(&plane, &[2, 2], &[2, 2], &[0, 0], false, true)
        .expect("compute default 2D gradient");
    let mut default_gradient2d_output = expected_default_gradient2d
        .clone()
        .expect("clone default 2D gradient output");
    default_gradient2d_output
        .avg_pool2d_backward_out(&gradient2d, &plane, &[2, 2], &[2, 2], &[0, 0], false, true)
        .expect("write default 2D gradient");
    assert_tensor_eq(
        &default_gradient2d_output,
        &expected_default_gradient2d,
        cuda,
        "default average 2D gradient",
    );
    let expected_gradient2d = gradient2d
        .avg_pool2d_backward_divisor(&plane, &[2, 2], &[2, 2], &[0, 0], false, true, 8)
        .expect("compute divided 2D gradient");
    let mut gradient2d_output = expected_gradient2d
        .clone()
        .expect("clone divided 2D gradient output");
    gradient2d_output
        .avg_pool2d_backward_out_divisor(
            &gradient2d,
            &plane,
            &[2, 2],
            &[2, 2],
            &[0, 0],
            false,
            true,
            8,
        )
        .expect("write divided 2D gradient");
    assert_tensor_eq(
        &gradient2d_output,
        &expected_gradient2d,
        cuda,
        "divided average 2D gradient",
    );

    let gradient3d = tensor(&[4.0], &[1, 1, 1, 1, 1], device_type, device_index);
    let expected_default_gradient3d = gradient3d
        .avg_pool3d_backward(&volume, &[2, 2, 2], &[2, 2, 2], &[0, 0, 0], false, true)
        .expect("compute default 3D gradient");
    let mut default_gradient3d_output = expected_default_gradient3d
        .clone()
        .expect("clone default 3D gradient output");
    default_gradient3d_output
        .avg_pool3d_backward_out(
            &gradient3d,
            &volume,
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
            false,
            true,
        )
        .expect("write default 3D gradient");
    assert_tensor_eq(
        &default_gradient3d_output,
        &expected_default_gradient3d,
        cuda,
        "default average 3D gradient",
    );
    let expected_gradient3d = gradient3d
        .avg_pool3d_backward_divisor(&volume, &[2, 2, 2], &[2, 2, 2], &[0, 0, 0], false, true, 16)
        .expect("compute divided 3D gradient");
    let mut gradient3d_output = expected_gradient3d
        .clone()
        .expect("clone divided 3D gradient output");
    gradient3d_output
        .avg_pool3d_backward_out_divisor(
            &gradient3d,
            &volume,
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
            false,
            true,
            16,
        )
        .expect("write divided 3D gradient");
    assert_tensor_eq(
        &gradient3d_output,
        &expected_gradient3d,
        cuda,
        "divided average 3D gradient",
    );
}

/// Proves adaptive and indexed maximum-pool outputs, gradients, and unpooling.
fn verify_maximum_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let plane = tensor(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        device_type,
        device_index,
    );
    let (expected_adaptive, expected_adaptive_indices) = plane
        .adaptive_max_pool2d(&[1, 1])
        .expect("compute adaptive 2D maximum pool");
    let mut adaptive_output = expected_adaptive.clone().expect("clone adaptive 2D output");
    let mut adaptive_indices = expected_adaptive_indices
        .clone()
        .expect("clone adaptive 2D indices");
    adaptive_output
        .adaptive_max_pool2d_out(&mut adaptive_indices, &plane, &[1, 1])
        .expect("write adaptive 2D maximum pool");
    assert_tensor_eq(
        &adaptive_output,
        &expected_adaptive,
        cuda,
        "adaptive 2D values",
    );
    assert_tensor_eq(
        &adaptive_indices,
        &expected_adaptive_indices,
        cuda,
        "adaptive 2D indices",
    );
    let adaptive_gradient = tensor(&[2.0], &[1, 1, 1, 1], device_type, device_index);
    let expected_adaptive_gradient = adaptive_gradient
        .adaptive_max_pool2d_backward(&plane, &expected_adaptive_indices)
        .expect("compute adaptive 2D gradient");
    let mut adaptive_gradient_output = expected_adaptive_gradient
        .clone()
        .expect("clone adaptive gradient output");
    adaptive_gradient_output
        .adaptive_max_pool2d_backward_out(&adaptive_gradient, &plane, &adaptive_indices)
        .expect("write adaptive 2D gradient");
    assert_tensor_eq(
        &adaptive_gradient_output,
        &expected_adaptive_gradient,
        cuda,
        "adaptive 2D gradient",
    );

    let volume = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 2, 2],
        device_type,
        device_index,
    );
    let (expected_volume, expected_volume_indices) = volume
        .adaptive_max_pool3d(&[1, 1, 1])
        .expect("compute adaptive 3D maximum pool");
    let mut volume_output = expected_volume.clone().expect("clone adaptive 3D output");
    let mut volume_indices = expected_volume_indices
        .clone()
        .expect("clone adaptive 3D indices");
    volume_output
        .adaptive_max_pool3d_out(&mut volume_indices, &volume, &[1, 1, 1])
        .expect("write adaptive 3D maximum pool");
    assert_tensor_eq(&volume_output, &expected_volume, cuda, "adaptive 3D values");
    assert_tensor_eq(
        &volume_indices,
        &expected_volume_indices,
        cuda,
        "adaptive 3D indices",
    );
    let volume_gradient = tensor(&[3.0], &[1, 1, 1, 1, 1], device_type, device_index);
    let expected_volume_gradient = volume_gradient
        .adaptive_max_pool3d_backward(&volume, &expected_volume_indices)
        .expect("compute adaptive 3D gradient");
    let mut volume_gradient_output = expected_volume_gradient
        .clone()
        .expect("clone adaptive 3D gradient output");
    volume_gradient_output
        .adaptive_max_pool3d_backward_out(&volume_gradient, &volume, &volume_indices)
        .expect("write adaptive 3D gradient");
    assert_tensor_eq(
        &volume_gradient_output,
        &expected_volume_gradient,
        cuda,
        "adaptive 3D gradient",
    );

    let (expected_pool, expected_indices) = plane
        .max_pool2d_with_indices(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
        .expect("compute indexed 2D maximum pool");
    let mut pool_output = expected_pool.clone().expect("clone indexed 2D output");
    let mut pool_indices = expected_indices.clone().expect("clone indexed 2D indices");
    pool_output
        .max_pool2d_with_indices_out(
            &mut pool_indices,
            &plane,
            &[2, 2],
            &[2, 2],
            &[0, 0],
            &[1, 1],
            false,
        )
        .expect("write indexed 2D maximum pool");
    assert_tensor_eq(&pool_output, &expected_pool, cuda, "indexed 2D values");
    assert_tensor_eq(&pool_indices, &expected_indices, cuda, "indexed 2D indices");
    let gradient = tensor(&[5.0], &[1, 1, 1, 1], device_type, device_index);
    let expected_gradient = gradient
        .max_pool2d_with_indices_backward(
            &plane,
            &[2, 2],
            &[2, 2],
            &[0, 0],
            &[1, 1],
            false,
            &expected_indices,
        )
        .expect("compute indexed 2D gradient");
    let mut gradient_output = expected_gradient
        .clone()
        .expect("clone indexed 2D gradient output");
    gradient_output
        .max_pool2d_with_indices_backward_out(
            &gradient,
            &plane,
            &[2, 2],
            &[2, 2],
            &[0, 0],
            &[1, 1],
            false,
            &pool_indices,
        )
        .expect("write indexed 2D gradient");
    assert_tensor_eq(
        &gradient_output,
        &expected_gradient,
        cuda,
        "indexed 2D gradient",
    );
    let expected_unpool = expected_pool
        .max_unpool2d(&expected_indices, &[2, 2])
        .expect("compute 2D maximum unpool");
    let mut unpool_output = expected_unpool.clone().expect("clone 2D unpool output");
    unpool_output
        .max_unpool2d_out(&pool_output, &pool_indices, &[2, 2])
        .expect("write 2D maximum unpool");
    assert_tensor_eq(&unpool_output, &expected_unpool, cuda, "maximum 2D unpool");

    let (expected_pool3d, expected_indices3d) = volume
        .max_pool3d_with_indices(&[2, 2, 2], &[2, 2, 2], &[0, 0, 0], &[1, 1, 1], false)
        .expect("compute indexed 3D maximum pool");
    let mut pool3d_output = expected_pool3d.clone().expect("clone indexed 3D output");
    let mut pool3d_indices = expected_indices3d
        .clone()
        .expect("clone indexed 3D indices");
    pool3d_output
        .max_pool3d_with_indices_out(
            &mut pool3d_indices,
            &volume,
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
            &[1, 1, 1],
            false,
        )
        .expect("write indexed 3D maximum pool");
    assert_tensor_eq(&pool3d_output, &expected_pool3d, cuda, "indexed 3D values");
    assert_tensor_eq(
        &pool3d_indices,
        &expected_indices3d,
        cuda,
        "indexed 3D indices",
    );
    let gradient3d = tensor(&[7.0], &[1, 1, 1, 1, 1], device_type, device_index);
    let expected_gradient3d = gradient3d
        .max_pool3d_with_indices_backward(
            &volume,
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
            &[1, 1, 1],
            false,
            &expected_indices3d,
        )
        .expect("compute indexed 3D gradient");
    let mut gradient3d_output = expected_gradient3d
        .clone()
        .expect("clone indexed 3D gradient output");
    gradient3d_output
        .max_pool3d_with_indices_backward_out(
            &gradient3d,
            &volume,
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
            &[1, 1, 1],
            false,
            &pool3d_indices,
        )
        .expect("write indexed 3D gradient");
    assert_tensor_eq(
        &gradient3d_output,
        &expected_gradient3d,
        cuda,
        "indexed 3D gradient",
    );
    let expected_unpool3d = expected_pool3d
        .max_unpool3d(&expected_indices3d, &[2, 2, 2], &[2, 2, 2], &[0, 0, 0])
        .expect("compute 3D maximum unpool");
    let mut unpool3d_output = expected_unpool3d.clone().expect("clone 3D unpool output");
    unpool3d_output
        .max_unpool3d_out(
            &pool3d_output,
            &pool3d_indices,
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
        )
        .expect("write 3D maximum unpool");
    assert_tensor_eq(
        &unpool3d_output,
        &expected_unpool3d,
        cuda,
        "maximum 3D unpool",
    );
}

/// Proves deterministic fractional maximum-pool outputs and gradients.
fn verify_fractional_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let plane_values = (1..=16).map(f64::from).collect::<Vec<_>>();
    let plane = tensor(&plane_values, &[1, 1, 4, 4], device_type, device_index);
    let samples2d = tensor(&[0.25, 0.75], &[1, 1, 2], device_type, device_index);
    let (expected2d, expected_indices2d) = plane
        .fractional_max_pool2d(&[2, 2], &[2, 2], &samples2d)
        .expect("compute fractional 2D pool");
    let mut output2d = expected2d.clone().expect("clone fractional 2D output");
    let mut indices2d = expected_indices2d
        .clone()
        .expect("clone fractional 2D indices");
    output2d
        .fractional_max_pool2d_out(&mut indices2d, &plane, &[2, 2], &[2, 2], &samples2d)
        .expect("write fractional 2D pool");
    assert_tensor_eq(&output2d, &expected2d, cuda, "fractional 2D values");
    assert_tensor_eq(
        &indices2d,
        &expected_indices2d,
        cuda,
        "fractional 2D indices",
    );
    let gradient2d = tensor(&[1.0; 4], &[1, 1, 2, 2], device_type, device_index);
    let expected_gradient2d = gradient2d
        .fractional_max_pool2d_backward(&plane, &[2, 2], &[2, 2], &expected_indices2d)
        .expect("compute fractional 2D gradient");
    let mut gradient2d_output = expected_gradient2d
        .clone()
        .expect("clone fractional 2D gradient output");
    gradient2d_output
        .fractional_max_pool2d_backward_out(&gradient2d, &plane, &[2, 2], &[2, 2], &indices2d)
        .expect("write fractional 2D gradient");
    assert_tensor_eq(
        &gradient2d_output,
        &expected_gradient2d,
        cuda,
        "fractional 2D gradient",
    );

    let volume_values = (1..=64).map(f64::from).collect::<Vec<_>>();
    let volume = tensor(&volume_values, &[1, 1, 4, 4, 4], device_type, device_index);
    let samples3d = tensor(&[0.2, 0.5, 0.8], &[1, 1, 3], device_type, device_index);
    let (expected3d, expected_indices3d) = volume
        .fractional_max_pool3d(&[2, 2, 2], &[2, 2, 2], &samples3d)
        .expect("compute fractional 3D pool");
    let mut output3d = expected3d.clone().expect("clone fractional 3D output");
    let mut indices3d = expected_indices3d
        .clone()
        .expect("clone fractional 3D indices");
    output3d
        .fractional_max_pool3d_out(&mut indices3d, &volume, &[2, 2, 2], &[2, 2, 2], &samples3d)
        .expect("write fractional 3D pool");
    assert_tensor_eq(&output3d, &expected3d, cuda, "fractional 3D values");
    assert_tensor_eq(
        &indices3d,
        &expected_indices3d,
        cuda,
        "fractional 3D indices",
    );
    let gradient3d = tensor(&[1.0; 8], &[1, 1, 2, 2, 2], device_type, device_index);
    let expected_gradient3d = gradient3d
        .fractional_max_pool3d_backward(&volume, &[2, 2, 2], &[2, 2, 2], &expected_indices3d)
        .expect("compute fractional 3D gradient");
    let mut gradient3d_output = expected_gradient3d
        .clone()
        .expect("clone fractional 3D gradient output");
    gradient3d_output
        .fractional_max_pool3d_backward_out(
            &gradient3d,
            &volume,
            &[2, 2, 2],
            &[2, 2, 2],
            &indices3d,
        )
        .expect("write fractional 3D gradient");
    assert_tensor_eq(
        &gradient3d_output,
        &expected_gradient3d,
        cuda,
        "fractional 3D gradient",
    );
}

/// Proves every pooling retained-output family on CPU.
#[test]
fn cpu_pooling_outputs_match_functional_results() {
    verify_average_outputs(0, -1, false);
    verify_maximum_outputs(0, -1, false);
    verify_fractional_outputs(0, -1, false);
}

/// Proves every pooling retained-output family on mandatory CUDA hardware.
#[test]
fn cuda_pooling_outputs_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_average_outputs(1, 0, true);
    verify_maximum_outputs(1, 0, true);
    verify_fractional_outputs(1, 0, true);
}

/// Proves invalid pooling dimensions fail without changing retained storage.
#[test]
fn invalid_pooling_output_fails_without_mutating_storage() {
    let source = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], 0, -1);
    let mut output = source.clone().expect("clone invalid output");
    let before = output.to_packet().expect("capture output before failure");
    let error = output
        .avg_pool2d_out(&source, &[0, 0], &[1, 1], &[0, 0], false, true)
        .expect_err("zero kernel must fail");
    assert_ne!(error.status, 0);
    assert_eq!(
        output.to_packet().expect("capture output after failure"),
        before
    );
}
