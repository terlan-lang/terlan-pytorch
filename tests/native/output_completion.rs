use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn empty(device: i64) -> Tensor {
    tensor(&[], &[0], device)
}

fn same(actual: &Tensor, expected: &Tensor, device: i64) {
    assert_eq!(actual.shape().unwrap(), expected.shape().unwrap());
    assert_eq!(actual.dtype().unwrap(), expected.dtype().unwrap());
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .allclose_native(expected, 1e-11, 1e-11, false)
        .unwrap());
}

fn verify(device: i64) {
    let real = tensor(&[1.0, 2.0], &[2], device);
    let imaginary = tensor(&[3.0, 4.0], &[2], device);
    let complex = real.complex(&imaginary).unwrap();
    let mut complex_output = real.complex(&imaginary).unwrap();
    complex_output.zero_in_place().unwrap();
    complex_output.complex_out(&real, &imaginary).unwrap();
    same(&complex_output, &complex, device);

    let left = tensor(&[1.0, 2.0], &[2], device);
    let right = tensor(&[4.0, 6.0], &[2], device);
    let distance = left.dist(&right, 2.0).unwrap();
    let mut distance_output = empty(device);
    distance_output.dist_out(&left, &right, 2.0).unwrap();
    same(&distance_output, &distance, device);

    let expandable = tensor(&[1.0, 2.0], &[1, 2], device);
    let expanded = expandable.expand_copy(&[3, 2], false).unwrap();
    let mut expanded_output = empty(device);
    expanded_output
        .expand_copy_out(&expandable, &[3, 2], false)
        .unwrap();
    same(&expanded_output, &expanded, device);

    let coefficient = tensor(&[2.0, 1.0, 1.0, 3.0], &[2, 2], device);
    let (factor, pivots) = coefficient.linalg_lu_factor(true).unwrap();
    let rhs = tensor(&[3.0, 4.0], &[2, 1], device);
    let solution = rhs.lu_solve(&factor, &pivots).unwrap();
    let mut solution_output = empty(device);
    solution_output
        .lu_solve_out(&rhs, &factor, &pivots)
        .unwrap();
    same(&solution_output, &solution, device);

    let gradient = tensor(&[1.0, 2.0], &[2], device);
    let mask = tensor(&[1.0, 0.0], &[2], device).to_dtype(11).unwrap();
    let dropout_gradient = gradient.native_dropout_backward(&mask, 2.0).unwrap();
    let mut dropout_output = empty(device);
    dropout_output
        .native_dropout_backward_out(&gradient, &mask, 2.0)
        .unwrap();
    same(&dropout_output, &dropout_gradient, device);

    let qr_source = tensor(&[1.0, 2.0, 3.0, 5.0], &[2, 2], device);
    let (qr_factor, tau) = qr_source.geqrf().unwrap();
    let other = tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2], device);
    let applied_q = qr_factor.ormqr(&tau, &other, true, false).unwrap();
    let mut ormqr_output = empty(device);
    ormqr_output
        .ormqr_out(&qr_factor, &tau, &other, true, false)
        .unwrap();
    same(&ormqr_output, &applied_q, device);

    let absolute = tensor(&[1.0, 2.0], &[2], device);
    let angle = tensor(&[0.0, 1.0], &[2], device);
    let polar = absolute.polar(&angle).unwrap();
    let mut polar_output = absolute.polar(&angle).unwrap();
    polar_output.zero_in_place().unwrap();
    polar_output.polar_out(&absolute, &angle).unwrap();
    same(&polar_output, &polar, device);

    let one_d = tensor(&[1.0, 2.0], &[1, 1, 2], device);
    let two_d = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], device);
    let three_d = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 2, 2],
        device,
    );
    let mut reflection1 = empty(device);
    let mut reflection2 = empty(device);
    let mut reflection3 = empty(device);
    let mut replication1 = empty(device);
    let mut replication2 = empty(device);
    let mut replication3 = empty(device);
    reflection1.reflection_pad1d_out(&one_d, &[1, 1]).unwrap();
    reflection2
        .reflection_pad2d_out(&two_d, &[1, 1, 1, 1])
        .unwrap();
    reflection3
        .reflection_pad3d_out(&three_d, &[1, 1, 1, 1, 1, 1])
        .unwrap();
    replication1.replication_pad1d_out(&one_d, &[1, 1]).unwrap();
    replication2
        .replication_pad2d_out(&two_d, &[1, 1, 1, 1])
        .unwrap();
    replication3
        .replication_pad3d_out(&three_d, &[1, 1, 1, 1, 1, 1])
        .unwrap();
    same(
        &reflection1,
        &one_d.reflection_pad1d(&[1, 1]).unwrap(),
        device,
    );
    same(
        &reflection2,
        &two_d.reflection_pad2d(&[1, 1, 1, 1]).unwrap(),
        device,
    );
    same(
        &reflection3,
        &three_d.reflection_pad3d(&[1, 1, 1, 1, 1, 1]).unwrap(),
        device,
    );
    same(
        &replication1,
        &one_d.replication_pad1d(&[1, 1]).unwrap(),
        device,
    );
    same(
        &replication2,
        &two_d.replication_pad2d(&[1, 1, 1, 1]).unwrap(),
        device,
    );
    same(
        &replication3,
        &three_d.replication_pad3d(&[1, 1, 1, 1, 1, 1]).unwrap(),
        device,
    );

    let renormed = coefficient.renorm(2.0, 0, 1.0).unwrap();
    let mut renorm_output = empty(device);
    renorm_output.renorm_out(&coefficient, 2.0, 0, 1.0).unwrap();
    same(&renorm_output, &renormed, device);

    let determinant_input = tensor(&[2.0, 0.0, 0.0, 3.0], &[2, 2], device);
    let (sign, log_absolute) = determinant_input.slogdet().unwrap();
    let mut sign_output = empty(device);
    let mut log_output = empty(device);
    sign_output
        .slogdet_out(&mut log_output, &determinant_input)
        .unwrap();
    same(&sign_output, &sign, device);
    same(&log_output, &log_absolute, device);

    let logits = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device);
    let probabilities = logits.softmax(1).unwrap();
    let mut softmax_output = empty(device);
    softmax_output.softmax_out(&logits, 1).unwrap();
    same(&softmax_output, &probabilities, device);

    assert!(reflection1.reflection_pad1d_out(&one_d, &[2, 0]).is_err());
    assert!(softmax_output.softmax_out(&logits, 7).is_err());
    same(&left, &tensor(&[1.0, 2.0], &[2], device), device);
    same(
        &coefficient,
        &tensor(&[2.0, 1.0, 1.0, 3.0], &[2, 2], device),
        device,
    );
}

#[test]
fn exact_retained_outputs_match_functional_kernels_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
