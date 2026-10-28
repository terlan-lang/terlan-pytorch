use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn check(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(&tensor(values, shape, device)).unwrap());
}

fn verify(device: i64) {
    let empty = Tensor::create_configured(
        0,
        &[2, 3],
        true,
        0,
        0.0,
        7,
        true,
        0,
        true,
        device,
        if device == 0 { -1 } else { 0 },
        true,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    assert_eq!(empty.shape().unwrap(), vec![2, 3]);
    assert_eq!(empty.dtype().unwrap(), 7);
    assert_eq!(empty.device_type().unwrap(), device);

    let integer_full = Tensor::create_configured(
        1,
        &[2, 2],
        true,
        -3,
        0.0,
        0,
        false,
        0,
        false,
        device,
        if device == 0 { -1 } else { 0 },
        true,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    assert_eq!(integer_full.dtype().unwrap(), 4);
    check(
        &integer_full.to_dtype(7).unwrap(),
        &[-3.; 4],
        &[2, 2],
        device,
    );

    let floating_full = Tensor::create_configured(
        1,
        &[2, 2],
        false,
        0,
        1.5,
        7,
        true,
        0,
        false,
        device,
        if device == 0 { -1 } else { 0 },
        true,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    check(&floating_full, &[1.5; 4], &[2, 2], device);

    let ones = Tensor::create_configured(
        2,
        &[2, 3],
        true,
        0,
        0.0,
        7,
        true,
        0,
        false,
        device,
        if device == 0 { -1 } else { 0 },
        true,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    let zeros = Tensor::create_configured(
        3,
        &[2, 3],
        true,
        0,
        0.0,
        7,
        true,
        0,
        false,
        device,
        if device == 0 { -1 } else { 0 },
        true,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    check(&ones, &[1.; 6], &[2, 3], device);
    check(&zeros, &[0.; 6], &[2, 3], device);

    let source = tensor(&[1., 2., 3., 4., 5., 6.], &[1, 2, 1, 3], device);
    let like_full = source
        .like_configured(
            1, false, 0, 2.5, 6, true, 0, false, 0, -1, false, false, false, 2, true,
        )
        .unwrap();
    assert_eq!(like_full.shape().unwrap(), vec![1, 2, 1, 3]);
    assert_eq!(like_full.dtype().unwrap(), 6);
    assert_eq!(like_full.device_type().unwrap(), device);
    assert_eq!(like_full.strides().unwrap(), vec![6, 1, 6, 2]);
    check(
        &like_full.to_dtype(7).unwrap(),
        &[2.5; 6],
        &[1, 2, 1, 3],
        device,
    );

    let inherited_zeros = source
        .like_configured(
            3, true, 0, 0.0, 0, false, 0, false, 0, -1, false, false, false, 0, false,
        )
        .unwrap();
    check(&inherited_zeros, &[0.; 6], &[1, 2, 1, 3], device);

    let mut output = tensor(&[9.], &[1], device);
    output
        .create_out(1, &[2, 3], true, 4, 0.0, 0, false)
        .unwrap();
    check(&output, &[4.; 6], &[2, 3], device);
    output.like_out(&source, 2, true, 0, 0.0, 0, false).unwrap();
    check(&output, &[1.; 6], &[1, 2, 1, 3], device);

    assert!(Tensor::create_configured(
        99,
        &[1],
        true,
        0,
        0.0,
        0,
        false,
        0,
        false,
        0,
        -1,
        false,
        false,
        false,
        0,
        false,
    )
    .is_err());
    assert!(source
        .like_configured(0, true, 0, 0.0, 0, false, 0, false, 0, -1, false, false, false, 99, true,)
        .is_err());
}

#[test]
fn cpu_creation_completion_covers_options_scalar_kinds_and_outputs() {
    verify(0);
}

#[test]
fn cuda_creation_completion_executes_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
