use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn check(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(&tensor(values, shape, device)).unwrap());
}

fn verify(device: i64) {
    let values = [-1., 2., -3., 4.];
    let shape = [2, 2];

    let mut alpha = tensor(&values, &shape, device);
    alpha.alpha_dropout_in_place(0., true).unwrap();
    check(&alpha, &values, &shape, device);
    alpha.alpha_dropout_in_place(1., true).unwrap();
    check(&alpha, &[0.; 4], &shape, device);

    let mut dropout = tensor(&values, &shape, device);
    dropout.dropout_in_place(0.7, false).unwrap();
    check(&dropout, &values, &shape, device);
    dropout.dropout_in_place(1., true).unwrap();
    check(&dropout, &[0.; 4], &shape, device);

    let source = tensor(&values, &shape, device);
    check(
        &source.feature_alpha_dropout(0., true).unwrap(),
        &values,
        &shape,
        device,
    );
    check(
        &source.feature_alpha_dropout(1., true).unwrap(),
        &[0.; 4],
        &shape,
        device,
    );
    check(&source, &values, &shape, device);

    let mut feature_alpha = tensor(&values, &shape, device);
    feature_alpha
        .feature_alpha_dropout_in_place(1., true)
        .unwrap();
    check(&feature_alpha, &[0.; 4], &shape, device);

    let mut feature = tensor(&values, &shape, device);
    feature.feature_dropout_in_place(0., true).unwrap();
    check(&feature, &values, &shape, device);
    feature.feature_dropout_in_place(1., true).unwrap();
    check(&feature, &[0.; 4], &shape, device);

    let mut invalid = tensor(&values, &shape, device);
    assert!(invalid.dropout_in_place(-0.1, true).is_err());
    check(&invalid, &values, &shape, device);
    assert!(invalid.feature_alpha_dropout_in_place(1.1, true).is_err());
    check(&invalid, &values, &shape, device);
}

#[test]
fn dropout_completion_values_mutation_errors_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
