use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn same(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    let expected = tensor(values, shape, device);
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .allclose_native(&expected, 1e-12, 1e-12, false)
        .unwrap());
}

fn verify(device: i64) {
    let y = tensor(&[1.0, 2.0, 4.0], &[3], device);
    let x = tensor(&[0.0, 1.0, 3.0], &[3], device);

    same(
        &y.cumulative_trapezoid_dx_int(1, -1).unwrap(),
        &[1.5, 4.5],
        &[2],
        device,
    );
    same(
        &y.cumulative_trapezoid_dx_float(0.5, 0).unwrap(),
        &[0.75, 2.25],
        &[2],
        device,
    );
    same(
        &y.cumulative_trapezoid_x(&x, 0).unwrap(),
        &[1.5, 7.5],
        &[2],
        device,
    );
    same(&y.trapezoid_dx_int(2, 0).unwrap(), &[9.0], &[], device);
    same(
        &y.trapezoid_dx_float(0.5, -1).unwrap(),
        &[2.25],
        &[],
        device,
    );
    same(&y.trapezoid_x(&x, 0).unwrap(), &[7.5], &[], device);
    same(&y.trapz_dx(1.0, -1).unwrap(), &[4.5], &[], device);
    same(&y.trapz_x(&x, 0).unwrap(), &[7.5], &[], device);

    let matrix = tensor(&[1.0, 2.0, 4.0, 2.0, 3.0, 5.0], &[2, 3], device);
    same(
        &matrix.cumulative_trapezoid_dx_int(1, 1).unwrap(),
        &[1.5, 4.5, 2.5, 6.5],
        &[2, 2],
        device,
    );

    let input = tensor(&[0.0, 0.0], &[2], device);
    let target = tensor(&[0.5, 0.5], &[2], device);
    let target_log = tensor(&[-0.6931471805599453, -0.6931471805599453], &[2], device);
    let probabilities = input.kl_div(&target, 2, false).unwrap();
    let logarithms = input.kl_div(&target_log, 2, true).unwrap();
    same(&probabilities, &[-0.6931471805599453], &[], device);
    assert!(probabilities
        .allclose_native(&logarithms, 1e-12, 1e-12, false)
        .unwrap());

    let leaf = tensor(&[1.0, 2.0, 4.0], &[3], device)
        .require_grad()
        .unwrap();
    leaf.trapezoid_x(&x, 0).unwrap().backward().unwrap();
    same(&leaf.gradient().unwrap(), &[0.5, 1.5, 1.0], &[3], device);

    assert!(y.cumulative_trapezoid_dx_int(1, 2).is_err());
    assert!(y
        .trapezoid_x(&tensor(&[0.0, 1.0], &[2], device), 0)
        .is_err());
    same(&y, &[1.0, 2.0, 4.0], &[3], device);
    same(&x, &[0.0, 1.0, 3.0], &[3], device);
}

#[test]
fn exact_integration_and_kl_div_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
