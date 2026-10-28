use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn output(length: usize, device: i64) -> Tensor {
    tensor(&vec![0.0; length], &[length as i64], device)
}

fn same(actual: &Tensor, values: &[f64], device: i64) {
    let expected = tensor(values, &[values.len() as i64], device);
    assert_eq!(actual.shape().unwrap(), [values.len() as i64]);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .allclose_native(&expected, 1e-12, 1e-12, false)
        .unwrap());
}

fn verify(device: i64) {
    let input = tensor(&[-2.0, -0.5, -0.25, 0.0, 0.25, 0.5, 2.0], &[7], device);
    let gradient = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[7], device);

    let hard_int = input.hardshrink_int(1).unwrap();
    let hard_float = input.hardshrink_float(0.5).unwrap();
    same(&hard_int, &[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], device);
    same(&hard_float, &[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], device);
    let mut hard_int_out = output(7, device);
    let mut hard_float_out = output(7, device);
    hard_int_out.hardshrink_out_int(&input, 1).unwrap();
    hard_float_out.hardshrink_out_float(&input, 0.5).unwrap();
    assert!(hard_int_out.equal_native(&hard_int).unwrap());
    assert!(hard_float_out.equal_native(&hard_float).unwrap());

    let hard_backward_int = gradient.hardshrink_backward_int(&input, 1).unwrap();
    let hard_backward_float = gradient.hardshrink_backward_float(&input, 0.5).unwrap();
    same(
        &hard_backward_int,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
        device,
    );
    same(
        &hard_backward_float,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
        device,
    );
    let mut hard_backward_int_out = output(7, device);
    let mut hard_backward_float_out = output(7, device);
    hard_backward_int_out
        .hardshrink_backward_out_int(&gradient, &input, 1)
        .unwrap();
    hard_backward_float_out
        .hardshrink_backward_out_float(&gradient, &input, 0.5)
        .unwrap();
    assert!(hard_backward_int_out
        .equal_native(&hard_backward_int)
        .unwrap());
    assert!(hard_backward_float_out
        .equal_native(&hard_backward_float)
        .unwrap());

    let soft_int = input.softshrink_int(1).unwrap();
    let soft_float = input.softshrink_float(0.5).unwrap();
    same(&soft_int, &[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device);
    same(&soft_float, &[-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5], device);
    let mut soft_int_out = output(7, device);
    let mut soft_float_out = output(7, device);
    soft_int_out.softshrink_out_int(&input, 1).unwrap();
    soft_float_out.softshrink_out_float(&input, 0.5).unwrap();
    assert!(soft_int_out.equal_native(&soft_int).unwrap());
    assert!(soft_float_out.equal_native(&soft_float).unwrap());

    let soft_backward_int = gradient.softshrink_backward_int(&input, 1).unwrap();
    let soft_backward_float = gradient.softshrink_backward_float(&input, 0.5).unwrap();
    same(
        &soft_backward_int,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
        device,
    );
    same(
        &soft_backward_float,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
        device,
    );
    let mut soft_backward_int_out = output(7, device);
    let mut soft_backward_float_out = output(7, device);
    soft_backward_int_out
        .softshrink_backward_out_int(&gradient, &input, 1)
        .unwrap();
    soft_backward_float_out
        .softshrink_backward_out_float(&gradient, &input, 0.5)
        .unwrap();
    assert!(soft_backward_int_out
        .equal_native(&soft_backward_int)
        .unwrap());
    assert!(soft_backward_float_out
        .equal_native(&soft_backward_float)
        .unwrap());

    let softplus_input = tensor(&[0.0, 2.0], &[2], device);
    let softplus_gradient = tensor(&[2.0, 3.0], &[2], device);
    let softplus_int = softplus_input.softplus_int(1, 1).unwrap();
    let softplus_float = softplus_input.softplus_float(0.5, 2.0).unwrap();
    same(&softplus_int, &[std::f64::consts::LN_2, 2.0], device);
    same(
        &softplus_float,
        &[1.3862943611198906, 2.6265233750364456],
        device,
    );
    let mut softplus_int_out = output(2, device);
    let mut softplus_float_out = output(2, device);
    softplus_int_out
        .softplus_out_int(&softplus_input, 1, 1)
        .unwrap();
    softplus_float_out
        .softplus_out_float(&softplus_input, 0.5, 2.0)
        .unwrap();
    assert!(softplus_int_out.equal_native(&softplus_int).unwrap());
    assert!(softplus_float_out.equal_native(&softplus_float).unwrap());

    let softplus_backward_int = softplus_gradient
        .softplus_backward_int(&softplus_input, 1, 1)
        .unwrap();
    let softplus_backward_float = softplus_gradient
        .softplus_backward_float(&softplus_input, 0.5, 2.0)
        .unwrap();
    same(&softplus_backward_int, &[1.0, 3.0], device);
    same(&softplus_backward_float, &[1.0, 2.193175735890015], device);
    let mut softplus_backward_int_out = output(2, device);
    let mut softplus_backward_float_out = output(2, device);
    softplus_backward_int_out
        .softplus_backward_out_int(&softplus_gradient, &softplus_input, 1, 1)
        .unwrap();
    softplus_backward_float_out
        .softplus_backward_out_float(&softplus_gradient, &softplus_input, 0.5, 2.0)
        .unwrap();
    assert!(softplus_backward_int_out
        .equal_native(&softplus_backward_int)
        .unwrap());
    assert!(softplus_backward_float_out
        .equal_native(&softplus_backward_float)
        .unwrap());

    let threshold_int = input.threshold_int(0, -1).unwrap();
    let threshold_float = input.threshold_float(0.25, -0.5).unwrap();
    same(
        &threshold_int,
        &[-1.0, -1.0, -1.0, -1.0, 0.25, 0.5, 2.0],
        device,
    );
    same(
        &threshold_float,
        &[-0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 2.0],
        device,
    );
    let mut threshold_int_out = output(7, device);
    let mut threshold_float_out = output(7, device);
    threshold_int_out.threshold_out_int(&input, 0, -1).unwrap();
    threshold_float_out
        .threshold_out_float(&input, 0.25, -0.5)
        .unwrap();
    assert!(threshold_int_out.equal_native(&threshold_int).unwrap());
    assert!(threshold_float_out.equal_native(&threshold_float).unwrap());

    let threshold_backward_int = gradient.threshold_backward_int(&input, 0).unwrap();
    let threshold_backward_float = gradient.threshold_backward_float(&input, 0.25).unwrap();
    same(
        &threshold_backward_int,
        &[0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0],
        device,
    );
    same(
        &threshold_backward_float,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0],
        device,
    );
    let mut threshold_backward_int_out = output(7, device);
    let mut threshold_backward_float_out = output(7, device);
    threshold_backward_int_out
        .threshold_backward_out_int(&gradient, &input, 0)
        .unwrap();
    threshold_backward_float_out
        .threshold_backward_out_float(&gradient, &input, 0.25)
        .unwrap();
    assert!(threshold_backward_int_out
        .equal_native(&threshold_backward_int)
        .unwrap());
    assert!(threshold_backward_float_out
        .equal_native(&threshold_backward_float)
        .unwrap());

    if device == 0 {
        assert!(input.softshrink_float(-0.5).is_err());
    }
    same(&input, &[-2.0, -0.5, -0.25, 0.0, 0.25, 0.5, 2.0], device);
    same(&gradient, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device);
}

#[test]
fn activation_variants_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
