use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], device: i64) -> Tensor {
    Tensor::from_floats(values, &[values.len() as i64])
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn zeros(length: usize, device: i64) -> Tensor {
    tensor(&vec![0.0; length], device)
}

fn same(actual: &Tensor, expected: &[f64], device: i64) {
    let expected_tensor = tensor(expected, device);
    assert_eq!(actual.shape().unwrap(), [expected.len() as i64]);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .allclose_native(&expected_tensor, 1e-12, 1e-12, false)
        .unwrap());
}

fn verify(device: i64) {
    let input = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], device);
    let gradient = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], device);

    let celu_int = input.celu_int(1).unwrap();
    let celu_float = input.celu_float(0.5).unwrap();
    same(
        &celu_int,
        &[-0.8646647167633873, -0.6321205588285577, 0.0, 1.0, 2.0],
        device,
    );
    same(
        &celu_float,
        &[-0.4908421805556329, -0.43233235838169365, 0.0, 1.0, 2.0],
        device,
    );
    let mut celu_int_out = zeros(5, device);
    let mut celu_float_out = zeros(5, device);
    celu_int_out.celu_out_int(&input, 1).unwrap();
    celu_float_out.celu_out_float(&input, 0.5).unwrap();
    assert!(celu_int_out.equal_native(&celu_int).unwrap());
    assert!(celu_float_out.equal_native(&celu_float).unwrap());
    let mut celu_int_in_place = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], device);
    let mut celu_float_in_place = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], device);
    celu_int_in_place.celu_in_place_int(1).unwrap();
    celu_float_in_place.celu_in_place_float(0.5).unwrap();
    assert!(celu_int_in_place.equal_native(&celu_int).unwrap());
    assert!(celu_float_in_place.equal_native(&celu_float).unwrap());

    let elu_int = input.elu_int(1, 1, 1).unwrap();
    let elu_float = input.elu_float(0.5, 2.0, 0.5).unwrap();
    same(
        &elu_int,
        &[-0.8646647167633873, -0.6321205588285577, 0.0, 1.0, 2.0],
        device,
    );
    same(
        &elu_float,
        &[-0.6321205588285577, -0.3934693402873666, 0.0, 2.0, 4.0],
        device,
    );
    let mut elu_int_out = zeros(5, device);
    let mut elu_float_out = zeros(5, device);
    elu_int_out.elu_out_int(&input, 1, 1, 1).unwrap();
    elu_float_out.elu_out_float(&input, 0.5, 2.0, 0.5).unwrap();
    assert!(elu_int_out.equal_native(&elu_int).unwrap());
    assert!(elu_float_out.equal_native(&elu_float).unwrap());
    let mut elu_int_in_place = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], device);
    let mut elu_float_in_place = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0], device);
    elu_int_in_place.elu_in_place_int(1, 1, 1).unwrap();
    elu_float_in_place
        .elu_in_place_float(0.5, 2.0, 0.5)
        .unwrap();
    assert!(elu_int_in_place.equal_native(&elu_int).unwrap());
    assert!(elu_float_in_place.equal_native(&elu_float).unwrap());
    let elu_backward_int = gradient.elu_backward_int(1, 1, 1, false, &input).unwrap();
    let elu_backward_float = gradient
        .elu_backward_float(0.5, 2.0, 0.5, false, &input)
        .unwrap();
    same(
        &elu_backward_int,
        &[0.1353352832366127, 0.7357588823428847, 3.0, 4.0, 5.0],
        device,
    );
    same(
        &elu_backward_float,
        &[0.18393972058572117, 0.6065306597126334, 1.5, 8.0, 10.0],
        device,
    );
    let elu_backward_result = gradient
        .elu_backward_float(0.5, 2.0, 0.5, true, &elu_float)
        .unwrap();
    assert!(elu_backward_result
        .allclose_native(&elu_backward_float, 1e-12, 1e-12, false)
        .unwrap());
    let mut elu_backward_int_out = zeros(5, device);
    let mut elu_backward_float_out = zeros(5, device);
    elu_backward_int_out
        .elu_backward_out_int(&gradient, 1, 1, 1, false, &input)
        .unwrap();
    elu_backward_float_out
        .elu_backward_out_float(&gradient, 0.5, 2.0, 0.5, false, &input)
        .unwrap();
    assert!(elu_backward_int_out
        .equal_native(&elu_backward_int)
        .unwrap());
    assert!(elu_backward_float_out
        .equal_native(&elu_backward_float)
        .unwrap());

    let hard_input = tensor(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], device);
    let hard_gradient = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device);
    let hard_int = hard_input.hardtanh_int(-1, 1).unwrap();
    let hard_float = hard_input.hardtanh_float(-0.5, 0.5).unwrap();
    same(&hard_int, &[-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0], device);
    same(&hard_float, &[-0.5, -0.5, -0.5, 0.0, 0.5, 0.5, 0.5], device);
    let mut hard_int_out = zeros(7, device);
    let mut hard_float_out = zeros(7, device);
    hard_int_out.hardtanh_out_int(&hard_input, -1, 1).unwrap();
    hard_float_out
        .hardtanh_out_float(&hard_input, -0.5, 0.5)
        .unwrap();
    assert!(hard_int_out.equal_native(&hard_int).unwrap());
    assert!(hard_float_out.equal_native(&hard_float).unwrap());
    let mut hard_int_in_place = tensor(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], device);
    let mut hard_float_in_place = tensor(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], device);
    hard_int_in_place.hardtanh_in_place_int(-1, 1).unwrap();
    hard_float_in_place
        .hardtanh_in_place_float(-0.5, 0.5)
        .unwrap();
    assert!(hard_int_in_place.equal_native(&hard_int).unwrap());
    assert!(hard_float_in_place.equal_native(&hard_float).unwrap());
    let hard_backward_int = hard_gradient
        .hardtanh_backward_int(&hard_input, -1, 1)
        .unwrap();
    let hard_backward_float = hard_gradient
        .hardtanh_backward_float(&hard_input, -0.5, 0.5)
        .unwrap();
    same(
        &hard_backward_int,
        &[0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0],
        device,
    );
    same(
        &hard_backward_float,
        &[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        device,
    );
    let mut hard_backward_int_out = zeros(7, device);
    let mut hard_backward_float_out = zeros(7, device);
    hard_backward_int_out
        .hardtanh_backward_out_int(&hard_gradient, &hard_input, -1, 1)
        .unwrap();
    hard_backward_float_out
        .hardtanh_backward_out_float(&hard_gradient, &hard_input, -0.5, 0.5)
        .unwrap();
    assert!(hard_backward_int_out
        .equal_native(&hard_backward_int)
        .unwrap());
    assert!(hard_backward_float_out
        .equal_native(&hard_backward_float)
        .unwrap());

    let leaky_input = tensor(&[-2.0, 0.0, 2.0], device);
    let leaky_gradient = tensor(&[1.0, 2.0, 3.0], device);
    let leaky_int = leaky_input.leaky_relu_int(1).unwrap();
    let leaky_float = leaky_input.leaky_relu_float(0.25).unwrap();
    same(&leaky_int, &[-2.0, 0.0, 2.0], device);
    same(&leaky_float, &[-0.5, 0.0, 2.0], device);
    let mut leaky_int_out = zeros(3, device);
    let mut leaky_float_out = zeros(3, device);
    leaky_int_out.leaky_relu_out_int(&leaky_input, 1).unwrap();
    leaky_float_out
        .leaky_relu_out_float(&leaky_input, 0.25)
        .unwrap();
    assert!(leaky_int_out.equal_native(&leaky_int).unwrap());
    assert!(leaky_float_out.equal_native(&leaky_float).unwrap());
    let mut leaky_int_in_place = tensor(&[-2.0, 0.0, 2.0], device);
    let mut leaky_float_in_place = tensor(&[-2.0, 0.0, 2.0], device);
    leaky_int_in_place.leaky_relu_in_place_int(1).unwrap();
    leaky_float_in_place
        .leaky_relu_in_place_float(0.25)
        .unwrap();
    assert!(leaky_int_in_place.equal_native(&leaky_int).unwrap());
    assert!(leaky_float_in_place.equal_native(&leaky_float).unwrap());
    let leaky_backward_int = leaky_gradient
        .leaky_relu_backward_int(&leaky_input, 1, false)
        .unwrap();
    let leaky_backward_float = leaky_gradient
        .leaky_relu_backward_float(&leaky_input, 0.25, false)
        .unwrap();
    same(&leaky_backward_int, &[1.0, 2.0, 3.0], device);
    same(&leaky_backward_float, &[0.25, 0.5, 3.0], device);
    let leaky_backward_result = leaky_gradient
        .leaky_relu_backward_float(&leaky_float, 0.25, true)
        .unwrap();
    assert!(leaky_backward_result
        .equal_native(&leaky_backward_float)
        .unwrap());
    let mut leaky_backward_int_out = zeros(3, device);
    let mut leaky_backward_float_out = zeros(3, device);
    leaky_backward_int_out
        .leaky_relu_backward_out_int(&leaky_gradient, &leaky_input, 1, false)
        .unwrap();
    leaky_backward_float_out
        .leaky_relu_backward_out_float(&leaky_gradient, &leaky_input, 0.25, false)
        .unwrap();
    assert!(leaky_backward_int_out
        .equal_native(&leaky_backward_int)
        .unwrap());
    assert!(leaky_backward_float_out
        .equal_native(&leaky_backward_float)
        .unwrap());

    let inverted_bounds = hard_input.hardtanh_int(2, 1).unwrap();
    same(
        &inverted_bounds,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        device,
    );
    same(&input, &[-2.0, -1.0, 0.0, 1.0, 2.0], device);
    same(&gradient, &[1.0, 2.0, 3.0, 4.0, 5.0], device);
}

#[test]
fn extended_activation_variants_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
