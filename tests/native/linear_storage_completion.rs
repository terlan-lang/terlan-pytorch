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
    let input1 = tensor(&[1., 2.], &[1, 2], device);
    let input2 = tensor(&[3., 4.], &[1, 2], device);
    let weight = tensor(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2], device);
    let bias = tensor(&[1., 2.], &[2], device);
    check(
        &input1.bilinear_no_bias(&input2, &weight).unwrap(),
        &[61., 145.],
        &[1, 2],
        device,
    );
    check(
        &input1.bilinear_with_bias(&input2, &weight, &bias).unwrap(),
        &[62., 147.],
        &[1, 2],
        device,
    );
    check(&input1, &[1., 2.], &[1, 2], device);
    check(&input2, &[3., 4.], &[1, 2], device);
    check(
        &weight,
        &[1., 2., 3., 4., 5., 6., 7., 8.],
        &[2, 2, 2],
        device,
    );

    let temporal_input = tensor(&[1., 2., 3.], &[3, 1, 1], device);
    let temporal_weight = tensor(&[2., 1.], &[2, 1, 1], device);
    let temporal_bias = tensor(&[1.], &[1], device);
    check(
        &temporal_input
            .conv_tbc(&temporal_weight, &temporal_bias, 0)
            .unwrap(),
        &[5., 8.],
        &[2, 1, 1],
        device,
    );
    let mut temporal_output = tensor(&[], &[0], device);
    temporal_output
        .conv_tbc_out(&temporal_input, &temporal_weight, &temporal_bias, 0)
        .unwrap();
    check(&temporal_output, &[5., 8.], &[2, 1, 1], device);
    temporal_output.zero_in_place().unwrap();
    check(&temporal_input, &[1., 2., 3.], &[3, 1, 1], device);

    let temporal_gradient = tensor(&[1., 1.], &[2, 1, 1], device);
    let (input_gradient, weight_gradient, bias_gradient) = temporal_gradient
        .conv_tbc_backward(&temporal_input, &temporal_weight, &temporal_bias, 0)
        .unwrap();
    check(&input_gradient, &[2., 3., 1.], &[3, 1, 1], device);
    check(&weight_gradient, &[3., 5.], &[2, 1, 1], device);
    check(&bias_gradient, &[2.], &[1], device);

    let cumulative_gradient = tensor(&[1., 2., 3.], &[3], device);
    let cumulative_input = tensor(&[3., 1., 2.], &[3], device);
    let cumulative_indices = tensor(&[0., 0., 0.], &[3], device).to_dtype(4).unwrap();
    check(
        &cumulative_gradient
            .cummaxmin_backward(&cumulative_input, &cumulative_indices, 0)
            .unwrap(),
        &[6., 0., 0.],
        &[3],
        device,
    );

    let reset_source = tensor(&[1., 2.], &[2], device);
    let reset = reset_source.set_empty().unwrap();
    check(&reset, &[], &[0], device);
    check(&reset_source, &[1., 2.], &[2], device);
    let mut reset_output = tensor(&[], &[0], device);
    reset_output.set_empty_out(&reset_source).unwrap();
    check(&reset_output, &[], &[0], device);
    check(&reset_source, &[1., 2.], &[2], device);
    let mut reset_mutation = tensor(&[1., 2.], &[2], device);
    reset_mutation.set_empty_in_place().unwrap();
    check(&reset_mutation, &[], &[0], device);

    let rebinding_input = tensor(&[1., 2.], &[2], device);
    let rebinding_source = tensor(&[3., 4., 5.], &[3], device);
    let mut rebound = rebinding_input
        .set_source_tensor(&rebinding_source)
        .unwrap();
    check(&rebound, &[3., 4., 5.], &[3], device);
    check(&rebinding_input, &[1., 2.], &[2], device);
    rebound.zero_in_place().unwrap();
    check(&rebinding_source, &[0., 0., 0.], &[3], device);

    let output_input = tensor(&[1., 2.], &[2], device);
    let output_source = tensor(&[6., 7., 8.], &[3], device);
    let mut rebound_output = tensor(&[], &[0], device);
    rebound_output
        .set_source_tensor_out(&output_input, &output_source)
        .unwrap();
    check(&rebound_output, &[6., 7., 8.], &[3], device);
    check(&output_input, &[1., 2.], &[2], device);
    rebound_output.zero_in_place().unwrap();
    check(&output_source, &[6., 7., 8.], &[3], device);

    let bad_weight = tensor(&[1., 2.], &[2], device);
    assert!(input1.bilinear_no_bias(&input2, &bad_weight).is_err());
    assert!(temporal_input
        .conv_tbc(&bad_weight, &temporal_bias, 0)
        .is_err());
    let float_indices = tensor(&[0., 0., 0.], &[3], device);
    assert!(cumulative_gradient
        .cummaxmin_backward(&cumulative_input, &float_indices, 0)
        .is_err());
    assert!(cumulative_gradient
        .cummaxmin_backward(&cumulative_input, &cumulative_indices, 1)
        .is_err());

    let leaf1 = tensor(&[1., 2.], &[1, 2], device).require_grad().unwrap();
    let leaf2 = tensor(&[3., 4.], &[1, 2], device).require_grad().unwrap();
    let leaf_weight = tensor(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2], device)
        .require_grad()
        .unwrap();
    let leaf_bias = tensor(&[1., 2.], &[2], device).require_grad().unwrap();
    leaf1
        .bilinear_with_bias(&leaf2, &leaf_weight, &leaf_bias)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(&leaf1.gradient().unwrap(), &[50., 78.], &[1, 2], device);
    check(&leaf2.gradient().unwrap(), &[26., 32.], &[1, 2], device);
    check(
        &leaf_weight.gradient().unwrap(),
        &[3., 4., 6., 8., 3., 4., 6., 8.],
        &[2, 2, 2],
        device,
    );
    check(&leaf_bias.gradient().unwrap(), &[1., 1.], &[2], device);
}

#[test]
fn linear_storage_values_aliasing_errors_and_gradients_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
