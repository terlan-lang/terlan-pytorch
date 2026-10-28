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
    let input1 = tensor(&[1., 2., 3.], &[1, 1, 3], device);
    let weight1 = tensor(&[1., 1.], &[1, 1, 2], device);
    let bias = tensor(&[1.], &[1], device);
    check(
        &input1
            .conv1d_no_bias(&weight1, &[1], &[0], &[1], 1)
            .unwrap(),
        &[3., 5.],
        &[1, 1, 2],
        device,
    );
    check(
        &input1
            .conv1d_with_bias(&weight1, &bias, &[1], &[0], &[1], 1)
            .unwrap(),
        &[4., 6.],
        &[1, 1, 2],
        device,
    );
    check(
        &input1
            .conv1d_padding_no_bias(&weight1, &[1], "valid", &[1], 1)
            .unwrap(),
        &[3., 5.],
        &[1, 1, 2],
        device,
    );
    check(
        &input1
            .conv1d_padding_with_bias(&weight1, &bias, &[1], "same", &[1], 1)
            .unwrap(),
        &[4., 6., 4.],
        &[1, 1, 3],
        device,
    );
    let grouped_input = tensor(&[1., 2., 3., 4.], &[1, 2, 2], device);
    let grouped_weight = tensor(&[2., 3.], &[2, 1, 1], device);
    check(
        &grouped_input
            .conv1d_no_bias(&grouped_weight, &[1], &[0], &[1], 2)
            .unwrap(),
        &[2., 4., 9., 12.],
        &[1, 2, 2],
        device,
    );

    let input2 = tensor(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], &[1, 1, 3, 3], device);
    let weight2 = tensor(&[1., 1., 1., 1.], &[1, 1, 2, 2], device);
    check(
        &input2
            .conv2d_no_bias(&weight2, &[1, 1], &[0, 0], &[1, 1], 1)
            .unwrap(),
        &[12., 16., 24., 28.],
        &[1, 1, 2, 2],
        device,
    );
    check(
        &input2
            .conv2d_with_bias(&weight2, &bias, &[1, 1], &[0, 0], &[1, 1], 1)
            .unwrap(),
        &[13., 17., 25., 29.],
        &[1, 1, 2, 2],
        device,
    );
    check(
        &input2
            .conv2d_padding_no_bias(&weight2, &[1, 1], "valid", &[1, 1], 1)
            .unwrap(),
        &[12., 16., 24., 28.],
        &[1, 1, 2, 2],
        device,
    );
    check(
        &input2
            .conv2d_padding_with_bias(&weight2, &bias, &[1, 1], "valid", &[1, 1], 1)
            .unwrap(),
        &[13., 17., 25., 29.],
        &[1, 1, 2, 2],
        device,
    );

    let input3 = tensor(&[1., 2., 3., 4., 5., 6., 7., 8.], &[1, 1, 2, 2, 2], device);
    let weight3 = tensor(&[1.; 8], &[1, 1, 2, 2, 2], device);
    check(
        &input3
            .conv3d_no_bias(&weight3, &[1, 1, 1], &[0, 0, 0], &[1, 1, 1], 1)
            .unwrap(),
        &[36.],
        &[1, 1, 1, 1, 1],
        device,
    );
    check(
        &input3
            .conv3d_with_bias(&weight3, &bias, &[1, 1, 1], &[0, 0, 0], &[1, 1, 1], 1)
            .unwrap(),
        &[37.],
        &[1, 1, 1, 1, 1],
        device,
    );
    check(
        &input3
            .conv3d_padding_no_bias(&weight3, &[1, 1, 1], "valid", &[1, 1, 1], 1)
            .unwrap(),
        &[36.],
        &[1, 1, 1, 1, 1],
        device,
    );
    check(
        &input3
            .conv3d_padding_with_bias(&weight3, &bias, &[1, 1, 1], "valid", &[1, 1, 1], 1)
            .unwrap(),
        &[37.],
        &[1, 1, 1, 1, 1],
        device,
    );

    let transposed1_input = tensor(&[1., 2.], &[1, 1, 2], device);
    check(
        &transposed1_input
            .conv_transpose1d_no_bias(&weight1, &[1], &[0], &[0], 1, &[1])
            .unwrap(),
        &[1., 3., 2.],
        &[1, 1, 3],
        device,
    );
    check(
        &transposed1_input
            .conv_transpose1d_with_bias(&weight1, &bias, &[1], &[0], &[0], 1, &[1])
            .unwrap(),
        &[2., 4., 3.],
        &[1, 1, 3],
        device,
    );
    let transposed2_input = tensor(&[1., 2., 3., 4.], &[1, 1, 2, 2], device);
    let transposed2 = [1., 3., 2., 4., 10., 6., 3., 7., 4.];
    check(
        &transposed2_input
            .conv_transpose2d_no_bias(&weight2, &[1, 1], &[0, 0], &[0, 0], 1, &[1, 1])
            .unwrap(),
        &transposed2,
        &[1, 1, 3, 3],
        device,
    );
    check(
        &transposed2_input
            .conv_transpose2d_with_bias(&weight2, &bias, &[1, 1], &[0, 0], &[0, 0], 1, &[1, 1])
            .unwrap(),
        &[2., 4., 3., 5., 11., 7., 4., 8., 5.],
        &[1, 1, 3, 3],
        device,
    );
    let transposed3_input = tensor(&[2.], &[1, 1, 1, 1, 1], device);
    check(
        &transposed3_input
            .conv_transpose3d_no_bias(&weight3, &[1, 1, 1], &[0, 0, 0], &[0, 0, 0], 1, &[1, 1, 1])
            .unwrap(),
        &[2.; 8],
        &[1, 1, 2, 2, 2],
        device,
    );
    check(
        &transposed3_input
            .conv_transpose3d_with_bias(
                &weight3,
                &bias,
                &[1, 1, 1],
                &[0, 0, 0],
                &[0, 0, 0],
                1,
                &[1, 1, 1],
            )
            .unwrap(),
        &[3.; 8],
        &[1, 1, 2, 2, 2],
        device,
    );

    assert!(input1
        .conv1d_no_bias(&weight1, &[1, 1], &[0], &[1], 1)
        .is_err());
    assert!(input1
        .conv1d_padding_no_bias(&weight1, &[2], "same", &[1], 1)
        .is_err());
    assert!(input1
        .conv1d_no_bias(&weight1, &[1], &[0], &[1], 0)
        .is_err());
    if device == 1 {
        let cpu_weight = tensor(&[1., 1.], &[1, 1, 2], 0);
        assert!(input1
            .conv1d_no_bias(&cpu_weight, &[1], &[0], &[1], 1)
            .is_err());
    }
    check(&input1, &[1., 2., 3.], &[1, 1, 3], device);
    check(&weight1, &[1., 1.], &[1, 1, 2], device);

    let leaf = tensor(&[1., 2., 3.], &[1, 1, 3], device)
        .require_grad()
        .unwrap();
    leaf.conv1d_no_bias(&weight1, &[1], &[1], &[1], 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(&leaf.gradient().unwrap(), &[2., 2., 2.], &[1, 1, 3], device);
}

#[test]
fn convolution_values_optionals_errors_and_gradients_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
