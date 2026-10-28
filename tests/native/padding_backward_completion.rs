use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn zeros(shape: &[i64], device: i64) -> Tensor {
    let length = shape.iter().product::<i64>() as usize;
    tensor(&vec![0.0; length], shape, device)
}

fn same(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    let expected = tensor(values, shape, device);
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(&expected).unwrap());
}

fn verify(device: i64) {
    let input1 = tensor(&[1.0, 2.0], &[1, 1, 2], device);
    let gradient1 = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], device);
    let reflection1 = gradient1
        .reflection_pad1d_backward(&input1, &[1, 1])
        .unwrap();
    let replication1 = gradient1
        .replication_pad1d_backward(&input1, &[1, 1])
        .unwrap();
    same(&reflection1, &[6.0, 4.0], &[1, 1, 2], device);
    same(&replication1, &[3.0, 7.0], &[1, 1, 2], device);
    let mut reflection1_out = zeros(&[1, 1, 2], device);
    let mut replication1_out = zeros(&[1, 1, 2], device);
    reflection1_out
        .reflection_pad1d_backward_out(&gradient1, &input1, &[1, 1])
        .unwrap();
    replication1_out
        .replication_pad1d_backward_out(&gradient1, &input1, &[1, 1])
        .unwrap();
    assert!(reflection1_out.equal_native(&reflection1).unwrap());
    assert!(replication1_out.equal_native(&replication1).unwrap());

    let input2 = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], device);
    let gradient2 = tensor(&[1.0; 16], &[1, 1, 4, 4], device);
    let reflection2 = gradient2
        .reflection_pad2d_backward(&input2, &[1, 1, 1, 1])
        .unwrap();
    let replication2 = gradient2
        .replication_pad2d_backward(&input2, &[1, 1, 1, 1])
        .unwrap();
    same(&reflection2, &[4.0; 4], &[1, 1, 2, 2], device);
    same(&replication2, &[4.0; 4], &[1, 1, 2, 2], device);
    let mut reflection2_out = zeros(&[1, 1, 2, 2], device);
    let mut replication2_out = zeros(&[1, 1, 2, 2], device);
    reflection2_out
        .reflection_pad2d_backward_out(&gradient2, &input2, &[1, 1, 1, 1])
        .unwrap();
    replication2_out
        .replication_pad2d_backward_out(&gradient2, &input2, &[1, 1, 1, 1])
        .unwrap();
    assert!(reflection2_out.equal_native(&reflection2).unwrap());
    assert!(replication2_out.equal_native(&replication2).unwrap());

    let input3 = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 2, 2],
        device,
    );
    let gradient3 = tensor(&[1.0; 64], &[1, 1, 4, 4, 4], device);
    let padding3 = [1, 1, 1, 1, 1, 1];
    let reflection3 = gradient3
        .reflection_pad3d_backward(&input3, &padding3)
        .unwrap();
    let replication3 = gradient3
        .replication_pad3d_backward(&input3, &padding3)
        .unwrap();
    same(&reflection3, &[8.0; 8], &[1, 1, 2, 2, 2], device);
    same(&replication3, &[8.0; 8], &[1, 1, 2, 2, 2], device);
    let mut reflection3_out = zeros(&[1, 1, 2, 2, 2], device);
    let mut replication3_out = zeros(&[1, 1, 2, 2, 2], device);
    reflection3_out
        .reflection_pad3d_backward_out(&gradient3, &input3, &padding3)
        .unwrap();
    replication3_out
        .replication_pad3d_backward_out(&gradient3, &input3, &padding3)
        .unwrap();
    assert!(reflection3_out.equal_native(&reflection3).unwrap());
    assert!(replication3_out.equal_native(&replication3).unwrap());

    if device == 0 {
        assert!(gradient1
            .reflection_pad1d_backward(&input1, &[2, 0])
            .is_err());
    }
    same(&input1, &[1.0, 2.0], &[1, 1, 2], device);
    same(&input2, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], device);
    same(
        &input3,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 2, 2],
        device,
    );
}

#[test]
fn padding_backward_variants_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
