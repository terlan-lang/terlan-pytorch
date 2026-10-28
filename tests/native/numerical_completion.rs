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
    let expected = tensor(values, shape, device)
        .to_dtype(actual.dtype().unwrap())
        .unwrap();
    assert!(actual.equal_native(&expected).unwrap());
}

fn verify(device: i64) {
    let order = tensor(&[3., 1., 1., 2.], &[4], device);
    let mut indices = tensor(&[], &[0], device).to_dtype(4).unwrap();
    indices.argsort_stable_out(&order, true, 0, false).unwrap();
    check(&indices, &[1., 2., 3., 0.], &[4], device);
    assert_eq!(indices.dtype().unwrap(), 4);
    indices.argsort_stable_out(&order, true, 0, true).unwrap();
    check(&indices, &[0., 3., 1., 2.], &[4], device);

    let left = tensor(&[0., 0., 3., 4.], &[2, 2], device);
    let right = tensor(&[0., 4.], &[1, 2], device);
    check(
        &left.cdist_default_compute(&right, 2.).unwrap(),
        &[4., 3.],
        &[2, 1],
        device,
    );
    check(
        &left.cdist_compute(&right, 2., 1).unwrap(),
        &[4., 3.],
        &[2, 1],
        device,
    );

    let first = tensor(&[1., 2., 3., 4.], &[2, 2], device);
    let identity = tensor(&[1., 0., 0., 1.], &[2, 2], device);
    let column = tensor(&[1., 1.], &[2, 1], device);
    let matrices = [&first, &identity, &column];
    check(
        &Tensor::chain_matmul_native(&matrices).unwrap(),
        &[3., 7.],
        &[2, 1],
        device,
    );
    let mut chain_output = tensor(&[], &[0], device);
    chain_output.chain_matmul_out_native(&matrices).unwrap();
    check(&chain_output, &[3., 7.], &[2, 1], device);

    let transposed = first.transpose(0, 1).unwrap();
    let mut cloned = tensor(&[], &[0], device);
    cloned.clone_out_default_memory(&transposed).unwrap();
    check(&cloned, &[1., 3., 2., 4.], &[2, 2], device);
    cloned.clone_out_memory(&transposed, 0).unwrap();
    check(&cloned, &[1., 3., 2., 4.], &[2, 2], device);
    let mut contiguous = transposed.contiguous_native(0).unwrap();
    check(&contiguous, &[1., 3., 2., 4.], &[2, 2], device);
    assert_eq!(contiguous.strides().unwrap(), [2, 1]);
    contiguous.zero_in_place().unwrap();
    check(&transposed, &[1., 3., 2., 4.], &[2, 2], device);
    let mut aliased = first.contiguous_native(0).unwrap();
    aliased.zero_in_place().unwrap();
    check(&first, &[0., 0., 0., 0.], &[2, 2], device);

    let x = tensor(&[1., 0., 0.], &[3], device);
    let y = tensor(&[0., 1., 0.], &[3], device);
    check(
        &x.cross_default_dimension(&y).unwrap(),
        &[0., 0., 1.],
        &[3],
        device,
    );
    check(
        &x.cross_dimension(&y, -1).unwrap(),
        &[0., 0., 1.],
        &[3],
        device,
    );
    let mut cross_output = tensor(&[], &[0], device);
    cross_output.cross_out_default_dimension(&x, &y).unwrap();
    check(&cross_output, &[0., 0., 1.], &[3], device);
    cross_output.cross_out_dimension(&x, &y, -1).unwrap();
    check(&cross_output, &[0., 0., 1.], &[3], device);

    assert!(indices.argsort_stable_out(&order, true, 1, false).is_err());
    assert!(order.cdist_default_compute(&right, 2.).is_err());
    check(
        &Tensor::chain_matmul_native(&[&identity]).unwrap(),
        &[1., 0., 0., 1.],
        &[2, 2],
        device,
    );
    assert!(Tensor::chain_matmul_native(&[]).is_err());
    assert!(transposed.contiguous_native(99).is_err());
    assert!(x.cross_dimension(&y, 1).is_err());
    check(&order, &[3., 1., 1., 2.], &[4], device);
    check(&x, &[1., 0., 0.], &[3], device);
    check(&y, &[0., 1., 0.], &[3], device);

    let leaf = tensor(&[1., 2., 3., 4.], &[2, 2], device)
        .require_grad()
        .unwrap();
    Tensor::chain_matmul_native(&[&leaf, &identity])
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(
        &leaf.gradient().unwrap(),
        &[1., 1., 1., 1.],
        &[2, 2],
        device,
    );
}

#[test]
fn numerical_completion_values_aliasing_errors_and_gradients_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
