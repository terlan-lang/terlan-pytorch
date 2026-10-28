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
    let scalar = tensor(&[5.], &[], device);
    let vector = tensor(&[1., 2.], &[2], device);
    let matrix = tensor(&[1., 2., 3., 4.], &[2, 2], device);

    let mut rank_one = scalar.atleast_1d_sequence(&[&vector, &matrix]).unwrap();
    let rank_two = scalar.atleast_2d_sequence(&[&vector, &matrix]).unwrap();
    let rank_three = scalar.atleast_3d_sequence(&[&vector, &matrix]).unwrap();
    assert_eq!(rank_one.len(), 3);
    assert_eq!(rank_two.len(), 3);
    assert_eq!(rank_three.len(), 3);
    check(&rank_one[0], &[5.], &[1], device);
    check(&rank_one[1], &[1., 2.], &[2], device);
    check(&rank_one[2], &[1., 2., 3., 4.], &[2, 2], device);
    check(&rank_two[0], &[5.], &[1, 1], device);
    check(&rank_two[1], &[1., 2.], &[1, 2], device);
    check(&rank_two[2], &[1., 2., 3., 4.], &[2, 2], device);
    check(&rank_three[0], &[5.], &[1, 1, 1], device);
    check(&rank_three[1], &[1., 2.], &[1, 2, 1], device);
    check(&rank_three[2], &[1., 2., 3., 4.], &[2, 2, 1], device);

    rank_one[0].zero_in_place().unwrap();
    check(&scalar, &[0.], &[], device);
    drop(scalar);
    drop(vector);
    drop(matrix);
    check(&rank_one[1], &[1., 2.], &[2], device);
    check(&rank_two[2], &[1., 2., 3., 4.], &[2, 2], device);

    let matrix = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3], device);
    let transposed = matrix.transpose(0, 1).unwrap();
    let non_contiguous = transposed.atleast_3d_sequence(&[]).unwrap();
    assert_eq!(non_contiguous.len(), 1);
    check(
        &non_contiguous[0],
        &[1., 4., 2., 5., 3., 6.],
        &[3, 2, 1],
        device,
    );

    let leaf = tensor(&[2., 3.], &[2], device).require_grad().unwrap();
    let normalized = leaf.atleast_3d_sequence(&[]).unwrap();
    normalized[0].sum().unwrap().backward().unwrap();
    check(&leaf.gradient().unwrap(), &[1., 1.], &[2], device);
}

#[test]
fn cpu_atleast_sequences_preserve_values_views_lifetimes_and_gradients() {
    verify(0);
}

#[test]
fn cuda_atleast_sequences_execute_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
