use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn empty(device: i64) -> Tensor {
    tensor(&[], &[0], device)
}

fn same(actual: &Tensor, values: &[f64], shape: &[i64], dtype: i64, device: i64) {
    let expected = tensor(values, shape, device).to_dtype(dtype).unwrap();
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), dtype);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual
        .allclose_native(&expected, 1e-6, 1e-6, false)
        .unwrap());
}

fn verify(device: i64) {
    let input = tensor(&[1.0, 2.0], &[1, 2], device);
    let weight = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device);
    let bias = tensor(&[0.5, -0.5], &[2], device);
    let linear_plain = input.linear_no_bias(&weight).unwrap();
    let linear_biased = input.linear_with_bias(&weight, &bias).unwrap();
    same(&linear_plain, &[5.0, 11.0], &[1, 2], 7, device);
    same(&linear_biased, &[5.5, 10.5], &[1, 2], 7, device);
    let mut linear_plain_out = empty(device);
    let mut linear_biased_out = empty(device);
    linear_plain_out
        .linear_out_no_bias(&input, &weight)
        .unwrap();
    linear_biased_out
        .linear_out_with_bias(&input, &weight, &bias)
        .unwrap();
    assert!(linear_plain_out.equal_native(&linear_plain).unwrap());
    assert!(linear_biased_out.equal_native(&linear_biased).unwrap());

    let logits = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device);
    let log_probabilities = logits.log_softmax_default_dtype(1).unwrap();
    let log_probabilities_float = logits.log_softmax_dtype(1, 6).unwrap();
    same(
        &log_probabilities,
        &[
            -1.3132616875182228,
            -0.31326168751822286,
            -1.3132616875182228,
            -0.31326168751822286,
        ],
        &[2, 2],
        7,
        device,
    );
    same(
        &log_probabilities_float,
        &[
            -1.3132616875182228,
            -0.31326168751822286,
            -1.3132616875182228,
            -0.31326168751822286,
        ],
        &[2, 2],
        6,
        device,
    );
    let mut log_default_out = empty(device);
    let mut log_float_out = empty(device).to_dtype(6).unwrap();
    log_default_out
        .log_softmax_out_default_dtype(&logits, 1)
        .unwrap();
    log_float_out.log_softmax_out_dtype(&logits, 1, 6).unwrap();
    assert!(log_default_out.equal_native(&log_probabilities).unwrap());
    assert!(log_float_out
        .equal_native(&log_probabilities_float)
        .unwrap());

    let normalized_input = tensor(&[1.0, 3.0, 2.0, 4.0], &[1, 2, 2], device);
    let affine_weight = tensor(&[2.0, 3.0], &[2], device);
    let affine_bias = tensor(&[1.0, -1.0], &[2], device);
    same(
        &normalized_input.group_norm_no_affine(2, 0.0, true).unwrap(),
        &[-1.0, 1.0, -1.0, 1.0],
        &[1, 2, 2],
        7,
        device,
    );
    same(
        &normalized_input
            .group_norm_weight(2, &affine_weight, 0.0, true)
            .unwrap(),
        &[-2.0, 2.0, -3.0, 3.0],
        &[1, 2, 2],
        7,
        device,
    );
    same(
        &normalized_input
            .group_norm_bias(2, &affine_bias, 0.0, true)
            .unwrap(),
        &[0.0, 2.0, -2.0, 0.0],
        &[1, 2, 2],
        7,
        device,
    );
    same(
        &normalized_input
            .group_norm_affine(2, &affine_weight, &affine_bias, 0.0, true)
            .unwrap(),
        &[-1.0, 3.0, -4.0, 2.0],
        &[1, 2, 2],
        7,
        device,
    );

    let sequence = tensor(&[1.0, 3.0, 6.0, 10.0], &[4], device);
    let prepend = tensor(&[0.0], &[1], device);
    let append = tensor(&[15.0], &[1], device);
    let basic = sequence.diff_basic(1, 0).unwrap();
    let prepended = sequence.diff_prepend(1, 0, &prepend).unwrap();
    let appended = sequence.diff_append(1, 0, &append).unwrap();
    let bounded = sequence.diff_both(1, 0, &prepend, &append).unwrap();
    same(&basic, &[2.0, 3.0, 4.0], &[3], 7, device);
    same(&prepended, &[1.0, 2.0, 3.0, 4.0], &[4], 7, device);
    same(&appended, &[2.0, 3.0, 4.0, 5.0], &[4], 7, device);
    same(&bounded, &[1.0, 2.0, 3.0, 4.0, 5.0], &[5], 7, device);
    let mut basic_out = empty(device);
    let mut prepended_out = empty(device);
    let mut appended_out = empty(device);
    let mut bounded_out = empty(device);
    basic_out.diff_out_basic(&sequence, 1, 0).unwrap();
    prepended_out
        .diff_out_prepend(&sequence, 1, 0, &prepend)
        .unwrap();
    appended_out
        .diff_out_append(&sequence, 1, 0, &append)
        .unwrap();
    bounded_out
        .diff_out_both(&sequence, 1, 0, &prepend, &append)
        .unwrap();
    assert!(basic_out.equal_native(&basic).unwrap());
    assert!(prepended_out.equal_native(&prepended).unwrap());
    assert!(appended_out.equal_native(&appended).unwrap());
    assert!(bounded_out.equal_native(&bounded).unwrap());

    let mask = tensor(&[1.0, 0.0, 1.0], &[3], device).to_dtype(11).unwrap();
    let scattered = tensor(&[10.0, 30.0, 20.0], &[3], device)
        .masked_scatter_backward(&mask, &[2])
        .unwrap();
    same(&scattered, &[10.0, 20.0], &[2], 7, device);

    let leaf = tensor(&[1.0, 2.0], &[1, 2], device).require_grad().unwrap();
    leaf.linear_no_bias(&weight)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    same(&leaf.gradient().unwrap(), &[4.0, 6.0], &[1, 2], 7, device);

    assert!(logits.log_softmax_default_dtype(7).is_err());
    assert!(sequence.diff_basic(-1, 0).is_err());
    assert!(normalized_input.group_norm_no_affine(3, 0.0, true).is_err());
    same(&input, &[1.0, 2.0], &[1, 2], 7, device);
    same(&sequence, &[1.0, 3.0, 6.0, 10.0], &[4], 7, device);
}

#[test]
fn functional_optionals_outputs_and_gradients_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
