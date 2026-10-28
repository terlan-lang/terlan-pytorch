use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn empty(device: i64) -> Tensor {
    tensor(&[0.0, 0.0], &[2], device)
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
    let input = tensor(&[0.25, 0.75], &[2], device);
    let target = tensor(&[0.0, 1.0], &[2], device);
    let weight = tensor(&[2.0, 3.0], &[2], device);
    let loss = input.binary_cross_entropy(&target, 0).unwrap();
    let weighted_loss = input
        .binary_cross_entropy_weight(&target, &weight, 0)
        .unwrap();
    same(
        &loss,
        &[0.2876820724517809, 0.2876820724517809],
        &[2],
        device,
    );
    same(
        &weighted_loss,
        &[0.5753641449035618, 0.8630462173553427],
        &[2],
        device,
    );
    let mut loss_out = empty(device);
    let mut weighted_loss_out = empty(device);
    loss_out
        .binary_cross_entropy_out(&input, &target, 0)
        .unwrap();
    weighted_loss_out
        .binary_cross_entropy_weight_out(&input, &target, &weight, 0)
        .unwrap();
    assert!(loss_out.equal_native(&loss).unwrap());
    assert!(weighted_loss_out.equal_native(&weighted_loss).unwrap());

    let gradient = tensor(&[2.0, 4.0], &[2], device);
    let backward = gradient
        .binary_cross_entropy_backward(&input, &target, 0)
        .unwrap();
    let weighted_backward = gradient
        .binary_cross_entropy_backward_weight(&input, &target, &weight, 0)
        .unwrap();
    same(
        &backward,
        &[2.6666666666666665, -5.333333333333333],
        &[2],
        device,
    );
    same(
        &weighted_backward,
        &[5.333333333333333, -16.0],
        &[2],
        device,
    );
    let mut backward_out = empty(device);
    let mut weighted_backward_out = empty(device);
    backward_out
        .binary_cross_entropy_backward_out(&gradient, &input, &target, 0)
        .unwrap();
    weighted_backward_out
        .binary_cross_entropy_backward_weight_out(&gradient, &input, &target, &weight, 0)
        .unwrap();
    assert!(backward_out.equal_native(&backward).unwrap());
    assert!(weighted_backward_out
        .equal_native(&weighted_backward)
        .unwrap());

    let logits = tensor(&[0.0, 0.0], &[2], device);
    let element_weight = tensor(&[4.0, 5.0], &[2], device);
    let positive_weight = tensor(&[2.0, 3.0], &[2], device);
    let plain_logits = logits.binary_cross_entropy_with_logits(&target, 0).unwrap();
    let weighted_logits = logits
        .binary_cross_entropy_with_logits_weight(&target, &element_weight, 0)
        .unwrap();
    let positive_logits = logits
        .binary_cross_entropy_with_logits_positive_weight(&target, &positive_weight, 0)
        .unwrap();
    let both_logits = logits
        .binary_cross_entropy_with_logits_both_weights(
            &target,
            &element_weight,
            &positive_weight,
            0,
        )
        .unwrap();
    let log_two = std::f64::consts::LN_2;
    same(&plain_logits, &[log_two, log_two], &[2], device);
    same(
        &weighted_logits,
        &[4.0 * log_two, 5.0 * log_two],
        &[2],
        device,
    );
    same(&positive_logits, &[log_two, 3.0 * log_two], &[2], device);
    same(&both_logits, &[4.0 * log_two, 15.0 * log_two], &[2], device);
    let mut plain_logits_out = empty(device);
    let mut weighted_logits_out = empty(device);
    let mut positive_logits_out = empty(device);
    let mut both_logits_out = empty(device);
    plain_logits_out
        .binary_cross_entropy_with_logits_out(&logits, &target, 0)
        .unwrap();
    weighted_logits_out
        .binary_cross_entropy_with_logits_weight_out(&logits, &target, &element_weight, 0)
        .unwrap();
    positive_logits_out
        .binary_cross_entropy_with_logits_positive_weight_out(&logits, &target, &positive_weight, 0)
        .unwrap();
    both_logits_out
        .binary_cross_entropy_with_logits_both_weights_out(
            &logits,
            &target,
            &element_weight,
            &positive_weight,
            0,
        )
        .unwrap();
    assert!(plain_logits_out.equal_native(&plain_logits).unwrap());
    assert!(weighted_logits_out.equal_native(&weighted_logits).unwrap());
    assert!(positive_logits_out.equal_native(&positive_logits).unwrap());
    assert!(both_logits_out.equal_native(&both_logits).unwrap());

    let leaf = tensor(&[0.0, 0.0], &[2], device).require_grad().unwrap();
    leaf.binary_cross_entropy_with_logits_positive_weight(&target, &positive_weight, 2)
        .unwrap()
        .backward()
        .unwrap();
    same(&leaf.gradient().unwrap(), &[0.5, -1.5], &[2], device);

    if device == 0 {
        assert!(tensor(&[1.5], &[1], device)
            .binary_cross_entropy(&tensor(&[0.0], &[1], device), 0)
            .is_err());
    }
    let broadcast_target = tensor(&[0.0], &[1], device);
    let broadcast_loss = input.binary_cross_entropy(&broadcast_target, 0).unwrap();
    same(
        &broadcast_loss,
        &[0.2876820724517809, 1.3862943611198906],
        &[2],
        device,
    );
    same(&input, &[0.25, 0.75], &[2], device);
    same(&target, &[0.0, 1.0], &[2], device);
}

#[test]
fn binary_cross_entropy_variants_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
