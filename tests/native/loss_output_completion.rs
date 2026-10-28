use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn same(actual: &Tensor, expected: &Tensor, shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(expected).unwrap());
}

fn verify(device: i64) {
    let input = tensor(&[-0.5, 2.0], &[2], device);
    let target = tensor(&[0.0, 0.0], &[2], device);
    let gradient = tensor(&[2.0, 3.0], &[2], device);
    let mut output = tensor(&[], &[0], device);

    let huber = input.huber_loss(&target, 0, 1.0).unwrap();
    output.huber_loss_out(&input, &target, 0, 1.0).unwrap();
    same(&output, &huber, &[2], device);
    let huber_gradient = gradient
        .huber_loss_backward(&input, &target, 0, 1.0)
        .unwrap();
    output
        .huber_loss_backward_out(&gradient, &input, &target, 0, 1.0)
        .unwrap();
    same(&output, &huber_gradient, &[2], device);

    let mse = input.mse_loss(&target, 0).unwrap();
    output.mse_loss_out(&input, &target, 0).unwrap();
    same(&output, &mse, &[2], device);
    let mse_gradient = gradient.mse_loss_gradient(&input, &target, 0).unwrap();
    output
        .mse_loss_backward_out(&gradient, &input, &target, 0)
        .unwrap();
    same(&output, &mse_gradient, &[2], device);

    let smooth = input.smooth_l1_loss(&target, 0, 1.0).unwrap();
    output.smooth_l1_loss_out(&input, &target, 0, 1.0).unwrap();
    same(&output, &smooth, &[2], device);
    let smooth_gradient = gradient
        .smooth_l1_loss_backward(&input, &target, 0, 1.0)
        .unwrap();
    output
        .smooth_l1_loss_backward_out(&gradient, &input, &target, 0, 1.0)
        .unwrap();
    same(&output, &smooth_gradient, &[2], device);

    let margin_input = tensor(&[0.0, 0.0], &[2], device);
    let margin_target = tensor(&[1.0, -1.0], &[2], device);
    let soft = margin_input.soft_margin_loss(&margin_target, 0).unwrap();
    output
        .soft_margin_loss_out(&margin_input, &margin_target, 0)
        .unwrap();
    same(&output, &soft, &[2], device);
    let soft_gradient = gradient
        .soft_margin_loss_backward(&margin_input, &margin_target, 0)
        .unwrap();
    output
        .soft_margin_loss_backward_out(&gradient, &margin_input, &margin_target, 0)
        .unwrap();
    same(&output, &soft_gradient, &[2], device);

    assert!(output.huber_loss_out(&input, &target, 0, 0.0).is_err());
    assert!(output.mse_loss_out(&input, &target, 9).is_err());
    let negative_beta = gradient
        .smooth_l1_loss_backward(&input, &target, 0, -1.0)
        .unwrap();
    output
        .smooth_l1_loss_backward_out(&gradient, &input, &target, 0, -1.0)
        .unwrap();
    same(&output, &negative_beta, &[2], device);
    same(&input, &tensor(&[-0.5, 2.0], &[2], device), &[2], device);
    same(&target, &tensor(&[0.0, 0.0], &[2], device), &[2], device);
}

#[test]
fn exact_loss_outputs_match_functional_kernels_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
