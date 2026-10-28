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
    assert!(actual
        .allclose_native(expected, 1e-12, 1e-12, false)
        .unwrap());
}

fn verify(device: i64) {
    let input = tensor(&[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], &[2, 4], device);
    let gradient_glu = tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2], device);
    let input_tangent = tensor(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], &[2, 4], device);
    let gradient_glu_tangent = tensor(&[0.25, 0.5, 0.75, 1.0], &[2, 2], device);
    let mut forward_output = tensor(&[], &[0], device);
    let mut backward_output = tensor(&[], &[0], device);
    let mut forward_jvp_output = tensor(&[], &[0], device);
    let mut backward_jvp_output = tensor(&[], &[0], device);

    let glu = input.glu(1).unwrap();
    forward_output.glu_out(&input, 1).unwrap();
    same(&forward_output, &glu, &[2, 2], device);
    same(
        &glu,
        &tensor(&[0.5, 1.0, 1.5, 2.0], &[2, 2], device),
        &[2, 2],
        device,
    );

    let gradient_input = gradient_glu.glu_backward(&input, 1).unwrap();
    backward_output
        .glu_backward_out(&gradient_glu, &input, 1)
        .unwrap();
    same(&backward_output, &gradient_input, &[2, 4], device);

    let forward_jvp = glu.glu_jvp(&input, &input_tangent, 1).unwrap();
    forward_jvp_output
        .glu_jvp_out(&glu, &input, &input_tangent, 1)
        .unwrap();
    same(&forward_jvp_output, &forward_jvp, &[2, 2], device);

    let backward_jvp = gradient_input
        .glu_backward_jvp(
            &gradient_glu,
            &input,
            &gradient_glu_tangent,
            &input_tangent,
            1,
        )
        .unwrap();
    backward_jvp_output
        .glu_backward_jvp_out(
            &gradient_input,
            &gradient_glu,
            &input,
            &gradient_glu_tangent,
            &input_tangent,
            1,
        )
        .unwrap();
    same(&backward_jvp_output, &backward_jvp, &[2, 4], device);

    let leaf = tensor(&[1.0, 2.0, 0.0, 0.0], &[1, 4], device)
        .require_grad()
        .unwrap();
    leaf.glu(1).unwrap().sum().unwrap().backward().unwrap();
    same(
        &leaf.gradient().unwrap(),
        &tensor(&[0.5, 0.5, 0.25, 0.5], &[1, 4], device),
        &[1, 4],
        device,
    );

    assert!(forward_output
        .glu_out(&tensor(&[1.0, 2.0, 3.0], &[1, 3], device), 1)
        .is_err());
    assert!(glu.glu_jvp(&input, &input_tangent, 9).is_err());
    same(
        &input,
        &tensor(&[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], &[2, 4], device),
        &[2, 4],
        device,
    );
}

#[test]
fn exact_glu_outputs_and_jvps_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
