use terlan_pytorch_cxx::Tensor;

/// Compares complete tensor packets after normalizing CUDA values to CPU.
fn assert_tensor_eq(actual: &Tensor, expected: &Tensor, cuda: bool, context: &str) {
    let actual = if cuda {
        actual.to_device(0, -1).expect("copy actual to CPU")
    } else {
        actual.clone().expect("clone actual")
    };
    let expected = if cuda {
        expected.to_device(0, -1).expect("copy expected to CPU")
    } else {
        expected.clone().expect("clone expected")
    };
    assert_eq!(
        actual.to_packet().expect(context),
        expected.to_packet().expect(context)
    );
}

/// Executes retained-output and in-place completion operations on one device.
fn verify_mutation_completion(device_type: i64, device_index: i64, cuda: bool) {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3])
        .expect("create input")
        .to_device(device_type, device_index)
        .expect("place input");
    let source = Tensor::from_floats(&[7.0, 8.0, 9.0], &[3])
        .expect("create source")
        .to_device(device_type, device_index)
        .expect("place source");

    let expected_copy = input.copy(&source, false).expect("compute copy");
    let mut copy_output = input.clone().expect("clone copy output");
    copy_output
        .copy_out(&input, &source, false)
        .expect("write copy output");
    let mut copy_mutation = input.clone().expect("clone copy mutation");
    copy_mutation
        .copy_in_place(&source, false)
        .expect("execute copy mutation");
    assert_tensor_eq(&copy_output, &expected_copy, cuda, "compare copy output");
    assert_tensor_eq(
        &copy_mutation,
        &expected_copy,
        cuda,
        "compare copy mutation",
    );

    let mask_values = Tensor::from_ints(&[0, 1, 0], &[3])
        .expect("create mask values")
        .to_device(device_type, device_index)
        .expect("place mask values");
    let mask = mask_values.to_dtype(11).expect("convert mask to bool");
    let replacement = Tensor::from_floats(&[11.0], &[1])
        .expect("create masked replacement")
        .to_device(device_type, device_index)
        .expect("place masked replacement");
    let expected_masked = input
        .masked_scatter(&mask, &replacement)
        .expect("compute masked scatter");
    let mut masked_output = input.clone().expect("clone masked output");
    masked_output
        .masked_scatter_out(&input, &mask, &replacement)
        .expect("write masked output");
    let mut masked_mutation = input.clone().expect("clone masked mutation");
    masked_mutation
        .masked_scatter_in_place(&mask, &replacement)
        .expect("execute masked mutation");
    assert_tensor_eq(
        &masked_output,
        &expected_masked,
        cuda,
        "compare masked output",
    );
    assert_tensor_eq(
        &masked_mutation,
        &expected_masked,
        cuda,
        "compare masked mutation",
    );

    let indices = Tensor::from_ints(&[1], &[1])
        .expect("create put index")
        .to_device(device_type, device_index)
        .expect("place put index");
    let put_source = Tensor::from_floats(&[13.0], &[1])
        .expect("create put source")
        .to_device(device_type, device_index)
        .expect("place put source");
    let expected_put = input
        .put(&indices, &put_source, false)
        .expect("compute put");
    let mut put_output = input.clone().expect("clone put output");
    put_output
        .put_out(&input, &indices, &put_source, false)
        .expect("write put output");
    let mut put_mutation = input.clone().expect("clone put mutation");
    put_mutation
        .put_in_place(&indices, &put_source, false)
        .expect("execute put mutation");
    assert_tensor_eq(&put_output, &expected_put, cuda, "compare put output");
    assert_tensor_eq(&put_mutation, &expected_put, cuda, "compare put mutation");

    let signs = Tensor::from_floats(&[-1.0, 1.0, -1.0], &[3])
        .expect("create signs")
        .to_device(device_type, device_index)
        .expect("place signs");
    let expected_copysign = input
        .copysign_tensor(&signs)
        .expect("compute tensor copysign");
    let mut copysign_mutation = input.clone().expect("clone copysign mutation");
    copysign_mutation
        .copysign_tensor_in_place(&signs)
        .expect("execute tensor copysign mutation");
    assert_tensor_eq(
        &copysign_mutation,
        &expected_copysign,
        cuda,
        "compare tensor copysign mutation",
    );

    let gamma_input = Tensor::from_floats(&[2.0, 3.0], &[2])
        .expect("create mvlgamma input")
        .to_device(device_type, device_index)
        .expect("place mvlgamma input");
    let expected_gamma = gamma_input.mvlgamma(2).expect("compute mvlgamma");
    let mut gamma_output = gamma_input.clone().expect("clone mvlgamma output");
    gamma_output
        .mvlgamma_out(&gamma_input, 2)
        .expect("write mvlgamma output");
    let mut gamma_mutation = gamma_input.clone().expect("clone mvlgamma mutation");
    gamma_mutation
        .mvlgamma_in_place(2)
        .expect("execute mvlgamma mutation");
    assert_tensor_eq(
        &gamma_output,
        &expected_gamma,
        cuda,
        "compare mvlgamma output",
    );
    assert_tensor_eq(
        &gamma_mutation,
        &expected_gamma,
        cuda,
        "compare mvlgamma mutation",
    );

    let end = Tensor::from_floats(&[5.0, 6.0, 7.0], &[3])
        .expect("create interpolation end")
        .to_device(device_type, device_index)
        .expect("place interpolation end");
    let weight = Tensor::from_floats(&[0.25, 0.5, 0.75], &[3])
        .expect("create interpolation weight")
        .to_device(device_type, device_index)
        .expect("place interpolation weight");
    let expected_lerp = input
        .lerp_tensor(&end, &weight)
        .expect("compute tensor lerp");
    let mut lerp_mutation = input.clone().expect("clone lerp mutation");
    lerp_mutation
        .lerp_tensor_in_place(&end, &weight)
        .expect("execute tensor lerp mutation");
    assert_tensor_eq(
        &lerp_mutation,
        &expected_lerp,
        cuda,
        "compare tensor lerp mutation",
    );

    assert_eq!(
        input
            .to_device(0, -1)
            .expect("copy input to CPU")
            .to_packet()
            .expect("export input"),
        Tensor::from_floats(&[1.0, 2.0, 3.0], &[3])
            .expect("create expected input")
            .to_packet()
            .expect("export expected input")
    );
}

/// Proves autograd participation can be enabled on retained leaf storage.
fn verify_requires_grad_mutation(device_type: i64, device_index: i64, cuda: bool) {
    let mut parameter = Tensor::from_float(2.0)
        .expect("create parameter")
        .to_device(device_type, device_index)
        .expect("place parameter");
    parameter
        .requires_grad_in_place(true)
        .expect("enable gradients");
    let loss = parameter.square().expect("square parameter");
    loss.backward().expect("run backward");
    let gradient = parameter.gradient().expect("read gradient");
    let expected = Tensor::from_float(4.0)
        .expect("create expected gradient")
        .to_device(device_type, device_index)
        .expect("place expected gradient");
    assert_tensor_eq(&gradient, &expected, cuda, "compare retained gradient");
}

/// Proves the completion batch on CPU.
#[test]
fn cpu_mutation_completion_matches_functional_results() {
    verify_mutation_completion(0, -1, false);
    verify_requires_grad_mutation(0, -1, false);
}

/// Proves the completion batch on mandatory CUDA hardware.
#[test]
fn cuda_mutation_completion_matches_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_mutation_completion(1, 0, true);
    verify_requires_grad_mutation(1, 0, true);
}

/// Proves LibTorch reports a short source after applying available masked values.
#[test]
fn short_masked_scatter_preserves_upstream_partial_mutation() {
    let mut input = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create input");
    let mask_values = Tensor::from_ints(&[1, 1], &[2]).expect("create mask values");
    let mask = mask_values.to_dtype(11).expect("convert mask");
    let source = Tensor::from_floats(&[9.0], &[1]).expect("create short source");
    let error = input
        .masked_scatter_in_place(&mask, &source)
        .expect_err("short masked source unexpectedly succeeded");
    let expected = Tensor::from_floats(&[9.0, 2.0], &[2]).expect("create partial result");

    assert_ne!(error.status, 0);
    assert_eq!(
        input.to_packet().expect("export input"),
        expected.to_packet().expect("export partial result")
    );
}
