use terlan_pytorch_cxx::Tensor;

fn assert_packet_eq(actual: &Tensor, expected: &Tensor, context: &str) {
    assert_eq!(
        actual.to_packet().expect(context),
        expected.to_packet().expect(context)
    );
}

fn assert_cuda_packet_eq(actual: &Tensor, expected: &Tensor, context: &str) {
    let actual = actual.to_device(0, -1).expect("copy actual to CPU");
    let expected = expected.to_device(0, -1).expect("copy expected to CPU");
    assert_packet_eq(&actual, &expected, context);
}

#[test]
fn cpu_binary_output_and_mutation_contracts_match_functional_results() {
    let left = Tensor::from_floats(&[7.0, -7.0], &[2]).expect("create left");
    let right = Tensor::from_floats(&[4.0, 4.0], &[2]).expect("create right");

    let expected_fmax = left.fmax(&right).expect("compute fmax");
    let mut fmax_output = left.clone().expect("clone fmax output");
    fmax_output
        .fmax_out(&left, &right)
        .expect("write fmax output");
    assert_packet_eq(&fmax_output, &expected_fmax, "compare fmax");

    let expected_fmin = left.fmin(&right).expect("compute fmin");
    let mut fmin_output = left.clone().expect("clone fmin output");
    fmin_output
        .fmin_out(&left, &right)
        .expect("write fmin output");
    assert_packet_eq(&fmin_output, &expected_fmin, "compare fmin");

    let expected_maximum = left.maximum(&right).expect("compute maximum");
    let mut maximum_output = left.clone().expect("clone maximum output");
    maximum_output
        .maximum_out(&left, &right)
        .expect("write maximum output");
    assert_packet_eq(&maximum_output, &expected_maximum, "compare maximum");

    let expected_minimum = left.minimum(&right).expect("compute minimum");
    let mut minimum_output = left.clone().expect("clone minimum output");
    minimum_output
        .minimum_out(&left, &right)
        .expect("write minimum output");
    assert_packet_eq(&minimum_output, &expected_minimum, "compare minimum");

    let expected_scalar = left
        .binary_scalar_right_float(17, 4.0)
        .expect("compute scalar fmod");
    let mut scalar_output = left.clone().expect("clone scalar output");
    scalar_output
        .binary_scalar_right_float_out(&left, 17, 4.0)
        .expect("write scalar fmod output");
    let mut scalar_mutation = left.clone().expect("clone scalar mutation");
    scalar_mutation
        .binary_scalar_right_float_in_place(17, 4.0)
        .expect("mutate scalar fmod");
    assert_packet_eq(&scalar_output, &expected_scalar, "compare scalar output");
    assert_packet_eq(
        &scalar_mutation,
        &expected_scalar,
        "compare scalar mutation",
    );

    let expected_copysign = left
        .binary_scalar_right_float(18, -1.0)
        .expect("compute scalar copysign");
    let mut copysign_output = left.clone().expect("clone copysign output");
    copysign_output
        .binary_scalar_right_float_out(&left, 18, -1.0)
        .expect("write scalar copysign output");
    let mut copysign_mutation = left.clone().expect("clone copysign mutation");
    copysign_mutation
        .binary_scalar_right_float_in_place(18, -1.0)
        .expect("mutate scalar copysign");
    let expected_positive_sign = left
        .binary_scalar_right_int(18, 1)
        .expect("compute integer scalar copysign");
    let mut integer_copysign_output = left.clone().expect("clone integer copysign output");
    integer_copysign_output
        .binary_scalar_right_int_out(&left, 18, 1)
        .expect("write integer scalar copysign output");
    let mut integer_copysign_mutation = left.clone().expect("clone integer copysign mutation");
    integer_copysign_mutation
        .binary_scalar_right_int_in_place(18, 1)
        .expect("mutate integer scalar copysign");
    assert_packet_eq(
        &copysign_output,
        &expected_copysign,
        "compare copysign output",
    );
    assert_packet_eq(
        &copysign_mutation,
        &expected_copysign,
        "compare copysign mutation",
    );
    assert_packet_eq(
        &integer_copysign_output,
        &expected_positive_sign,
        "compare integer copysign output",
    );
    assert_packet_eq(
        &integer_copysign_mutation,
        &expected_positive_sign,
        "compare integer copysign mutation",
    );
    assert_eq!(
        expected_positive_sign
            .sum()
            .expect("sum positive copysign")
            .item()
            .expect("read positive copysign"),
        14
    );

    let expected_tensor = left.fmod_tensor(&right).expect("compute tensor fmod");
    let mut tensor_output = left.clone().expect("clone tensor output");
    tensor_output
        .fmod_tensor_out(&left, &right)
        .expect("write tensor fmod output");
    let mut tensor_mutation = left.clone().expect("clone tensor mutation");
    tensor_mutation
        .fmod_tensor_in_place(&right)
        .expect("mutate tensor fmod");
    assert_packet_eq(&tensor_output, &expected_tensor, "compare tensor output");
    assert_packet_eq(
        &tensor_mutation,
        &expected_tensor,
        "compare tensor mutation",
    );

    assert_eq!(left.sum().expect("sum left").item().expect("read left"), 0);
    assert_eq!(
        right.sum().expect("sum right").item().expect("read right"),
        8
    );
}

#[test]
fn cpu_logical_output_and_mutation_contracts_match_functional_results() {
    let left = Tensor::from_bool(true).expect("create left");
    let right = Tensor::from_bool(false).expect("create right");

    let expected_and = left.logical_and(&right).expect("compute logical and");
    let mut and_output = left.clone().expect("clone and output");
    and_output
        .logical_and_out(&left, &right)
        .expect("write logical and output");
    let mut and_mutation = left.clone().expect("clone and mutation");
    and_mutation
        .logical_and_in_place(&right)
        .expect("mutate logical and");
    assert_packet_eq(&and_output, &expected_and, "compare and output");
    assert_packet_eq(&and_mutation, &expected_and, "compare and mutation");

    let expected_or = left.logical_or(&right).expect("compute logical or");
    let mut or_output = left.clone().expect("clone or output");
    or_output
        .logical_or_out(&left, &right)
        .expect("write logical or output");
    let mut or_mutation = left.clone().expect("clone or mutation");
    or_mutation
        .logical_or_in_place(&right)
        .expect("mutate logical or");
    assert_packet_eq(&or_output, &expected_or, "compare or output");
    assert_packet_eq(&or_mutation, &expected_or, "compare or mutation");

    let expected_xor = left.logical_xor(&right).expect("compute logical xor");
    let mut xor_output = left.clone().expect("clone xor output");
    xor_output
        .logical_xor_out(&left, &right)
        .expect("write logical xor output");
    let mut xor_mutation = left.clone().expect("clone xor mutation");
    xor_mutation
        .logical_xor_in_place(&right)
        .expect("mutate logical xor");
    assert_packet_eq(&xor_output, &expected_xor, "compare xor output");
    assert_packet_eq(&xor_mutation, &expected_xor, "compare xor mutation");
}

#[test]
fn invalid_binary_storage_operations_fail_without_mutating_inputs() {
    let left = Tensor::from_float(7.0).expect("create left");
    let right = Tensor::from_float(4.0).expect("create right");
    let mut output = Tensor::from_int(11).expect("create incompatible output");
    let output_error = output
        .fmax_out(&left, &right)
        .expect_err("integer output unexpectedly accepted float result");
    let mut mutation = Tensor::from_bool(true).expect("create bool mutation");
    let mutation_error = mutation
        .binary_scalar_right_float_in_place(17, 0.0)
        .expect_err("boolean fmod unexpectedly succeeded");

    assert_ne!(output_error.status, 0);
    assert_ne!(mutation_error.status, 0);
    assert_eq!(output.item().expect("read output"), 11);
    assert!(mutation.item_bool().expect("read mutation"));
    assert_eq!(left.item_float().expect("read left"), 7.0);
    assert_eq!(right.item_float().expect("read right"), 4.0);
}

#[test]
fn cuda_binary_storage_contracts_match_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let left = Tensor::from_floats(&[7.0, -7.0], &[2])
        .expect("create left")
        .to_device(1, 0)
        .expect("place left on CUDA");
    let right = Tensor::from_floats(&[4.0, 4.0], &[2])
        .expect("create right")
        .to_device(1, 0)
        .expect("place right on CUDA");

    let expected_fmax = left.fmax(&right).expect("compute CUDA fmax");
    let mut fmax_output = left.clone().expect("clone CUDA fmax output");
    fmax_output
        .fmax_out(&left, &right)
        .expect("write CUDA fmax output");
    assert_cuda_packet_eq(&fmax_output, &expected_fmax, "compare CUDA fmax");

    let expected_fmin = left.fmin(&right).expect("compute CUDA fmin");
    let mut fmin_output = left.clone().expect("clone CUDA fmin output");
    fmin_output
        .fmin_out(&left, &right)
        .expect("write CUDA fmin output");
    assert_cuda_packet_eq(&fmin_output, &expected_fmin, "compare CUDA fmin");

    let expected_maximum = left.maximum(&right).expect("compute CUDA maximum");
    let mut maximum_output = left.clone().expect("clone CUDA maximum output");
    maximum_output
        .maximum_out(&left, &right)
        .expect("write CUDA maximum output");
    assert_cuda_packet_eq(&maximum_output, &expected_maximum, "compare CUDA maximum");

    let expected_minimum = left.minimum(&right).expect("compute CUDA minimum");
    let mut minimum_output = left.clone().expect("clone CUDA minimum output");
    minimum_output
        .minimum_out(&left, &right)
        .expect("write CUDA minimum output");
    assert_cuda_packet_eq(&minimum_output, &expected_minimum, "compare CUDA minimum");

    let expected_scalar = left
        .binary_scalar_right_float(17, 4.0)
        .expect("compute CUDA scalar fmod");
    let mut scalar_output = left.clone().expect("clone CUDA scalar output");
    scalar_output
        .binary_scalar_right_float_out(&left, 17, 4.0)
        .expect("write CUDA scalar fmod output");
    let mut scalar_mutation = left.clone().expect("clone CUDA scalar mutation");
    scalar_mutation
        .binary_scalar_right_float_in_place(17, 4.0)
        .expect("mutate CUDA scalar fmod");
    assert_cuda_packet_eq(
        &scalar_output,
        &expected_scalar,
        "compare CUDA scalar output",
    );
    assert_cuda_packet_eq(
        &scalar_mutation,
        &expected_scalar,
        "compare CUDA scalar mutation",
    );

    let expected_copysign = left
        .binary_scalar_right_float(18, -1.0)
        .expect("compute CUDA scalar copysign");
    let mut copysign_output = left.clone().expect("clone CUDA copysign output");
    copysign_output
        .binary_scalar_right_float_out(&left, 18, -1.0)
        .expect("write CUDA scalar copysign output");
    let mut copysign_mutation = left.clone().expect("clone CUDA copysign mutation");
    copysign_mutation
        .binary_scalar_right_float_in_place(18, -1.0)
        .expect("mutate CUDA scalar copysign");
    let expected_positive_sign = left
        .binary_scalar_right_int(18, 1)
        .expect("compute CUDA integer scalar copysign");
    let mut integer_copysign_output = left.clone().expect("clone CUDA integer copysign output");
    integer_copysign_output
        .binary_scalar_right_int_out(&left, 18, 1)
        .expect("write CUDA integer scalar copysign output");
    let mut integer_copysign_mutation = left.clone().expect("clone CUDA integer copysign mutation");
    integer_copysign_mutation
        .binary_scalar_right_int_in_place(18, 1)
        .expect("mutate CUDA integer scalar copysign");
    assert_cuda_packet_eq(
        &copysign_output,
        &expected_copysign,
        "compare CUDA copysign output",
    );
    assert_cuda_packet_eq(
        &copysign_mutation,
        &expected_copysign,
        "compare CUDA copysign mutation",
    );
    assert_cuda_packet_eq(
        &integer_copysign_output,
        &expected_positive_sign,
        "compare CUDA integer copysign output",
    );
    assert_cuda_packet_eq(
        &integer_copysign_mutation,
        &expected_positive_sign,
        "compare CUDA integer copysign mutation",
    );
    assert_eq!(
        expected_positive_sign
            .to_device(0, -1)
            .expect("copy positive copysign to CPU")
            .sum()
            .expect("sum positive copysign")
            .item()
            .expect("read positive copysign"),
        14
    );

    let expected_tensor = left.fmod_tensor(&right).expect("compute CUDA tensor fmod");
    let mut tensor_output = left.clone().expect("clone CUDA tensor output");
    tensor_output
        .fmod_tensor_out(&left, &right)
        .expect("write CUDA tensor fmod output");
    let mut tensor_mutation = left.clone().expect("clone CUDA tensor mutation");
    tensor_mutation
        .fmod_tensor_in_place(&right)
        .expect("mutate CUDA tensor fmod");
    assert_cuda_packet_eq(
        &tensor_output,
        &expected_tensor,
        "compare CUDA tensor output",
    );
    assert_cuda_packet_eq(
        &tensor_mutation,
        &expected_tensor,
        "compare CUDA tensor mutation",
    );

    let truth = Tensor::from_bool(true)
        .expect("create truth")
        .to_device(1, 0)
        .expect("place truth on CUDA");
    let mask = Tensor::from_bool(false)
        .expect("create mask")
        .to_device(1, 0)
        .expect("place mask on CUDA");
    let expected_and = truth.logical_and(&mask).expect("compute CUDA and");
    let mut and_output = truth.clone().expect("clone CUDA and output");
    and_output
        .logical_and_out(&truth, &mask)
        .expect("write CUDA and output");
    let mut and_mutation = truth.clone().expect("clone CUDA and mutation");
    and_mutation
        .logical_and_in_place(&mask)
        .expect("mutate CUDA and");
    assert_cuda_packet_eq(&and_output, &expected_and, "compare CUDA and output");
    assert_cuda_packet_eq(&and_mutation, &expected_and, "compare CUDA and mutation");

    let expected_or = truth.logical_or(&mask).expect("compute CUDA or");
    let mut or_output = truth.clone().expect("clone CUDA or output");
    or_output
        .logical_or_out(&truth, &mask)
        .expect("write CUDA or output");
    let mut or_mutation = truth.clone().expect("clone CUDA or mutation");
    or_mutation
        .logical_or_in_place(&mask)
        .expect("mutate CUDA or");
    assert_cuda_packet_eq(&or_output, &expected_or, "compare CUDA or output");
    assert_cuda_packet_eq(&or_mutation, &expected_or, "compare CUDA or mutation");

    let expected_xor = truth.logical_xor(&mask).expect("compute CUDA xor");
    let mut xor_output = truth.clone().expect("clone CUDA xor output");
    xor_output
        .logical_xor_out(&truth, &mask)
        .expect("write CUDA xor output");
    let mut xor_mutation = truth.clone().expect("clone CUDA xor mutation");
    xor_mutation
        .logical_xor_in_place(&mask)
        .expect("mutate CUDA xor");
    assert_cuda_packet_eq(&xor_output, &expected_xor, "compare CUDA xor output");
    assert_cuda_packet_eq(&xor_mutation, &expected_xor, "compare CUDA xor mutation");
}
