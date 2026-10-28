use terlan_pytorch_cxx::{CAbiError, Tensor};

fn flat_item(tensor: &Tensor, index: i64) -> f64 {
    tensor
        .flatten(0, -1)
        .expect("flatten tensor")
        .narrow(0, index, 1)
        .expect("select flattened item")
        .item_float()
        .expect("read flattened item")
}

fn values(tensor: &Tensor) -> Vec<f64> {
    let count = tensor.numel().expect("read element count");
    (0..count).map(|index| flat_item(tensor, index)).collect()
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn bitwise_scalar_forms_preserve_operand_order() {
    let input = Tensor::from_ints(&[5, 6], &[2]).expect("create bitwise input");
    let and_result = input
        .binary_scalar_right_int(0, 3)
        .expect("bitwise and scalar right");
    let or_result = Tensor::binary_scalar_left_int(1, 1, &input).expect("bitwise or scalar left");
    let xor_result = input
        .binary_scalar_right_int(2, 3)
        .expect("bitwise xor scalar right");
    let left_result = Tensor::binary_scalar_left_int(3, 1, &input).expect("left shift scalar left");
    let right_result =
        Tensor::binary_scalar_left_int(4, 32, &input).expect("right shift scalar left");

    assert_eq!(values(&and_result), vec![1.0, 2.0]);
    assert_eq!(values(&or_result), vec![5.0, 7.0]);
    assert_eq!(values(&xor_result), vec![6.0, 5.0]);
    assert_eq!(values(&left_result), vec![32.0, 64.0]);
    assert_eq!(values(&right_result), vec![1.0, 0.0]);
    assert_eq!(values(&input), vec![5.0, 6.0]);
}

#[test]
fn bitwise_output_and_in_place_forms_retain_expected_storage() {
    let source = Tensor::from_ints(&[5, 6], &[2]).expect("create source");
    let right = Tensor::from_ints(&[3, 1], &[2]).expect("create right operand");
    let mut output = Tensor::from_ints(&[], &[0]).expect("create output");
    output
        .binary_tensor_out(&source, 0, &right)
        .expect("write tensor bitwise and");
    let mut scalar_output = Tensor::from_ints(&[], &[0]).expect("create scalar output");
    scalar_output
        .binary_scalar_left_int_out(3, 1, &right)
        .expect("write scalar-left shift");
    let mut tensor_mutation = Tensor::from_ints(&[5, 6], &[2]).expect("create tensor mutation");
    tensor_mutation
        .binary_tensor_in_place(2, &right)
        .expect("mutate tensor xor");
    let mut scalar_mutation = Tensor::from_ints(&[5, 6], &[2]).expect("create scalar mutation");
    scalar_mutation
        .binary_scalar_right_int_in_place(1, 1)
        .expect("mutate scalar or");

    assert_eq!(values(&output), vec![1.0, 0.0]);
    assert_eq!(values(&scalar_output), vec![8.0, 2.0]);
    assert_eq!(values(&tensor_mutation), vec![6.0, 7.0]);
    assert_eq!(values(&scalar_mutation), vec![5.0, 7.0]);
    assert_eq!(values(&source), vec![5.0, 6.0]);
    assert_eq!(values(&right), vec![3.0, 1.0]);
}

#[test]
fn remainder_and_power_cover_integer_and_float_scalars() {
    let integers = Tensor::from_ints(&[-5, 5], &[2]).expect("create integer input");
    let remainder_right = integers
        .binary_scalar_right_int(5, 3)
        .expect("integer remainder right");
    let divisors = Tensor::from_ints(&[3, 4], &[2]).expect("create divisors");
    let remainder_left =
        Tensor::binary_scalar_left_int(5, 7, &divisors).expect("integer remainder left");
    let powers = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create powers");
    let pow_right = powers
        .binary_scalar_right_float(6, 2.0)
        .expect("power scalar right");
    let pow_left = Tensor::binary_scalar_left_int(6, 2, &powers).expect("power scalar left");
    let float_power = powers
        .binary_scalar_right_float(7, 0.5)
        .expect("floating power scalar right");

    assert_eq!(values(&remainder_right), vec![1.0, 2.0]);
    assert_eq!(values(&remainder_left), vec![1.0, 3.0]);
    assert_eq!(values(&pow_right), vec![4.0, 9.0]);
    assert_eq!(values(&pow_left), vec![4.0, 8.0]);
    assert!((flat_item(&float_power, 0) - 2.0_f64.sqrt()).abs() < 1.0e-12);
    assert!((flat_item(&float_power, 1) - 3.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn numeric_output_and_mutation_forms_follow_libtorch_dtype_rules() {
    let source = Tensor::from_floats(&[5.5, 7.5], &[2]).expect("create source");
    let right = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create right");
    let mut remainder_output = Tensor::from_floats(&[], &[0]).expect("create remainder output");
    remainder_output
        .binary_scalar_right_float_out(&source, 5, 2.0)
        .expect("write remainder output");
    let mut pow_output = Tensor::from_floats(&[], &[0]).expect("create power output");
    pow_output
        .binary_scalar_left_float_out(6, 2.0, &right)
        .expect("write scalar-left power output");
    let mut tensor_mutation = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create mutation");
    tensor_mutation
        .binary_tensor_in_place(6, &right)
        .expect("mutate tensor power");
    let mut float_mutation = Tensor::from_floats(&[4.0, 9.0], &[2]).expect("create float power");
    float_mutation
        .binary_scalar_right_float_in_place(7, 0.5)
        .expect("mutate floating power");

    assert_eq!(values(&remainder_output), vec![1.5, 1.5]);
    assert_eq!(values(&pow_output), vec![4.0, 8.0]);
    assert_eq!(values(&tensor_mutation), vec![4.0, 27.0]);
    assert_eq!(values(&float_mutation), vec![2.0, 3.0]);
    assert_eq!(values(&source), vec![5.5, 7.5]);
}

#[test]
fn multiplication_reuses_all_binary_storage_contracts() {
    let source = Tensor::from_floats(&[2.5, -3.0], &[2]).expect("create source");
    let right = Tensor::from_floats(&[4.0, 2.0], &[2]).expect("create right");
    let integer = source
        .binary_scalar_right_int(8, 3)
        .expect("multiply by integer");
    let floating = source
        .binary_scalar_right_float(8, 0.5)
        .expect("multiply by float");
    let mut tensor_output = Tensor::from_floats(&[], &[0]).expect("create tensor output");
    tensor_output
        .binary_tensor_out(&source, 8, &right)
        .expect("write tensor multiplication");
    let mut scalar_output = Tensor::from_floats(&[], &[0]).expect("create scalar output");
    scalar_output
        .binary_scalar_right_float_out(&source, 8, 2.0)
        .expect("write scalar multiplication");
    let mut tensor_mutation = Tensor::from_floats(&[2.5, -3.0], &[2]).expect("create mutation");
    tensor_mutation
        .binary_tensor_in_place(8, &right)
        .expect("multiply by tensor in place");
    let mut scalar_mutation = Tensor::from_floats(&[2.5, -3.0], &[2]).expect("create mutation");
    scalar_mutation
        .binary_scalar_right_int_in_place(8, 2)
        .expect("multiply by scalar in place");

    assert_eq!(values(&integer), vec![7.5, -9.0]);
    assert_eq!(values(&floating), vec![1.25, -1.5]);
    assert_eq!(values(&tensor_output), vec![10.0, -6.0]);
    assert_eq!(values(&scalar_output), vec![5.0, -6.0]);
    assert_eq!(values(&tensor_mutation), vec![10.0, -6.0]);
    assert_eq!(values(&scalar_mutation), vec![5.0, -6.0]);
    assert_eq!(values(&source), vec![2.5, -3.0]);
    assert_eq!(values(&right), vec![4.0, 2.0]);
}

#[test]
fn floor_division_reuses_all_binary_storage_contracts() {
    let source = Tensor::from_floats(&[-7.5, 7.5], &[2]).expect("create source");
    let right = Tensor::from_floats(&[2.0, -2.0], &[2]).expect("create right");
    let integer = source
        .binary_scalar_right_int(9, 2)
        .expect("floor divide by integer");
    let floating = source
        .binary_scalar_right_float(9, -2.0)
        .expect("floor divide by float");
    let mut tensor_output = Tensor::from_floats(&[], &[0]).expect("create tensor output");
    tensor_output
        .binary_tensor_out(&source, 9, &right)
        .expect("write tensor floor division");
    let mut scalar_output = Tensor::from_floats(&[], &[0]).expect("create scalar output");
    scalar_output
        .binary_scalar_right_float_out(&source, 9, 2.0)
        .expect("write scalar floor division");
    let mut tensor_mutation =
        Tensor::from_floats(&[-7.5, 7.5], &[2]).expect("create tensor mutation");
    tensor_mutation
        .binary_tensor_in_place(9, &right)
        .expect("floor divide by tensor in place");
    let mut scalar_mutation =
        Tensor::from_floats(&[-7.5, 7.5], &[2]).expect("create scalar mutation");
    scalar_mutation
        .binary_scalar_right_int_in_place(9, 2)
        .expect("floor divide by scalar in place");

    assert_eq!(values(&integer), vec![-4.0, 3.0]);
    assert_eq!(values(&floating), vec![3.0, -4.0]);
    assert_eq!(values(&tensor_output), vec![-4.0, -4.0]);
    assert_eq!(values(&scalar_output), vec![-4.0, 3.0]);
    assert_eq!(values(&tensor_mutation), vec![-4.0, -4.0]);
    assert_eq!(values(&scalar_mutation), vec![-4.0, 3.0]);
    assert_eq!(values(&source), vec![-7.5, 7.5]);
    assert_eq!(values(&right), vec![2.0, -2.0]);
}

#[test]
fn xlogy_preserves_scalar_operand_order_and_storage_contracts() {
    let source = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create source");
    let right = Tensor::from_floats(&[4.0, 2.0], &[2]).expect("create right");
    let scalar_right = source
        .binary_scalar_right_float(10, 4.0)
        .expect("xlogy scalar right");
    let scalar_left = Tensor::binary_scalar_left_int(10, 2, &right).expect("xlogy scalar left");
    let mut tensor_output = Tensor::from_floats(&[], &[0]).expect("create tensor output");
    tensor_output
        .binary_tensor_out(&source, 10, &right)
        .expect("write tensor xlogy");
    let mut right_output = Tensor::from_floats(&[], &[0]).expect("create right output");
    right_output
        .binary_scalar_right_int_out(&source, 10, 1)
        .expect("write scalar-right xlogy");
    let mut left_output = Tensor::from_floats(&[], &[0]).expect("create left output");
    left_output
        .binary_scalar_left_float_out(10, 0.0, &right)
        .expect("write scalar-left xlogy");
    let mut tensor_mutation = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create mutation");
    tensor_mutation
        .binary_tensor_in_place(10, &right)
        .expect("apply tensor xlogy in place");
    let mut scalar_mutation = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create mutation");
    scalar_mutation
        .binary_scalar_right_float_in_place(10, 4.0)
        .expect("apply scalar xlogy in place");

    let expected_tensor = vec![2.0 * 4.0_f64.ln(), 3.0 * 2.0_f64.ln()];
    let expected_right = vec![2.0 * 4.0_f64.ln(), 3.0 * 4.0_f64.ln()];
    let expected_left = vec![2.0 * 4.0_f64.ln(), 2.0 * 2.0_f64.ln()];
    for (actual, expected) in values(&tensor_output)
        .into_iter()
        .zip(expected_tensor.iter())
    {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    for (actual, expected) in values(&scalar_right).into_iter().zip(expected_right.iter()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    for (actual, expected) in values(&scalar_left).into_iter().zip(expected_left.iter()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    assert_eq!(values(&right_output), vec![0.0, 0.0]);
    assert_eq!(values(&left_output), vec![0.0, 0.0]);
    assert_eq!(values(&source), vec![2.0, 3.0]);
    assert_eq!(values(&right), vec![4.0, 2.0]);
    for (actual, expected) in values(&tensor_mutation)
        .into_iter()
        .zip(expected_tensor.iter())
    {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    for (actual, expected) in values(&scalar_mutation)
        .into_iter()
        .zip(expected_right.iter())
    {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn comparisons_cover_every_operation_and_storage_contract() {
    let left = Tensor::from_float(3.0).expect("create left");
    let right = Tensor::from_float(3.0).expect("create right");
    for (operation, expected) in [
        (11, true),
        (12, false),
        (13, false),
        (14, true),
        (15, false),
        (16, true),
    ] {
        let scalar_int = left
            .binary_scalar_right_int(operation, 3)
            .expect("compare integer scalar");
        let scalar_float = left
            .binary_scalar_right_float(operation, 3.0)
            .expect("compare floating scalar");
        let mut tensor_output = Tensor::from_bool(false).expect("create tensor output");
        tensor_output
            .binary_tensor_out(&left, operation, &right)
            .expect("write tensor comparison");
        let mut scalar_output = Tensor::from_bool(false).expect("create scalar output");
        scalar_output
            .binary_scalar_right_int_out(&left, operation, 3)
            .expect("write scalar comparison");
        let mut tensor_mutation = Tensor::from_float(3.0).expect("create tensor mutation");
        tensor_mutation
            .binary_tensor_in_place(operation, &right)
            .expect("mutate tensor comparison");
        let mut scalar_mutation = Tensor::from_float(3.0).expect("create scalar mutation");
        scalar_mutation
            .binary_scalar_right_float_in_place(operation, 3.0)
            .expect("mutate scalar comparison");

        assert_eq!(
            scalar_int.item_bool().expect("read integer result"),
            expected
        );
        assert_eq!(
            scalar_float.item_bool().expect("read floating result"),
            expected
        );
        assert_eq!(
            tensor_output.item_bool().expect("read tensor output"),
            expected
        );
        assert_eq!(
            scalar_output.item_bool().expect("read scalar output"),
            expected
        );
        assert_eq!(
            flat_item(&tensor_mutation, 0),
            if expected { 1.0 } else { 0.0 }
        );
        assert_eq!(
            flat_item(&scalar_mutation, 0),
            if expected { 1.0 } else { 0.0 }
        );
    }
    assert_eq!(flat_item(&left, 0), 3.0);
    assert_eq!(flat_item(&right, 0), 3.0);
}

#[test]
fn invalid_operations_and_dtypes_fail_without_mutating_inputs() {
    let integers = Tensor::from_ints(&[5, 6], &[2]).expect("create integer input");
    let invalid_operation = required_tensor_error(
        integers.binary_scalar_right_int(99, 1),
        "unknown operation unexpectedly succeeded",
    );
    let floats = Tensor::from_floats(&[1.5, 2.5], &[2]).expect("create float input");
    let invalid_dtype = required_tensor_error(
        floats.binary_scalar_right_float(0, 1.0),
        "floating bitwise operation unexpectedly succeeded",
    );
    let mut mutation = Tensor::from_floats(&[1.5, 2.5], &[2]).expect("create mutation input");
    let mutation_error = mutation
        .binary_scalar_right_float_in_place(0, 1.0)
        .expect_err("floating bitwise mutation unexpectedly succeeded");

    assert_eq!(invalid_operation.status, 7120);
    assert_eq!(invalid_dtype.status, 7121);
    assert_eq!(mutation_error.status, 7121);
    assert_eq!(values(&integers), vec![5.0, 6.0]);
    assert_eq!(values(&floats), vec![1.5, 2.5]);
    assert_eq!(values(&mutation), vec![1.5, 2.5]);
}
