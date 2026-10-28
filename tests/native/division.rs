use terlan_pytorch_cxx::{CAbiError, Tensor};

fn item(tensor: &Tensor) -> f64 {
    tensor.item_float().expect("read scalar tensor")
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "{actual} != {expected}"
    );
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn div_scalar_forms_cover_typed_values_outputs_and_mutation() {
    let source = Tensor::from_float(-7.5).expect("create source");
    let plain_int = source.div_right_int(2).expect("divide by integer");
    let floor_int = source
        .div_right_int_mode(2, "floor")
        .expect("floor divide by integer");
    let plain_float = source.div_right_float(2.0).expect("divide by float");
    let trunc_float = source
        .div_right_float_mode(2.0, "trunc")
        .expect("truncate divide by float");
    let mut plain_int_output = Tensor::from_float(0.0).expect("create integer output");
    plain_int_output
        .div_right_int_out(&source, 2)
        .expect("write integer output");
    let mut floor_int_output = Tensor::from_float(0.0).expect("create floor integer output");
    floor_int_output
        .div_right_int_mode_out(&source, 2, "floor")
        .expect("write floor integer output");
    let mut plain_float_output = Tensor::from_float(0.0).expect("create float output");
    plain_float_output
        .div_right_float_out(&source, 2.0)
        .expect("write float output");
    let mut trunc_float_output = Tensor::from_float(0.0).expect("create trunc float output");
    trunc_float_output
        .div_right_float_mode_out(&source, 2.0, "trunc")
        .expect("write trunc float output");
    let mut plain_int_mutation = Tensor::from_float(-7.5).expect("create integer mutation");
    plain_int_mutation
        .div_right_int_in_place(2)
        .expect("mutate by integer");
    let mut floor_int_mutation = Tensor::from_float(-7.5).expect("create floor mutation");
    floor_int_mutation
        .div_right_int_mode_in_place(2, "floor")
        .expect("floor mutate by integer");
    let mut plain_float_mutation = Tensor::from_float(-7.5).expect("create float mutation");
    plain_float_mutation
        .div_right_float_in_place(2.0)
        .expect("mutate by float");
    let mut trunc_float_mutation = Tensor::from_float(-7.5).expect("create trunc mutation");
    trunc_float_mutation
        .div_right_float_mode_in_place(2.0, "trunc")
        .expect("truncate mutate by float");

    for tensor in [
        &plain_int,
        &plain_float,
        &plain_int_output,
        &plain_float_output,
        &plain_int_mutation,
        &plain_float_mutation,
    ] {
        assert_close(item(tensor), -3.75);
    }
    for tensor in [&floor_int, &floor_int_output, &floor_int_mutation] {
        assert_eq!(item(tensor), -4.0);
    }
    for tensor in [&trunc_float, &trunc_float_output, &trunc_float_mutation] {
        assert_eq!(item(tensor), -3.0);
    }
    assert_eq!(item(&source), -7.5);
}

#[test]
fn div_tensor_forms_cover_rounding_outputs_and_mutation() {
    let left = Tensor::from_float(-7.5).expect("create left");
    let right = Tensor::from_float(2.0).expect("create right");
    let floor = left
        .div_tensor_mode(&right, "floor")
        .expect("floor tensor division");
    let mut output = Tensor::from_float(0.0).expect("create output");
    output
        .div_tensor_out(&left, &right)
        .expect("write tensor division");
    let mut floor_output = Tensor::from_float(0.0).expect("create floor output");
    floor_output
        .div_tensor_mode_out(&left, &right, "floor")
        .expect("write floor tensor division");
    let mut mutation = Tensor::from_float(-7.5).expect("create mutation");
    mutation
        .div_tensor_in_place(&right)
        .expect("mutate tensor division");
    let mut floor_mutation = Tensor::from_float(-7.5).expect("create floor mutation");
    floor_mutation
        .div_tensor_mode_in_place(&right, "floor")
        .expect("floor mutate tensor division");

    assert_eq!(item(&floor), -4.0);
    assert_close(item(&output), -3.75);
    assert_eq!(item(&floor_output), -4.0);
    assert_close(item(&mutation), -3.75);
    assert_eq!(item(&floor_mutation), -4.0);
    assert_eq!(item(&left), -7.5);
    assert_eq!(item(&right), 2.0);
}

#[test]
fn divide_alias_covers_every_scalar_and_tensor_shape() {
    let source = Tensor::from_float(-7.5).expect("create source");
    let right = Tensor::from_float(2.0).expect("create right");
    let int_value = source.divide_right_int(2).expect("integer alias");
    let int_floor = source
        .divide_right_int_mode(2, "floor")
        .expect("integer floor alias");
    let float_value = source.divide_right_float(2.0).expect("float alias");
    let float_trunc = source
        .divide_right_float_mode(2.0, "trunc")
        .expect("float trunc alias");
    let tensor_floor = source
        .divide_tensor_mode(&right, "floor")
        .expect("tensor floor alias");
    let mut int_mutation = Tensor::from_float(-7.5).expect("create int mutation");
    int_mutation
        .divide_right_int_in_place(2)
        .expect("integer alias mutation");
    let mut int_floor_mutation = Tensor::from_float(-7.5).expect("create int floor mutation");
    int_floor_mutation
        .divide_right_int_mode_in_place(2, "floor")
        .expect("integer floor alias mutation");
    let mut float_mutation = Tensor::from_float(-7.5).expect("create float mutation");
    float_mutation
        .divide_right_float_in_place(2.0)
        .expect("float alias mutation");
    let mut float_trunc_mutation = Tensor::from_float(-7.5).expect("create trunc mutation");
    float_trunc_mutation
        .divide_right_float_mode_in_place(2.0, "trunc")
        .expect("float trunc alias mutation");
    let mut output = Tensor::from_float(0.0).expect("create output");
    output
        .divide_tensor_out(&source, &right)
        .expect("write alias output");
    let mut floor_output = Tensor::from_float(0.0).expect("create floor output");
    floor_output
        .divide_tensor_mode_out(&source, &right, "floor")
        .expect("write floor alias output");
    let mut tensor_mutation = Tensor::from_float(-7.5).expect("create tensor mutation");
    tensor_mutation
        .divide_tensor_in_place(&right)
        .expect("tensor alias mutation");
    let mut tensor_floor_mutation = Tensor::from_float(-7.5).expect("create tensor floor mutation");
    tensor_floor_mutation
        .divide_tensor_mode_in_place(&right, "floor")
        .expect("tensor floor alias mutation");

    for tensor in [
        &int_value,
        &float_value,
        &int_mutation,
        &float_mutation,
        &output,
        &tensor_mutation,
    ] {
        assert_close(item(tensor), -3.75);
    }
    for tensor in [
        &int_floor,
        &tensor_floor,
        &int_floor_mutation,
        &floor_output,
        &tensor_floor_mutation,
    ] {
        assert_eq!(item(tensor), -4.0);
    }
    assert_eq!(item(&float_trunc), -3.0);
    assert_eq!(item(&float_trunc_mutation), -3.0);
    assert_eq!(item(&source), -7.5);
}

#[test]
fn true_divide_covers_scalars_tensors_outputs_and_mutation() {
    let source = Tensor::from_float(7.5).expect("create source");
    let right = Tensor::from_float(2.0).expect("create right");
    let int_value = source
        .true_divide_right_int(2)
        .expect("true divide integer");
    let float_value = source
        .true_divide_right_float(2.0)
        .expect("true divide float");
    let mut output = Tensor::from_float(0.0).expect("create output");
    output
        .true_divide_tensor_out(&source, &right)
        .expect("write true division");
    let mut int_mutation = Tensor::from_float(7.5).expect("create int mutation");
    int_mutation
        .true_divide_right_int_in_place(2)
        .expect("true divide integer in place");
    let mut float_mutation = Tensor::from_float(7.5).expect("create float mutation");
    float_mutation
        .true_divide_right_float_in_place(2.0)
        .expect("true divide float in place");
    let mut tensor_mutation = Tensor::from_float(7.5).expect("create tensor mutation");
    tensor_mutation
        .true_divide_tensor_in_place(&right)
        .expect("true divide tensor in place");

    for tensor in [
        &int_value,
        &float_value,
        &output,
        &int_mutation,
        &float_mutation,
        &tensor_mutation,
    ] {
        assert_eq!(item(tensor), 3.75);
    }
    assert_eq!(item(&source), 7.5);
}

#[test]
fn invalid_rounding_modes_fail_without_mutating_sources() {
    let source = Tensor::from_float(7.5).expect("create source");
    let right = Tensor::from_float(2.0).expect("create right");
    let functional_error = required_tensor_error(
        source.div_right_int_mode(2, "nearest"),
        "invalid scalar rounding mode unexpectedly succeeded",
    );
    let mut mutation = Tensor::from_float(7.5).expect("create mutation");
    let mutation_error = mutation
        .divide_tensor_mode_in_place(&right, "nearest")
        .expect_err("invalid tensor rounding mode unexpectedly succeeded");

    assert_ne!(functional_error.status, 0);
    assert_ne!(mutation_error.status, 0);
    assert_eq!(item(&source), 7.5);
    assert_eq!(item(&mutation), 7.5);
}
