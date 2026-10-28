use terlan_pytorch_cxx::{CAbiError, Tensor};

fn sum(tensor: &Tensor) -> f64 {
    tensor
        .sum()
        .expect("sum tensor")
        .item_float()
        .expect("read sum")
}

fn mask(values: &[f64], shape: &[i64]) -> Tensor {
    let values = Tensor::from_floats(values, shape).expect("create mask values");
    let zero = Tensor::from_float(0.0).expect("create mask threshold");
    values.greater_tensor(&zero).expect("create boolean mask")
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn full_fill_forms_preserve_values_and_storage_contracts() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create input");
    let integer = input.fill_scalar(true, 3, 0.0).expect("integer fill");
    let floating = input.fill_scalar(false, 0, 0.5).expect("floating fill");
    let mut integer_output = Tensor::from_float(-1.0).expect("create integer output");
    integer_output
        .fill_scalar_out(&input, true, 3, 0.0)
        .expect("integer output fill");
    let mut floating_mutation = input.clone().expect("clone input");
    floating_mutation
        .fill_scalar_in_place(false, 0, 0.5)
        .expect("floating in-place fill");
    let value = Tensor::from_float(2.5).expect("create scalar tensor");
    let mut tensor_output = Tensor::from_float(-1.0).expect("create tensor output");
    tensor_output
        .fill_tensor_out(&input, &value)
        .expect("tensor output fill");
    let mut tensor_mutation = input.clone().expect("clone tensor mutation");
    tensor_mutation
        .fill_tensor_in_place(&value)
        .expect("tensor in-place fill");

    assert_eq!(sum(&integer), 12.0);
    assert_eq!(sum(&floating), 2.0);
    assert_eq!(sum(&integer_output), 12.0);
    assert_eq!(sum(&floating_mutation), 2.0);
    assert_eq!(sum(&tensor_output), 10.0);
    assert_eq!(sum(&tensor_mutation), 10.0);
    assert_eq!(sum(&input), 10.0);
    assert_eq!(sum(&value), 2.5);
}

#[test]
fn diagonal_and_masked_fill_forms_match_libtorch() {
    let mut diagonal = Tensor::from_floats(&[0.0; 9], &[3, 3]).expect("create diagonal input");
    diagonal
        .fill_diagonal_in_place(true, 2, 0.0, false)
        .expect("integer diagonal fill");
    let mut wrapped = Tensor::from_floats(&[0.0; 10], &[5, 2]).expect("create wrapped input");
    wrapped
        .fill_diagonal_in_place(false, 0, 1.5, true)
        .expect("wrapped diagonal fill");
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create input");
    let selected = mask(&[0.0, 1.0, 0.0, 1.0], &[2, 2]);
    let integer = input
        .masked_fill_scalar(&selected, true, 9, 0.0)
        .expect("integer masked fill");
    let floating = input
        .masked_fill_scalar(&selected, false, 0, 0.5)
        .expect("floating masked fill");
    let mut scalar_output = Tensor::from_float(-1.0).expect("create scalar output");
    scalar_output
        .masked_fill_scalar_out(&input, &selected, true, 9, 0.0)
        .expect("scalar masked output");
    let mut scalar_mutation = input.clone().expect("clone scalar mutation");
    scalar_mutation
        .masked_fill_scalar_in_place(&selected, false, 0, 0.5)
        .expect("scalar masked mutation");
    let value = Tensor::from_float(7.0).expect("create tensor value");
    let mut tensor_output = Tensor::from_float(-1.0).expect("create tensor output");
    tensor_output
        .masked_fill_tensor_out(&input, &selected, &value)
        .expect("tensor masked output");
    let mut tensor_mutation = input.clone().expect("clone tensor mutation");
    tensor_mutation
        .masked_fill_tensor_in_place(&selected, &value)
        .expect("tensor masked mutation");

    assert_eq!(sum(&diagonal), 6.0);
    assert_eq!(sum(&wrapped), 6.0);
    assert_eq!(sum(&integer), 22.0);
    assert_eq!(sum(&floating), 5.0);
    assert_eq!(sum(&scalar_output), 22.0);
    assert_eq!(sum(&scalar_mutation), 5.0);
    assert_eq!(sum(&tensor_output), 18.0);
    assert_eq!(sum(&tensor_mutation), 18.0);
    assert_eq!(sum(&input), 10.0);
}

#[test]
fn malformed_masks_and_values_fail_without_partial_mutation() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create input");
    let invalid_mask = mask(&[0.0, 1.0, 0.0], &[3]);
    let error = required_tensor_error(
        input.masked_fill_scalar(&invalid_mask, true, 9, 0.0),
        "invalid mask unexpectedly succeeded",
    );
    let mut output = Tensor::from_float(-1.0).expect("create output");
    let output_error = output
        .masked_fill_scalar_out(&input, &invalid_mask, true, 9, 0.0)
        .expect_err("invalid masked output unexpectedly succeeded");
    let mut mutation = input.clone().expect("clone mutation");
    let mutation_error = mutation
        .masked_fill_scalar_in_place(&invalid_mask, true, 9, 0.0)
        .expect_err("invalid masked mutation unexpectedly succeeded");
    let invalid_value = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create invalid value");
    let mut fill_output = Tensor::from_float(-2.0).expect("create fill output");
    let fill_error = fill_output
        .fill_tensor_out(&input, &invalid_value)
        .expect_err("non-scalar fill tensor unexpectedly succeeded");

    assert_eq!(error.status, 7201);
    assert_eq!(output_error.status, 7201);
    assert_eq!(mutation_error.status, 7201);
    assert_eq!(fill_error.status, 7201);
    assert_eq!(sum(&output), -1.0);
    assert_eq!(sum(&mutation), 10.0);
    assert_eq!(sum(&fill_output), -2.0);
    assert_eq!(sum(&input), 10.0);
}

#[test]
fn cuda_executes_every_fill_primitive_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create input")
        .to_device(1, 0)
        .expect("place input on CUDA");
    let mask = mask(&[0.0, 1.0, 0.0, 1.0], &[2, 2])
        .to_device(1, 0)
        .expect("place mask on CUDA");
    let value = Tensor::from_float(7.0)
        .expect("create value")
        .to_device(1, 0)
        .expect("place value on CUDA");
    let filled = input.fill_scalar(true, 3, 0.0).expect("functional fill");
    let mut fill_output = input.clone().expect("clone fill output");
    fill_output
        .fill_scalar_out(&input, false, 0, 0.5)
        .expect("scalar fill output");
    let mut fill_mutation = input.clone().expect("clone fill mutation");
    fill_mutation
        .fill_scalar_in_place(true, 3, 0.0)
        .expect("scalar fill mutation");
    let mut tensor_output = input.clone().expect("clone tensor output");
    tensor_output
        .fill_tensor_out(&input, &value)
        .expect("tensor fill output");
    let mut tensor_mutation = input.clone().expect("clone tensor mutation");
    tensor_mutation
        .fill_tensor_in_place(&value)
        .expect("tensor fill mutation");
    let mut diagonal = input.clone().expect("clone diagonal");
    diagonal
        .fill_diagonal_in_place(false, 0, 2.5, false)
        .expect("diagonal fill");
    let masked = input
        .masked_fill_scalar(&mask, true, 9, 0.0)
        .expect("functional masked fill");
    let mut masked_output = input.clone().expect("clone masked output");
    masked_output
        .masked_fill_scalar_out(&input, &mask, false, 0, 0.5)
        .expect("scalar masked output");
    let mut masked_mutation = input.clone().expect("clone masked mutation");
    masked_mutation
        .masked_fill_scalar_in_place(&mask, true, 9, 0.0)
        .expect("scalar masked mutation");
    let mut masked_tensor_output = input.clone().expect("clone tensor output");
    masked_tensor_output
        .masked_fill_tensor_out(&input, &mask, &value)
        .expect("tensor masked output");
    let mut masked_tensor_mutation = input.clone().expect("clone tensor mutation");
    masked_tensor_mutation
        .masked_fill_tensor_in_place(&mask, &value)
        .expect("tensor masked mutation");

    for (tensor, expected) in [
        (&filled, 12.0),
        (&fill_output, 2.0),
        (&fill_mutation, 12.0),
        (&tensor_output, 28.0),
        (&tensor_mutation, 28.0),
        (&diagonal, 10.0),
        (&masked, 22.0),
        (&masked_output, 5.0),
        (&masked_mutation, 22.0),
        (&masked_tensor_output, 18.0),
        (&masked_tensor_mutation, 18.0),
    ] {
        let host = tensor.to_device(0, -1).expect("copy result to CPU");
        assert_eq!(sum(&host), expected);
    }
}
