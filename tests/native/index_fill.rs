use terlan_pytorch_cxx::{CAbiError, Tensor};

fn sum(tensor: &Tensor) -> f64 {
    tensor
        .sum()
        .expect("sum tensor")
        .item_float()
        .expect("read sum")
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn scalar_forms_preserve_integer_float_and_storage_contracts() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create input");
    let index = Tensor::from_ints(&[1], &[1]).expect("create index");
    let integer = input
        .index_fill_scalar(1, &index, true, 9, 0.0)
        .expect("fill integer scalar");
    let floating = input
        .index_fill_scalar(0, &index, false, 0, 0.5)
        .expect("fill floating scalar");
    let mut output = Tensor::from_float(-1.0).expect("create output");
    output
        .index_fill_scalar_out(&input, 1, &index, false, 0, 2.5)
        .expect("write scalar output");
    let mut mutation =
        Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create mutation");
    mutation
        .index_fill_scalar_in_place(1, &index, true, 7, 0.0)
        .expect("fill scalar in place");

    assert_eq!(sum(&integer), 22.0);
    assert_eq!(sum(&floating), 4.0);
    assert_eq!(sum(&output), 9.0);
    assert_eq!(sum(&mutation), 18.0);
    assert_eq!(sum(&input), 10.0);
    assert_eq!(sum(&index), 1.0);
}

#[test]
fn tensor_forms_preserve_value_and_caller_owned_storage() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create input");
    let index = Tensor::from_ints(&[0], &[1]).expect("create index");
    let value = Tensor::from_float(5.0).expect("create value");
    let functional = input
        .index_fill_tensor(1, &index, &value)
        .expect("fill tensor value");
    let mut output = Tensor::from_float(-1.0).expect("create output");
    output
        .index_fill_tensor_out(&input, 1, &index, &value)
        .expect("write tensor output");
    let mut mutation =
        Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create mutation");
    mutation
        .index_fill_tensor_in_place(1, &index, &value)
        .expect("fill tensor in place");

    assert_eq!(sum(&functional), 16.0);
    assert_eq!(sum(&output), 16.0);
    assert_eq!(sum(&mutation), 16.0);
    assert_eq!(sum(&input), 10.0);
    assert_eq!(sum(&index), 0.0);
    assert_eq!(sum(&value), 5.0);
}

#[test]
fn invalid_indexes_fail_without_mutating_output_or_input() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create input");
    let invalid_index = Tensor::from_ints(&[5], &[1]).expect("create invalid index");
    let value = Tensor::from_float(9.0).expect("create value");
    let error = required_tensor_error(
        input.index_fill_tensor(1, &invalid_index, &value),
        "out-of-range index unexpectedly succeeded",
    );
    let mut output = Tensor::from_float(-1.0).expect("create output");
    let output_error = output
        .index_fill_scalar_out(&input, 1, &invalid_index, true, 9, 0.0)
        .expect_err("invalid output fill unexpectedly succeeded");
    let mut mutation =
        Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create mutation");
    let mutation_error = mutation
        .index_fill_tensor_in_place(1, &invalid_index, &value)
        .expect_err("invalid in-place fill unexpectedly succeeded");

    assert_eq!(error.status, 7181);
    assert_eq!(output_error.status, 7181);
    assert_eq!(mutation_error.status, 7181);
    assert_eq!(sum(&output), -1.0);
    assert_eq!(sum(&mutation), 10.0);
    assert_eq!(sum(&input), 10.0);
}

#[test]
fn cuda_executes_scalar_and_tensor_index_fill_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let input = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create input")
        .to_device(1, 0)
        .expect("place input on CUDA");
    let index = Tensor::from_ints(&[1], &[1])
        .expect("create index")
        .to_device(1, 0)
        .expect("place index on CUDA");
    let value = Tensor::from_float(5.0)
        .expect("create value")
        .to_device(1, 0)
        .expect("place value on CUDA");
    let scalar = input
        .index_fill_scalar(1, &index, true, 9, 0.0)
        .expect("execute scalar fill on CUDA");
    let tensor = input
        .index_fill_tensor(1, &index, &value)
        .expect("execute tensor fill on CUDA");
    let scalar_host = scalar.to_device(0, -1).expect("copy scalar result");
    let tensor_host = tensor.to_device(0, -1).expect("copy tensor result");

    assert_eq!(sum(&scalar_host), 22.0);
    assert_eq!(sum(&tensor_host), 14.0);
}
