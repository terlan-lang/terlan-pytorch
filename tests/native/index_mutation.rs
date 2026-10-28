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
fn index_add_preserves_alpha_and_storage_contracts() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create input");
    let index = Tensor::from_ints(&[0, 2], &[2]).expect("create index");
    let source = Tensor::from_floats(&[10.0, 20.0], &[2]).expect("create source");
    let integer = input
        .index_add(0, &index, &source, true, 2, 0.0)
        .expect("integer alpha");
    let floating = input
        .index_add(0, &index, &source, false, 0, 0.5)
        .expect("floating alpha");
    let mut output = Tensor::from_float(-1.0).expect("create output");
    output
        .index_add_out(&input, 0, &index, &source, true, 2, 0.0)
        .expect("output add");
    let mut mutation = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create mutation");
    mutation
        .index_add_in_place(0, &index, &source, false, 0, 0.5)
        .expect("in-place add");

    assert_eq!(sum(&integer), 66.0);
    assert_eq!(sum(&floating), 21.0);
    assert_eq!(sum(&output), 66.0);
    assert_eq!(sum(&mutation), 21.0);
    assert_eq!(sum(&input), 6.0);
    assert_eq!(sum(&source), 30.0);
}

#[test]
fn copy_select_and_backward_preserve_storage_contracts() {
    let input = Tensor::from_floats(&[0.0, 0.0, 0.0], &[3]).expect("create input");
    let index = Tensor::from_ints(&[2, 0], &[2]).expect("create index");
    let source = Tensor::from_floats(&[5.0, 6.0], &[2]).expect("create source");
    let mut copy_output = Tensor::from_float(-1.0).expect("create copy output");
    copy_output
        .index_copy_out(&input, 0, &index, &source)
        .expect("copy output");
    let mut copy_mutation =
        Tensor::from_floats(&[0.0, 0.0, 0.0], &[3]).expect("create copy mutation");
    copy_mutation
        .index_copy_in_place(0, &index, &source)
        .expect("copy in place");
    let values = Tensor::from_floats(&[10.0, 20.0, 30.0], &[3]).expect("create values");
    let mut select_output = Tensor::from_float(-1.0).expect("create select output");
    select_output
        .index_select_out(&values, 0, &index)
        .expect("select output");
    let gradient = Tensor::from_floats(&[2.0, 3.0], &[2]).expect("create gradient");
    let backward = gradient
        .index_select_backward(&[3], 0, &index)
        .expect("selection backward");

    assert_eq!(sum(&copy_output), 11.0);
    assert_eq!(sum(&copy_mutation), 11.0);
    assert_eq!(sum(&select_output), 40.0);
    assert_eq!(sum(&backward), 5.0);
    assert_eq!(sum(&input), 0.0);
    assert_eq!(sum(&values), 60.0);
    assert_eq!(sum(&gradient), 5.0);
}

#[test]
fn invalid_indexes_fail_without_partial_mutation() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create input");
    let invalid = Tensor::from_ints(&[5], &[1]).expect("create invalid index");
    let source = Tensor::from_float(9.0).expect("create source");
    let error = required_tensor_error(
        input.index_add(0, &invalid, &source, true, 1, 0.0),
        "invalid indexed add unexpectedly succeeded",
    );
    let mut output = Tensor::from_float(-1.0).expect("create output");
    let output_error = output
        .index_copy_out(&input, 0, &invalid, &source)
        .expect_err("invalid indexed copy unexpectedly succeeded");
    let mut mutation = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create mutation");
    let mutation_error = mutation
        .index_add_in_place(0, &invalid, &source, true, 1, 0.0)
        .expect_err("invalid in-place add unexpectedly succeeded");

    assert_eq!(error.status, 7191);
    assert_eq!(output_error.status, 7191);
    assert_eq!(mutation_error.status, 7191);
    assert_eq!(sum(&output), -1.0);
    assert_eq!(sum(&mutation), 6.0);
    assert_eq!(sum(&input), 6.0);
}

#[test]
fn cuda_executes_every_index_mutation_primitive_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3])
        .expect("create input")
        .to_device(1, 0)
        .expect("place input on CUDA");
    let index = Tensor::from_ints(&[0, 2], &[2])
        .expect("create index")
        .to_device(1, 0)
        .expect("place index on CUDA");
    let source = Tensor::from_floats(&[10.0, 20.0], &[2])
        .expect("create source")
        .to_device(1, 0)
        .expect("place source on CUDA");
    let added = input
        .index_add(0, &index, &source, true, 2, 0.0)
        .expect("functional add");
    let mut add_output = Tensor::from_float(-1.0)
        .expect("create output")
        .to_device(1, 0)
        .expect("place output on CUDA");
    add_output
        .index_add_out(&input, 0, &index, &source, false, 0, 0.5)
        .expect("output add");
    let mut add_mutation = input.clone().expect("clone input");
    add_mutation
        .index_add_in_place(0, &index, &source, true, 2, 0.0)
        .expect("in-place add");
    let mut copy_output = Tensor::from_float(-1.0)
        .expect("create copy output")
        .to_device(1, 0)
        .expect("place copy output on CUDA");
    copy_output
        .index_copy_out(&input, 0, &index, &source)
        .expect("copy output");
    let mut copy_mutation = input.clone().expect("clone input");
    copy_mutation
        .index_copy_in_place(0, &index, &source)
        .expect("copy in place");
    let mut select_output = Tensor::from_float(-1.0)
        .expect("create select output")
        .to_device(1, 0)
        .expect("place select output on CUDA");
    select_output
        .index_select_out(&input, 0, &index)
        .expect("select output");
    let gradient = Tensor::from_floats(&[2.0, 3.0], &[2])
        .expect("create gradient")
        .to_device(1, 0)
        .expect("place gradient on CUDA");
    let backward = gradient
        .index_select_backward(&[3], 0, &index)
        .expect("selection backward");

    for (tensor, expected) in [
        (&added, 66.0),
        (&add_output, 21.0),
        (&add_mutation, 66.0),
        (&copy_output, 32.0),
        (&copy_mutation, 32.0),
        (&select_output, 4.0),
        (&backward, 5.0),
    ] {
        let host = tensor.to_device(0, -1).expect("copy result to CPU");
        assert_eq!(sum(&host), expected);
    }
}
