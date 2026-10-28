use terlan_pytorch_cxx::{CAbiError, Tensor};

fn sum(tensor: &Tensor) -> f64 {
    tensor
        .sum()
        .expect("sum tensor")
        .item_float()
        .expect("read sum")
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn tensor_scatter_modes_preserve_storage_contracts() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create input");
    let index = Tensor::from_ints(&[0, 2, 0], &[3]).expect("create index");
    let source = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create source");
    let legacy = input
        .scatter_tensor(1, 0, &index, &source, 0, true)
        .expect("legacy additive scatter");
    let reduced = input
        .scatter_tensor(3, 0, &index, &source, 0, false)
        .expect("modern reduced scatter");
    let mut plain_output = input.clone().expect("clone plain output");
    plain_output
        .scatter_tensor_out(&input, 0, 0, &index, &source, 0, true)
        .expect("plain scatter output");
    let mut legacy_output = input.clone().expect("clone legacy output");
    legacy_output
        .scatter_tensor_out(&input, 1, 0, &index, &source, 0, true)
        .expect("legacy scatter output");
    let mut additive_output = input.clone().expect("clone additive output");
    additive_output
        .scatter_tensor_out(&input, 2, 0, &index, &source, 0, true)
        .expect("additive scatter output");
    let mut plain_mutation = input.clone().expect("clone plain mutation");
    plain_mutation
        .scatter_tensor_in_place(0, 0, &index, &source, 0, true)
        .expect("plain scatter mutation");
    let mut additive_mutation = input.clone().expect("clone additive mutation");
    additive_mutation
        .scatter_tensor_in_place(2, 0, &index, &source, 0, true)
        .expect("additive scatter mutation");
    let mut reduced_mutation = input.clone().expect("clone reduced mutation");
    reduced_mutation
        .scatter_tensor_in_place(3, 0, &index, &source, 3, false)
        .expect("maximum scatter mutation");

    assert_eq!(sum(&plain_output), 7.0);
    assert_eq!(sum(&plain_mutation), 7.0);
    assert_eq!(sum(&legacy), 12.0);
    assert_eq!(sum(&legacy_output), 12.0);
    assert_eq!(sum(&additive_output), 12.0);
    assert_eq!(sum(&additive_mutation), 12.0);
    assert_eq!(sum(&reduced), 8.0);
    assert_eq!(sum(&reduced_mutation), 7.0);
    assert_eq!(sum(&input), 6.0);
    assert_eq!(sum(&source), 6.0);
}

#[test]
fn scalar_and_modern_reductions_cover_every_identifier() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create input");
    let index = Tensor::from_ints(&[0, 2], &[2]).expect("create index");
    let integer = input
        .scatter_scalar(0, 0, &index, true, 5, 0.0, 0)
        .expect("integer scalar scatter");
    let floating = input
        .scatter_scalar(0, 0, &index, false, 0, 0.5, 0)
        .expect("floating scalar scatter");
    let reduced_integer = input
        .scatter_scalar(1, 0, &index, true, 5, 0.0, 0)
        .expect("integer reduced scatter");
    let mut scalar_output = input.clone().expect("clone scalar output");
    scalar_output
        .scatter_scalar_out(&input, 0, 0, &index, false, 0, 0.5, 0)
        .expect("scalar output");
    let mut reduced_output = input.clone().expect("clone reduced output");
    reduced_output
        .scatter_scalar_out(&input, 1, 0, &index, true, 2, 0.0, 1)
        .expect("reduced scalar output");
    let mut scalar_mutation = input.clone().expect("clone scalar mutation");
    scalar_mutation
        .scatter_scalar_in_place(0, 0, &index, true, 5, 0.0, 0)
        .expect("scalar mutation");
    let source = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create source");
    let reductions = [
        (0, true, 9.0),
        (1, true, 9.0),
        (2, true, 5.5),
        (3, false, 5.0),
        (4, false, 5.0),
    ];

    assert_eq!(sum(&integer), 12.0);
    assert_eq!(sum(&floating), 3.0);
    assert_eq!(sum(&reduced_integer), 16.0);
    assert_eq!(sum(&scalar_output), 3.0);
    assert_eq!(sum(&reduced_output), 10.0);
    assert_eq!(sum(&scalar_mutation), 12.0);
    for (reduction, include_self, expected) in reductions {
        let result = input
            .scatter_tensor(3, 0, &index, &source, reduction, include_self)
            .expect("modern reduction");
        assert_close(sum(&result), expected);
    }
}

#[test]
fn invalid_modes_reductions_and_indices_are_atomic() {
    let input = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create input");
    let index = Tensor::from_ints(&[0], &[1]).expect("create index");
    let source = Tensor::from_float(2.0).expect("create source");
    let invalid_mode = required_tensor_error(
        input.scatter_tensor(9, 0, &index, &source, 0, true),
        "invalid tensor mode unexpectedly succeeded",
    );
    let invalid_reduction = required_tensor_error(
        input.scatter_tensor(3, 0, &index, &source, 9, true),
        "invalid reduction unexpectedly succeeded",
    );
    let invalid_index = Tensor::from_ints(&[5], &[1]).expect("create invalid index");
    let mut output = input.clone().expect("clone output");
    let output_error = output
        .scatter_tensor_out(&input, 0, 0, &invalid_index, &source, 0, true)
        .expect_err("invalid output scatter unexpectedly succeeded");
    let mut mutation = input.clone().expect("clone mutation");
    let mutation_error = mutation
        .scatter_scalar_in_place(0, 0, &invalid_index, true, 5, 0.0, 0)
        .expect_err("invalid scalar mutation unexpectedly succeeded");

    assert_eq!(invalid_mode.status, 7210);
    assert_eq!(invalid_reduction.status, 7211);
    assert_eq!(output_error.status, 7211);
    assert_eq!(mutation_error.status, 7211);
    assert_eq!(sum(&output), 6.0);
    assert_eq!(sum(&mutation), 6.0);
    assert_eq!(sum(&input), 6.0);
}

#[test]
fn cuda_executes_every_scatter_primitive_when_required() {
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
    let source = Tensor::from_floats(&[1.0, 2.0], &[2])
        .expect("create source")
        .to_device(1, 0)
        .expect("place source on CUDA");
    let tensor = input
        .scatter_tensor(1, 0, &index, &source, 0, true)
        .expect("tensor functional");
    let mut tensor_output = input.clone().expect("clone tensor output");
    tensor_output
        .scatter_tensor_out(&input, 2, 0, &index, &source, 0, true)
        .expect("tensor output");
    let mut tensor_mutation = input.clone().expect("clone tensor mutation");
    tensor_mutation
        .scatter_tensor_in_place(3, 0, &index, &source, 0, false)
        .expect("tensor mutation");
    let scalar = input
        .scatter_scalar(0, 0, &index, true, 5, 0.0, 0)
        .expect("scalar functional");
    let mut scalar_output = input.clone().expect("clone scalar output");
    scalar_output
        .scatter_scalar_out(&input, 1, 0, &index, true, 2, 0.0, 1)
        .expect("scalar output");
    let mut scalar_mutation = input.clone().expect("clone scalar mutation");
    scalar_mutation
        .scatter_scalar_in_place(0, 0, &index, false, 0, 0.5, 0)
        .expect("scalar mutation");

    for (tensor, expected) in [
        (&tensor, 9.0),
        (&tensor_output, 9.0),
        (&tensor_mutation, 5.0),
        (&scalar, 12.0),
        (&scalar_output, 10.0),
        (&scalar_mutation, 3.0),
    ] {
        let host = tensor.to_device(0, -1).expect("copy result to CPU");
        assert_eq!(sum(&host), expected);
    }
}
