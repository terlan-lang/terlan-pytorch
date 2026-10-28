use terlan_pytorch_cxx::{CAbiError, Tensor};

fn item(tensor: &Tensor) -> f64 {
    tensor.item_float().expect("read scalar tensor")
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn scalar_bounds_preserve_numeric_types_and_optional_positions() {
    let high = Tensor::from_float(10.5).expect("create high input");
    let low = Tensor::from_float(-10.5).expect("create low input");
    let integer_bounds = high
        .clamp_scalar(true, true, -2, 0.0, true, true, 3, 0.0)
        .expect("clamp integer bounds");
    let floating_bounds = low
        .clamp_scalar(true, false, 0, -2.5, true, false, 0, 3.5)
        .expect("clamp floating bounds");
    let mixed_bounds = high
        .clamp_scalar(true, true, -2, 0.0, true, false, 0, 3.5)
        .expect("clamp mixed bounds");
    let minimum_only = low
        .clamp_scalar(true, false, 0, -4.5, false, true, 0, 0.0)
        .expect("clamp minimum only");
    let maximum_only = high
        .clamp_scalar(false, true, 0, 0.0, true, true, 4, 0.0)
        .expect("clamp maximum only");

    assert_eq!(item(&integer_bounds), 3.0);
    assert_eq!(item(&floating_bounds), -2.5);
    assert_eq!(item(&mixed_bounds), 3.5);
    assert_eq!(item(&minimum_only), -4.5);
    assert_eq!(item(&maximum_only), 4.0);
    assert_eq!(item(&high), 10.5);
    assert_eq!(item(&low), -10.5);
}

#[test]
fn scalar_output_and_mutation_forms_retain_their_storage_contracts() {
    let source = Tensor::from_float(10.0).expect("create source");
    let mut output = Tensor::from_float(-99.0).expect("create output");
    output
        .clamp_scalar_out(&source, true, true, -2, 0.0, true, false, 0, 3.25)
        .expect("write scalar clamp output");
    let mut minimum = Tensor::from_float(-10.0).expect("create minimum mutation");
    minimum
        .clamp_scalar_in_place(true, true, -3, 0.0, false, true, 0, 0.0)
        .expect("apply minimum in place");
    let mut maximum = Tensor::from_float(10.0).expect("create maximum mutation");
    maximum
        .clamp_scalar_in_place(false, true, 0, 0.0, true, false, 0, 2.5)
        .expect("apply maximum in place");

    assert_eq!(item(&output), 3.25);
    assert_eq!(item(&minimum), -3.0);
    assert_eq!(item(&maximum), 2.5);
    assert_eq!(item(&source), 10.0);
}

#[test]
fn tensor_bounds_cover_minimum_maximum_and_paired_forms() {
    let high = Tensor::from_float(10.0).expect("create high input");
    let low = Tensor::from_float(-10.0).expect("create low input");
    let minimum = Tensor::from_float(-3.0).expect("create minimum");
    let maximum = Tensor::from_float(4.0).expect("create maximum");
    let minimum_only = low
        .clamp_tensor(0, &minimum, &minimum)
        .expect("clamp by minimum tensor");
    let maximum_only = high
        .clamp_tensor(1, &maximum, &maximum)
        .expect("clamp by maximum tensor");
    let paired = high
        .clamp_tensor(2, &minimum, &maximum)
        .expect("clamp by paired tensors");
    let mut output = Tensor::from_float(0.0).expect("create output");
    output
        .clamp_tensor_out(&low, 2, &minimum, &maximum)
        .expect("write paired tensor clamp");
    let mut mutation = Tensor::from_float(10.0).expect("create mutation");
    mutation
        .clamp_tensor_in_place(1, &maximum, &maximum)
        .expect("clamp tensor in place");

    assert_eq!(item(&minimum_only), -3.0);
    assert_eq!(item(&maximum_only), 4.0);
    assert_eq!(item(&paired), 4.0);
    assert_eq!(item(&output), -3.0);
    assert_eq!(item(&mutation), 4.0);
    assert_eq!(item(&high), 10.0);
    assert_eq!(item(&low), -10.0);
    assert_eq!(item(&minimum), -3.0);
    assert_eq!(item(&maximum), 4.0);
}

#[test]
fn invalid_contracts_fail_before_mutating_caller_storage() {
    let source = Tensor::from_float(10.0).expect("create source");
    let bound = Tensor::from_float(3.0).expect("create bound");
    let absent = required_tensor_error(
        source.clamp_scalar(false, true, 0, 0.0, false, true, 0, 0.0),
        "absent scalar bounds unexpectedly succeeded",
    );
    let mode = required_tensor_error(
        source.clamp_tensor(99, &bound, &bound),
        "invalid tensor mode unexpectedly succeeded",
    );
    let mut output = Tensor::from_float(7.0).expect("create output");
    let output_error = output
        .clamp_tensor_out(&source, -1, &bound, &bound)
        .expect_err("invalid output mode unexpectedly succeeded");
    let mut mutation = Tensor::from_float(8.0).expect("create mutation");
    let mutation_error = mutation
        .clamp_scalar_in_place(false, false, 0, 0.0, false, false, 0, 0.0)
        .expect_err("absent mutation bounds unexpectedly succeeded");

    assert_eq!(absent.status, 7150);
    assert_eq!(mode.status, 7150);
    assert_eq!(output_error.status, 7150);
    assert_eq!(mutation_error.status, 7150);
    assert_eq!(item(&output), 7.0);
    assert_eq!(item(&mutation), 8.0);
    assert_eq!(item(&source), 10.0);
    assert_eq!(item(&bound), 3.0);
}

#[test]
fn cuda_executes_scalar_and_tensor_clamp_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let host = Tensor::from_float(10.0).expect("create host tensor");
    let host_minimum = Tensor::from_float(-2.0).expect("create host minimum");
    let host_maximum = Tensor::from_float(3.0).expect("create host maximum");
    let device = host.to_device(1, 0).expect("copy input to CUDA");
    let minimum = host_minimum.to_device(1, 0).expect("copy minimum to CUDA");
    let maximum = host_maximum.to_device(1, 0).expect("copy maximum to CUDA");
    let scalar = device
        .clamp_scalar(true, false, 0, -2.0, true, false, 0, 4.0)
        .expect("execute scalar clamp on CUDA");
    let tensor = device
        .clamp_tensor(2, &minimum, &maximum)
        .expect("execute tensor clamp on CUDA");
    let scalar_host = scalar.to_device(0, -1).expect("copy scalar result to CPU");
    let tensor_host = tensor.to_device(0, -1).expect("copy tensor result to CPU");

    assert_eq!(item(&scalar_host), 4.0);
    assert_eq!(item(&tensor_host), 3.0);
}
