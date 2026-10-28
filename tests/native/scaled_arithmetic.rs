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
fn functional_forms_preserve_operations_and_alpha_types() {
    let left = Tensor::from_float(10.0).expect("create left");
    let right = Tensor::from_float(3.0).expect("create right");
    let add_tensor = left
        .scaled_tensor(0, &right, true, 2, 0.0)
        .expect("add tensor with integer alpha");
    let sub_tensor = left
        .scaled_tensor(1, &right, false, 0, 0.5)
        .expect("subtract tensor with float alpha");
    let rsub_tensor = left
        .scaled_tensor(2, &right, true, 2, 0.0)
        .expect("reverse subtract tensor");
    let add_int = left
        .scaled_scalar_int(0, 3, true, 2, 0.0)
        .expect("add integer scalar");
    let sub_float = left
        .scaled_scalar_float(1, 3.0, false, 0, 0.5)
        .expect("subtract float scalar");
    let rsub_int = left
        .scaled_scalar_int(2, 3, true, 2, 0.0)
        .expect("reverse subtract integer scalar");

    assert_eq!(item(&add_tensor), 16.0);
    assert_eq!(item(&sub_tensor), 8.5);
    assert_eq!(item(&rsub_tensor), -17.0);
    assert_eq!(item(&add_int), 16.0);
    assert_eq!(item(&sub_float), 8.5);
    assert_eq!(item(&rsub_int), -17.0);
    assert_eq!(item(&left), 10.0);
    assert_eq!(item(&right), 3.0);
}

#[test]
fn output_forms_retain_storage_and_sources() {
    let left = Tensor::from_float(10.0).expect("create left");
    let right = Tensor::from_float(3.0).expect("create right");
    let mut add_output = Tensor::from_float(0.0).expect("create add output");
    add_output
        .scaled_tensor_out(&left, 0, &right, true, 2, 0.0)
        .expect("write add output");
    let mut sub_output = Tensor::from_float(0.0).expect("create sub output");
    sub_output
        .scaled_scalar_int_out(&left, 1, 3, false, 0, 0.5)
        .expect("write sub output");
    let mut rsub_output = Tensor::from_float(0.0).expect("create rsub output");
    rsub_output
        .scaled_scalar_float_out(&left, 2, 3.0, true, 2, 0.0)
        .expect("write rsub output");

    assert_eq!(item(&add_output), 16.0);
    assert_eq!(item(&sub_output), 8.5);
    assert_eq!(item(&rsub_output), -17.0);
    assert_eq!(item(&left), 10.0);
    assert_eq!(item(&right), 3.0);
}

#[test]
fn mutation_forms_apply_add_and_subtract_only() {
    let right = Tensor::from_float(3.0).expect("create right");
    let mut add_tensor = Tensor::from_float(10.0).expect("create add mutation");
    add_tensor
        .scaled_tensor_in_place(0, &right, true, 2, 0.0)
        .expect("mutate tensor add");
    let mut sub_int = Tensor::from_float(10.0).expect("create int mutation");
    sub_int
        .scaled_scalar_int_in_place(1, 3, false, 0, 0.5)
        .expect("mutate integer subtract");
    let mut add_float = Tensor::from_float(10.0).expect("create float mutation");
    add_float
        .scaled_scalar_float_in_place(0, 3.0, true, 2, 0.0)
        .expect("mutate float add");

    assert_eq!(item(&add_tensor), 16.0);
    assert_eq!(item(&sub_int), 8.5);
    assert_eq!(item(&add_float), 16.0);
    assert_eq!(item(&right), 3.0);
}

#[test]
fn invalid_operations_fail_without_mutating_storage() {
    let left = Tensor::from_float(10.0).expect("create left");
    let right = Tensor::from_float(3.0).expect("create right");
    let functional = required_tensor_error(
        left.scaled_tensor(99, &right, true, 1, 0.0),
        "unknown operation unexpectedly succeeded",
    );
    let mut mutation = Tensor::from_float(10.0).expect("create mutation");
    let reverse_mutation = mutation
        .scaled_tensor_in_place(2, &right, true, 1, 0.0)
        .expect_err("reverse subtraction mutation unexpectedly succeeded");
    let mut output = Tensor::from_float(7.0).expect("create output");
    let output_error = output
        .scaled_scalar_float_out(&left, -1, 3.0, true, 1, 0.0)
        .expect_err("negative operation unexpectedly succeeded");

    assert_eq!(functional.status, 7140);
    assert_eq!(reverse_mutation.status, 7140);
    assert_eq!(output_error.status, 7140);
    assert_eq!(item(&mutation), 10.0);
    assert_eq!(item(&output), 7.0);
    assert_eq!(item(&left), 10.0);
    assert_eq!(item(&right), 3.0);
}

#[test]
fn cuda_executes_scaled_and_binary_arithmetic_when_required() {
    let host_left = Tensor::from_float(10.0).expect("create host left");
    let host_right = Tensor::from_float(2.0).expect("create host right");
    let device_left = match host_left.to_device(1, 0) {
        Ok(tensor) => tensor,
        Err(error) => {
            if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
                return;
            }
            panic!("required CUDA placement failed: {error:?}");
        }
    };
    let device_right = host_right
        .to_device(1, 0)
        .expect("place right tensor on CUDA");
    let add = device_left
        .scaled_tensor(0, &device_right, true, 3, 0.0)
        .expect("execute scaled add on CUDA");
    let division = device_left
        .div_right_float(2.0)
        .expect("execute division on CUDA");
    let multiplication = device_left
        .binary_scalar_right_float(8, 2.0)
        .expect("execute multiplication on CUDA");
    let floor_division = device_left
        .binary_scalar_right_float(9, 2.0)
        .expect("execute floor division on CUDA");
    let xlogy = device_left
        .binary_scalar_right_float(10, 2.0)
        .expect("execute xlogy on CUDA");
    let host_add = add.to_device(0, -1).expect("copy add result to CPU");
    let host_division = division
        .to_device(0, -1)
        .expect("copy division result to CPU");
    let host_multiplication = multiplication
        .to_device(0, -1)
        .expect("copy multiplication result to CPU");
    let host_floor = floor_division
        .to_device(0, -1)
        .expect("copy floor division result to CPU");
    let host_xlogy = xlogy.to_device(0, -1).expect("copy xlogy result to CPU");

    assert_eq!(item(&host_add), 16.0);
    assert_eq!(item(&host_division), 5.0);
    assert_eq!(item(&host_multiplication), 20.0);
    assert_eq!(item(&host_floor), 5.0);
    assert!((item(&host_xlogy) - 10.0 * 2.0_f64.ln()).abs() < 1.0e-12);
    for (operation, expected) in [
        (11, true),
        (12, false),
        (13, false),
        (14, true),
        (15, false),
        (16, true),
    ] {
        let functional = device_left
            .binary_scalar_right_float(operation, 10.0)
            .expect("execute comparison on CUDA");
        let host_functional = functional.to_device(0, -1).expect("copy comparison to CPU");
        let mut output = Tensor::from_bool(false)
            .expect("create comparison output")
            .to_device(1, 0)
            .expect("place comparison output on CUDA");
        output
            .binary_scalar_right_float_out(&device_left, operation, 10.0)
            .expect("write comparison output on CUDA");
        let host_output = output
            .to_device(0, -1)
            .expect("copy comparison output to CPU");
        let mut mutation = host_left
            .to_device(1, 0)
            .expect("create comparison mutation on CUDA");
        mutation
            .binary_scalar_right_float_in_place(operation, 10.0)
            .expect("mutate comparison on CUDA");
        let host_mutation = mutation
            .to_device(0, -1)
            .expect("copy comparison mutation to CPU");
        assert_eq!(
            host_functional.item_bool().expect("read comparison"),
            expected
        );
        assert_eq!(
            host_output.item_bool().expect("read comparison output"),
            expected
        );
        assert_eq!(item(&host_mutation), if expected { 1.0 } else { 0.0 });
    }
    assert_eq!(device_left.device_type().expect("read device type"), 1);
    assert_eq!(device_left.device_index().expect("read device index"), 0);
    assert_eq!(item(&host_left), 10.0);
    assert_eq!(item(&host_right), 2.0);
}
