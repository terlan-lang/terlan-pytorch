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
fn index_extraction_returns_independently_owned_dimension_tensors() {
    let values = Tensor::from_ints(&[0, 1, 0, 1], &[4]).expect("create values");
    let condition = values
        .binary_scalar_right_int(11, 1)
        .expect("create condition");
    let indices = condition.where_indices().expect("extract true indices");

    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].shape().expect("index shape"), vec![2]);
    assert_eq!(item(&indices[0].sum().expect("sum indices")), 4.0);
    assert_eq!(condition.shape().expect("condition shape"), vec![4]);
    drop(condition);
    assert_eq!(item(&indices[0].sum().expect("sum owned indices")), 4.0);
}

#[test]
fn scalar_selection_preserves_integer_float_and_mixed_values() {
    let true_condition = Tensor::from_bool(true).expect("create true condition");
    let false_condition = Tensor::from_bool(false).expect("create false condition");
    let int_int = true_condition
        .where_scalar(true, 3, 0.0, true, 7, 0.0)
        .expect("select integer scalars");
    let float_float = false_condition
        .where_scalar(false, 0, 3.5, false, 0, 7.5)
        .expect("select floating scalars");
    let int_float = true_condition
        .where_scalar(true, 3, 0.0, false, 0, 7.5)
        .expect("select mixed scalars");
    let float_int = false_condition
        .where_scalar(false, 0, 3.5, true, 7, 0.0)
        .expect("select reverse mixed scalars");

    assert_eq!(item(&int_int), 3.0);
    assert_eq!(item(&float_float), 7.5);
    assert_eq!(item(&int_float), 3.0);
    assert_eq!(item(&float_int), 7.0);
}

#[test]
fn tensor_scalar_and_output_forms_retain_every_source() {
    let true_condition = Tensor::from_bool(true).expect("create true condition");
    let false_condition = Tensor::from_bool(false).expect("create false condition");
    let when_true = Tensor::from_float(3.0).expect("create true value");
    let when_false = Tensor::from_float(7.0).expect("create false value");
    let tensor_scalar = true_condition
        .where_scalar_other(&when_true, false, 0, 9.5)
        .expect("select tensor or scalar");
    let scalar_tensor = false_condition
        .where_scalar_self(true, 2, 0.0, &when_false)
        .expect("select scalar or tensor");
    let mut output = Tensor::from_float(-1.0).expect("create output");
    output
        .where_tensor_out(&false_condition, &when_true, &when_false)
        .expect("write tensor selection");

    assert_eq!(item(&tensor_scalar), 3.0);
    assert_eq!(item(&scalar_tensor), 7.0);
    assert_eq!(item(&output), 7.0);
    assert_eq!(item(&when_true), 3.0);
    assert_eq!(item(&when_false), 7.0);
}

#[test]
fn invalid_conditions_fail_without_mutating_output_or_sources() {
    let invalid_condition = Tensor::from_int(1).expect("create invalid condition");
    let when_true = Tensor::from_float(3.0).expect("create true value");
    let when_false = Tensor::from_float(7.0).expect("create false value");
    let scalar_error = required_tensor_error(
        invalid_condition.where_scalar(true, 1, 0.0, true, 2, 0.0),
        "non-boolean scalar condition unexpectedly succeeded",
    );
    let mut output = Tensor::from_float(-1.0).expect("create output");
    let output_error = output
        .where_tensor_out(&invalid_condition, &when_true, &when_false)
        .expect_err("non-boolean output condition unexpectedly succeeded");

    assert_eq!(scalar_error.status, 7171);
    assert_eq!(output_error.status, 7171);
    assert_eq!(item(&output), -1.0);
    assert_eq!(item(&when_true), 3.0);
    assert_eq!(item(&when_false), 7.0);
}

#[test]
fn cuda_executes_where_forms_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let condition = Tensor::from_bool(true)
        .expect("create condition")
        .to_device(1, 0)
        .expect("place condition on CUDA");
    let when_true = Tensor::from_float(3.0)
        .expect("create true value")
        .to_device(1, 0)
        .expect("place true value on CUDA");
    let when_false = Tensor::from_float(7.0)
        .expect("create false value")
        .to_device(1, 0)
        .expect("place false value on CUDA");
    let scalar = condition
        .where_scalar_other(&when_true, false, 0, 9.0)
        .expect("execute scalar where on CUDA");
    let mut output = Tensor::from_float(0.0)
        .expect("create output")
        .to_device(1, 0)
        .expect("place output on CUDA");
    output
        .where_tensor_out(&condition, &when_true, &when_false)
        .expect("execute output where on CUDA");
    let scalar_host = scalar.to_device(0, -1).expect("copy scalar result");
    let output_host = output.to_device(0, -1).expect("copy output result");

    assert_eq!(item(&scalar_host), 3.0);
    assert_eq!(item(&output_host), 3.0);
}
