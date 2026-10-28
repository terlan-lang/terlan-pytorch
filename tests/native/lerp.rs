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
fn scalar_weights_preserve_numeric_type_and_extrapolation() {
    let start = Tensor::from_float(2.0).expect("create start");
    let end = Tensor::from_float(10.0).expect("create end");
    let integer = start
        .lerp_scalar(&end, true, 2, 0.0)
        .expect("interpolate with integer weight");
    let floating = start
        .lerp_scalar(&end, false, 0, 0.25)
        .expect("interpolate with floating weight");
    let negative = start
        .lerp_scalar(&end, true, -1, 0.0)
        .expect("extrapolate with negative weight");

    assert_eq!(item(&integer), 18.0);
    assert_eq!(item(&floating), 4.0);
    assert_eq!(item(&negative), -6.0);
    assert_eq!(item(&start), 2.0);
    assert_eq!(item(&end), 10.0);
}

#[test]
fn output_and_tensor_weight_forms_retain_all_sources() {
    let start = Tensor::from_float(2.0).expect("create start");
    let end = Tensor::from_float(10.0).expect("create end");
    let weight = Tensor::from_float(0.25).expect("create weight");
    let tensor = start
        .lerp_tensor(&end, &weight)
        .expect("interpolate with tensor weight");
    let mut scalar_output = Tensor::from_float(0.0).expect("create scalar output");
    scalar_output
        .lerp_scalar_out(&start, &end, false, 0, 0.5)
        .expect("write scalar interpolation");
    let mut tensor_output = Tensor::from_float(0.0).expect("create tensor output");
    tensor_output
        .lerp_tensor_out(&start, &end, &weight)
        .expect("write tensor interpolation");

    assert_eq!(item(&tensor), 4.0);
    assert_eq!(item(&scalar_output), 6.0);
    assert_eq!(item(&tensor_output), 4.0);
    assert_eq!(item(&start), 2.0);
    assert_eq!(item(&end), 10.0);
    assert_eq!(item(&weight), 0.25);
}

#[test]
fn incompatible_shapes_fail_without_mutating_output_or_sources() {
    let start = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create start");
    let end = Tensor::from_floats(&[3.0, 4.0, 5.0], &[3]).expect("create end");
    let weight = Tensor::from_float(0.5).expect("create weight");
    let error = required_tensor_error(
        start.lerp_tensor(&end, &weight),
        "incompatible functional interpolation unexpectedly succeeded",
    );
    let mut output = Tensor::from_float(7.0).expect("create output");
    let output_error = output
        .lerp_scalar_out(&start, &end, false, 0, 0.5)
        .expect_err("incompatible output interpolation unexpectedly succeeded");

    assert_eq!(error.status, 7161);
    assert_eq!(output_error.status, 7161);
    assert_eq!(item(&output), 7.0);
    assert_eq!(start.shape().expect("start shape"), vec![2]);
    assert_eq!(end.shape().expect("end shape"), vec![3]);
    assert_eq!(item(&weight), 0.5);
}

#[test]
fn cuda_executes_scalar_and_tensor_lerp_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let start = Tensor::from_float(2.0)
        .expect("create start")
        .to_device(1, 0)
        .expect("place start on CUDA");
    let end = Tensor::from_float(10.0)
        .expect("create end")
        .to_device(1, 0)
        .expect("place end on CUDA");
    let weight = Tensor::from_float(0.25)
        .expect("create weight")
        .to_device(1, 0)
        .expect("place weight on CUDA");
    let scalar = start
        .lerp_scalar(&end, false, 0, 0.5)
        .expect("execute scalar lerp on CUDA");
    let tensor = start
        .lerp_tensor(&end, &weight)
        .expect("execute tensor lerp on CUDA");
    let scalar_host = scalar.to_device(0, -1).expect("copy scalar result");
    let tensor_host = tensor.to_device(0, -1).expect("copy tensor result");

    assert_eq!(item(&scalar_host), 6.0);
    assert_eq!(item(&tensor_host), 4.0);
}
