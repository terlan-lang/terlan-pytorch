use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn tensor(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let cpu_tensor = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(
        !cpu_tensor.is_null(),
        "LibTorch tensor construction returned null"
    );
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        ffi::terlan_owned_value_pytorch_tensor_to_device(
            cpu_tensor.as_ref().expect("constructed CPU tensor"),
            1,
            0,
        )
    } else {
        cpu_tensor
    }
}

fn shape(tensor: &ffi::Tensor) -> Vec<i64> {
    ffi::terlan_collection_pytorch_tensor_shape(tensor)
        .as_ref()
        .expect("tensor shape projection")
        .iter()
        .copied()
        .collect()
}

fn assert_allclose(label: &str, actual: &ffi::Tensor, values: &[f64], dimensions: &[i64]) {
    let expected = tensor(values, dimensions);
    let result = ffi::terlan_exception_pytorch_typerelationsnative_allclose_native(
        actual,
        expected.as_ref().expect("expected tensor"),
        1e-6,
        1e-6,
        false,
    );
    let result = result.as_ref().expect("contained allclose result");
    assert!(result.is_ok(), "allclose failed: {}", result.message());
    assert!(result.bool_value(), "{label}: Tensor values differ");
}

fn loss(
    input: &ffi::Tensor,
    target: &ffi::Tensor,
    weight: Option<&ffi::Tensor>,
    has_size_average: bool,
    size_average: bool,
    has_reduce: bool,
    reduce: bool,
    reduction: &str,
) -> UniquePtr<ffi::Tensor> {
    let fallback_weight = input;
    ffi::terlan_owned_value_pytorch_functionalcompletion_multilabel_soft_margin_loss_configured(
        input,
        target,
        weight.is_some(),
        weight.unwrap_or(fallback_weight),
        has_size_average,
        size_average,
        has_reduce,
        reduce,
        reduction,
    )
}

#[test]
fn multilabel_soft_margin_loss_matches_weight_and_legacy_reductions() {
    let input = tensor(&[0.0, 1.0, -1.0, 2.0, -2.0, 0.5], &[2, 3]);
    let target = tensor(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]);
    let weight = tensor(&[1.0, 2.0, 0.5], &[3]);
    let input_ref = input.as_ref().expect("input tensor");
    let target_ref = target.as_ref().expect("target tensor");
    let weight_ref = weight.as_ref().expect("weight tensor");

    let none = loss(
        input_ref,
        target_ref,
        Some(weight_ref),
        false,
        true,
        false,
        true,
        "none",
    );
    let none = none.as_ref().expect("none reduction");
    assert_eq!(shape(none), [2]);
    assert_allclose(
        "weighted none",
        none,
        &[1.3254337997851675, 2.2059408417396571],
        &[2],
    );

    let mean = loss(
        input_ref,
        target_ref,
        Some(weight_ref),
        false,
        true,
        false,
        true,
        "mean",
    );
    assert_allclose(
        "weighted mean",
        mean.as_ref().expect("mean reduction"),
        &[1.7656873207624124],
        &[],
    );

    let sum = loss(
        input_ref,
        target_ref,
        Some(weight_ref),
        false,
        true,
        false,
        true,
        "sum",
    );
    assert_allclose(
        "weighted sum",
        sum.as_ref().expect("sum reduction"),
        &[3.5313746415248248],
        &[],
    );

    let legacy_sum = loss(
        input_ref, target_ref, None, true, false, false, true, "mean",
    );
    assert_allclose(
        "legacy size_average=false",
        legacy_sum.as_ref().expect("legacy sum reduction"),
        &[2.6825345206208144],
        &[],
    );

    let legacy_none = loss(
        input_ref, target_ref, None, false, true, true, false, "mean",
    );
    let legacy_none = legacy_none.as_ref().expect("legacy none reduction");
    assert_eq!(shape(legacy_none), [2]);
    assert_allclose(
        "legacy reduce=false",
        legacy_none,
        &[1.1065568518654636, 1.5759776687553506],
        &[2],
    );
}
