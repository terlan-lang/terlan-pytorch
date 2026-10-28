use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn tensor(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let tensor = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(
        !tensor.is_null(),
        "LibTorch tensor construction returned null"
    );
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        ffi::terlan_owned_value_pytorch_tensor_to_device(
            tensor.as_ref().expect("constructed tensor"),
            1,
            0,
        )
    } else {
        tensor
    }
}

fn shape(tensor: &ffi::Tensor) -> Vec<i64> {
    let values = ffi::terlan_collection_pytorch_tensor_shape(tensor);
    values
        .as_ref()
        .expect("tensor shape projection")
        .iter()
        .copied()
        .collect()
}

fn assert_tensor_equal(actual: &ffi::Tensor, expected: &ffi::Tensor) {
    let result = ffi::terlan_exception_pytorch_typerelationsnative_equal_native(actual, expected);
    let result = result.as_ref().expect("contained tensor equality result");
    assert!(
        result.is_ok(),
        "tensor equality failed: {}",
        result.message()
    );
    assert!(result.bool_value(), "Tensor values differ");
}

fn assert_ok(result: &UniquePtr<ffi::TerlanExceptionEnvelope>) {
    let result = result.as_ref().expect("contained LibTorch result");
    assert!(
        result.is_ok(),
        "LibTorch operation failed: {}",
        result.message()
    );
}

#[test]
fn histogram_count_respects_range_and_weights_for_values_and_out() {
    let input = tensor(&[0.1, 0.2, 0.8, 1.2], &[4]);
    let weight = tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let expected_hist = tensor(&[3.0, 3.0], &[2]);
    let expected_edges = tensor(&[0.0, 0.5, 1.0], &[3]);

    let result = ffi::terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct(
        input.as_ref().expect("input tensor"),
        2,
        true,
        &[0.0, 1.0],
        true,
        weight.as_ref().expect("weight tensor"),
        false,
    );
    let mut result = result;
    assert!(!result.is_null(), "histogram result returned null");
    let hist = ffi::terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct_take_0(
        result.pin_mut(),
    );
    let edges = ffi::terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct_take_1(
        result.pin_mut(),
    );
    let hist = hist.as_ref().expect("histogram tensor");
    let edges = edges.as_ref().expect("histogram edges");
    assert_eq!(shape(hist), [2]);
    assert_eq!(shape(edges), [3]);
    assert_tensor_equal(hist, expected_hist.as_ref().expect("expected histogram"));
    assert_tensor_equal(edges, expected_edges.as_ref().expect("expected edges"));

    let mut out_hist = tensor(&[0.0], &[1]);
    let mut out_edges = tensor(&[0.0], &[1]);
    let result = ffi::terlan_mutation_pytorch_histogramcompletion_histogram_bin_ct_out(
        out_hist.pin_mut(),
        out_edges.pin_mut(),
        input.as_ref().expect("input tensor"),
        2,
        true,
        &[0.0, 1.0],
        true,
        weight.as_ref().expect("weight tensor"),
        false,
    );
    assert_ok(&result);
    assert_tensor_equal(
        out_hist.as_ref().expect("retained histogram"),
        expected_hist.as_ref().expect("expected histogram"),
    );
    assert_tensor_equal(
        out_edges.as_ref().expect("retained edges"),
        expected_edges.as_ref().expect("expected edges"),
    );
}

#[test]
fn histogram_tensor_bins_respect_weights_for_values_and_out() {
    let input = tensor(&[0.1, 0.2, 0.8, 1.2], &[4]);
    let weight = tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let bins = tensor(&[0.0, 0.5, 1.5], &[3]);
    let expected_hist = tensor(&[3.0, 7.0], &[2]);

    let result = ffi::terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor(
        input.as_ref().expect("input tensor"),
        bins.as_ref().expect("bin edges"),
        true,
        weight.as_ref().expect("weight tensor"),
        false,
    );
    let mut result = result;
    assert!(!result.is_null(), "histogram result returned null");
    let hist = ffi::terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor_take_0(
        result.pin_mut(),
    );
    let edges = ffi::terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor_take_1(
        result.pin_mut(),
    );
    let hist = hist.as_ref().expect("histogram tensor");
    let edges = edges.as_ref().expect("histogram edges");
    assert_eq!(shape(hist), [2]);
    assert_tensor_equal(hist, expected_hist.as_ref().expect("expected histogram"));
    assert_tensor_equal(edges, bins.as_ref().expect("bin edges"));

    let mut out_hist = tensor(&[0.0], &[1]);
    let mut out_edges = tensor(&[0.0], &[1]);
    let result = ffi::terlan_mutation_pytorch_histogramcompletion_histogram_bins_tensor_out(
        out_hist.pin_mut(),
        out_edges.pin_mut(),
        input.as_ref().expect("input tensor"),
        bins.as_ref().expect("bin edges"),
        true,
        weight.as_ref().expect("weight tensor"),
        false,
    );
    assert_ok(&result);
    assert_tensor_equal(
        out_hist.as_ref().expect("retained histogram"),
        expected_hist.as_ref().expect("expected histogram"),
    );
    assert_tensor_equal(
        out_edges.as_ref().expect("retained edges"),
        bins.as_ref().expect("bin edges"),
    );
}
