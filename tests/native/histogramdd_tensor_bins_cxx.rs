use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn tensor(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let tensor = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(
        !tensor.is_null(),
        "LibTorch tensor construction returned null"
    );
    tensor
}

fn shape(tensor: &ffi::Tensor) -> Vec<i64> {
    let shape = ffi::terlan_collection_pytorch_tensor_shape(tensor);
    shape
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

fn verify_histogramdd_tensor_bins() {
    let input = tensor(&[0.1, 0.2, 0.8, 1.2], &[4, 1]);
    let bins = tensor(&[0.0, 0.5, 1.0, 1.5], &[4]);
    let expected_hist = tensor(&[2.0, 1.0, 1.0], &[3]);

    let mut bin_list = ffi::terlan_resource_list_pytorch_tensor_tensor_new();
    assert!(
        !bin_list.is_null(),
        "Tensor resource-list allocation failed"
    );
    assert!(ffi::terlan_resource_list_pytorch_tensor_tensor_push(
        bin_list.pin_mut(),
        bins.as_ref().expect("bin edges tensor"),
    ));

    let result = ffi::terlan_owned_value_pytorch_histogramddnative_histogramdd_tensor_bins_result(
        input.as_ref().expect("input tensor"),
        bin_list.as_ref().expect("Tensor resource list"),
        false,
        &[],
        false,
        input.as_ref().expect("weight placeholder tensor"),
        false,
    );
    let result = result.as_ref().expect("histogramdd result");

    let hist = ffi::terlan_owned_value_pytorch_histogramddnative_histogramdd_result_hist(result);
    let hist = hist.as_ref().expect("histogram projection");
    assert_eq!(shape(hist), [3]);
    assert_tensor_equal(hist, expected_hist.as_ref().expect("expected histogram"));

    let edges =
        ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources(
            result,
        );
    let edges = edges.as_ref().expect("bin edge resource-list result");
    assert!(ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_is_ok(edges));
    assert_eq!(
        ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_len(
            edges
        ),
        1
    );
    let edge = ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_element(edges, 0);
    let edge = edge.as_ref().expect("projected bin-edge tensor");
    assert_eq!(shape(edge), [4]);
    assert_tensor_equal(edge, bins.as_ref().expect("bin edges tensor"));
}

#[test]
fn generated_cxx_histogramdd_tensor_bins_matches_values_and_edges() {
    verify_histogramdd_tensor_bins();
}

#[test]
fn generated_cxx_histogramdd_tensor_bin_list_preserves_owned_results() {
    let input = tensor(&[0.1, 0.1, 0.2, 0.8, 0.8, 0.2, 1.2, 1.2], &[4, 2]);
    let first_bins = tensor(&[0.0, 0.5, 1.5], &[3]);
    let second_bins = tensor(&[0.0, 0.5, 1.5], &[3]);
    let expected_hist = tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let expected_first_edges = tensor(&[0.0, 0.5, 1.5], &[3]);
    let expected_second_edges = tensor(&[0.0, 0.5, 1.5], &[3]);

    let mut bin_list = ffi::terlan_resource_list_pytorch_tensor_tensor_new();
    assert!(
        !bin_list.is_null(),
        "Tensor resource-list allocation failed"
    );
    assert!(ffi::terlan_resource_list_pytorch_tensor_tensor_push(
        bin_list.pin_mut(),
        first_bins.as_ref().expect("first bin edges tensor"),
    ));
    assert!(ffi::terlan_resource_list_pytorch_tensor_tensor_push(
        bin_list.pin_mut(),
        second_bins.as_ref().expect("second bin edges tensor"),
    ));

    let result = ffi::terlan_owned_value_pytorch_histogramddnative_histogramdd_tensor_bins_result(
        input.as_ref().expect("input tensor"),
        bin_list.as_ref().expect("Tensor resource list"),
        false,
        &[],
        false,
        input.as_ref().expect("weight placeholder tensor"),
        false,
    );
    let result = result.as_ref().expect("histogramdd result");

    drop(bin_list);
    drop(second_bins);
    drop(first_bins);
    drop(input);

    let hist = ffi::terlan_owned_value_pytorch_histogramddnative_histogramdd_result_hist(result);
    let hist = hist.as_ref().expect("histogram projection");
    assert_eq!(shape(hist), [2, 2]);
    assert_tensor_equal(hist, expected_hist.as_ref().expect("expected histogram"));

    let edges =
        ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources(
            result,
        );
    let edges = edges.as_ref().expect("bin edge resource-list result");
    assert!(ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_is_ok(edges));
    assert_eq!(
        ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_len(
            edges
        ),
        2
    );
    for (index, expected) in [
        expected_first_edges.as_ref().expect("expected first edges"),
        expected_second_edges
            .as_ref()
            .expect("expected second edges"),
    ]
    .into_iter()
    .enumerate()
    {
        let edge = ffi::terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_element(edges, index);
        let edge = edge.as_ref().expect("projected bin-edge tensor");
        assert_eq!(shape(edge), [3]);
        assert_tensor_equal(edge, expected);
    }
}
