use terlan_pytorch_cxx::{CAbiError, Tensor};

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

fn flat_item(tensor: &Tensor, index: i64) -> f64 {
    tensor
        .flatten(0, -1)
        .expect("flatten tensor")
        .narrow(0, index, 1)
        .expect("select flattened item")
        .item_float()
        .expect("read flattened item")
}

fn sum_value(tensor: &Tensor) -> f64 {
    tensor
        .sum()
        .expect("sum tensor")
        .item_float()
        .expect("read sum")
}

#[test]
fn embedding_and_gather_outputs_retain_storage_and_sources() {
    let weight = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .expect("create embedding weight");
    let embedding_indices = Tensor::from_ints(&[2, 0], &[2]).expect("create embedding indices");
    let mut embedded = Tensor::from_floats(&[], &[0]).expect("create embedding output");
    embedded
        .embedding_out(&weight, &embedding_indices, -1, false, false)
        .expect("write embedding output");
    let gathered_source =
        Tensor::from_floats(&[10.0, 20.0, 30.0], &[3]).expect("create gather source");
    let gather_indices = Tensor::from_ints(&[2, 0], &[2]).expect("create gather indices");
    let mut gathered = Tensor::from_floats(&[], &[0]).expect("create gather output");
    gathered
        .gather_out(&gathered_source, 0, &gather_indices, false)
        .expect("write gather output");

    assert_eq!(embedded.shape().expect("embedding shape"), vec![2, 2]);
    assert_eq!(flat_item(&embedded, 0), 5.0);
    assert_eq!(flat_item(&embedded, 3), 2.0);
    assert_eq!(gathered.shape().expect("gather shape"), vec![2]);
    assert_eq!(flat_item(&gathered, 0), 30.0);
    assert_eq!(flat_item(&gathered, 1), 10.0);
    assert_eq!(sum_value(&weight), 21.0);
    assert_eq!(sum_value(&gathered_source), 60.0);
}

#[test]
fn embedding_bag_optionals_return_four_independent_handles() {
    let weight = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .expect("create embedding weight");
    let indices = Tensor::from_ints(&[0, 1, 2, 3], &[4]).expect("create embedding indices");
    let offsets = Tensor::from_ints(&[0, 2], &[2]).expect("create embedding offsets");
    let sample_weights =
        Tensor::from_floats(&[1.0, 0.5, 2.0, 0.25], &[4]).expect("create per-sample weights");
    let plain = weight
        .embedding_bag(&indices, &offsets, false, 0, false, false)
        .expect("plain embedding bag");
    let weighted = weight
        .embedding_bag_weighted(&indices, &offsets, false, 0, false, &sample_weights, false)
        .expect("weighted embedding bag");
    let padded = weight
        .embedding_bag_padding(&indices, &offsets, false, 0, false, false, 0)
        .expect("padded embedding bag");
    let weighted_padded = weight
        .embedding_bag_padding_weighted(
            &indices,
            &offsets,
            false,
            0,
            false,
            &sample_weights,
            false,
            0,
        )
        .expect("weighted padded embedding bag");

    assert_eq!(plain.0.shape().expect("plain output shape"), vec![2, 2]);
    assert_eq!(sum_value(&plain.0), 36.0);
    assert_eq!(sum_value(&weighted.0), 32.25);
    assert_eq!(sum_value(&padded.0), 33.0);
    assert_eq!(sum_value(&weighted_padded.0), 29.25);
    for auxiliary in [
        &plain.1,
        &plain.2,
        &plain.3,
        &weighted.1,
        &weighted.2,
        &weighted.3,
        &padded.1,
        &padded.2,
        &padded.3,
        &weighted_padded.1,
        &weighted_padded.2,
        &weighted_padded.3,
    ] {
        auxiliary.shape().expect("read auxiliary output shape");
    }
    assert_eq!(sum_value(&weight), 36.0);
    assert_eq!(sum_value(&sample_weights), 3.75);
}

#[test]
fn embedding_renorm_distinguishes_functional_output_and_in_place_mutation() {
    let weight = Tensor::from_floats(&[3.0, 4.0, 0.0, 2.0], &[2, 2]).expect("create source weight");
    let output_weight =
        Tensor::from_floats(&[3.0, 4.0, 0.0, 2.0], &[2, 2]).expect("create output source");
    let mut mutable_weight =
        Tensor::from_floats(&[3.0, 4.0, 0.0, 2.0], &[2, 2]).expect("create mutable weight");
    let indices = Tensor::from_ints(&[0], &[1]).expect("create selected row");
    let bounded = weight
        .embedding_renorm(&indices, 1.0, 2.0)
        .expect("functional embedding renorm");
    let mut output = Tensor::from_floats(&[], &[0]).expect("create renorm output");
    output
        .embedding_renorm_out(&output_weight, &indices, 1.0, 2.0)
        .expect("output embedding renorm");
    mutable_weight
        .embedding_renorm_in_place(&indices, 1.0, 2.0)
        .expect("in-place embedding renorm");

    for tensor in [&bounded, &output, &mutable_weight] {
        assert!((flat_item(tensor, 0) - 0.6).abs() < 1.0e-6);
        assert!((flat_item(tensor, 1) - 0.8).abs() < 1.0e-6);
        assert_eq!(flat_item(tensor, 3), 2.0);
    }
    assert_eq!(sum_value(&weight), 9.0);
    assert_eq!(sum_value(&output_weight), 9.0);
}

#[test]
fn embedding_gradient_outputs_cover_dense_sparse_and_failure_paths() {
    let gradient = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create gradient");
    let indices = Tensor::from_ints(&[0, 2], &[2]).expect("create gradient indices");
    let mut dense_output = Tensor::from_floats(&[], &[0]).expect("create dense gradient output");
    dense_output
        .embedding_dense_backward_out(&gradient, &indices, 3, -1, false)
        .expect("write dense embedding gradient");
    let sparse_gradient = gradient
        .embedding_sparse_backward(&indices, 3, -1, false)
        .expect("compute sparse embedding gradient");
    let sparse_dense = sparse_gradient
        .to_dense()
        .expect("materialize sparse gradient");
    let invalid_indices = Tensor::from_ints(&[3], &[1]).expect("create invalid index");
    let invalid = required_tensor_error(
        gradient.embedding_sparse_backward(&invalid_indices, 3, -1, false),
        "out-of-range sparse embedding index unexpectedly succeeded",
    );

    assert_eq!(
        dense_output.shape().expect("dense gradient shape"),
        vec![3, 2]
    );
    assert_eq!(
        sparse_gradient.shape().expect("sparse gradient shape"),
        vec![3, 2]
    );
    assert_eq!(sum_value(&dense_output), 10.0);
    assert_eq!(sum_value(&sparse_dense), 10.0);
    assert_eq!(flat_item(&dense_output, 0), 1.0);
    assert_eq!(flat_item(&dense_output, 4), 3.0);
    assert_ne!(invalid.status, 0);
    assert_eq!(sum_value(&gradient), 10.0);
}
