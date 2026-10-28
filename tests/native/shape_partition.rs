use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn integer_tensor(values: &[i64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_ints(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn check(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.dtype().unwrap(), 7);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(&tensor(values, shape, device)).unwrap());
}

fn check_list(
    actual: &[Tensor],
    expected_values: &[&[f64]],
    expected_shapes: &[&[i64]],
    device: i64,
) {
    assert_eq!(actual.len(), expected_values.len());
    assert_eq!(actual.len(), expected_shapes.len());
    for ((value, expected), shape) in actual
        .iter()
        .zip(expected_values.iter())
        .zip(expected_shapes.iter())
    {
        check(value, expected, shape, device);
    }
}

fn verify(device: i64) {
    let values = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let matrix = tensor(&values, &[2, 6], device);

    let chunks = matrix.chunk(3, -1).unwrap();
    check_list(
        &chunks,
        &[&[1., 2., 7., 8.], &[3., 4., 9., 10.], &[5., 6., 11., 12.]],
        &[&[2, 2], &[2, 2], &[2, 2]],
        device,
    );

    let scalar_splits = matrix.split_size(4, 1).unwrap();
    check_list(
        &scalar_splits,
        &[&[1., 2., 3., 4., 7., 8., 9., 10.], &[5., 6., 11., 12.]],
        &[&[2, 4], &[2, 2]],
        device,
    );
    let sized_splits = matrix.split_sizes(&[1, 2, 3], 1).unwrap();
    check_list(
        &sized_splits,
        &[&[1., 7.], &[2., 3., 8., 9.], &[4., 5., 6., 10., 11., 12.]],
        &[&[2, 1], &[2, 2], &[2, 3]],
        device,
    );

    let section_splits = matrix.tensor_split_sections(4, 1).unwrap();
    check_list(
        &section_splits,
        &[
            &[1., 2., 7., 8.],
            &[3., 4., 9., 10.],
            &[5., 11.],
            &[6., 12.],
        ],
        &[&[2, 2], &[2, 2], &[2, 1], &[2, 1]],
        device,
    );
    let indexed_splits = matrix.tensor_split_indices(&[1, 5], 1).unwrap();
    check_list(
        &indexed_splits,
        &[&[1., 7.], &[2., 3., 4., 5., 8., 9., 10., 11.], &[6., 12.]],
        &[&[2, 1], &[2, 4], &[2, 1]],
        device,
    );
    // Pinned ATen requires tensor-valued split controls to remain on CPU even
    // when the partitioned data tensor is CUDA-resident.
    let scalar_sections = integer_tensor(&[3], &[], 0);
    let tensor_section_splits = matrix.tensor_split_tensor(&scalar_sections, 1).unwrap();
    check_list(
        &tensor_section_splits,
        &[&[1., 2., 7., 8.], &[3., 4., 9., 10.], &[5., 6., 11., 12.]],
        &[&[2, 2], &[2, 2], &[2, 2]],
        device,
    );
    let tensor_indices = integer_tensor(&[1, 5], &[2], 0);
    let tensor_index_splits = matrix.tensor_split_tensor(&tensor_indices, 1).unwrap();
    check_list(
        &tensor_index_splits,
        &[&[1., 7.], &[2., 3., 4., 5., 8., 9., 10., 11.], &[6., 12.]],
        &[&[2, 1], &[2, 4], &[2, 1]],
        device,
    );

    let horizontal_sections = matrix.hsplit_sections(3).unwrap();
    check_list(
        &horizontal_sections,
        &[&[1., 2., 7., 8.], &[3., 4., 9., 10.], &[5., 6., 11., 12.]],
        &[&[2, 2], &[2, 2], &[2, 2]],
        device,
    );
    let horizontal_indices = matrix.hsplit_indices(&[1, 5]).unwrap();
    check_list(
        &horizontal_indices,
        &[&[1., 7.], &[2., 3., 4., 5., 8., 9., 10., 11.], &[6., 12.]],
        &[&[2, 1], &[2, 4], &[2, 1]],
        device,
    );
    let vertical_sections = matrix.vsplit_sections(2).unwrap();
    check_list(
        &vertical_sections,
        &[&[1., 2., 3., 4., 5., 6.], &[7., 8., 9., 10., 11., 12.]],
        &[&[1, 6], &[1, 6]],
        device,
    );
    let vertical_indices = matrix.vsplit_indices(&[1]).unwrap();
    check_list(
        &vertical_indices,
        &[&[1., 2., 3., 4., 5., 6.], &[7., 8., 9., 10., 11., 12.]],
        &[&[1, 6], &[1, 6]],
        device,
    );

    let volume = tensor(&values, &[1, 2, 6], device);
    let depth_sections = volume.dsplit_sections(3).unwrap();
    check_list(
        &depth_sections,
        &[&[1., 2., 7., 8.], &[3., 4., 9., 10.], &[5., 6., 11., 12.]],
        &[&[1, 2, 2], &[1, 2, 2], &[1, 2, 2]],
        device,
    );
    let depth_indices = volume.dsplit_indices(&[1, 5]).unwrap();
    check_list(
        &depth_indices,
        &[&[1., 7.], &[2., 3., 4., 5., 8., 9., 10., 11.], &[6., 12.]],
        &[&[1, 2, 1], &[1, 2, 4], &[1, 2, 1]],
        device,
    );

    let rows = matrix.unbind(0).unwrap();
    check_list(
        &rows,
        &[&[1., 2., 3., 4., 5., 6.], &[7., 8., 9., 10., 11., 12.]],
        &[&[6], &[6]],
        device,
    );

    let transposed = matrix.transpose(0, 1).unwrap();
    let non_contiguous = transposed.split_size(1, 1).unwrap();
    check_list(
        &non_contiguous,
        &[&[1., 2., 3., 4., 5., 6.], &[7., 8., 9., 10., 11., 12.]],
        &[&[6, 1], &[6, 1]],
        device,
    );

    let alias_source = tensor(&values, &[2, 6], device);
    let mut alias_views = alias_source.split_sizes(&[2, 4], 1).unwrap();
    alias_views[0].zero_in_place().unwrap();
    check(
        &alias_source,
        &[0., 0., 3., 4., 5., 6., 0., 0., 9., 10., 11., 12.],
        &[2, 6],
        device,
    );
    drop(alias_source);
    check(
        &alias_views[1],
        &[3., 4., 5., 6., 9., 10., 11., 12.],
        &[2, 4],
        device,
    );

    let leaf = tensor(&[1., 2., 3., 4.], &[4], device)
        .require_grad()
        .unwrap();
    let gradient_views = leaf.chunk(2, 0).unwrap();
    gradient_views[1].sum().unwrap().backward().unwrap();
    check(&leaf.gradient().unwrap(), &[0., 0., 1., 1.], &[4], device);

    assert!(matrix.chunk(0, 0).is_err());
    assert!(matrix.split_size(0, 1).is_err());
    assert!(matrix.tensor_split_sections(0, 1).is_err());
    assert!(matrix.tensor_split_indices(&[1], 3).is_err());
    if device == 1 {
        let cuda_sections = integer_tensor(&[3], &[], device);
        let cuda_indices = integer_tensor(&[1, 5], &[2], device);
        assert!(matrix.tensor_split_tensor(&cuda_sections, 1).is_err());
        assert!(matrix.tensor_split_tensor(&cuda_indices, 1).is_err());
    }
    assert!(matrix.vsplit_sections(3).is_err());
    assert!(volume.dsplit_sections(4).is_err());
    check(&matrix, &values, &[2, 6], device);
}

#[test]
fn cpu_partition_values_views_ownership_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_partition_values_views_ownership_errors_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
