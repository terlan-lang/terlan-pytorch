use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn check(actual: &Tensor, values: &[f64], shape: &[i64], device: i64) {
    assert_eq!(actual.shape().unwrap(), shape);
    assert_eq!(actual.device_type().unwrap(), device);
    assert!(actual.equal_native(&tensor(values, shape, device)).unwrap());
}

fn verify(device: i64) {
    let mut column = tensor(&[1., 2., 3.], &[3, 1], device);
    let row = tensor(&[10., 20.], &[1, 2], device);
    let broadcast = column.broadcast_tensors(&[&row]).unwrap();
    assert_eq!(broadcast.len(), 2);
    check(&broadcast[0], &[1., 1., 2., 2., 3., 3.], &[3, 2], device);
    check(
        &broadcast[1],
        &[10., 20., 10., 20., 10., 20.],
        &[3, 2],
        device,
    );
    column.zero_in_place().unwrap();
    check(&broadcast[0], &[0.; 6], &[3, 2], device);

    let x = tensor(&[1., 2.], &[2], device);
    let y = tensor(&[10., 20., 30.], &[3], device);
    let default_grid = x.meshgrid(&[&y]).unwrap();
    let ij_grid = x.meshgrid_indexing(&[&y], "ij").unwrap();
    let xy_grid = x.meshgrid_indexing(&[&y], "xy").unwrap();
    assert_eq!(default_grid.len(), 2);
    assert_eq!(ij_grid.len(), 2);
    assert_eq!(xy_grid.len(), 2);
    check(&default_grid[0], &[1., 1., 1., 2., 2., 2.], &[2, 3], device);
    check(
        &default_grid[1],
        &[10., 20., 30., 10., 20., 30.],
        &[2, 3],
        device,
    );
    check(&ij_grid[0], &[1., 1., 1., 2., 2., 2.], &[2, 3], device);
    check(&xy_grid[0], &[1., 2., 1., 2., 1., 2.], &[3, 2], device);
    check(
        &xy_grid[1],
        &[10., 10., 20., 20., 30., 30.],
        &[3, 2],
        device,
    );

    let matrix = tensor(&[1., 2., 3., 4.], &[2, 2], device);
    let vector = tensor(&[5., 6.], &[2], device);
    let mut flat = Tensor::flatten_dense_tensors(&[&matrix, &vector]).unwrap();
    check(&flat, &[1., 2., 3., 4., 5., 6.], &[6], device);
    let pieces = flat.unflatten_dense_tensors(&matrix, &[&vector]).unwrap();
    assert_eq!(pieces.len(), 2);
    check(&pieces[0], &[1., 2., 3., 4.], &[2, 2], device);
    check(&pieces[1], &[5., 6.], &[2], device);
    flat.zero_in_place().unwrap();
    check(&pieces[0], &[0.; 4], &[2, 2], device);
    check(&pieces[1], &[0.; 2], &[2], device);
    check(&matrix, &[1., 2., 3., 4.], &[2, 2], device);

    let base = tensor(&[1., 2., 3., 4.], &[2, 2], device);
    let transposed = base.transpose(0, 1).unwrap();
    let non_contiguous = Tensor::flatten_dense_tensors(&[&transposed]).unwrap();
    check(&non_contiguous, &[1., 3., 2., 4.], &[4], device);

    let leaf_matrix = tensor(&[1., 2., 3., 4.], &[2, 2], device)
        .require_grad()
        .unwrap();
    let leaf_vector = tensor(&[5., 6.], &[2], device).require_grad().unwrap();
    Tensor::flatten_dense_tensors(&[&leaf_matrix, &leaf_vector])
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(&leaf_matrix.gradient().unwrap(), &[1.; 4], &[2, 2], device);
    check(&leaf_vector.gradient().unwrap(), &[1.; 2], &[2], device);

    let incompatible = tensor(&[1., 2., 3.], &[3], device);
    assert!(x.broadcast_tensors(&[&incompatible]).is_err());
    assert!(x.meshgrid_indexing(&[&y], "invalid").is_err());
    assert!(Tensor::flatten_dense_tensors(&[]).is_err());
    let short = tensor(&[1.], &[1], device);
    assert!(short.unflatten_dense_tensors(&matrix, &[&vector]).is_err());
    if device == 1 {
        let host = tensor(&[1., 2.], &[2], 0);
        let mixed = x.broadcast_tensors(&[&host]).unwrap();
        assert_eq!(mixed.len(), 2);
        assert_eq!(mixed[0].device_type().unwrap(), 1);
        assert_eq!(mixed[1].device_type().unwrap(), 0);
        assert!(Tensor::flatten_dense_tensors(&[&x, &host]).is_err());
    }
}

#[test]
fn cpu_tensor_list_transforms_cover_values_views_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_tensor_list_transforms_execute_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
