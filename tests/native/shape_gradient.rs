use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
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

macro_rules! check_backward {
    ($gradient:ident, $operation:ident, $out:ident, [$($arg:expr),*], $values:expr, $shape:expr, $device:ident) => {{
        let result = $gradient.$operation($($arg),*).unwrap();
        check(&result, $values, $shape, $device);
        let mut output = tensor(&[], &[0], $device);
        output.$out(&$gradient, $($arg),*).unwrap();
        check(&output, $values, $shape, $device);
        output.zero_in_place().unwrap();
        check(&result, $values, $shape, $device);
    }};
}

fn verify(device: i64) {
    let matrix = tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], device);
    let mut view = matrix.diagonal(1, 0, 1).unwrap();
    check(&view, &[1., 5.], &[2], device);
    assert_eq!(view.strides().unwrap(), [4]);
    assert_eq!(view.storage_offset().unwrap(), 1);
    let reversed = matrix.diagonal(-1, 1, 0).unwrap();
    check(&reversed, &[1., 5.], &[2], device);
    let empty = matrix.diagonal(4, 0, 1).unwrap();
    check(&empty, &[], &[0], device);
    view.zero_in_place().unwrap();
    check(&matrix, &[0., 0., 2., 3., 4., 0.], &[2, 3], device);
    drop(matrix);
    check(&view, &[0., 0.], &[2], device);

    let gradient = tensor(&[10., 20.], &[2], device);
    check_backward!(
        gradient,
        diagonal_backward,
        diagonal_backward_out,
        [&[2, 3], 1, 0, 1],
        &[0., 10., 0., 0., 0., 20.],
        &[2, 3],
        device
    );
    check_backward!(
        gradient,
        select_backward,
        select_backward_out,
        [&[2, 3], -1, 1],
        &[0., 10., 0., 0., 20., 0.],
        &[2, 3],
        device
    );
    check_backward!(
        gradient,
        slice_backward,
        slice_backward_out,
        [&[5], 0, 1, 5, 2],
        &[0., 10., 0., 20., 0.],
        &[5],
        device
    );
    let windows = tensor(&[1., 2., 3., 4., 5., 6.], &[3, 2], device);
    check_backward!(
        windows,
        unfold_backward,
        unfold_backward_out,
        [&[4], 0, 2, 1],
        &[1., 5., 9., 6.],
        &[4],
        device
    );
    check(&gradient, &[10., 20.], &[2], device);
    check(&windows, &[1., 2., 3., 4., 5., 6.], &[3, 2], device);

    let matrix = tensor(&[0., 1., 2., 3., 4., 5.], &[2, 3], device);
    let mut output = tensor(&[], &[0], device);
    output
        .diagonal_scatter_out(&matrix, &gradient, 1, 0, 1)
        .unwrap();
    check(&output, &[0., 10., 2., 3., 4., 20.], &[2, 3], device);
    output.zero_in_place().unwrap();
    check(&matrix, &[0., 1., 2., 3., 4., 5.], &[2, 3], device);
    assert!(matrix.diagonal(0, 0, 0).is_err());
    assert!(gradient.diagonal_backward(&[2, 3], 0, 0, 0).is_err());
    assert!(gradient.select_backward(&[2, 3], 2, 0).is_err());
    assert!(gradient.slice_backward(&[5], 0, 0, 5, 0).is_err());
    assert!(windows.unfold_backward(&[4], 2, 2, 1).is_err());

    let leaf = matrix.require_grad().unwrap();
    leaf.diagonal(1, 0, 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(
        &leaf.gradient().unwrap(),
        &[0., 1., 0., 0., 0., 1.],
        &[2, 3],
        device,
    );
}

#[test]
fn cpu_shape_gradients_and_diagonal_aliases() {
    verify(0);
}

#[test]
fn cuda_shape_gradients_and_diagonal_aliases_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
