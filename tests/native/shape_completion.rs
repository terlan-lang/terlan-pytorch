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

fn verify(device: i64) {
    let values = [1., 2., 3., 4., 5., 6.];
    let matrix = tensor(&values, &[2, 3], device);
    macro_rules! output_case {
        ($method:ident, ($($arg:expr),*), $values:expr, $shape:expr) => {{
            let mut output = tensor(&[], &[0], device);
            output.$method($($arg),*).unwrap();
            check(&output, $values, $shape, device);
            output.zero_in_place().unwrap();
            check(&matrix, &values, &[2, 3], device);
        }};
    }
    output_case!(
        transpose_copy_out,
        (&matrix, 0, 1),
        &[1., 4., 2., 5., 3., 6.],
        &[3, 2]
    );
    output_case!(select_copy_out, (&matrix, -1, 1), &[2., 5.], &[2]);
    // The pinned native_functions.yaml registers narrow_copy.out only for CPU.
    if device == 0 {
        output_case!(
            narrow_copy_out,
            (&matrix, -1, 1, 2),
            &[2., 3., 5., 6.],
            &[2, 2]
        );
    } else {
        let mut unsupported = tensor(&[42.], &[1], device);
        assert!(unsupported.narrow_copy_out(&matrix, -1, 1, 2).is_err());
        check(&unsupported, &[42.], &[1], device);
    }
    let patch = tensor(&[10., 20.], &[2], device);
    output_case!(
        select_scatter_out,
        (&matrix, &patch, 1, -1),
        &[1., 2., 10., 4., 5., 20.],
        &[2, 3]
    );
    check(&patch, &[10., 20.], &[2], device);
    output_case!(zero_out, (&matrix), &[0.; 6], &[2, 3]);
    let mut zeroed = matrix.zero().unwrap();
    check(&zeroed, &[0.; 6], &[2, 3], device);
    zeroed.select_scatter_out(&matrix, &patch, 1, 0).unwrap();
    check(&matrix, &values, &[2, 3], device);

    let channels = tensor(&[0., 1., 2., 3., 4., 5.], &[1, 6, 1, 1], device);
    output_case!(
        channel_shuffle_out,
        (&channels, 2),
        &[0., 3., 1., 4., 2., 5.],
        &[1, 6, 1, 1]
    );
    check(&channels, &[0., 1., 2., 3., 4., 5.], &[1, 6, 1, 1], device);

    let narrowed = matrix.narrow_view(-1, -2, 2).unwrap();
    check(&narrowed, &[2., 3., 5., 6.], &[2, 2], device);
    assert_eq!(narrowed.strides().unwrap(), [3, 1]);
    assert_eq!(narrowed.storage_offset().unwrap(), 1);
    let index = tensor(&[-2.], &[], device).to_dtype(4).unwrap();
    let mut indexed = matrix.narrow_tensor(-1, &index, 2).unwrap();
    check(&indexed, &[2., 3., 5., 6.], &[2, 2], device);
    indexed.zero_in_place().unwrap();
    check(&matrix, &[1., 0., 0., 4., 0., 0.], &[2, 3], device);
    check(&narrowed, &[0.; 4], &[2, 2], device);
    drop(matrix);
    check(&indexed, &[0.; 4], &[2, 2], device);

    let mut base = tensor(&[1., 2., 3., 4.], &[4], device);
    let windows = base.unfold(0, 2, 1).unwrap();
    check(&windows, &[1., 2., 2., 3., 3., 4.], &[3, 2], device);
    assert_eq!(windows.strides().unwrap(), [1, 1]);
    let mut copied = tensor(&[], &[0], device);
    copied.unfold_copy_out(&base, 0, 2, 1).unwrap();
    base.zero_in_place().unwrap();
    check(&windows, &[0.; 6], &[3, 2], device);
    check(&copied, &[1., 2., 2., 3., 3., 4.], &[3, 2], device);
    drop(base);
    check(&windows, &[0.; 6], &[3, 2], device);
    let scalar = tensor(&[3.], &[], device);
    check(
        &scalar.trace_backward(&[2, 3]).unwrap(),
        &[3., 0., 0., 0., 3., 0.],
        &[2, 3],
        device,
    );

    let image = tensor(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], &[1, 1, 3, 3], device);
    let column_values = [
        1., 2., 4., 5., 2., 3., 5., 6., 4., 5., 7., 8., 5., 6., 8., 9.,
    ];
    let columns = image.im2col(&[2, 2], &[1, 1], &[0, 0], &[1, 1]).unwrap();
    check(&columns, &column_values, &[1, 4, 4], device);
    let mut column_output = tensor(&[], &[0], device);
    column_output
        .im2col_out(&image, &[2, 2], &[1, 1], &[0, 0], &[1, 1])
        .unwrap();
    check(&column_output, &column_values, &[1, 4, 4], device);
    let folded_values = [1., 4., 3., 8., 20., 12., 7., 16., 9.];
    let folded = columns
        .col2im(&[3, 3], &[2, 2], &[1, 1], &[0, 0], &[1, 1])
        .unwrap();
    check(&folded, &folded_values, &[1, 1, 3, 3], device);
    let mut folded_output = tensor(&[], &[0], device);
    folded_output
        .col2im_out(&columns, &[3, 3], &[2, 2], &[1, 1], &[0, 0], &[1, 1])
        .unwrap();
    check(&folded_output, &folded_values, &[1, 1, 3, 3], device);
    column_output.zero_in_place().unwrap();
    folded_output.zero_in_place().unwrap();
    check(&columns, &column_values, &[1, 4, 4], device);
    check(&folded, &folded_values, &[1, 1, 3, 3], device);
    check(
        &image,
        &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        &[1, 1, 3, 3],
        device,
    );
    check(
        &image.im2col(&[2, 2], &[2, 2], &[0, 0], &[1, 1]).unwrap(),
        &[1., 3., 7., 9.],
        &[1, 4, 1],
        device,
    );
    check(
        &image.im2col(&[1, 1], &[1, 1], &[1, 1], &[2, 2]).unwrap(),
        &[0., 0., 0., 0., 5., 0., 0., 0., 0.],
        &[1, 1, 9],
        device,
    );

    let left = tensor(&values, &[2, 3], device);
    let right = tensor(&[1., 2., 3., 4., 5., 6.], &[3, 2], device);
    let contracted = left.tensordot(&right, &[-1], &[0]).unwrap();
    check(&contracted, &[22., 28., 49., 64.], &[2, 2], device);
    let mut contracted_output = tensor(&[], &[0], device);
    contracted_output
        .tensordot_out(&left, &right, &[1], &[0])
        .unwrap();
    check(&contracted_output, &[22., 28., 49., 64.], &[2, 2], device);
    contracted_output.zero_in_place().unwrap();
    check(&contracted, &[22., 28., 49., 64.], &[2, 2], device);
    check(
        &left.tensordot(&left, &[0, 1], &[0, 1]).unwrap(),
        &[91.],
        &[],
        device,
    );
    check(
        &patch.tensordot(&scalar, &[], &[]).unwrap(),
        &[30., 60.],
        &[2],
        device,
    );
    check(&left, &values, &[2, 3], device);

    assert!(left.narrow_view(0, 2, 1).is_err());
    assert!(left.narrow_tensor(0, &patch, 1).is_err());
    assert!(left.unfold(0, 3, 1).is_err());
    assert!(left.unfold(0, 1, 0).is_err());
    assert!(image.im2col(&[0, 2], &[1, 1], &[0, 0], &[1, 1]).is_err());
    assert!(columns
        .col2im(&[4, 4], &[2, 2], &[1, 1], &[0, 0], &[1, 1])
        .is_err());
    assert!(left.tensordot(&right, &[0], &[0]).is_err());
    assert!(left.tensordot(&right, &[1], &[0, 1]).is_err());
    assert!(contracted_output.channel_shuffle_out(&channels, 4).is_err());
    check(&left, &values, &[2, 3], device);

    let leaf = tensor(&[1., 2., 3., 4.], &[4], device)
        .require_grad()
        .unwrap();
    leaf.unfold(0, 2, 1)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(&leaf.gradient().unwrap(), &[1., 2., 2., 1.], &[4], device);
    let image_leaf = image.require_grad().unwrap();
    image_leaf
        .im2col(&[2, 2], &[1, 1], &[0, 0], &[1, 1])
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(
        &image_leaf.gradient().unwrap(),
        &[1., 2., 1., 2., 4., 2., 1., 2., 1.],
        &[1, 1, 3, 3],
        device,
    );
}

#[test]
fn cpu_shape_completion_values_ownership_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_shape_completion_values_ownership_errors_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
