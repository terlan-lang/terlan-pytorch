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
    let left = tensor(&[1., 2.], &[2], device);
    let right = tensor(&[3., 4.], &[2], device);
    let inputs = [&left, &right];
    let joined = [1., 2., 3., 4.];
    let columns = [1., 3., 2., 4.];
    let depth = [1., 3., 2., 4.];

    macro_rules! functional_out {
        ($function:ident, $out:ident, ($($arg:expr),*), $values:expr, $shape:expr) => {{
            let result = Tensor::$function(&inputs, $($arg),*).unwrap();
            let mut output = tensor(&[], &[0], device);
            output.$out(&inputs, $($arg),*).unwrap();
            check(&result, $values, $shape, device);
            check(&output, $values, $shape, device);
            output.zero_in_place().unwrap();
            check(&result, $values, $shape, device);
        }};
    }

    functional_out!(cat, cat_out, (0), &joined, &[4]);
    functional_out!(concat, concat_out, (0), &joined, &[4]);
    functional_out!(concatenate, concatenate_out, (0), &joined, &[4]);
    functional_out!(stack, stack_out, (0), &joined, &[2, 2]);
    functional_out!(stack, stack_out, (1), &columns, &[2, 2]);
    functional_out!(vstack, vstack_out, (), &joined, &[2, 2]);
    functional_out!(row_stack, row_stack_out, (), &joined, &[2, 2]);
    functional_out!(hstack, hstack_out, (), &joined, &[4]);
    functional_out!(column_stack, column_stack_out, (), &columns, &[2, 2]);
    functional_out!(dstack, dstack_out, (), &depth, &[1, 2, 2]);
    functional_out!(
        block_diag,
        block_diag_out,
        (),
        &[1., 2., 0., 0., 0., 0., 3., 4.],
        &[2, 4]
    );
    check(
        &Tensor::cartesian_prod(&inputs).unwrap(),
        &[1., 3., 1., 4., 2., 3., 2., 4.],
        &[4, 2],
        device,
    );
    check(&left, &[1., 2.], &[2], device);
    check(&right, &[3., 4.], &[2], device);

    let matrix = tensor(&[1., 2., 3., 4.], &[2, 2], device);
    let transposed = matrix.transpose(0, 1).unwrap();
    check(
        &Tensor::cat(&[&transposed, &transposed], 1).unwrap(),
        &[1., 3., 1., 3., 2., 4., 2., 4.],
        &[2, 4],
        device,
    );
    check(
        &Tensor::stack(&[&transposed, &transposed], -1).unwrap(),
        &[1., 1., 3., 3., 2., 2., 4., 4.],
        &[2, 2, 2],
        device,
    );
    let scalar = tensor(&[5.], &[], device);
    check(
        &Tensor::block_diag(&[&scalar, &left, &matrix]).unwrap(),
        &[
            5., 0., 0., 0., 0., 0., 1., 2., 0., 0., 0., 0., 0., 1., 2., 0., 0., 0., 3., 4.,
        ],
        &[4, 5],
        device,
    );
    let empty = tensor(&[], &[0, 2], device);
    check(
        &Tensor::cat(&[&empty, &matrix], 0).unwrap(),
        &[1., 2., 3., 4.],
        &[2, 2],
        device,
    );

    assert!(Tensor::cat(&[], 0).is_err());
    assert!(Tensor::stack(&[], 0).is_err());
    assert!(Tensor::cat(&inputs, 2).is_err());
    assert!(Tensor::stack(&[&left, &matrix], 0).is_err());
    assert!(Tensor::cartesian_prod(&[]).is_err());
    if device == 1 {
        let other_device = tensor(&[3., 4.], &[2], 0);
        assert!(Tensor::cat(&[&left, &other_device], 0).is_err());
    }
    check(&left, &[1., 2.], &[2], device);

    let leaf_left = tensor(&[1., 2.], &[2], device).require_grad().unwrap();
    let leaf_right = tensor(&[3., 4.], &[2], device).require_grad().unwrap();
    Tensor::stack(&[&leaf_left, &leaf_right], 0)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(&leaf_left.gradient().unwrap(), &[1., 1.], &[2], device);
    check(&leaf_right.gradient().unwrap(), &[1., 1.], &[2], device);
}

#[test]
fn cpu_tensor_joining_values_ownership_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_tensor_joining_values_ownership_errors_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
