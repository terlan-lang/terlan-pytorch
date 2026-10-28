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
    macro_rules! bounds_case {
        ($view:ident, $copy:ident, $copy_out:ident, $scatter:ident, $scatter_out:ident,
         ($($bound:expr),*), $indices:expr) => {{
            let values = [1., 2., 3., 4., 5., 6.];
            let indices: &[usize] = $indices;
            let selected: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
            let patch_values: Vec<f64> = (1..=indices.len()).map(|i| (i * 10) as f64).collect();
            let shape = [indices.len() as i64];
            let mut replaced = values;
            let mut aliased = values;
            for (i, &index) in indices.iter().enumerate() {
                replaced[index] = patch_values[i];
                aliased[index] = 0.;
            }
            let base = tensor(&values, &[6], device);
            let patch = tensor(&patch_values, &shape, device);
            let mut view = base.$view(-1, $($bound,)* 2).unwrap();
            let copy = base.$copy(-1, $($bound,)* 2).unwrap();
            let mut copy_output = tensor(&[], &[0], device);
            copy_output.$copy_out(&base, -1, $($bound,)* 2).unwrap();
            let scatter = base.$scatter(&patch, -1, $($bound,)* 2).unwrap();
            let mut scatter_output = tensor(&[], &[0], device);
            scatter_output.$scatter_out(&base, &patch, -1, $($bound,)* 2).unwrap();
            check(&view, &selected, &shape, device);
            check(&copy, &selected, &shape, device);
            check(&copy_output, &selected, &shape, device);
            check(&scatter, &replaced, &[6], device);
            check(&scatter_output, &replaced, &[6], device);
            assert_eq!(view.strides().unwrap(), [2]);
            if let Some(&first) = indices.first() {
                assert_eq!(view.storage_offset().unwrap(), first as i64);
            }
            check(&base, &values, &[6], device);
            copy_output.zero_in_place().unwrap();
            scatter_output.zero_in_place().unwrap();
            view.zero_in_place().unwrap();
            check(&base, &aliased, &[6], device);
            check(&copy, &selected, &shape, device);
            check(&scatter, &replaced, &[6], device);
            check(&patch, &patch_values, &shape, device);
            drop(base);
            check(&view, &vec![0.; indices.len()], &shape, device);
        }};
    }
    bounds_case!(
        slice_all,
        slice_copy_all,
        slice_copy_out_all,
        slice_scatter_all,
        slice_scatter_out_all,
        (),
        &[0, 2, 4]
    );
    bounds_case!(
        slice_from,
        slice_copy_from,
        slice_copy_out_from,
        slice_scatter_from,
        slice_scatter_out_from,
        (1),
        &[1, 3, 5]
    );
    bounds_case!(
        slice_until,
        slice_copy_until,
        slice_copy_out_until,
        slice_scatter_until,
        slice_scatter_out_until,
        (3),
        &[0, 2]
    );
    bounds_case!(
        slice_bounds,
        slice_copy_bounds,
        slice_copy_out_bounds,
        slice_scatter_bounds,
        slice_scatter_out_bounds,
        (1, 5),
        &[1, 3]
    );
    bounds_case!(
        slice_bounds,
        slice_copy_bounds,
        slice_copy_out_bounds,
        slice_scatter_bounds,
        slice_scatter_out_bounds,
        (-5, -1),
        &[1, 3]
    );
    bounds_case!(
        slice_bounds,
        slice_copy_bounds,
        slice_copy_out_bounds,
        slice_scatter_bounds,
        slice_scatter_out_bounds,
        (-100, 100),
        &[0, 2, 4]
    );
    bounds_case!(
        slice_until,
        slice_copy_until,
        slice_copy_out_until,
        slice_scatter_until,
        slice_scatter_out_until,
        (0),
        &[]
    );

    let matrix = tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3], device);
    let transposed = matrix.transpose(0, 1).unwrap();
    let sliced = transposed.slice_from(0, 1, 1).unwrap();
    check(&sliced, &[2., 5., 3., 6.], &[2, 2], device);
    assert_eq!(sliced.strides().unwrap(), [1, 3]);
    assert_eq!(sliced.storage_offset().unwrap(), 1);
    let integers = matrix.to_dtype(4).unwrap().slice_all(-1, 2).unwrap();
    assert_eq!(integers.dtype().unwrap(), 4);
    assert_eq!(integers.shape().unwrap(), [2, 2]);
    assert!(integers
        .equal_native(
            &tensor(&[1., 3., 4., 6.], &[2, 2], device)
                .to_dtype(4)
                .unwrap()
        )
        .unwrap());
    assert!(matrix.slice_all(2, 1).is_err());
    assert!(matrix.slice_all(0, 0).is_err());
    assert!(matrix.slice_all(0, -1).is_err());
    assert!(tensor(&[1.], &[], device).slice_all(0, 1).is_err());
    let wrong = tensor(&[1.], &[1], device);
    assert!(matrix.slice_scatter_all(&wrong, 0, 1).is_err());
    let mut output = tensor(&[42.], &[1], device);
    assert!(output.slice_copy_out_all(&matrix, 0, 0).is_err());
    check(&output, &[42.], &[1], device);
    check(&matrix, &[1., 2., 3., 4., 5., 6.], &[2, 3], device);
    let leaf = tensor(&[1., 2., 3., 4., 5., 6.], &[6], device)
        .require_grad()
        .unwrap();
    leaf.slice_bounds(0, 1, 5, 2)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(
        &leaf.gradient().unwrap(),
        &[0., 1., 0., 1., 0., 0.],
        &[6],
        device,
    );
}

#[test]
fn cpu_slice_optional_bounds_values_ownership_errors_and_gradients() {
    verify(0);
}

#[test]
fn cuda_slice_optional_bounds_values_ownership_errors_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
