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
    let values = [0., 1., 2., 3., 4., 5., 6., 7.];
    for explicit in [false, true] {
        let base = tensor(&values, &[8], device);
        let source = base.narrow(0, 1, 6).unwrap();
        let mut view = if explicit {
            source.as_strided_offset(&[2, 2], &[3, 1], 2).unwrap()
        } else {
            source.as_strided(&[2, 2], &[3, 1]).unwrap()
        };
        let expected = if explicit {
            [2., 3., 5., 6.]
        } else {
            [1., 2., 4., 5.]
        };
        check(&view, &expected, &[2, 2], device);
        assert_eq!(view.strides().unwrap(), [3, 1]);
        assert_eq!(view.storage_offset().unwrap(), if explicit { 2 } else { 1 });
        view.zero_in_place().unwrap();
        let changed = if explicit {
            [0., 1., 0., 0., 4., 0., 0., 7.]
        } else {
            [0., 0., 0., 3., 0., 0., 6., 7.]
        };
        check(&base, &changed, &[8], device);
        drop(source);
        drop(base);
        check(&view, &[0.; 4], &[2, 2], device);

        let base = tensor(&values, &[8], device);
        let source = base.narrow(0, 1, 6).unwrap();
        let mut copy = if explicit {
            source.as_strided_copy_offset(&[2, 2], &[3, 1], 2).unwrap()
        } else {
            source.as_strided_copy(&[2, 2], &[3, 1]).unwrap()
        };
        check(&copy, &expected, &[2, 2], device);
        assert_eq!(copy.strides().unwrap(), [2, 1]);
        let mut output = tensor(&[], &[0], device);
        if explicit {
            output
                .as_strided_copy_out_offset(&source, &[2, 2], &[3, 1], 2)
                .unwrap();
        } else {
            output
                .as_strided_copy_out(&source, &[2, 2], &[3, 1])
                .unwrap();
        }
        check(&output, &expected, &[2, 2], device);
        copy.zero_in_place().unwrap();
        output.zero_in_place().unwrap();
        check(&base, &values, &[8], device);

        let mut mutation = base.narrow(0, 1, 6).unwrap();
        if explicit {
            mutation
                .as_strided_in_place_offset(&[2, 2], &[3, 1], 2)
                .unwrap();
        } else {
            mutation.as_strided_in_place(&[2, 2], &[3, 1]).unwrap();
        }
        check(&mutation, &expected, &[2, 2], device);
        assert_eq!(mutation.strides().unwrap(), [3, 1]);
        assert_eq!(
            mutation.storage_offset().unwrap(),
            if explicit { 2 } else { 1 }
        );
        mutation.zero_in_place().unwrap();
        check(&base, &changed, &[8], device);

        let base = tensor(&values, &[8], device);
        let patch = tensor(&[10., 11., 12., 13.], &[2, 2], device);
        let mut scattered = if explicit {
            base.as_strided_scatter_offset(&patch, &[2, 2], &[3, 1], 2)
                .unwrap()
        } else {
            base.as_strided_scatter(&patch, &[2, 2], &[3, 1]).unwrap()
        };
        let scattered_values = if explicit {
            [0., 1., 10., 11., 4., 12., 13., 7.]
        } else {
            [10., 11., 2., 12., 13., 5., 6., 7.]
        };
        check(&scattered, &scattered_values, &[8], device);
        let mut output = tensor(&[], &[0], device);
        if explicit {
            output
                .as_strided_scatter_out_offset(&base, &patch, &[2, 2], &[3, 1], 2)
                .unwrap();
        } else {
            output
                .as_strided_scatter_out(&base, &patch, &[2, 2], &[3, 1])
                .unwrap();
        }
        check(&output, &scattered_values, &[8], device);
        scattered.zero_in_place().unwrap();
        output.zero_in_place().unwrap();
        check(&base, &values, &[8], device);
        check(&patch, &[10., 11., 12., 13.], &[2, 2], device);
    }

    let source = tensor(&values, &[8], device);
    let scalar = source.as_strided_offset(&[], &[], 3).unwrap();
    check(&scalar, &[3.], &[], device);
    let empty = source.as_strided(&[0, 2], &[2, 1]).unwrap();
    check(&empty, &[], &[0, 2], device);
    let overlap = source.as_strided(&[2, 2], &[0, 1]).unwrap();
    check(&overlap, &[0., 1., 0., 1.], &[2, 2], device);
    let integer = source
        .to_dtype(4)
        .unwrap()
        .as_strided_offset(&[2], &[2], 1)
        .unwrap();
    assert_eq!(integer.dtype().unwrap(), 4);
    assert!(integer
        .equal_native(&tensor(&[1., 3.], &[2], device).to_dtype(4).unwrap())
        .unwrap());

    let leaf = source.require_grad().unwrap();
    let overlap = leaf.as_strided_offset(&[2, 2], &[1, 1], 1).unwrap();
    overlap.sum().unwrap().backward().unwrap();
    check(
        &leaf.gradient().unwrap(),
        &[0., 1., 2., 1., 0., 0., 0., 0.],
        &[8],
        device,
    );
}

fn verify_errors(device: i64) {
    let mut source = tensor(&[0., 1., 2., 3.], &[4], device);
    let before = source.to_device(0, -1).unwrap().to_packet().unwrap();
    assert!(source.as_strided(&[5], &[1]).is_err());
    assert!(source.as_strided(&[2], &[-1]).is_err());
    assert!(source.as_strided(&[2, 2], &[1]).is_err());
    assert!(source.as_strided_offset(&[2], &[1], -1).is_err());
    assert!(source.as_strided_copy(&[5], &[1]).is_err());
    assert!(source.as_strided_in_place_offset(&[2], &[1], 4).is_err());
    assert_eq!(source.shape().unwrap(), [4]);
    assert_eq!(source.strides().unwrap(), [1]);
    assert_eq!(source.storage_offset().unwrap(), 0);
    assert_eq!(
        source.to_device(0, -1).unwrap().to_packet().unwrap(),
        before
    );
    let wrong_patch = tensor(&[8.], &[1], device);
    assert!(source.as_strided_scatter(&wrong_patch, &[2], &[1]).is_err());
    let mut output = tensor(&[9., 9.], &[2], device);
    assert!(output.as_strided_copy_out(&source, &[5], &[1]).is_err());
    check(&output, &[9., 9.], &[2], device);
    assert!(output
        .as_strided_scatter_out(&source, &wrong_patch, &[2], &[1])
        .is_err());
    check(&source, &[0., 1., 2., 3.], &[4], device);
}

#[test]
fn cpu_strided_values_aliasing_copies_outputs_and_autograd() {
    verify(0);
}

#[test]
fn cpu_strided_invalid_requests_preserve_inputs() {
    verify_errors(0);
}

#[test]
fn cuda_strided_values_and_invalid_requests_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
        verify_errors(1);
    }
}
