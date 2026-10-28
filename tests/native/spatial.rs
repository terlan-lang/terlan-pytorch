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
    assert!(actual
        .allclose_native(&tensor(values, shape, device), 1e-12, 1e-12, false)
        .unwrap());
}

fn verify(device: i64) {
    let theta = tensor(&[1., 0., 0., 0., 1., 0.], &[1, 2, 3], device);
    let input = tensor(&[1., 2., 3., 4.], &[1, 1, 2, 2], device);
    for align in [false, true] {
        let a = if align { 1. } else { 0.5 };
        let expected = [-a, -a, a, -a, -a, a, a, a];
        let grid = theta.affine_grid_generator(&[1, 1, 2, 2], align).unwrap();
        check(&grid, &expected, &[1, 2, 2, 2], device);
        let mut grid_output = tensor(&[], &[0], device);
        grid_output
            .affine_grid_generator_out(&theta, &[1, 1, 2, 2], align)
            .unwrap();
        check(&grid_output, &expected, &[1, 2, 2, 2], device);
        let grad = tensor(&[1.; 8], &[1, 2, 2, 2], device);
        check(
            &grad
                .affine_grid_generator_backward(&[1, 1, 2, 2], align)
                .unwrap(),
            &[0., 0., 4., 0., 0., 4.],
            &[1, 2, 3],
            device,
        );
        for mode in [0, 1, 2] {
            for padding in [0, 1, 2] {
                let generic = input.grid_sampler(&grid, mode, padding, align).unwrap();
                let sampled = input.grid_sampler_2d(&grid, mode, padding, align).unwrap();
                check(&generic, &[1., 2., 3., 4.], &[1, 1, 2, 2], device);
                check(&sampled, &[1., 2., 3., 4.], &[1, 1, 2, 2], device);
                let mut output = tensor(&[], &[0], device);
                output
                    .grid_sampler_2d_out(&input, &grid, mode, padding, align)
                    .unwrap();
                check(&output, &[1., 2., 3., 4.], &[1, 1, 2, 2], device);
            }
        }
        grid_output.zero_in_place().unwrap();
        check(&theta, &[1., 0., 0., 0., 1., 0.], &[1, 2, 3], device);
        check(&grid, &expected, &[1, 2, 2, 2], device);
    }
    let center = tensor(&[0., 0.], &[1, 1, 1, 2], device);
    check(
        &input.grid_sampler_2d(&center, 0, 0, true).unwrap(),
        &[2.5],
        &[1, 1, 1, 1],
        device,
    );
    let outside = tensor(&[3., 3.], &[1, 1, 1, 2], device);
    for (padding, expected) in [(0, 0.), (1, 4.), (2, 1.)] {
        check(
            &input.grid_sampler_2d(&outside, 0, padding, true).unwrap(),
            &[expected],
            &[1, 1, 1, 1],
            device,
        );
    }

    let theta3 = tensor(
        &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        &[1, 3, 4],
        device,
    );
    let volume_values = [1., 2., 3., 4., 5., 6., 7., 8.];
    let volume = tensor(&volume_values, &[1, 1, 2, 2, 2], device);
    for align in [false, true] {
        let a = if align { 1. } else { 0.5 };
        let expected = [
            -a, -a, -a, a, -a, -a, -a, a, -a, a, a, -a, -a, -a, a, a, -a, a, -a, a, a, a, a, a,
        ];
        let grid = theta3
            .affine_grid_generator(&[1, 1, 2, 2, 2], align)
            .unwrap();
        check(&grid, &expected, &[1, 2, 2, 2, 3], device);
        let mut grid_output = tensor(&[], &[0], device);
        grid_output
            .affine_grid_generator_out(&theta3, &[1, 1, 2, 2, 2], align)
            .unwrap();
        check(&grid_output, &expected, &[1, 2, 2, 2, 3], device);
        let grad = tensor(&[1.; 24], &[1, 2, 2, 2, 3], device);
        check(
            &grad
                .affine_grid_generator_backward(&[1, 1, 2, 2, 2], align)
                .unwrap(),
            &[0., 0., 0., 8., 0., 0., 0., 8., 0., 0., 0., 8.],
            &[1, 3, 4],
            device,
        );
        for mode in [0, 1] {
            for padding in [0, 1, 2] {
                check(
                    &volume.grid_sampler(&grid, mode, padding, align).unwrap(),
                    &volume_values,
                    &[1, 1, 2, 2, 2],
                    device,
                );
                check(
                    &volume.grid_sampler_3d(&grid, mode, padding, align).unwrap(),
                    &volume_values,
                    &[1, 1, 2, 2, 2],
                    device,
                );
                let mut output = tensor(&[], &[0], device);
                output
                    .grid_sampler_3d_out(&volume, &grid, mode, padding, align)
                    .unwrap();
                check(&output, &volume_values, &[1, 1, 2, 2, 2], device);
            }
        }
    }
    let outside3 = tensor(&[3., 3., 3.], &[1, 1, 1, 1, 3], device);
    for (padding, expected) in [(0, 0.), (1, 8.), (2, 1.)] {
        check(
            &volume.grid_sampler_3d(&outside3, 0, padding, true).unwrap(),
            &[expected],
            &[1, 1, 1, 1, 1],
            device,
        );
    }
    let center3 = tensor(&[0., 0., 0.], &[1, 1, 1, 1, 3], device);
    check(
        &volume.grid_sampler_3d(&center3, 0, 0, true).unwrap(),
        &[4.5],
        &[1, 1, 1, 1, 1],
        device,
    );
    assert!(volume.grid_sampler_3d(&center3, 2, 0, true).is_err());
    assert!(input.grid_sampler_2d(&center3, 0, 0, true).is_err());
    assert!(theta.affine_grid_generator(&[1, 2, 2], true).is_err());
    check(&input, &[1., 2., 3., 4.], &[1, 1, 2, 2], device);
    check(&volume, &volume_values, &[1, 1, 2, 2, 2], device);

    let leaf = input.require_grad().unwrap();
    leaf.grid_sampler_2d(&center, 0, 0, true)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    check(&leaf.gradient().unwrap(), &[0.25; 4], &[1, 1, 2, 2], device);
}

#[test]
fn cpu_spatial_transform_values_options_outputs_and_gradients() {
    verify(0);
}

#[test]
fn cuda_spatial_transform_values_options_outputs_and_gradients_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
}
