use terlan_pytorch_cxx::Tensor;

fn tensor(values: &[f64], shape: &[i64], device: i64) -> Tensor {
    Tensor::from_floats(values, shape)
        .unwrap()
        .to_device(device, if device == 0 { -1 } else { 0 })
        .unwrap()
}

fn output(device: i64) -> Tensor {
    tensor(&[0.0], &[1], device)
}

fn same(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape().unwrap(), expected.shape().unwrap());
    assert_eq!(actual.dtype().unwrap(), expected.dtype().unwrap());
    assert_eq!(
        actual.device_type().unwrap(),
        expected.device_type().unwrap()
    );
    assert!(actual
        .allclose_native(expected, 1e-12, 1e-12, false)
        .unwrap());
}

fn verify(device: i64) {
    let source = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device);
    let input = source.transpose(0, 1).unwrap();
    assert_eq!(input.strides().unwrap(), [1, 2]);

    let std_default = input.std_correction_default(false).unwrap();
    let std_dimensions = input.std_correction_dimensions(&[1], false).unwrap();
    let std_float = input.std_correction_float(0.5, false).unwrap();
    let std_int = input.std_correction_int(0, false).unwrap();
    let std_dimensions_int = input.std_correction_dimensions_int(&[1], 0, true).unwrap();
    let std_dimensions_float = input
        .std_correction_dimensions_float(&[1], 0.5, false)
        .unwrap();

    same(&std_default, &tensor(&[1.2909944487358056], &[], device));
    same(
        &std_dimensions,
        &tensor(&[1.4142135623730951, 1.4142135623730951], &[2], device),
    );
    same(&std_int, &tensor(&[1.118033988749895], &[], device));
    same(&std_float, &tensor(&[1.1952286093343936], &[], device));
    same(&std_dimensions_int, &tensor(&[1.0, 1.0], &[2, 1], device));
    same(
        &std_dimensions_float,
        &tensor(&[1.1547005383792515, 1.1547005383792515], &[2], device),
    );

    let mut std_out_default = output(device);
    let mut std_out_dimensions = output(device);
    let mut std_out_int = output(device);
    let mut std_out_float = output(device);
    let mut std_out_dimensions_int = output(device);
    let mut std_out_dimensions_float = output(device);
    std_out_default
        .std_correction_out_default(&input, false)
        .unwrap();
    std_out_dimensions
        .std_correction_out_dimensions(&input, &[1], false)
        .unwrap();
    std_out_int
        .std_correction_out_int(&input, 0, false)
        .unwrap();
    std_out_float
        .std_correction_out_float(&input, 0.5, false)
        .unwrap();
    std_out_dimensions_int
        .std_correction_out_dimensions_int(&input, &[1], 0, true)
        .unwrap();
    std_out_dimensions_float
        .std_correction_out_dimensions_float(&input, &[1], 0.5, false)
        .unwrap();
    same(&std_out_default, &std_default);
    same(&std_out_dimensions, &std_dimensions);
    same(&std_out_int, &std_int);
    same(&std_out_float, &std_float);
    same(&std_out_dimensions_int, &std_dimensions_int);
    same(&std_out_dimensions_float, &std_dimensions_float);

    let var_default = input.var_correction_default(false).unwrap();
    let var_dimensions = input.var_correction_dimensions(&[1], false).unwrap();
    let var_int = input.var_correction_int(0, false).unwrap();
    let var_float = input.var_correction_float(0.5, false).unwrap();
    let var_dimensions_int = input.var_correction_dimensions_int(&[1], 0, true).unwrap();
    let var_dimensions_float = input
        .var_correction_dimensions_float(&[1], 0.5, false)
        .unwrap();
    same(&var_default, &tensor(&[5.0 / 3.0], &[], device));
    same(&var_dimensions, &tensor(&[2.0, 2.0], &[2], device));
    same(&var_int, &tensor(&[1.25], &[], device));
    same(&var_float, &tensor(&[10.0 / 7.0], &[], device));
    same(&var_dimensions_int, &tensor(&[1.0, 1.0], &[2, 1], device));
    same(
        &var_dimensions_float,
        &tensor(&[4.0 / 3.0, 4.0 / 3.0], &[2], device),
    );

    let mut var_out_default = output(device);
    let mut var_out_dimensions = output(device);
    let mut var_out_int = output(device);
    let mut var_out_float = output(device);
    let mut var_out_dimensions_int = output(device);
    let mut var_out_dimensions_float = output(device);
    var_out_default
        .var_correction_out_default(&input, false)
        .unwrap();
    var_out_dimensions
        .var_correction_out_dimensions(&input, &[1], false)
        .unwrap();
    var_out_int
        .var_correction_out_int(&input, 0, false)
        .unwrap();
    var_out_float
        .var_correction_out_float(&input, 0.5, false)
        .unwrap();
    var_out_dimensions_int
        .var_correction_out_dimensions_int(&input, &[1], 0, true)
        .unwrap();
    var_out_dimensions_float
        .var_correction_out_dimensions_float(&input, &[1], 0.5, false)
        .unwrap();
    same(&var_out_default, &var_default);
    same(&var_out_dimensions, &var_dimensions);
    same(&var_out_int, &var_int);
    same(&var_out_float, &var_float);
    same(&var_out_dimensions_int, &var_dimensions_int);
    same(&var_out_dimensions_float, &var_dimensions_float);

    let (std_pair_default, mean_pair_default) = input.std_mean_correction_default(false).unwrap();
    let (std_pair_dimensions, mean_pair_dimensions) =
        input.std_mean_correction_dimensions(&[1], false).unwrap();
    same(&std_pair_default, &std_default);
    same(&std_pair_dimensions, &std_dimensions);
    same(&mean_pair_default, &tensor(&[2.5], &[], device));
    same(&mean_pair_dimensions, &tensor(&[2.0, 3.0], &[2], device));

    let mut std_pair_out_default = output(device);
    let mut std_mean_out_default = output(device);
    let mut std_pair_out_dimensions = output(device);
    let mut std_mean_out_dimensions = output(device);
    std_pair_out_default
        .std_mean_correction_out_default(&mut std_mean_out_default, &input, false)
        .unwrap();
    std_pair_out_dimensions
        .std_mean_correction_out_dimensions(&mut std_mean_out_dimensions, &input, &[1], false)
        .unwrap();
    same(&std_pair_out_default, &std_pair_default);
    same(&std_mean_out_default, &mean_pair_default);
    same(&std_pair_out_dimensions, &std_pair_dimensions);
    same(&std_mean_out_dimensions, &mean_pair_dimensions);

    let std_int_pair = input.std_mean_correction_int_pair(0, false).unwrap();
    let std_int_pair_value = std_int_pair.tensor_pair_first().unwrap();
    let std_int_pair_mean = std_int_pair.tensor_pair_second().unwrap();
    let std_float_pair = input.std_mean_correction_float_pair(0.5, false).unwrap();
    let std_float_pair_value = std_float_pair.tensor_pair_first().unwrap();
    let std_float_pair_mean = std_float_pair.tensor_pair_second().unwrap();
    let std_dimensions_int_pair = input
        .std_mean_correction_dimensions_int_pair(&[1], 0, true)
        .unwrap();
    let std_dimensions_int_pair_value = std_dimensions_int_pair.tensor_pair_first().unwrap();
    let std_dimensions_int_pair_mean = std_dimensions_int_pair.tensor_pair_second().unwrap();
    let std_dimensions_float_pair = input
        .std_mean_correction_dimensions_float_pair(&[1], 0.5, false)
        .unwrap();
    let std_dimensions_float_pair_value = std_dimensions_float_pair.tensor_pair_first().unwrap();
    let std_dimensions_float_pair_mean = std_dimensions_float_pair.tensor_pair_second().unwrap();
    same(&std_int_pair_value, &std_int);
    same(&std_int_pair_mean, &mean_pair_default);
    same(&std_float_pair_value, &std_float);
    same(&std_float_pair_mean, &mean_pair_default);
    same(&std_dimensions_int_pair_value, &std_dimensions_int);
    same(
        &std_dimensions_int_pair_mean,
        &tensor(&[2.0, 3.0], &[2, 1], device),
    );
    same(&std_dimensions_float_pair_value, &std_dimensions_float);
    same(&std_dimensions_float_pair_mean, &mean_pair_dimensions);

    let mut std_int_pair_out = output(device);
    let mut std_int_pair_mean_out = output(device);
    let mut std_float_pair_out = output(device);
    let mut std_float_pair_mean_out = output(device);
    let mut std_dimensions_int_pair_out = output(device);
    let mut std_dimensions_int_pair_mean_out = output(device);
    let mut std_dimensions_float_pair_out = output(device);
    let mut std_dimensions_float_pair_mean_out = output(device);
    std_int_pair_out
        .std_mean_correction_out_int(&mut std_int_pair_mean_out, &input, 0, false)
        .unwrap();
    std_float_pair_out
        .std_mean_correction_out_float(&mut std_float_pair_mean_out, &input, 0.5, false)
        .unwrap();
    std_dimensions_int_pair_out
        .std_mean_correction_out_dimensions_int(
            &mut std_dimensions_int_pair_mean_out,
            &input,
            &[1],
            0,
            true,
        )
        .unwrap();
    std_dimensions_float_pair_out
        .std_mean_correction_out_dimensions_float(
            &mut std_dimensions_float_pair_mean_out,
            &input,
            &[1],
            0.5,
            false,
        )
        .unwrap();
    same(&std_int_pair_out, &std_int_pair_value);
    same(&std_int_pair_mean_out, &std_int_pair_mean);
    same(&std_float_pair_out, &std_float_pair_value);
    same(&std_float_pair_mean_out, &std_float_pair_mean);
    same(&std_dimensions_int_pair_out, &std_dimensions_int_pair_value);
    same(
        &std_dimensions_int_pair_mean_out,
        &std_dimensions_int_pair_mean,
    );
    same(
        &std_dimensions_float_pair_out,
        &std_dimensions_float_pair_value,
    );
    same(
        &std_dimensions_float_pair_mean_out,
        &std_dimensions_float_pair_mean,
    );

    let (var_pair_default, var_mean_default) = input.var_mean_correction_default(false).unwrap();
    let (var_pair_dimensions, var_mean_dimensions) =
        input.var_mean_correction_dimensions(&[1], false).unwrap();
    same(&var_pair_default, &var_default);
    same(&var_pair_dimensions, &var_dimensions);
    same(&var_mean_default, &mean_pair_default);
    same(&var_mean_dimensions, &mean_pair_dimensions);

    let mut var_pair_out_default = output(device);
    let mut var_mean_out_default = output(device);
    let mut var_pair_out_dimensions = output(device);
    let mut var_mean_out_dimensions = output(device);
    var_pair_out_default
        .var_mean_correction_out_default(&mut var_mean_out_default, &input, false)
        .unwrap();
    var_pair_out_dimensions
        .var_mean_correction_out_dimensions(&mut var_mean_out_dimensions, &input, &[1], false)
        .unwrap();
    same(&var_pair_out_default, &var_pair_default);
    same(&var_mean_out_default, &var_mean_default);
    same(&var_pair_out_dimensions, &var_pair_dimensions);
    same(&var_mean_out_dimensions, &var_mean_dimensions);

    let var_int_pair = input.var_mean_correction_int_pair(0, false).unwrap();
    let var_int_pair_value = var_int_pair.tensor_pair_first().unwrap();
    let var_int_pair_mean = var_int_pair.tensor_pair_second().unwrap();
    let var_float_pair = input.var_mean_correction_float_pair(0.5, false).unwrap();
    let var_float_pair_value = var_float_pair.tensor_pair_first().unwrap();
    let var_float_pair_mean = var_float_pair.tensor_pair_second().unwrap();
    let var_dimensions_int_pair = input
        .var_mean_correction_dimensions_int_pair(&[1], 0, true)
        .unwrap();
    let var_dimensions_int_pair_value = var_dimensions_int_pair.tensor_pair_first().unwrap();
    let var_dimensions_int_pair_mean = var_dimensions_int_pair.tensor_pair_second().unwrap();
    let var_dimensions_float_pair = input
        .var_mean_correction_dimensions_float_pair(&[1], 0.5, false)
        .unwrap();
    let var_dimensions_float_pair_value = var_dimensions_float_pair.tensor_pair_first().unwrap();
    let var_dimensions_float_pair_mean = var_dimensions_float_pair.tensor_pair_second().unwrap();
    same(&var_int_pair_value, &var_int);
    same(&var_int_pair_mean, &var_mean_default);
    same(&var_float_pair_value, &var_float);
    same(&var_float_pair_mean, &var_mean_default);
    same(&var_dimensions_int_pair_value, &var_dimensions_int);
    same(
        &var_dimensions_int_pair_mean,
        &tensor(&[2.0, 3.0], &[2, 1], device),
    );
    same(&var_dimensions_float_pair_value, &var_dimensions_float);
    same(&var_dimensions_float_pair_mean, &var_mean_dimensions);

    let mut var_int_pair_out = output(device);
    let mut var_int_pair_mean_out = output(device);
    let mut var_float_pair_out = output(device);
    let mut var_float_pair_mean_out = output(device);
    let mut var_dimensions_int_pair_out = output(device);
    let mut var_dimensions_int_pair_mean_out = output(device);
    let mut var_dimensions_float_pair_out = output(device);
    let mut var_dimensions_float_pair_mean_out = output(device);
    var_int_pair_out
        .var_mean_correction_out_int(&mut var_int_pair_mean_out, &input, 0, false)
        .unwrap();
    var_float_pair_out
        .var_mean_correction_out_float(&mut var_float_pair_mean_out, &input, 0.5, false)
        .unwrap();
    var_dimensions_int_pair_out
        .var_mean_correction_out_dimensions_int(
            &mut var_dimensions_int_pair_mean_out,
            &input,
            &[1],
            0,
            true,
        )
        .unwrap();
    var_dimensions_float_pair_out
        .var_mean_correction_out_dimensions_float(
            &mut var_dimensions_float_pair_mean_out,
            &input,
            &[1],
            0.5,
            false,
        )
        .unwrap();
    same(&var_int_pair_out, &var_int_pair_value);
    same(&var_int_pair_mean_out, &var_int_pair_mean);
    same(&var_float_pair_out, &var_float_pair_value);
    same(&var_float_pair_mean_out, &var_float_pair_mean);
    same(&var_dimensions_int_pair_out, &var_dimensions_int_pair_value);
    same(
        &var_dimensions_int_pair_mean_out,
        &var_dimensions_int_pair_mean,
    );
    same(
        &var_dimensions_float_pair_out,
        &var_dimensions_float_pair_value,
    );
    same(
        &var_dimensions_float_pair_mean_out,
        &var_dimensions_float_pair_mean,
    );

    assert!(input.std_correction_dimensions(&[2], false).is_err());
    same(&source, &tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device));
}

#[test]
fn correction_statistics_variants_match_on_cpu_and_required_cuda() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
        verify(1);
    }
    verify(0);
}
