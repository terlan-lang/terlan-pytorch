use terlan_pytorch_cxx::Tensor;

/// Compares complete tensor packets after normalizing CUDA values to CPU.
fn assert_tensor_eq(actual: &Tensor, expected: &Tensor, cuda: bool, context: &str) {
    let actual = if cuda {
        actual.to_device(0, -1).expect("copy actual to CPU")
    } else {
        actual.clone().expect("clone actual")
    };
    let expected = if cuda {
        expected.to_device(0, -1).expect("copy expected to CPU")
    } else {
        expected.clone().expect("clone expected")
    };
    assert_eq!(
        actual.to_packet().expect(context),
        expected.to_packet().expect(context)
    );
}

/// Compares one retained-output method with its functional counterpart.
macro_rules! assert_shape_output {
    ($source:ident, $functional:ident, $output:ident, [$($argument:expr),*], $cuda:ident) => {{
        let before = $source
            .to_device(0, -1)
            .expect("normalize source before operation")
            .to_packet()
            .expect("export source before operation");
        let expected = $source
            .$functional($($argument),*)
            .expect(concat!("compute ", stringify!($functional)));
        let mut actual = expected.clone().expect("clone output storage");
        actual
            .$output(&$source $(, $argument)*)
            .expect(concat!("execute ", stringify!($output)));
        assert_tensor_eq(
            &actual,
            &expected,
            $cuda,
            concat!("compare ", stringify!($output)),
        );
        assert_eq!(
            $source
                .to_device(0, -1)
                .expect("normalize source after operation")
                .to_packet()
                .expect("export source after operation"),
            before,
            "{} mutated its source",
            stringify!($output)
        );
    }};
}

/// Executes every shape retained-output wrapper in the slice on one device.
fn verify_shape_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let matrix = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("create matrix")
        .to_device(device_type, device_index)
        .expect("place matrix");
    assert_shape_output!(matrix, diag, diag_out, [0], cuda);
    assert_shape_output!(matrix, flip, flip_out, [&[1]], cuda);
    assert_shape_output!(matrix, permute_copy, permute_copy_out, [&[1, 0]], cuda);
    assert_shape_output!(matrix, repeat, repeat_out, [&[2, 1]], cuda);
    assert_shape_output!(matrix, roll, roll_out, [&[1], &[1]], cuda);
    assert_shape_output!(matrix, rot90, rot90_out, [1, &[0, 1]], cuda);
    assert_shape_output!(matrix, view_copy, view_copy_out, [&[3, 2]], cuda);
    assert_shape_output!(matrix, unsqueeze_copy, unsqueeze_copy_out, [0], cuda);

    let square = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create square matrix")
        .to_device(device_type, device_index)
        .expect("place square matrix");
    assert_shape_output!(square, diagonal_copy, diagonal_copy_out, [0, 0, 1], cuda);
    assert_shape_output!(square, matrix_power, matrix_power_out, [2], cuda);
    assert_shape_output!(square, tril, tril_out, [0], cuda);
    assert_shape_output!(square, triu, triu_out, [0], cuda);

    let vector = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3])
        .expect("create vector")
        .to_device(device_type, device_index)
        .expect("place vector");
    assert_shape_output!(vector, diag_embed, diag_embed_out, [0, -2, -1], cuda);

    let pixels = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1])
        .expect("create pixel channels")
        .to_device(device_type, device_index)
        .expect("place pixel channels");
    assert_shape_output!(pixels, pixel_shuffle, pixel_shuffle_out, [2], cuda);
    let shuffled = pixels.pixel_shuffle(2).expect("prepare shuffled pixels");
    assert_shape_output!(shuffled, pixel_unshuffle, pixel_unshuffle_out, [2], cuda);

    let singleton = Tensor::from_floats(&[1.0, 2.0], &[1, 2, 1])
        .expect("create singleton tensor")
        .to_device(device_type, device_index)
        .expect("place singleton tensor");
    assert_shape_output!(singleton, squeeze_copy, squeeze_copy_out, [], cuda);
    assert_shape_output!(
        singleton,
        squeeze_copy_dim,
        squeeze_copy_dimension_out,
        [0],
        cuda
    );
    assert_shape_output!(
        singleton,
        squeeze_copy_dims,
        squeeze_copy_dimensions_out,
        [&[0, 2]],
        cuda
    );
}

#[test]
fn cpu_shape_outputs_match_functional_results() {
    verify_shape_outputs(0, -1, false);
}

#[test]
fn cuda_shape_outputs_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_shape_outputs(1, 0, true);
}
