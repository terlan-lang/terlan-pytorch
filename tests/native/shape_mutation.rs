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

/// Compares one in-place shape method with its functional counterpart.
macro_rules! assert_shape_mutation {
    ($source:ident, $functional:ident, $mutation:ident, [$($argument:expr),*], $cuda:ident) => {{
        let before = $source
            .to_device(0, -1)
            .expect("normalize source before operation")
            .to_packet()
            .expect("export source before operation");
        let expected = $source
            .$functional($($argument),*)
            .expect(concat!("compute ", stringify!($functional)));
        let mut actual = $source.clone().expect("clone mutation storage");
        actual
            .$mutation($($argument),*)
            .expect(concat!("execute ", stringify!($mutation)));
        assert_tensor_eq(
            &actual,
            &expected,
            $cuda,
            concat!("compare ", stringify!($mutation)),
        );
        assert_eq!(
            $source
                .to_device(0, -1)
                .expect("normalize source after operation")
                .to_packet()
                .expect("export source after operation"),
            before,
            "{} mutated its source handle",
            stringify!($mutation)
        );
    }};
}

/// Executes every shape-mutation wrapper in this slice on one device.
fn verify_shape_mutations(device_type: i64, device_index: i64, cuda: bool) {
    let singleton = Tensor::from_floats(&[1.0, 2.0], &[1, 2, 1])
        .expect("create singleton tensor")
        .to_device(device_type, device_index)
        .expect("place singleton tensor");
    assert_shape_mutation!(singleton, squeeze, squeeze_in_place, [], cuda);
    assert_shape_mutation!(
        singleton,
        squeeze_dim,
        squeeze_dimension_in_place,
        [0],
        cuda
    );
    assert_shape_mutation!(
        singleton,
        squeeze_dims,
        squeeze_dimensions_in_place,
        [&[0, 2]],
        cuda
    );
    assert_shape_mutation!(singleton, unsqueeze, unsqueeze_in_place, [0], cuda);

    let matrix = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("create matrix")
        .to_device(device_type, device_index)
        .expect("place matrix");
    assert_shape_mutation!(matrix, swapaxes, swapaxes_in_place, [0, 1], cuda);
    assert_shape_mutation!(matrix, swapdims, swapdims_in_place, [0, 1], cuda);
    assert_shape_mutation!(matrix, transpose, transpose_in_place, [0, 1], cuda);

    let square = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create square matrix")
        .to_device(device_type, device_index)
        .expect("place square matrix");
    assert_shape_mutation!(square, tril, tril_in_place, [0], cuda);
    assert_shape_mutation!(square, triu, triu_in_place, [0], cuda);

    let detached_source = Tensor::from_float(2.0)
        .expect("create detach source")
        .to_device(device_type, device_index)
        .expect("place detach source")
        .require_grad()
        .expect("create trainable source");
    let mut connected = detached_source.square().expect("create connected value");
    let expected_detached = connected.detach().expect("create functional detach");
    connected.detach_in_place().expect("detach in place");
    assert_tensor_eq(&connected, &expected_detached, cuda, "compare detach");
    assert!(
        connected.backward().is_err(),
        "detached tensor unexpectedly retained an autograd graph"
    );

    let mut zeroed = square.clone().expect("clone zero storage");
    zeroed.zero_in_place().expect("zero tensor in place");
    let expected_zero = Tensor::from_floats(&[0.0, 0.0, 0.0, 0.0], &[2, 2])
        .expect("create zero tensor")
        .to_device(device_type, device_index)
        .expect("place zero tensor");
    assert_tensor_eq(&zeroed, &expected_zero, cuda, "compare zero mutation");
}

#[test]
fn cpu_shape_mutations_match_functional_results() {
    verify_shape_mutations(0, -1, false);
}

#[test]
fn cuda_shape_mutations_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_shape_mutations(1, 0, true);
}

#[test]
fn invalid_shape_mutations_fail_atomically() {
    let source = Tensor::from_floats(&[1.0, 2.0], &[1, 2, 1]).expect("create source");
    let mut mutation = source.clone().expect("clone mutation");
    let before = mutation.to_packet().expect("snapshot mutation");

    let error = mutation
        .squeeze_dimension_in_place(7)
        .expect_err("invalid squeeze dimension unexpectedly succeeded");

    assert_ne!(error.status, 0);
    assert_eq!(mutation.to_packet().expect("export mutation"), before);
    assert_eq!(source.shape().expect("source shape"), vec![1, 2, 1]);
}
