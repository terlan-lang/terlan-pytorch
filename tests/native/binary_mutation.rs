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

/// Compares one binary mutation with its immutable counterpart.
macro_rules! assert_binary_mutation {
    ($left:ident, $right:ident, $functional:ident, $mutation:ident, $cuda:ident) => {{
        let left_before = $left
            .to_device(0, -1)
            .expect("normalize left before operation")
            .to_packet()
            .expect("export left before operation");
        let right_before = $right
            .to_device(0, -1)
            .expect("normalize right before operation")
            .to_packet()
            .expect("export right before operation");
        let expected = $left
            .$functional(&$right)
            .expect(concat!("compute ", stringify!($functional)));
        let mut actual = $left.clone().expect("clone mutation storage");
        actual
            .$mutation(&$right)
            .expect(concat!("execute ", stringify!($mutation)));
        assert_tensor_eq(
            &actual,
            &expected,
            $cuda,
            concat!("compare ", stringify!($mutation)),
        );
        assert_eq!(
            $left
                .to_device(0, -1)
                .expect("normalize left after operation")
                .to_packet()
                .expect("export left after operation"),
            left_before,
            "{} mutated the source handle",
            stringify!($mutation)
        );
        assert_eq!(
            $right
                .to_device(0, -1)
                .expect("normalize right after operation")
                .to_packet()
                .expect("export right after operation"),
            right_before,
            "{} mutated its right operand",
            stringify!($mutation)
        );
    }};
}

/// Executes every binary mutation in this slice on one device.
fn verify_binary_mutations(device_type: i64, device_index: i64, cuda: bool) {
    let floats = Tensor::from_floats(&[0.5, 1.5], &[2])
        .expect("create float left")
        .to_device(device_type, device_index)
        .expect("place float left");
    let positive = Tensor::from_floats(&[1.0, 2.0], &[2])
        .expect("create float right")
        .to_device(device_type, device_index)
        .expect("place float right");
    assert_binary_mutation!(floats, positive, arctan2, arctan2_in_place, cuda);
    assert_binary_mutation!(floats, positive, atan2, atan2_in_place, cuda);
    assert_binary_mutation!(floats, positive, heaviside, heaviside_in_place, cuda);
    assert_binary_mutation!(floats, positive, hypot, hypot_in_place, cuda);
    assert_binary_mutation!(floats, positive, igamma, igamma_in_place, cuda);
    assert_binary_mutation!(floats, positive, igammac, igammac_in_place, cuda);
    assert_binary_mutation!(floats, positive, nextafter, nextafter_in_place, cuda);

    let integers = Tensor::from_ints(&[6, 15], &[2])
        .expect("create integer left")
        .to_device(device_type, device_index)
        .expect("place integer left");
    let divisors = Tensor::from_ints(&[4, 10], &[2])
        .expect("create integer right")
        .to_device(device_type, device_index)
        .expect("place integer right");
    assert_binary_mutation!(integers, divisors, gcd, gcd_in_place, cuda);
    assert_binary_mutation!(integers, divisors, lcm, lcm_in_place, cuda);

    let exponents = Tensor::from_ints(&[2, -1], &[2])
        .expect("create exponents")
        .to_device(device_type, device_index)
        .expect("place exponents");
    assert_binary_mutation!(floats, exponents, ldexp_tensor, ldexp_tensor_in_place, cuda);
}

/// Proves all binary mutations against their CPU functional counterparts.
#[test]
fn cpu_binary_mutations_match_functional_results() {
    verify_binary_mutations(0, -1, false);
}

/// Proves all binary mutations against their CUDA functional counterparts.
#[test]
fn cuda_binary_mutations_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_binary_mutations(1, 0, true);
}

/// Proves a failed broadcast leaves both caller-owned operands unchanged.
#[test]
fn invalid_binary_mutation_fails_atomically() {
    let mut left = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create left");
    let right = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create right");
    let left_before = left.to_packet().expect("snapshot left");
    let right_before = right.to_packet().expect("snapshot right");

    let error = left
        .atan2_in_place(&right)
        .expect_err("invalid broadcast unexpectedly succeeded");

    assert_ne!(error.status, 0);
    assert_eq!(left.to_packet().expect("export left"), left_before);
    assert_eq!(right.to_packet().expect("export right"), right_before);
}
