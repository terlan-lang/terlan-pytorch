use terlan_pytorch_cxx::Tensor;

/// Compares tensor packets after normalizing CUDA values onto the CPU.
fn assert_tensor_eq(actual: &Tensor, expected: &Tensor, cuda: bool, context: &str) {
    if cuda {
        let actual = actual.to_device(0, -1).expect("copy actual to CPU");
        let expected = expected.to_device(0, -1).expect("copy expected to CPU");
        assert_eq!(
            actual.to_packet().expect(context),
            expected.to_packet().expect(context)
        );
    } else {
        assert_eq!(
            actual.to_packet().expect(context),
            expected.to_packet().expect(context)
        );
    }
}

macro_rules! assert_output_matches {
    ($left:ident, $right:ident, $functional:ident, $output:ident, $cuda:ident, $device_type:ident, $device_index:ident) => {{
        let expected = $left
            .$functional(&$right)
            .expect(concat!("compute ", stringify!($functional)));
        let dtype = $left.dtype().expect("read output dtype");
        let mut actual = Tensor::empty_typed_native(&[0], dtype, $device_type, $device_index)
            .expect(concat!("allocate ", stringify!($output), " output"));
        actual
            .$output(&$left, &$right)
            .expect(concat!("execute ", stringify!($output)));
        assert_tensor_eq(
            &actual,
            &expected,
            $cuda,
            concat!("compare ", stringify!($output)),
        );
    }};
}

/// Exercises every floating two-tensor retained-output schema in the slice.
fn verify_float_outputs(
    left: Tensor,
    right: Tensor,
    cuda: bool,
    device_type: i64,
    device_index: i64,
) {
    assert_output_matches!(
        left,
        right,
        arctan2,
        arctan2_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        atan2,
        atan2_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        copysign_tensor,
        copysign_tensor_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(left, right, dot, dot_out, cuda, device_type, device_index);
    assert_output_matches!(
        left,
        right,
        heaviside,
        heaviside_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        hypot,
        hypot_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        igamma,
        igamma_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        igammac,
        igammac_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        inner,
        inner_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(left, right, kron, kron_out, cuda, device_type, device_index);
    assert_output_matches!(
        left,
        right,
        logaddexp,
        logaddexp_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        logaddexp2,
        logaddexp2_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        max_other,
        max_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        min_other,
        min_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left,
        right,
        nextafter,
        nextafter_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(left, right, vdot, vdot_out, cuda, device_type, device_index);
}

/// Exercises integer, exponent, mask, and index retained-output schemas.
fn verify_specialized_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let integers = Tensor::from_ints(&[6, 15], &[2])
        .expect("create integer left")
        .to_device(device_type, device_index)
        .expect("place integer left");
    let divisors = Tensor::from_ints(&[4, 10], &[2])
        .expect("create integer right")
        .to_device(device_type, device_index)
        .expect("place integer right");
    assert_output_matches!(
        integers,
        divisors,
        gcd,
        gcd_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        integers,
        divisors,
        lcm,
        lcm_out,
        cuda,
        device_type,
        device_index
    );

    let mantissas = Tensor::from_floats(&[1.5, 2.0], &[2])
        .expect("create mantissas")
        .to_device(device_type, device_index)
        .expect("place mantissas");
    let exponents = Tensor::from_ints(&[2, -1], &[2])
        .expect("create exponents")
        .to_device(device_type, device_index)
        .expect("place exponents");
    assert_output_matches!(
        mantissas,
        exponents,
        ldexp_tensor,
        ldexp_out,
        cuda,
        device_type,
        device_index
    );

    let values = Tensor::from_floats(&[-2.0, 3.0, 4.0], &[3])
        .expect("create selection values")
        .to_device(device_type, device_index)
        .expect("place selection values");
    let mask = Tensor::from_bool(true)
        .expect("create mask")
        .to_device(device_type, device_index)
        .expect("place mask");
    assert_output_matches!(
        values,
        mask,
        masked_select,
        masked_select_out,
        cuda,
        device_type,
        device_index
    );

    let indices = Tensor::from_ints(&[2, 0], &[2])
        .expect("create indices")
        .to_device(device_type, device_index)
        .expect("place indices");
    assert_output_matches!(
        values,
        indices,
        take,
        take_out,
        cuda,
        device_type,
        device_index
    );
}

/// Exercises retained tuple outputs for dimension-wise extrema.
fn verify_dimension_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let input = Tensor::from_floats(&[3.0, 1.0, 2.0, 4.0], &[2, 2])
        .expect("create dimension input")
        .to_device(device_type, device_index)
        .expect("place dimension input");
    let (expected_max, expected_max_indices) =
        input.max_dim(1, false).expect("compute dimension maximum");
    let mut max_values = Tensor::from_floats(&[], &[0])
        .expect("create maximum values")
        .to_device(device_type, device_index)
        .expect("place maximum values");
    let mut max_indices = Tensor::from_ints(&[], &[0])
        .expect("create maximum indices")
        .to_device(device_type, device_index)
        .expect("place maximum indices");
    max_values
        .max_dim_out(&mut max_indices, &input, 1, false)
        .expect("execute max_dim_out");
    assert_tensor_eq(&max_values, &expected_max, cuda, "compare maximum values");
    assert_tensor_eq(
        &max_indices,
        &expected_max_indices,
        cuda,
        "compare maximum indices",
    );

    let (expected_min, expected_min_indices) =
        input.min_dim(1, false).expect("compute dimension minimum");
    let mut min_values = Tensor::from_floats(&[], &[0])
        .expect("create minimum values")
        .to_device(device_type, device_index)
        .expect("place minimum values");
    let mut min_indices = Tensor::from_ints(&[], &[0])
        .expect("create minimum indices")
        .to_device(device_type, device_index)
        .expect("place minimum indices");
    min_values
        .min_dim_out(&mut min_indices, &input, 1, false)
        .expect("execute min_dim_out");
    assert_tensor_eq(&min_values, &expected_min, cuda, "compare minimum values");
    assert_tensor_eq(
        &min_indices,
        &expected_min_indices,
        cuda,
        "compare minimum indices",
    );
}

/// Exercises retained outputs for dense matrix and vector operations.
fn verify_dense_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let matrix = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create matrix")
        .to_device(device_type, device_index)
        .expect("place matrix");
    let identity = Tensor::from_floats(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
        .expect("create identity")
        .to_device(device_type, device_index)
        .expect("place identity");
    assert_output_matches!(
        matrix,
        identity,
        matmul,
        matmul_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        matrix,
        identity,
        mm,
        mm_out,
        cuda,
        device_type,
        device_index
    );

    let batch = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2])
        .expect("create batch")
        .to_device(device_type, device_index)
        .expect("place batch");
    let batch_identity = Tensor::from_floats(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2])
        .expect("create batch identity")
        .to_device(device_type, device_index)
        .expect("place batch identity");
    assert_output_matches!(
        batch,
        batch_identity,
        bmm,
        bmm_out,
        cuda,
        device_type,
        device_index
    );

    let left_vector = Tensor::from_floats(&[1.0, 2.0], &[2])
        .expect("create left vector")
        .to_device(device_type, device_index)
        .expect("place left vector");
    let right_vector = Tensor::from_floats(&[3.0, 4.0], &[2])
        .expect("create right vector")
        .to_device(device_type, device_index)
        .expect("place right vector");
    assert_output_matches!(
        left_vector,
        right_vector,
        ger,
        ger_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        left_vector,
        right_vector,
        outer,
        outer_out,
        cuda,
        device_type,
        device_index
    );
    assert_output_matches!(
        matrix,
        left_vector,
        mv,
        mv_out,
        cuda,
        device_type,
        device_index
    );

    let reflectors = Tensor::from_floats(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
        .expect("create reflectors")
        .to_device(device_type, device_index)
        .expect("place reflectors");
    let tau = Tensor::from_floats(&[0.0, 0.0], &[2])
        .expect("create tau")
        .to_device(device_type, device_index)
        .expect("place tau");
    assert_output_matches!(
        reflectors,
        tau,
        orgqr,
        orgqr_out,
        cuda,
        device_type,
        device_index
    );
}

/// Compares retained tuple outputs with their functional values and indices.
fn assert_selection_pair(
    values: &Tensor,
    indices: &Tensor,
    expected_values: &Tensor,
    expected_indices: &Tensor,
    cuda: bool,
    context: &str,
) {
    assert_tensor_eq(values, expected_values, cuda, context);
    assert_tensor_eq(indices, expected_indices, cuda, context);
}

/// Exercises retained tuple outputs for ordering and ranked selection.
fn verify_selection_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let input = Tensor::from_floats(&[3.0, 1.0, 4.0, 2.0, 6.0, 8.0, 7.0, 5.0], &[2, 4])
        .expect("create selection input")
        .to_device(device_type, device_index)
        .expect("place selection input");
    let modal = Tensor::from_floats(&[2.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 4.0], &[2, 4])
        .expect("create modal input")
        .to_device(device_type, device_index)
        .expect("place modal input");

    let (expected_values, expected_indices) = input.cummax(1).expect("compute cummax");
    let mut values = expected_values.clone().expect("clone cummax values");
    let mut indices = expected_indices.clone().expect("clone cummax indices");
    values
        .cummax_out(&mut indices, &input, 1)
        .expect("execute cummax_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare cummax output",
    );

    let (expected_values, expected_indices) = input.cummin(1).expect("compute cummin");
    let mut values = expected_values.clone().expect("clone cummin values");
    let mut indices = expected_indices.clone().expect("clone cummin indices");
    values
        .cummin_out(&mut indices, &input, 1)
        .expect("execute cummin_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare cummin output",
    );

    let (expected_values, expected_indices) =
        input.kthvalue(2, 1, false).expect("compute kthvalue");
    let mut values = expected_values.clone().expect("clone kthvalue values");
    let mut indices = expected_indices.clone().expect("clone kthvalue indices");
    values
        .kthvalue_out(&mut indices, &input, 2, 1, false)
        .expect("execute kthvalue_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare kthvalue output",
    );

    let (expected_values, expected_indices) = input.median_dim(1, false).expect("compute median");
    let mut values = expected_values.clone().expect("clone median values");
    let mut indices = expected_indices.clone().expect("clone median indices");
    values
        .median_dim_out(&mut indices, &input, 1, false)
        .expect("execute median_dim_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare median output",
    );

    let (expected_values, expected_indices) = modal.mode(1, false).expect("compute mode");
    let mut values = expected_values.clone().expect("clone mode values");
    let mut indices = expected_indices.clone().expect("clone mode indices");
    values
        .mode_out(&mut indices, &modal, 1, false)
        .expect("execute mode_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare mode output",
    );

    let (expected_values, expected_indices) =
        input.nanmedian_dim(1, false).expect("compute nanmedian");
    let mut values = expected_values.clone().expect("clone nanmedian values");
    let mut indices = expected_indices.clone().expect("clone nanmedian indices");
    values
        .nanmedian_dim_out(&mut indices, &input, 1, false)
        .expect("execute nanmedian_dim_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare nanmedian output",
    );

    let (expected_values, expected_indices) = input.sort(1, false).expect("compute sort");
    let mut values = expected_values.clone().expect("clone sort values");
    let mut indices = expected_indices.clone().expect("clone sort indices");
    values
        .sort_out(&mut indices, &input, 1, false)
        .expect("execute sort_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare sort output",
    );

    let (expected_values, expected_indices) = input.topk(2, 1, true, true).expect("compute topk");
    let mut values = expected_values.clone().expect("clone topk values");
    let mut indices = expected_indices.clone().expect("clone topk indices");
    values
        .topk_out(&mut indices, &input, 2, 1, true, true)
        .expect("execute topk_out");
    assert_selection_pair(
        &values,
        &indices,
        &expected_values,
        &expected_indices,
        cuda,
        "compare topk output",
    );
}

#[test]
fn cpu_binary_output_families_match_functional_results() {
    let left = Tensor::from_floats(&[0.5, 1.5], &[2]).expect("create float left");
    let right = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create float right");
    verify_float_outputs(left, right, false, 0, -1);
    verify_specialized_outputs(0, -1, false);
    verify_dimension_outputs(0, -1, false);
    verify_dense_outputs(0, -1, false);
    verify_selection_outputs(0, -1, false);
}

#[test]
fn cuda_binary_output_families_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    let left = Tensor::from_floats(&[0.5, 1.5], &[2])
        .expect("create float left")
        .to_device(1, 0)
        .expect("place float left");
    let right = Tensor::from_floats(&[1.0, 2.0], &[2])
        .expect("create float right")
        .to_device(1, 0)
        .expect("place float right");
    verify_float_outputs(left, right, true, 1, 0);
    verify_specialized_outputs(1, 0, true);
    verify_dimension_outputs(1, 0, true);
    verify_dense_outputs(1, 0, true);
    verify_selection_outputs(1, 0, true);
}
