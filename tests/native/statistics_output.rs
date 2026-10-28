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

/// Executes every reduction output wrapper in the slice on one device.
fn verify_statistics_outputs(device_type: i64, device_index: i64, cuda: bool) {
    let values = Tensor::from_floats(&[3.0, 1.0, 4.0, 2.0, 6.0, 5.0], &[2, 3])
        .expect("create values")
        .to_device(device_type, device_index)
        .expect("place values");
    let logical = Tensor::from_ints(&[1, 0, 1, 1], &[2, 2])
        .expect("create logical values")
        .to_device(device_type, device_index)
        .expect("place logical values");
    let values_before = values
        .to_device(0, -1)
        .expect("normalize values")
        .to_packet()
        .expect("export values before operations");
    let logical_before = logical
        .to_device(0, -1)
        .expect("normalize logical values")
        .to_packet()
        .expect("export logical values before operations");

    let expected = values.amax(&[1], false).expect("compute amax");
    let mut output = expected.clone().expect("clone amax output");
    output
        .amax_out(&values, &[1], false)
        .expect("execute amax_out");
    assert_tensor_eq(&output, &expected, cuda, "compare amax output");

    let expected = values.amin(&[1], false).expect("compute amin");
    let mut output = expected.clone().expect("clone amin output");
    output
        .amin_out(&values, &[1], false)
        .expect("execute amin_out");
    assert_tensor_eq(&output, &expected, cuda, "compare amin output");

    let expected = logical
        .all_dimensions(&[1], false)
        .expect("compute all dimensions");
    let mut output = expected.clone().expect("clone all dimensions output");
    output
        .all_dimensions_out(&logical, &[1], false)
        .expect("execute all_dimensions_out");
    assert_tensor_eq(&output, &expected, cuda, "compare all dimensions output");

    let expected = logical
        .any_dimensions(&[1], false)
        .expect("compute any dimensions");
    let mut output = expected.clone().expect("clone any dimensions output");
    output
        .any_dimensions_out(&logical, &[1], false)
        .expect("execute any_dimensions_out");
    assert_tensor_eq(&output, &expected, cuda, "compare any dimensions output");

    let expected = logical.all_dim(1, false).expect("compute all dimension");
    let mut output = expected.clone().expect("clone all dimension output");
    output
        .all_dimension_out(&logical, 1, false)
        .expect("execute all_dimension_out");
    assert_tensor_eq(&output, &expected, cuda, "compare all dimension output");

    let expected = logical.any_dim(1, false).expect("compute any dimension");
    let mut output = expected.clone().expect("clone any dimension output");
    output
        .any_dimension_out(&logical, 1, false)
        .expect("execute any_dimension_out");
    assert_tensor_eq(&output, &expected, cuda, "compare any dimension output");

    let expected = values.argmax(1, false).expect("compute argmax");
    let mut output = expected.clone().expect("clone argmax output");
    output
        .argmax_out(&values, 1, false)
        .expect("execute argmax_out");
    assert_tensor_eq(&output, &expected, cuda, "compare argmax output");

    let expected = values.argmax_all(false).expect("compute global argmax");
    let mut output = expected.clone().expect("clone global argmax output");
    output
        .argmax_all_out(&values, false)
        .expect("execute argmax_all_out");
    assert_tensor_eq(&output, &expected, cuda, "compare global argmax output");

    let expected = values.argmin(1, false).expect("compute argmin");
    let mut output = expected.clone().expect("clone argmin output");
    output
        .argmin_out(&values, 1, false)
        .expect("execute argmin_out");
    assert_tensor_eq(&output, &expected, cuda, "compare argmin output");

    let expected = values.argmin_all(false).expect("compute global argmin");
    let mut output = expected.clone().expect("clone global argmin output");
    output
        .argmin_all_out(&values, false)
        .expect("execute argmin_all_out");
    assert_tensor_eq(&output, &expected, cuda, "compare global argmin output");

    let expected = logical.count_nonzero(1).expect("compute count_nonzero");
    let mut output = expected.clone().expect("clone count output");
    output
        .count_nonzero_out(&logical, 1)
        .expect("execute count_nonzero_out");
    assert_tensor_eq(&output, &expected, cuda, "compare count output");
    let mut list_output = expected.clone().expect("clone count list output");
    list_output
        .count_nonzero_dimensions_out(&logical, &[1])
        .expect("execute count_nonzero_dimensions_out");
    assert_tensor_eq(&list_output, &expected, cuda, "compare count list output");

    let expected = logical
        .count_nonzero_all()
        .expect("compute global count_nonzero");
    let mut output = expected.clone().expect("clone global count output");
    output
        .count_nonzero_all_out(&logical)
        .expect("execute count_nonzero_all_out");
    assert_tensor_eq(&output, &expected, cuda, "compare global count output");

    let (expected_minimum, expected_maximum) = values
        .aminmax_dim(1, false)
        .expect("compute dimension aminmax");
    let mut minimum = expected_minimum.clone().expect("clone dimension minimum");
    let mut maximum = expected_maximum.clone().expect("clone dimension maximum");
    minimum
        .aminmax_dim_out(&mut maximum, &values, 1, false)
        .expect("execute aminmax_dim_out");
    assert_tensor_eq(
        &minimum,
        &expected_minimum,
        cuda,
        "compare dimension minimum",
    );
    assert_tensor_eq(
        &maximum,
        &expected_maximum,
        cuda,
        "compare dimension maximum",
    );

    let (expected_minimum, expected_maximum) =
        values.aminmax(false).expect("compute global aminmax");
    let mut minimum = expected_minimum.clone().expect("clone global minimum");
    let mut maximum = expected_maximum.clone().expect("clone global maximum");
    minimum
        .aminmax_all_out(&mut maximum, &values, false)
        .expect("execute aminmax_all_out");
    assert_tensor_eq(&minimum, &expected_minimum, cuda, "compare global minimum");
    assert_tensor_eq(&maximum, &expected_maximum, cuda, "compare global maximum");

    let expected = values.nanmedian().expect("compute nanmedian");
    let mut output = expected.clone().expect("clone nanmedian output");
    output
        .nanmedian_out(&values)
        .expect("execute nanmedian_out");
    assert_tensor_eq(&output, &expected, cuda, "compare nanmedian output");

    let expected = values.logsumexp(&[1], false).expect("compute logsumexp");
    let mut output = expected.clone().expect("clone logsumexp output");
    output
        .logsumexp_out(&values, &[1], false)
        .expect("execute logsumexp_out");
    assert_tensor_eq(&output, &expected, cuda, "compare logsumexp output");

    let expected = values.mean().expect("compute mean");
    let mut output = expected.clone().expect("clone mean output");
    output.mean_out(&values).expect("execute mean_out");
    assert_tensor_eq(&output, &expected, cuda, "compare mean output");
    let mut all_output = expected
        .clone()
        .expect("clone omitted-dimension mean output");
    all_output
        .mean_all_dimensions_out(&values, false)
        .expect("execute mean_all_dimensions_out");
    assert_tensor_eq(
        &all_output,
        &expected,
        cuda,
        "compare omitted-dimension mean output",
    );

    let expected = values
        .mean_dimensions(&[1], false)
        .expect("compute dimension mean");
    let mut output = expected.clone().expect("clone dimension mean output");
    output
        .mean_dimensions_out(&values, &[1], false)
        .expect("execute mean_dimensions_out");
    assert_tensor_eq(&output, &expected, cuda, "compare dimension mean output");

    let expected = logical.prod().expect("compute product");
    let mut output = expected.clone().expect("clone product output");
    output.prod_out(&logical).expect("execute prod_out");
    assert_tensor_eq(&output, &expected, cuda, "compare product output");

    let expected = logical
        .prod_dim(1, false)
        .expect("compute dimension product");
    let mut output = expected.clone().expect("clone dimension product output");
    output
        .prod_dimension_out(&logical, 1, false)
        .expect("execute prod_dimension_out");
    assert_tensor_eq(&output, &expected, cuda, "compare dimension product output");

    let expected = logical.sum().expect("compute sum");
    let mut output = expected.clone().expect("clone sum output");
    output.sum_out(&logical).expect("execute sum_out");
    assert_tensor_eq(&output, &expected, cuda, "compare sum output");
    let mut all_output = expected
        .clone()
        .expect("clone omitted-dimension sum output");
    all_output
        .sum_all_dimensions_out(&logical, false)
        .expect("execute sum_all_dimensions_out");
    assert_tensor_eq(
        &all_output,
        &expected,
        cuda,
        "compare omitted-dimension sum output",
    );

    let expected = logical
        .sum_dimensions(&[1], false)
        .expect("compute dimension sum");
    let mut output = expected.clone().expect("clone dimension sum output");
    output
        .sum_dimensions_out(&logical, &[1], false)
        .expect("execute sum_dimensions_out");
    assert_tensor_eq(&output, &expected, cuda, "compare dimension sum output");

    let expected = values
        .std_dimensions(&[1], true, false)
        .expect("compute dimension standard deviation");
    let mut output = expected
        .clone()
        .expect("clone dimension standard deviation output");
    output
        .std_dimensions_out(&values, &[1], true, false)
        .expect("execute std_dimensions_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare dimension standard deviation output",
    );

    let expected = values.std(true).expect("compute standard deviation");
    let mut output = expected.clone().expect("clone standard deviation output");
    output
        .std_all_dimensions_out(&values, true, false)
        .expect("execute std_all_dimensions_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare omitted-dimension standard deviation output",
    );

    let expected = values
        .var_dimensions(&[1], true, false)
        .expect("compute dimension variance");
    let mut output = expected.clone().expect("clone dimension variance output");
    output
        .var_dimensions_out(&values, &[1], true, false)
        .expect("execute var_dimensions_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare dimension variance output",
    );

    let expected = values.var(true).expect("compute variance");
    let mut output = expected.clone().expect("clone variance output");
    output
        .var_all_dimensions_out(&values, true, false)
        .expect("execute var_all_dimensions_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare omitted-dimension variance output",
    );

    let expected = logical.cumsum(1).expect("compute cumulative sum");
    let mut output = expected.clone().expect("clone cumulative sum output");
    output.cumsum_out(&logical, 1).expect("execute cumsum_out");
    assert_tensor_eq(&output, &expected, cuda, "compare cumulative sum output");

    let expected = logical.cumprod(1).expect("compute cumulative product");
    let mut output = expected.clone().expect("clone cumulative product output");
    output
        .cumprod_out(&logical, 1)
        .expect("execute cumprod_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare cumulative product output",
    );

    let expected = values
        .logcumsumexp(1)
        .expect("compute logarithmic cumulative sum");
    let mut output = expected
        .clone()
        .expect("clone logarithmic cumulative sum output");
    output
        .logcumsumexp_out(&values, 1)
        .expect("execute logcumsumexp_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare logarithmic cumulative sum output",
    );

    let positive_definite = Tensor::from_floats(&[4.0, 2.0, 2.0, 3.0], &[2, 2])
        .expect("create positive-definite matrix")
        .to_device(device_type, device_index)
        .expect("place positive-definite matrix");
    let positive_definite_before = positive_definite
        .to_device(0, -1)
        .expect("normalize positive-definite matrix")
        .to_packet()
        .expect("export positive-definite matrix before operations");
    let expected_factor = positive_definite
        .cholesky(false)
        .expect("compute Cholesky factor");
    let mut factor_output = expected_factor.clone().expect("clone Cholesky output");
    factor_output
        .cholesky_out(&positive_definite, false)
        .expect("execute cholesky_out");
    assert_tensor_eq(
        &factor_output,
        &expected_factor,
        cuda,
        "compare Cholesky output",
    );

    let expected_inverse = expected_factor
        .cholesky_inverse(false)
        .expect("compute Cholesky inverse");
    let mut inverse_output = expected_inverse.clone().expect("clone inverse output");
    inverse_output
        .cholesky_inverse_out(&expected_factor, false)
        .expect("execute cholesky_inverse_out");
    assert_tensor_eq(
        &inverse_output,
        &expected_inverse,
        cuda,
        "compare Cholesky inverse output",
    );

    let right_hand_side = Tensor::from_floats(&[1.0, 0.0], &[2, 1])
        .expect("create right-hand side")
        .to_device(device_type, device_index)
        .expect("place right-hand side");
    let right_hand_side_before = right_hand_side
        .to_device(0, -1)
        .expect("normalize right-hand side")
        .to_packet()
        .expect("export right-hand side before operations");
    let expected_solution = right_hand_side
        .cholesky_solve(&expected_factor, false)
        .expect("compute Cholesky solution");
    let mut solution_output = expected_solution.clone().expect("clone solution output");
    solution_output
        .cholesky_solve_out(&right_hand_side, &expected_factor, false)
        .expect("execute cholesky_solve_out");
    assert_tensor_eq(
        &solution_output,
        &expected_solution,
        cuda,
        "compare Cholesky solution output",
    );

    let expected = positive_definite
        .frobenius_norm_dimensions(&[0, 1], false)
        .expect("compute Frobenius norm");
    let mut output = expected.clone().expect("clone Frobenius norm output");
    output
        .frobenius_norm_dimensions_out(&positive_definite, &[0, 1], false)
        .expect("execute frobenius_norm_dimensions_out");
    assert_tensor_eq(&output, &expected, cuda, "compare Frobenius norm output");

    let expected = positive_definite
        .nuclear_norm(false)
        .expect("compute nuclear norm");
    let mut output = expected.clone().expect("clone nuclear norm output");
    output
        .nuclear_norm_out(&positive_definite, false)
        .expect("execute nuclear_norm_out");
    assert_tensor_eq(&output, &expected, cuda, "compare nuclear norm output");

    let expected = positive_definite
        .nuclear_norm_dimensions(&[0, 1], false)
        .expect("compute dimension nuclear norm");
    let mut output = expected
        .clone()
        .expect("clone dimension nuclear norm output");
    output
        .nuclear_norm_dimensions_out(&positive_definite, &[0, 1], false)
        .expect("execute nuclear_norm_dimensions_out");
    assert_tensor_eq(
        &output,
        &expected,
        cuda,
        "compare dimension nuclear norm output",
    );

    assert_eq!(
        values
            .to_device(0, -1)
            .expect("normalize final values")
            .to_packet()
            .expect("export final values"),
        values_before
    );
    assert_eq!(
        logical
            .to_device(0, -1)
            .expect("normalize final logical values")
            .to_packet()
            .expect("export final logical values"),
        logical_before
    );
    assert_eq!(
        positive_definite
            .to_device(0, -1)
            .expect("normalize final positive-definite matrix")
            .to_packet()
            .expect("export final positive-definite matrix"),
        positive_definite_before
    );
    assert_eq!(
        right_hand_side
            .to_device(0, -1)
            .expect("normalize final right-hand side")
            .to_packet()
            .expect("export final right-hand side"),
        right_hand_side_before
    );
}

#[test]
fn cpu_statistics_outputs_match_functional_results() {
    verify_statistics_outputs(0, -1, false);
}

#[test]
fn cuda_statistics_outputs_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_statistics_outputs(1, 0, true);
}
