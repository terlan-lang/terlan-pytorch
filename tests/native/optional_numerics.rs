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

/// Proves cumulative reductions preserve omitted and explicit dtype behavior.
fn verify_cumulative_variants(device_type: i64, device_index: i64, cuda: bool) {
    let values = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create cumulative input")
        .to_device(device_type, device_index)
        .expect("place cumulative input");

    let expected_sum = values.cumsum(1).expect("compute cumulative sum");
    let mut sum_output = values.clone().expect("clone sum output");
    sum_output
        .cumsum_out(&values, 1)
        .expect("write cumulative sum output");
    let mut sum_mutation = values.clone().expect("clone sum mutation");
    sum_mutation
        .cumsum_in_place(1)
        .expect("mutate cumulative sum");
    let expected_sum_dtype = values
        .cumsum_dtype(1, 7)
        .expect("compute typed cumulative sum");
    let mut sum_dtype_output = values.clone().expect("clone typed sum output");
    sum_dtype_output
        .cumsum_out_dtype(&values, 1, 7)
        .expect("write typed cumulative sum output");
    let mut sum_dtype_mutation = values.clone().expect("clone typed sum mutation");
    sum_dtype_mutation
        .cumsum_in_place_dtype(1, 7)
        .expect("mutate typed cumulative sum");

    let expected_product = values.cumprod(1).expect("compute cumulative product");
    let mut product_output = values.clone().expect("clone product output");
    product_output
        .cumprod_out(&values, 1)
        .expect("write cumulative product output");
    let mut product_mutation = values.clone().expect("clone product mutation");
    product_mutation
        .cumprod_in_place(1)
        .expect("mutate cumulative product");
    let expected_product_dtype = values
        .cumprod_dtype(1, 7)
        .expect("compute typed cumulative product");
    let mut product_dtype_output = values.clone().expect("clone typed product output");
    product_dtype_output
        .cumprod_out_dtype(&values, 1, 7)
        .expect("write typed cumulative product output");
    let mut product_dtype_mutation = values.clone().expect("clone typed product mutation");
    product_dtype_mutation
        .cumprod_in_place_dtype(1, 7)
        .expect("mutate typed cumulative product");

    for (actual, expected, context) in [
        (&sum_output, &expected_sum, "compare cumulative sum output"),
        (
            &sum_mutation,
            &expected_sum,
            "compare cumulative sum mutation",
        ),
        (
            &sum_dtype_output,
            &expected_sum_dtype,
            "compare typed cumulative sum output",
        ),
        (
            &sum_dtype_mutation,
            &expected_sum_dtype,
            "compare typed cumulative sum mutation",
        ),
        (
            &product_output,
            &expected_product,
            "compare cumulative product output",
        ),
        (
            &product_mutation,
            &expected_product,
            "compare cumulative product mutation",
        ),
        (
            &product_dtype_output,
            &expected_product_dtype,
            "compare typed cumulative product output",
        ),
        (
            &product_dtype_mutation,
            &expected_product_dtype,
            "compare typed cumulative product mutation",
        ),
    ] {
        assert_tensor_eq(actual, expected, cuda, context);
    }
}

/// Proves GELU and polygamma retained-output and mutation behavior.
fn verify_approximation_and_order(device_type: i64, device_index: i64, cuda: bool) {
    let input = Tensor::from_floats(&[-1.0, 0.0, 1.0], &[3])
        .expect("create GELU input")
        .to_device(device_type, device_index)
        .expect("place GELU input");
    let expected_gelu = input.gelu("tanh").expect("compute tanh GELU");
    let mut gelu_output = input.clone().expect("clone GELU output");
    gelu_output
        .gelu_out(&input, "tanh")
        .expect("write tanh GELU output");
    let mut gelu_mutation = input.clone().expect("clone GELU mutation");
    gelu_mutation
        .gelu_in_place("tanh")
        .expect("mutate tanh GELU");
    assert_tensor_eq(&gelu_output, &expected_gelu, cuda, "compare GELU output");
    assert_tensor_eq(
        &gelu_mutation,
        &expected_gelu,
        cuda,
        "compare GELU mutation",
    );

    let gradient = Tensor::from_floats(&[1.0, 1.0, 1.0], &[3])
        .expect("create GELU gradient")
        .to_device(device_type, device_index)
        .expect("place GELU gradient");
    let expected_gradient = gradient
        .gelu_backward(&input, "tanh")
        .expect("compute GELU gradient");
    let mut gradient_output = input.clone().expect("clone gradient output");
    gradient_output
        .gelu_backward_out(&gradient, &input, "tanh")
        .expect("write GELU gradient output");
    assert_tensor_eq(
        &gradient_output,
        &expected_gradient,
        cuda,
        "compare GELU gradient output",
    );

    let positive = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3])
        .expect("create polygamma input")
        .to_device(device_type, device_index)
        .expect("place polygamma input");
    let expected_polygamma = positive.polygamma(1).expect("compute polygamma");
    let mut polygamma_output = positive.clone().expect("clone polygamma output");
    polygamma_output
        .polygamma_out(&positive, 1)
        .expect("write polygamma output");
    let mut polygamma_mutation = positive.clone().expect("clone polygamma mutation");
    polygamma_mutation
        .polygamma_in_place(1)
        .expect("mutate polygamma");
    assert_tensor_eq(
        &polygamma_output,
        &expected_polygamma,
        cuda,
        "compare polygamma output",
    );
    assert_tensor_eq(
        &polygamma_mutation,
        &expected_polygamma,
        cuda,
        "compare polygamma mutation",
    );
}

/// Proves logit optional epsilon behavior across forward and backward storage forms.
fn verify_logit_variants(device_type: i64, device_index: i64, cuda: bool) {
    let probabilities = Tensor::from_floats(&[0.2, 0.5, 0.8], &[3])
        .expect("create probabilities")
        .to_device(device_type, device_index)
        .expect("place probabilities");
    let expected = probabilities.logit().expect("compute unclamped logit");
    let mut output = probabilities.clone().expect("clone logit output");
    output
        .logit_out(&probabilities)
        .expect("write unclamped logit output");
    let mut mutation = probabilities.clone().expect("clone logit mutation");
    mutation.logit_in_place().expect("mutate unclamped logit");
    assert_tensor_eq(&output, &expected, cuda, "compare logit output");
    assert_tensor_eq(&mutation, &expected, cuda, "compare logit mutation");

    let clamped_input = Tensor::from_floats(&[0.0, 0.5, 1.0], &[3])
        .expect("create clamped probabilities")
        .to_device(device_type, device_index)
        .expect("place clamped probabilities");
    let expected_clamped = clamped_input
        .logit_epsilon(0.1)
        .expect("compute clamped logit");
    let mut clamped_output = probabilities.clone().expect("clone clamped output");
    clamped_output
        .logit_epsilon_out(&clamped_input, 0.1)
        .expect("write clamped logit output");
    let mut clamped_mutation = clamped_input.clone().expect("clone clamped mutation");
    clamped_mutation
        .logit_epsilon_in_place(0.1)
        .expect("mutate clamped logit");
    assert_tensor_eq(
        &clamped_output,
        &expected_clamped,
        cuda,
        "compare clamped logit output",
    );
    assert_tensor_eq(
        &clamped_mutation,
        &expected_clamped,
        cuda,
        "compare clamped logit mutation",
    );

    let gradient = Tensor::from_floats(&[1.0, 2.0, 3.0], &[3])
        .expect("create logit gradient")
        .to_device(device_type, device_index)
        .expect("place logit gradient");
    let expected_backward = gradient
        .logit_backward(&probabilities)
        .expect("compute unclamped logit backward");
    let mut backward_output = probabilities.clone().expect("clone backward output");
    backward_output
        .logit_backward_out(&gradient, &probabilities)
        .expect("write unclamped logit backward output");
    let expected_backward_clamped = gradient
        .logit_backward_epsilon(&probabilities, 0.1)
        .expect("compute clamped logit backward");
    let mut backward_clamped_output = probabilities
        .clone()
        .expect("clone clamped backward output");
    backward_clamped_output
        .logit_backward_epsilon_out(&gradient, &probabilities, 0.1)
        .expect("write clamped logit backward output");
    assert_tensor_eq(
        &backward_output,
        &expected_backward,
        cuda,
        "compare logit backward output",
    );
    assert_tensor_eq(
        &backward_clamped_output,
        &expected_backward_clamped,
        cuda,
        "compare clamped logit backward output",
    );
}

/// Proves all optional numerical paths on CPU.
#[test]
fn cpu_optional_numerics_match_functional_results() {
    verify_cumulative_variants(0, -1, false);
    verify_approximation_and_order(0, -1, false);
    verify_logit_variants(0, -1, false);
}

/// Proves all optional numerical paths on mandatory CUDA hardware.
#[test]
fn cuda_optional_numerics_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }
    verify_cumulative_variants(1, 0, true);
    verify_approximation_and_order(1, 0, true);
    verify_logit_variants(1, 0, true);
}

/// Proves invalid string and dtype options report errors without mutating storage.
#[test]
fn invalid_options_fail_atomically() {
    let mut gelu_input =
        Tensor::from_floats(&[-1.0, 0.0, 1.0], &[3]).expect("create invalid GELU input");
    let gelu_before = gelu_input.to_packet().expect("snapshot GELU input");
    let gelu_error = gelu_input
        .gelu_in_place("unsupported")
        .expect_err("invalid GELU approximation unexpectedly succeeded");
    assert_ne!(gelu_error.status, 0);
    assert_eq!(
        gelu_input.to_packet().expect("export GELU input"),
        gelu_before
    );

    let mut cumulative =
        Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create invalid cumulative input");
    let cumulative_before = cumulative.to_packet().expect("snapshot cumulative input");
    let dtype_error = cumulative
        .cumsum_in_place_dtype(0, 999)
        .expect_err("invalid cumulative dtype unexpectedly succeeded");
    assert_ne!(dtype_error.status, 0);
    assert_eq!(
        cumulative.to_packet().expect("export cumulative input"),
        cumulative_before
    );
}
