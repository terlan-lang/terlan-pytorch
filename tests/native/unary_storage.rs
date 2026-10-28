use terlan_pytorch_cxx::Tensor;

macro_rules! assert_cpu_storage {
    ($source:expr, $functional:ident, $out:ident, $in_place:ident) => {{
        let expected = $source.$functional().expect("compute functional result");
        let mut output = $source.clone().expect("clone output storage");
        output.$out(&$source).expect("write output result");
        let mut mutation = $source.clone().expect("clone mutation storage");
        mutation.$in_place().expect("mutate in place");
        assert_eq!(
            output.to_packet().expect("export output result"),
            expected.to_packet().expect("export functional result")
        );
        assert_eq!(
            mutation.to_packet().expect("export mutation result"),
            expected.to_packet().expect("export functional result")
        );
    }};
}

macro_rules! assert_cuda_storage {
    ($source:expr, $functional:ident, $out:ident, $in_place:ident) => {{
        let expected = $source
            .$functional()
            .expect("compute CUDA functional result");
        let mut output = $source.clone().expect("clone CUDA output storage");
        output.$out(&$source).expect("write CUDA output result");
        let mut mutation = $source.clone().expect("clone CUDA mutation storage");
        mutation.$in_place().expect("mutate CUDA tensor");
        let expected = expected.to_device(0, -1).expect("copy expected to CPU");
        let output = output.to_device(0, -1).expect("copy output to CPU");
        let mutation = mutation.to_device(0, -1).expect("copy mutation to CPU");
        assert_eq!(
            output.to_packet().expect("export output result"),
            expected.to_packet().expect("export functional result")
        );
        assert_eq!(
            mutation.to_packet().expect("export mutation result"),
            expected.to_packet().expect("export functional result")
        );
    }};
}

macro_rules! assert_cpu_in_place {
    ($source:expr, $functional:ident, $in_place:ident) => {{
        let expected = $source.$functional().expect("compute functional result");
        let mut mutation = $source.clone().expect("clone mutation storage");
        mutation.$in_place().expect("mutate in place");
        assert_eq!(
            mutation.to_packet().expect("export mutation result"),
            expected.to_packet().expect("export functional result")
        );
    }};
}

macro_rules! assert_cuda_in_place {
    ($source:expr, $functional:ident, $in_place:ident) => {{
        let expected = $source
            .$functional()
            .expect("compute CUDA functional result");
        let mut mutation = $source.clone().expect("clone CUDA mutation storage");
        mutation.$in_place().expect("mutate CUDA tensor");
        let expected = expected.to_device(0, -1).expect("copy expected to CPU");
        let mutation = mutation.to_device(0, -1).expect("copy mutation to CPU");
        assert_eq!(
            mutation.to_packet().expect("export mutation result"),
            expected.to_packet().expect("export functional result")
        );
    }};
}

macro_rules! assert_cpu_out {
    ($source:expr, $functional:ident, $out:ident) => {{
        let expected = $source.$functional().expect("compute functional result");
        let mut output = expected.clone().expect("clone output storage");
        output.$out(&$source).expect("write output result");
        assert_eq!(
            output.to_packet().expect("export output result"),
            expected.to_packet().expect("export functional result")
        );
    }};
}

macro_rules! assert_cuda_out {
    ($source:expr, $functional:ident, $out:ident) => {{
        let expected = $source
            .$functional()
            .expect("compute CUDA functional result");
        let mut output = expected.clone().expect("clone CUDA output storage");
        output.$out(&$source).expect("write CUDA output result");
        let expected = expected.to_device(0, -1).expect("copy expected to CPU");
        let output = output.to_device(0, -1).expect("copy output to CPU");
        assert_eq!(
            output.to_packet().expect("export output result"),
            expected.to_packet().expect("export functional result")
        );
    }};
}

macro_rules! assert_cpu_backward_out {
    ($gradient:expr, $input:expr, $functional:ident, $out:ident) => {{
        let expected = $gradient
            .$functional(&$input)
            .expect("compute functional backward result");
        let mut output = expected.clone().expect("clone gradient output storage");
        output
            .$out(&$gradient, &$input)
            .expect("write gradient output");
        assert_eq!(
            output.to_packet().expect("export output gradient"),
            expected.to_packet().expect("export expected gradient")
        );
    }};
}

macro_rules! assert_cuda_backward_out {
    ($gradient:expr, $input:expr, $functional:ident, $out:ident) => {{
        let expected = $gradient
            .$functional(&$input)
            .expect("compute CUDA backward result");
        let mut output = expected.clone().expect("clone CUDA gradient storage");
        output
            .$out(&$gradient, &$input)
            .expect("write CUDA gradient output");
        let expected = expected.to_device(0, -1).expect("copy expected to CPU");
        let output = output.to_device(0, -1).expect("copy output to CPU");
        assert_eq!(
            output.to_packet().expect("export output gradient"),
            expected.to_packet().expect("export expected gradient")
        );
    }};
}

#[test]
fn cpu_output_and_in_place_forms_match_functional_results() {
    let zero = Tensor::from_float(0.0).expect("create zero");
    let one = Tensor::from_float(1.0).expect("create one");

    assert_cpu_storage!(one, acos, acos_out, acos_in_place);
    assert_cpu_storage!(one, acosh, acosh_out, acosh_in_place);
    assert_cpu_storage!(zero, asin, asin_out, asin_in_place);
    assert_cpu_storage!(zero, asinh, asinh_out, asinh_in_place);
    assert_cpu_storage!(zero, atan, atan_out, atan_in_place);
    assert_cpu_storage!(zero, atanh, atanh_out, atanh_in_place);
    assert_cpu_storage!(zero, cos, cos_out, cos_in_place);
    assert_cpu_storage!(zero, cosh, cosh_out, cosh_in_place);
    assert_cpu_storage!(zero, sin, sin_out, sin_in_place);
    assert_cpu_storage!(zero, sinh, sinh_out, sinh_in_place);
    assert_cpu_storage!(zero, tan, tan_out, tan_in_place);
    assert_cpu_storage!(zero, tanh, tanh_out, tanh_in_place);

    assert_eq!(zero.item_float().expect("read zero"), 0.0);
    assert_eq!(one.item_float().expect("read one"), 1.0);
}

#[test]
fn invalid_output_and_in_place_dtypes_fail_atomically() {
    let source = Tensor::from_float(0.0).expect("create source");
    let mut output = Tensor::from_int(7).expect("create invalid output");
    let output_error = output
        .sin_out(&source)
        .expect_err("integer output unexpectedly accepted floating result");
    let mut mutation = Tensor::from_int(7).expect("create invalid mutation");
    let mutation_error = mutation
        .acos_in_place()
        .expect_err("integer inverse cosine unexpectedly succeeded");

    assert_ne!(output_error.status, 0);
    assert_ne!(mutation_error.status, 0);
    assert_eq!(output.item().expect("read unchanged output"), 7);
    assert_eq!(mutation.item().expect("read unchanged mutation"), 7);
    assert_eq!(source.item_float().expect("read unchanged source"), 0.0);
}

#[test]
fn cuda_output_and_in_place_forms_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let zero = Tensor::from_float(0.0)
        .expect("create zero")
        .to_device(1, 0)
        .expect("place zero on CUDA");
    let one = Tensor::from_float(1.0)
        .expect("create one")
        .to_device(1, 0)
        .expect("place one on CUDA");

    assert_cuda_storage!(one, acos, acos_out, acos_in_place);
    assert_cuda_storage!(one, acosh, acosh_out, acosh_in_place);
    assert_cuda_storage!(zero, asin, asin_out, asin_in_place);
    assert_cuda_storage!(zero, asinh, asinh_out, asinh_in_place);
    assert_cuda_storage!(zero, atan, atan_out, atan_in_place);
    assert_cuda_storage!(zero, atanh, atanh_out, atanh_in_place);
    assert_cuda_storage!(zero, cos, cos_out, cos_in_place);
    assert_cuda_storage!(zero, cosh, cosh_out, cosh_in_place);
    assert_cuda_storage!(zero, sin, sin_out, sin_in_place);
    assert_cuda_storage!(zero, sinh, sinh_out, sinh_in_place);
    assert_cuda_storage!(zero, tan, tan_out, tan_in_place);
    assert_cuda_storage!(zero, tanh, tanh_out, tanh_in_place);
}

#[test]
fn remaining_cpu_storage_forms_match_functional_results() {
    let negative = Tensor::from_float(-2.0).expect("create negative");
    let fraction = Tensor::from_float(1.5).expect("create fraction");
    let zero = Tensor::from_float(0.0).expect("create zero");
    let two = Tensor::from_float(2.0).expect("create two");
    let three = Tensor::from_float(3.0).expect("create three");
    let four = Tensor::from_float(4.0).expect("create four");
    let eight = Tensor::from_float(8.0).expect("create eight");
    let hundred = Tensor::from_float(100.0).expect("create hundred");
    let integer = Tensor::from_int(1).expect("create integer");
    let boolean = Tensor::from_bool(false).expect("create boolean");

    assert_cpu_storage!(negative, abs, abs_out, abs_in_place);
    assert_cpu_storage!(integer, bitwise_not, bitwise_not_out, bitwise_not_in_place);
    assert_cpu_storage!(fraction, ceil, ceil_out, ceil_in_place);
    assert_cpu_storage!(zero, deg2rad, deg2rad_out, deg2rad_in_place);
    assert_cpu_storage!(zero, erf, erf_out, erf_in_place);
    assert_cpu_storage!(zero, erfc, erfc_out, erfc_in_place);
    assert_cpu_storage!(zero, erfinv, erfinv_out, erfinv_in_place);
    assert_cpu_storage!(three, exp2, exp2_out, exp2_in_place);
    assert_cpu_storage!(zero, expm1, expm1_out, expm1_in_place);
    assert_cpu_storage!(fraction, floor, floor_out, floor_in_place);
    assert_cpu_storage!(fraction, frac, frac_out, frac_in_place);
    assert_cpu_storage!(zero, hardsigmoid, hardsigmoid_out, hardsigmoid_in_place);
    assert_cpu_storage!(zero, hardswish, hardswish_out, hardswish_in_place);
    assert_cpu_storage!(hundred, log10, log10_out, log10_in_place);
    assert_cpu_storage!(zero, log1p, log1p_out, log1p_in_place);
    assert_cpu_storage!(eight, log2, log2_out, log2_in_place);
    assert_cpu_storage!(boolean, logical_not, logical_not_out, logical_not_in_place);
    assert_cpu_storage!(zero, mish, mish_out, mish_in_place);
    assert_cpu_storage!(zero, rad2deg, rad2deg_out, rad2deg_in_place);
    assert_cpu_storage!(two, reciprocal, reciprocal_out, reciprocal_in_place);
    assert_cpu_storage!(fraction, round, round_out, round_in_place);
    assert_cpu_storage!(four, rsqrt, rsqrt_out, rsqrt_in_place);
    assert_cpu_storage!(zero, sigmoid, sigmoid_out, sigmoid_in_place);
    assert_cpu_storage!(negative, sign, sign_out, sign_in_place);
    assert_cpu_storage!(zero, silu, silu_out, silu_in_place);
    assert_cpu_storage!(zero, sinc, sinc_out, sinc_in_place);
    assert_cpu_storage!(three, square, square_out, square_in_place);
    assert_cpu_storage!(fraction, trunc, trunc_out, trunc_in_place);
    assert_cpu_in_place!(negative, relu6, relu6_in_place);
    assert_cpu_in_place!(zero, selu, selu_in_place);
}

#[test]
fn remaining_cuda_storage_forms_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let cuda_float = |value| {
        Tensor::from_float(value)
            .expect("create float")
            .to_device(1, 0)
            .expect("place float on CUDA")
    };
    let negative = cuda_float(-2.0);
    let fraction = cuda_float(1.5);
    let zero = cuda_float(0.0);
    let two = cuda_float(2.0);
    let three = cuda_float(3.0);
    let four = cuda_float(4.0);
    let eight = cuda_float(8.0);
    let hundred = cuda_float(100.0);
    let integer = Tensor::from_int(1)
        .expect("create integer")
        .to_device(1, 0)
        .expect("place integer on CUDA");
    let boolean = Tensor::from_bool(false)
        .expect("create boolean")
        .to_device(1, 0)
        .expect("place boolean on CUDA");

    assert_cuda_storage!(negative, abs, abs_out, abs_in_place);
    assert_cuda_storage!(integer, bitwise_not, bitwise_not_out, bitwise_not_in_place);
    assert_cuda_storage!(fraction, ceil, ceil_out, ceil_in_place);
    assert_cuda_storage!(zero, deg2rad, deg2rad_out, deg2rad_in_place);
    assert_cuda_storage!(zero, erf, erf_out, erf_in_place);
    assert_cuda_storage!(zero, erfc, erfc_out, erfc_in_place);
    assert_cuda_storage!(zero, erfinv, erfinv_out, erfinv_in_place);
    assert_cuda_storage!(three, exp2, exp2_out, exp2_in_place);
    assert_cuda_storage!(zero, expm1, expm1_out, expm1_in_place);
    assert_cuda_storage!(fraction, floor, floor_out, floor_in_place);
    assert_cuda_storage!(fraction, frac, frac_out, frac_in_place);
    assert_cuda_storage!(zero, hardsigmoid, hardsigmoid_out, hardsigmoid_in_place);
    assert_cuda_storage!(zero, hardswish, hardswish_out, hardswish_in_place);
    assert_cuda_storage!(hundred, log10, log10_out, log10_in_place);
    assert_cuda_storage!(zero, log1p, log1p_out, log1p_in_place);
    assert_cuda_storage!(eight, log2, log2_out, log2_in_place);
    assert_cuda_storage!(boolean, logical_not, logical_not_out, logical_not_in_place);
    assert_cuda_storage!(zero, mish, mish_out, mish_in_place);
    assert_cuda_storage!(zero, rad2deg, rad2deg_out, rad2deg_in_place);
    assert_cuda_storage!(two, reciprocal, reciprocal_out, reciprocal_in_place);
    assert_cuda_storage!(fraction, round, round_out, round_in_place);
    assert_cuda_storage!(four, rsqrt, rsqrt_out, rsqrt_in_place);
    assert_cuda_storage!(zero, sigmoid, sigmoid_out, sigmoid_in_place);
    assert_cuda_storage!(negative, sign, sign_out, sign_in_place);
    assert_cuda_storage!(zero, silu, silu_out, silu_in_place);
    assert_cuda_storage!(zero, sinc, sinc_out, sinc_in_place);
    assert_cuda_storage!(three, square, square_out, square_in_place);
    assert_cuda_storage!(fraction, trunc, trunc_out, trunc_in_place);
    assert_cuda_in_place!(negative, relu6, relu6_in_place);
    assert_cuda_in_place!(zero, selu, selu_in_place);
}

#[test]
fn residual_cpu_unary_contracts_match_functional_results() {
    let zero = Tensor::from_float(0.0).expect("create zero");
    let one = Tensor::from_float(1.0).expect("create one");
    let truth = Tensor::from_bool(true).expect("create truth");
    let negative = Tensor::from_float(-1.0).expect("create negative");
    let infinity = Tensor::from_float(f64::INFINITY).expect("create infinity");
    let negative_infinity =
        Tensor::from_float(f64::NEG_INFINITY).expect("create negative infinity");
    let nan = Tensor::from_float(f64::NAN).expect("create NaN");

    assert_cpu_storage!(negative, absolute, absolute_out, absolute_in_place);
    assert_cpu_storage!(one, arccos, arccos_out, arccos_in_place);
    assert_cpu_storage!(one, arccosh, arccosh_out, arccosh_in_place);
    assert_cpu_storage!(zero, arcsin, arcsin_out, arcsin_in_place);
    assert_cpu_storage!(zero, arcsinh, arcsinh_out, arcsinh_in_place);
    assert_cpu_storage!(zero, arctan, arctan_out, arctan_in_place);
    assert_cpu_storage!(zero, arctanh, arctanh_out, arctanh_in_place);
    assert_cpu_out!(truth, all, all_all_out);
    assert_cpu_out!(truth, any, any_all_out);
    assert_cpu_out!(infinity, isinf, isinf_out);
    assert_cpu_out!(nan, isnan, isnan_out);
    assert_cpu_out!(negative_infinity, isneginf, isneginf_out);
    assert_cpu_out!(infinity, isposinf, isposinf_out);
    assert_cpu_out!(negative, signbit, signbit_out);

    let values = Tensor::from_floats(&[3.0, 1.0, 2.0], &[3]).expect("create values");
    assert_cpu_out!(values, max, max_unary_out);
    assert_cpu_out!(values, median, median_out);
    assert_cpu_out!(values, min, min_unary_out);
    assert_cpu_out!(values, nonzero, nonzero_out);
    let matrix = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create matrix");
    assert_cpu_out!(matrix, trace, trace_out);

    let decimal = Tensor::from_float(1.234).expect("create decimal");
    let expected_decimal = decimal.round_decimals(2).expect("round decimal");
    let mut decimal_output = expected_decimal.clone().expect("clone decimal output");
    decimal_output
        .round_decimals_out(&decimal, 2)
        .expect("write decimal output");
    let mut decimal_mutation = decimal.clone().expect("clone decimal mutation");
    decimal_mutation
        .round_decimals_in_place(2)
        .expect("round decimal in place");
    assert_eq!(
        decimal_output.to_packet().expect("export decimal output"),
        expected_decimal
            .to_packet()
            .expect("export expected decimal")
    );
    assert_eq!(
        decimal_mutation
            .to_packet()
            .expect("export decimal mutation"),
        expected_decimal
            .to_packet()
            .expect("export expected decimal")
    );

    let gradient = Tensor::from_float(1.0).expect("create gradient");
    assert_cpu_backward_out!(
        gradient,
        zero,
        hardsigmoid_backward,
        hardsigmoid_backward_out
    );
    assert_cpu_backward_out!(gradient, zero, hardswish_backward, hardswish_backward_out);
    let sigmoid_output = zero.sigmoid().expect("compute sigmoid output");
    assert_cpu_backward_out!(
        gradient,
        sigmoid_output,
        sigmoid_backward,
        sigmoid_backward_out
    );
    assert_cpu_backward_out!(gradient, zero, silu_backward, silu_backward_out);
    let tanh_output = zero.tanh().expect("compute tanh output");
    assert_cpu_backward_out!(gradient, tanh_output, tanh_backward, tanh_backward_out);

    let eight = Tensor::from_float(8.0).expect("create decomposition input");
    let (expected_mantissa, expected_exponent) = eight.frexp().expect("decompose input");
    let mut mantissa = expected_mantissa.clone().expect("clone mantissa output");
    let mut exponent = expected_exponent.clone().expect("clone exponent output");
    mantissa
        .frexp_out(&mut exponent, &eight)
        .expect("write decomposition outputs");
    assert_eq!(
        mantissa.to_packet().expect("export mantissa"),
        expected_mantissa
            .to_packet()
            .expect("export expected mantissa")
    );
    assert_eq!(
        exponent.to_packet().expect("export exponent"),
        expected_exponent
            .to_packet()
            .expect("export expected exponent")
    );
}

#[test]
fn residual_cuda_unary_contracts_match_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let cuda_float = |value| {
        Tensor::from_float(value)
            .expect("create float")
            .to_device(1, 0)
            .expect("place float on CUDA")
    };
    let zero = cuda_float(0.0);
    let one = cuda_float(1.0);
    let negative = cuda_float(-1.0);
    let infinity = cuda_float(f64::INFINITY);
    let negative_infinity = cuda_float(f64::NEG_INFINITY);
    let nan = cuda_float(f64::NAN);
    let truth = Tensor::from_bool(true)
        .expect("create truth")
        .to_device(1, 0)
        .expect("place truth on CUDA");

    assert_cuda_storage!(negative, absolute, absolute_out, absolute_in_place);
    assert_cuda_storage!(one, arccos, arccos_out, arccos_in_place);
    assert_cuda_storage!(one, arccosh, arccosh_out, arccosh_in_place);
    assert_cuda_storage!(zero, arcsin, arcsin_out, arcsin_in_place);
    assert_cuda_storage!(zero, arcsinh, arcsinh_out, arcsinh_in_place);
    assert_cuda_storage!(zero, arctan, arctan_out, arctan_in_place);
    assert_cuda_storage!(zero, arctanh, arctanh_out, arctanh_in_place);
    assert_cuda_out!(truth, all, all_all_out);
    assert_cuda_out!(truth, any, any_all_out);
    assert_cuda_out!(infinity, isinf, isinf_out);
    assert_cuda_out!(nan, isnan, isnan_out);
    assert_cuda_out!(negative_infinity, isneginf, isneginf_out);
    assert_cuda_out!(infinity, isposinf, isposinf_out);
    assert_cuda_out!(negative, signbit, signbit_out);

    let values = Tensor::from_floats(&[3.0, 1.0, 2.0], &[3])
        .expect("create values")
        .to_device(1, 0)
        .expect("place values on CUDA");
    assert_cuda_out!(values, max, max_unary_out);
    assert_cuda_out!(values, median, median_out);
    assert_cuda_out!(values, min, min_unary_out);
    assert_cuda_out!(values, nonzero, nonzero_out);
    let matrix = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create matrix")
        .to_device(1, 0)
        .expect("place matrix on CUDA");
    assert_cuda_out!(matrix, trace, trace_out);

    let decimal = cuda_float(1.234);
    let expected_decimal = decimal.round_decimals(2).expect("round CUDA decimal");
    let mut decimal_output = expected_decimal.clone().expect("clone CUDA decimal output");
    decimal_output
        .round_decimals_out(&decimal, 2)
        .expect("write CUDA decimal output");
    let mut decimal_mutation = decimal.clone().expect("clone CUDA decimal mutation");
    decimal_mutation
        .round_decimals_in_place(2)
        .expect("round CUDA decimal in place");
    assert_eq!(
        decimal_output
            .to_device(0, -1)
            .expect("copy output")
            .to_packet()
            .expect("export output"),
        expected_decimal
            .to_device(0, -1)
            .expect("copy expected")
            .to_packet()
            .expect("export expected")
    );
    assert_eq!(
        decimal_mutation
            .to_device(0, -1)
            .expect("copy mutation")
            .to_packet()
            .expect("export mutation"),
        expected_decimal
            .to_device(0, -1)
            .expect("copy expected")
            .to_packet()
            .expect("export expected")
    );

    let gradient = cuda_float(1.0);
    assert_cuda_backward_out!(
        gradient,
        zero,
        hardsigmoid_backward,
        hardsigmoid_backward_out
    );
    assert_cuda_backward_out!(gradient, zero, hardswish_backward, hardswish_backward_out);
    let sigmoid_output = zero.sigmoid().expect("compute CUDA sigmoid output");
    assert_cuda_backward_out!(
        gradient,
        sigmoid_output,
        sigmoid_backward,
        sigmoid_backward_out
    );
    assert_cuda_backward_out!(gradient, zero, silu_backward, silu_backward_out);
    let tanh_output = zero.tanh().expect("compute CUDA tanh output");
    assert_cuda_backward_out!(gradient, tanh_output, tanh_backward, tanh_backward_out);

    let eight = cuda_float(8.0);
    let (expected_mantissa, expected_exponent) = eight.frexp().expect("decompose CUDA input");
    let mut mantissa = expected_mantissa.clone().expect("clone CUDA mantissa");
    let mut exponent = expected_exponent.clone().expect("clone CUDA exponent");
    mantissa
        .frexp_out(&mut exponent, &eight)
        .expect("write CUDA decomposition");
    assert_eq!(
        mantissa
            .to_device(0, -1)
            .expect("copy mantissa")
            .to_packet()
            .expect("export mantissa"),
        expected_mantissa
            .to_device(0, -1)
            .expect("copy expected mantissa")
            .to_packet()
            .expect("export expected mantissa")
    );
    assert_eq!(
        exponent
            .to_device(0, -1)
            .expect("copy exponent")
            .to_packet()
            .expect("export exponent"),
        expected_exponent
            .to_device(0, -1)
            .expect("copy expected exponent")
            .to_packet()
            .expect("export expected exponent")
    );
}

#[test]
fn final_cpu_unary_storage_family_matches_functional_results() {
    let one = Tensor::from_float(1.0).expect("create one");
    let fraction = Tensor::from_float(1.75).expect("create fraction");
    let two = Tensor::from_float(2.0).expect("create two");

    assert_cpu_out!(one, alias_copy, alias_copy_out);
    assert_cpu_out!(one, angle, angle_out);
    assert_cpu_storage!(
        one,
        conj_physical,
        conj_physical_out,
        conj_physical_in_place
    );
    assert_cpu_out!(one, detach_copy, detach_copy_out);
    assert_cpu_storage!(one, exp, exp_out, exp_in_place);
    assert_cpu_storage!(fraction, fix, fix_out, fix_in_place);
    assert_cpu_storage!(one, i0, i0_out, i0_in_place);
    assert_cpu_storage!(two, lgamma, lgamma_out, lgamma_in_place);
    assert_cpu_out!(one, lift, lift_out);
    assert_cpu_out!(one, lift_fresh_copy, lift_fresh_copy_out);
    assert_cpu_storage!(two, log, log_out, log_in_place);
}

#[test]
fn final_cuda_unary_storage_family_matches_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let cuda_float = |value| {
        Tensor::from_float(value)
            .expect("create CPU source")
            .to_device(1, 0)
            .expect("place source on CUDA")
    };
    let one = cuda_float(1.0);
    let fraction = cuda_float(1.75);
    let two = cuda_float(2.0);

    assert_cuda_out!(one, alias_copy, alias_copy_out);
    assert_cuda_out!(one, angle, angle_out);
    assert_cuda_storage!(
        one,
        conj_physical,
        conj_physical_out,
        conj_physical_in_place
    );
    assert_cuda_out!(one, detach_copy, detach_copy_out);
    assert_cuda_storage!(one, exp, exp_out, exp_in_place);
    assert_cuda_storage!(fraction, fix, fix_out, fix_in_place);
    assert_cuda_storage!(one, i0, i0_out, i0_in_place);
    assert_cuda_storage!(two, lgamma, lgamma_out, lgamma_in_place);
    assert_cuda_out!(one, lift, lift_out);
    assert_cuda_out!(one, lift_fresh_copy, lift_fresh_copy_out);
    assert_cuda_storage!(two, log, log_out, log_in_place);
}

#[test]
fn retained_output_completion_family_matches_cpu_functional_results() {
    let scalar = Tensor::from_float(2.0).expect("create scalar");
    let negative = Tensor::from_float(-2.0).expect("create negative scalar");
    let four = Tensor::from_float(4.0).expect("create square-root source");
    let values = Tensor::from_floats(&[3.0, 1.0, 2.0], &[3]).expect("create sortable values");
    let matrix =
        Tensor::from_floats(&[4.0, 7.0, 2.0, 6.0], &[2, 2]).expect("create invertible matrix");
    let real_pairs = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create real and imaginary pairs");
    let complex = real_pairs
        .view_as_complex_copy()
        .expect("create complex tensor");

    assert_cpu_out!(scalar, digamma, digamma_out);
    assert_cpu_out!(matrix, inverse, inverse_out);
    assert_cpu_out!(scalar, log_sigmoid, log_sigmoid_out);
    assert_cpu_out!(values, msort, msort_out);
    assert_cpu_out!(negative, negate, neg_out);
    assert_cpu_out!(negative, negative, negative_out);
    assert_cpu_out!(negative, relu, relu_out);
    assert_cpu_out!(negative, sgn, sgn_out);
    assert_cpu_out!(four, sqrt, sqrt_out);
    assert_cpu_out!(matrix, t_copy, t_copy_out);
    let expected_complex = real_pairs
        .view_as_complex_copy()
        .expect("compute complex view copy");
    let mut output_complex = expected_complex
        .clone()
        .expect("clone complex output storage");
    output_complex
        .view_as_complex_copy_out(&real_pairs)
        .expect("write complex view copy");
    assert_eq!(
        output_complex
            .view_as_real_copy()
            .expect("convert output to real")
            .to_packet()
            .expect("export real output"),
        expected_complex
            .view_as_real_copy()
            .expect("convert expected to real")
            .to_packet()
            .expect("export expected real output")
    );
    assert_cpu_out!(complex, view_as_real_copy, view_as_real_copy_out);
    assert_cpu_in_place!(scalar, digamma, digamma_in_place);
    assert_cpu_in_place!(negative, negate, neg_in_place);
    assert_cpu_in_place!(negative, negative, negative_in_place);
    assert_cpu_in_place!(negative, relu, relu_in_place);
    assert_cpu_in_place!(negative, sgn, sgn_in_place);
    assert_cpu_in_place!(four, sqrt, sqrt_in_place);
    assert_cpu_in_place!(matrix, t, t_in_place);
}

#[test]
fn retained_output_completion_family_matches_cuda_functional_results_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let cuda_float = |value| {
        Tensor::from_float(value)
            .expect("create CPU scalar")
            .to_device(1, 0)
            .expect("place scalar on CUDA")
    };
    let scalar = cuda_float(2.0);
    let negative = cuda_float(-2.0);
    let four = cuda_float(4.0);
    let values = Tensor::from_floats(&[3.0, 1.0, 2.0], &[3])
        .expect("create sortable values")
        .to_device(1, 0)
        .expect("place sortable values on CUDA");
    let matrix = Tensor::from_floats(&[4.0, 7.0, 2.0, 6.0], &[2, 2])
        .expect("create invertible matrix")
        .to_device(1, 0)
        .expect("place matrix on CUDA");
    let real_pairs = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("create real and imaginary pairs")
        .to_device(1, 0)
        .expect("place pairs on CUDA");
    let complex = real_pairs
        .view_as_complex_copy()
        .expect("create CUDA complex tensor");

    assert_cuda_out!(scalar, digamma, digamma_out);
    assert_cuda_out!(matrix, inverse, inverse_out);
    assert_cuda_out!(scalar, log_sigmoid, log_sigmoid_out);
    assert_cuda_out!(values, msort, msort_out);
    assert_cuda_out!(negative, negate, neg_out);
    assert_cuda_out!(negative, negative, negative_out);
    assert_cuda_out!(negative, relu, relu_out);
    assert_cuda_out!(negative, sgn, sgn_out);
    assert_cuda_out!(four, sqrt, sqrt_out);
    assert_cuda_out!(matrix, t_copy, t_copy_out);
    let expected_complex = real_pairs
        .view_as_complex_copy()
        .expect("compute CUDA complex view copy");
    let mut output_complex = expected_complex
        .clone()
        .expect("clone CUDA complex output storage");
    output_complex
        .view_as_complex_copy_out(&real_pairs)
        .expect("write CUDA complex view copy");
    assert_eq!(
        output_complex
            .view_as_real_copy()
            .expect("convert CUDA output to real")
            .to_device(0, -1)
            .expect("copy real output to CPU")
            .to_packet()
            .expect("export real output"),
        expected_complex
            .view_as_real_copy()
            .expect("convert CUDA expected to real")
            .to_device(0, -1)
            .expect("copy expected real output to CPU")
            .to_packet()
            .expect("export expected real output")
    );
    assert_cuda_out!(complex, view_as_real_copy, view_as_real_copy_out);
    assert_cuda_in_place!(scalar, digamma, digamma_in_place);
    assert_cuda_in_place!(negative, negate, neg_in_place);
    assert_cuda_in_place!(negative, negative, negative_in_place);
    assert_cuda_in_place!(negative, relu, relu_in_place);
    assert_cuda_in_place!(negative, sgn, sgn_in_place);
    assert_cuda_in_place!(four, sqrt, sqrt_in_place);
    assert_cuda_in_place!(matrix, t, t_in_place);
}

#[test]
fn retained_output_completion_rejects_invalid_storage_atomically() {
    let source = Tensor::from_float(2.0).expect("create source");
    let source_before = source.to_packet().expect("snapshot source");
    let mut output = Tensor::from_int(17).expect("create incompatible output");
    let output_before = output.to_packet().expect("snapshot output");

    let error = output
        .digamma_out(&source)
        .expect_err("integer output unexpectedly accepted digamma result");

    assert_ne!(error.status, 0);
    assert_eq!(source.to_packet().expect("export source"), source_before);
    assert_eq!(output.to_packet().expect("export output"), output_before);

    let invalid_pairs =
        Tensor::from_floats(&[1.0, 2.0, 3.0], &[3]).expect("create invalid complex-view source");
    let invalid_before = invalid_pairs.to_packet().expect("snapshot invalid source");
    let mut complex_output = Tensor::from_float(23.0).expect("create complex output");
    let complex_output_before = complex_output.to_packet().expect("snapshot complex output");
    let error = complex_output
        .view_as_complex_copy_out(&invalid_pairs)
        .expect_err("invalid trailing dimension unexpectedly produced a complex view");

    assert_ne!(error.status, 0);
    assert_eq!(
        invalid_pairs.to_packet().expect("export invalid source"),
        invalid_before
    );
    assert_eq!(
        complex_output.to_packet().expect("export complex output"),
        complex_output_before
    );

    let mut invalid_mutation = Tensor::from_int(29).expect("create invalid mutation");
    let invalid_mutation_before = invalid_mutation.to_packet().expect("snapshot mutation");
    let error = invalid_mutation
        .digamma_in_place()
        .expect_err("integer digamma mutation unexpectedly succeeded");

    assert_ne!(error.status, 0);
    assert_eq!(
        invalid_mutation.to_packet().expect("export mutation"),
        invalid_mutation_before
    );
}
