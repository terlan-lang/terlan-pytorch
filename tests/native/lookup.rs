use terlan_pytorch_cxx::{CAbiError, Tensor};

fn sum_int(tensor: &Tensor) -> i64 {
    tensor.sum().expect("sum tensor").item().expect("read sum")
}

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn membership_forms_preserve_direction_flags_and_outputs() {
    let tests = Tensor::from_ints(&[1, 3, 5], &[3]).expect("create tests");
    let scalar_member = tests
        .isin_scalar(1, true, 3, 0.0, false, false)
        .expect("scalar in tensor");
    let scalar_missing = tests
        .isin_scalar(1, false, 0, 2.5, false, true)
        .expect("inverted scalar in tensor");
    let elements = Tensor::from_ints(&[1, 2, 1], &[3]).expect("create elements");
    let tensor_scalar = elements
        .isin_scalar(0, true, 1, 0.0, false, false)
        .expect("tensor against scalar");
    let mut scalar_output =
        Tensor::empty_typed_native(&[0], 11, 0, -1).expect("create scalar output");
    scalar_output
        .isin_scalar_out(&tests, 1, true, 3, 0.0, false, false)
        .expect("scalar membership output");
    let mut tensor_scalar_output =
        Tensor::empty_typed_native(&[3], 11, 0, -1).expect("create tensor scalar output");
    tensor_scalar_output
        .isin_scalar_out(&elements, 0, true, 1, 0.0, false, false)
        .expect("tensor scalar output");
    let mut tensor_output = Tensor::empty_typed_native(&[3], 11, 0, -1).expect("create output");
    tensor_output
        .isin_tensor_out(&elements, &tests, false, false)
        .expect("tensor membership output");

    assert!(scalar_member.item_bool().expect("read scalar member"));
    assert!(scalar_missing.item_bool().expect("read scalar missing"));
    assert!(scalar_output.item_bool().expect("read scalar output"));
    assert_eq!(sum_int(&tensor_scalar), 2);
    assert_eq!(sum_int(&tensor_scalar_output), 2);
    assert_eq!(sum_int(&tensor_output), 2);
    assert_eq!(sum_int(&elements), 4);
    assert_eq!(sum_int(&tests), 9);
}

#[test]
fn searchsorted_and_bucketize_cover_side_sorter_and_width() {
    let sorted = Tensor::from_ints(&[1, 3, 5, 7], &[4]).expect("create sorted sequence");
    let values = Tensor::from_ints(&[0, 3, 8], &[3]).expect("create values");
    let left = sorted
        .searchsorted_tensor(&values, false, false, 1, &sorted, false)
        .expect("left tensor search");
    let right = sorted
        .searchsorted_tensor(&values, true, true, 2, &sorted, false)
        .expect("right tensor search");
    let mut tensor_output = Tensor::from_ints(&[0, 0, 0], &[3]).expect("create output");
    tensor_output
        .searchsorted_tensor_out(&sorted, &values, false, false, 0, &sorted, false)
        .expect("tensor search output");
    let unsorted = Tensor::from_ints(&[5, 1, 7, 3], &[4]).expect("create unsorted sequence");
    let sorter = Tensor::from_ints(&[1, 3, 0, 2], &[4]).expect("create sorter");
    let sorted_values = Tensor::from_ints(&[3, 6], &[2]).expect("create sorted values");
    let with_sorter = unsorted
        .searchsorted_tensor(&sorted_values, false, false, 1, &sorter, true)
        .expect("search with sorter");
    let scalar = sorted
        .searchsorted_scalar(true, 3, 0.0, false, false, 1, &sorted, false)
        .expect("integer scalar search");
    let mut scalar_output = Tensor::from_int(0).expect("create scalar output");
    scalar_output
        .searchsorted_scalar_out(&sorted, false, 0, 3.0, false, true, 2, &sorted, false)
        .expect("scalar search output");
    let mut bucket_output = Tensor::from_ints(&[0, 0, 0], &[3]).expect("create bucket output");
    bucket_output
        .bucketize_tensor_out(&values, &sorted, false, false)
        .expect("tensor bucket output");
    let bucket = sorted
        .bucketize_scalar(true, 3, 0.0, false, false)
        .expect("integer scalar bucket");
    let mut bucket_scalar_output = Tensor::from_int(0).expect("create bucket scalar output");
    bucket_scalar_output
        .bucketize_scalar_out(&sorted, false, 0, 3.0, false, true)
        .expect("scalar bucket output");

    assert_eq!(sum_int(&left), 5);
    assert_eq!(sum_int(&right), 6);
    assert_eq!(right.dtype().expect("right dtype"), 3);
    assert_eq!(sum_int(&tensor_output), 5);
    assert_eq!(sum_int(&with_sorter), 4);
    assert_eq!(scalar.item().expect("scalar search value"), 1);
    assert_eq!(scalar_output.item().expect("scalar output value"), 2);
    assert_eq!(sum_int(&bucket_output), 5);
    assert_eq!(bucket.item().expect("bucket value"), 1);
    assert_eq!(bucket_scalar_output.item().expect("bucket output value"), 2);
}

#[test]
fn invalid_modes_sides_and_sorters_fail_atomically() {
    let sorted = Tensor::from_ints(&[1, 3, 5], &[3]).expect("create sorted sequence");
    let value = Tensor::from_int(3).expect("create value");
    let invalid_mode = required_tensor_error(
        sorted.isin_scalar(9, true, 3, 0.0, false, false),
        "invalid membership mode unexpectedly succeeded",
    );
    let invalid_side = required_tensor_error(
        sorted.searchsorted_tensor(&value, false, false, 9, &sorted, false),
        "invalid search side unexpectedly succeeded",
    );
    let invalid_sorter = Tensor::from_ints(&[0, 1], &[2]).expect("create invalid sorter");
    let mut output = Tensor::from_int(-1).expect("create output");
    let output_error = output
        .searchsorted_tensor_out(&sorted, &value, false, false, 0, &invalid_sorter, true)
        .expect_err("invalid sorter unexpectedly succeeded");

    assert_eq!(invalid_mode.status, 7220);
    assert_eq!(invalid_side.status, 7221);
    assert_eq!(output_error.status, 7221);
    assert_eq!(output.item().expect("unchanged output"), -1);
    assert_eq!(sum_int(&sorted), 9);
}

#[test]
fn cuda_executes_every_lookup_primitive_when_required() {
    if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_none() {
        return;
    }

    let sorted = Tensor::from_ints(&[1, 3, 5, 7], &[4])
        .expect("create sorted sequence")
        .to_device(1, 0)
        .expect("place sorted sequence on CUDA");
    let values = Tensor::from_ints(&[0, 3, 8], &[3])
        .expect("create values")
        .to_device(1, 0)
        .expect("place values on CUDA");
    let member = sorted
        .isin_scalar(0, true, 3, 0.0, false, false)
        .expect("scalar membership");
    let mut member_output =
        Tensor::empty_typed_native(&[4], 11, 1, 0).expect("create CUDA member output");
    member_output
        .isin_scalar_out(&sorted, 0, true, 3, 0.0, false, false)
        .expect("scalar membership output");
    let mut tensor_member_output =
        Tensor::empty_typed_native(&[3], 11, 1, 0).expect("create CUDA tensor output");
    tensor_member_output
        .isin_tensor_out(&values, &sorted, false, false)
        .expect("tensor membership output");
    let searched = sorted
        .searchsorted_tensor(&values, false, false, 0, &sorted, false)
        .expect("tensor search");
    let mut search_output = values.clone().expect("clone search output");
    search_output
        .searchsorted_tensor_out(&sorted, &values, false, false, 0, &sorted, false)
        .expect("tensor search output");
    let scalar_search = sorted
        .searchsorted_scalar(true, 3, 0.0, false, false, 0, &sorted, false)
        .expect("scalar search");
    let mut scalar_search_output = Tensor::from_int(0)
        .expect("create scalar output")
        .to_device(1, 0)
        .expect("place scalar output on CUDA");
    scalar_search_output
        .searchsorted_scalar_out(&sorted, true, 3, 0.0, false, true, 0, &sorted, false)
        .expect("scalar search output");
    let mut bucket_output = values.clone().expect("clone bucket output");
    bucket_output
        .bucketize_tensor_out(&values, &sorted, false, false)
        .expect("tensor bucket output");
    let bucket = sorted
        .bucketize_scalar(true, 3, 0.0, false, false)
        .expect("scalar bucket");
    let mut bucket_scalar_output = Tensor::from_int(0)
        .expect("create bucket output")
        .to_device(1, 0)
        .expect("place bucket output on CUDA");
    bucket_scalar_output
        .bucketize_scalar_out(&sorted, true, 3, 0.0, false, true)
        .expect("scalar bucket output");

    for (tensor, expected) in [
        (&member, 1),
        (&member_output, 1),
        (&tensor_member_output, 1),
        (&searched, 5),
        (&search_output, 5),
        (&scalar_search, 1),
        (&scalar_search_output, 2),
        (&bucket_output, 5),
        (&bucket, 1),
        (&bucket_scalar_output, 2),
    ] {
        let host = tensor.to_device(0, -1).expect("copy result to CPU");
        assert_eq!(sum_int(&host), expected);
    }
}
