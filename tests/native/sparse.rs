use std::ptr;

use terlan_pytorch_cxx::{ffi, CAbiError, Tensor};

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

fn scalar_float(tensor: &Tensor) -> f64 {
    tensor
        .sum()
        .expect("sum tensor")
        .item_float()
        .expect("read scalar")
}

#[test]
fn coo_conversion_metadata_and_accessors_match_libtorch() {
    let dense = Tensor::from_floats(&[1.0, 0.0, 0.0, 2.0], &[2, 2]).expect("create dense");
    let sparse = dense.to_sparse_dimensions(2).expect("convert dense to COO");
    let coalesced = sparse.coalesce().expect("coalesce COO");
    let indices = coalesced.indices().expect("borrow indices");
    let indices_copy = coalesced.indices_copy().expect("copy indices");
    let values = coalesced.values().expect("borrow values");
    let values_copy = coalesced.values_copy().expect("copy values");
    let round_trip = coalesced.to_dense().expect("convert COO to dense");
    let mut indices_output = Tensor::from_ints(&[], &[0]).expect("create index output");
    let mut values_output = Tensor::from_floats(&[], &[0]).expect("create value output");

    indices_output
        .indices_copy_out(&coalesced)
        .expect("write copied indices");
    values_output
        .values_copy_out(&coalesced)
        .expect("write copied values");

    assert_eq!(coalesced.sparse_dim().expect("sparse dimensions"), 2);
    assert_eq!(coalesced.dense_dim().expect("dense dimensions"), 0);
    assert!(coalesced.is_coalesced().expect("coalesced state"));
    assert_eq!(indices.shape().expect("index shape"), vec![2, 2]);
    assert_eq!(
        indices_copy.shape().expect("copied index shape"),
        vec![2, 2]
    );
    assert_eq!(
        indices_output.shape().expect("output index shape"),
        vec![2, 2]
    );
    assert_eq!(values.shape().expect("value shape"), vec![2]);
    assert_eq!(values_copy.shape().expect("copied value shape"), vec![2]);
    assert_eq!(values_output.shape().expect("output value shape"), vec![2]);
    assert_eq!(scalar_float(&values), 3.0);
    assert_eq!(scalar_float(&values_copy), 3.0);
    assert_eq!(scalar_float(&values_output), 3.0);
    assert_eq!(scalar_float(&round_trip), 3.0);
    assert_eq!(dense.shape().expect("source shape"), vec![2, 2]);
}

#[test]
fn compressed_sparse_conversions_expose_their_native_indices() {
    let dense = Tensor::from_floats(&[1.0, 0.0, 0.0, 2.0], &[2, 2]).expect("create dense");
    let csr = dense.to_sparse_csr().expect("convert to CSR");
    let csc = dense.to_sparse_csc().expect("convert to CSC");
    let bsr = dense.to_sparse_bsr(&[1, 1]).expect("convert to BSR");
    let bsc = dense.to_sparse_bsc(&[1, 1]).expect("convert to BSC");
    let configured_csr = dense
        .to_sparse_csr_configured(0)
        .expect("convert to configured CSR");
    let configured_csc = dense
        .to_sparse_csc_configured(0)
        .expect("convert to configured CSC");
    let configured_bsr = dense
        .to_sparse_bsr_configured(&[1, 1], 0)
        .expect("convert to configured BSR");
    let configured_bsc = dense
        .to_sparse_bsc_configured(&[1, 1], 0)
        .expect("convert to configured BSC");
    let csr_rows_alias = csr.crow_indices().expect("borrow CSR row pointers");
    let csr_columns_alias = csr.col_indices().expect("borrow CSR columns");
    let csc_columns_alias = csc.ccol_indices().expect("borrow CSC column pointers");
    let csc_rows_alias = csc.row_indices().expect("borrow CSC rows");
    let csr_rows = csr.crow_indices_copy().expect("copy CSR row pointers");
    let csr_columns = csr.col_indices_copy().expect("copy CSR columns");
    let csc_columns = csc.ccol_indices_copy().expect("copy CSC column pointers");
    let csc_rows = csc.row_indices_copy().expect("copy CSC rows");
    let bsr_values = bsr.values_copy().expect("copy BSR values");
    let bsc_values = bsc.values_copy().expect("copy BSC values");
    let mut csr_rows_output = Tensor::from_ints(&[], &[0]).expect("create CSR row output");
    let mut csr_columns_output = Tensor::from_ints(&[], &[0]).expect("create CSR column output");
    let mut csc_columns_output = Tensor::from_ints(&[], &[0]).expect("create CSC column output");
    let mut csc_rows_output = Tensor::from_ints(&[], &[0]).expect("create CSC row output");

    csr_rows_output
        .crow_indices_copy_out(&csr)
        .expect("write CSR row pointers");
    csr_columns_output
        .col_indices_copy_out(&csr)
        .expect("write CSR columns");
    csc_columns_output
        .ccol_indices_copy_out(&csc)
        .expect("write CSC column pointers");
    csc_rows_output
        .row_indices_copy_out(&csc)
        .expect("write CSC rows");

    assert_eq!(
        csr_rows_alias.shape().expect("CSR row alias shape"),
        vec![3]
    );
    assert_eq!(
        csr_columns_alias.shape().expect("CSR column alias shape"),
        vec![2]
    );
    assert_eq!(
        csc_columns_alias.shape().expect("CSC column alias shape"),
        vec![3]
    );
    assert_eq!(
        csc_rows_alias.shape().expect("CSC row alias shape"),
        vec![2]
    );
    assert_eq!(csr_rows.shape().expect("CSR row pointer shape"), vec![3]);
    assert_eq!(csr_columns.shape().expect("CSR column shape"), vec![2]);
    assert_eq!(
        csc_columns.shape().expect("CSC column pointer shape"),
        vec![3]
    );
    assert_eq!(csc_rows.shape().expect("CSC row shape"), vec![2]);
    assert_eq!(
        csr_rows_output.shape().expect("CSR row output shape"),
        vec![3]
    );
    assert_eq!(
        csr_columns_output.shape().expect("CSR column output shape"),
        vec![2]
    );
    assert_eq!(
        csc_columns_output.shape().expect("CSC column output shape"),
        vec![3]
    );
    assert_eq!(
        csc_rows_output.shape().expect("CSC row output shape"),
        vec![2]
    );
    assert_eq!(bsr_values.shape().expect("BSR values shape"), vec![2, 1, 1]);
    assert_eq!(bsc_values.shape().expect("BSC values shape"), vec![2, 1, 1]);
    assert_eq!(scalar_float(&csr.to_dense().expect("CSR to dense")), 3.0);
    assert_eq!(scalar_float(&csc.to_dense().expect("CSC to dense")), 3.0);
    assert_eq!(scalar_float(&bsr.to_dense().expect("BSR to dense")), 3.0);
    assert_eq!(scalar_float(&bsc.to_dense().expect("BSC to dense")), 3.0);
    assert_eq!(
        scalar_float(&configured_csr.to_dense().expect("configured CSR to dense")),
        3.0
    );
    assert_eq!(
        scalar_float(&configured_csc.to_dense().expect("configured CSC to dense")),
        3.0
    );
    assert_eq!(
        scalar_float(&configured_bsr.to_dense().expect("configured BSR to dense")),
        3.0
    );
    assert_eq!(
        scalar_float(&configured_bsc.to_dense().expect("configured BSC to dense")),
        3.0
    );
}

#[test]
fn sparse_mask_and_out_preserve_source_ownership() {
    let dense = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create dense");
    let mask_source =
        Tensor::from_floats(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("create mask source");
    let mask = mask_source
        .to_sparse_dimensions(2)
        .expect("convert sparse mask");
    let selected = dense.sparse_mask(&mask).expect("apply sparse mask");
    let zero = Tensor::from_floats(&[0.0, 0.0, 0.0, 0.0], &[2, 2]).expect("create zero");
    let mut output = zero.to_sparse_dimensions(2).expect("create sparse output");

    output
        .sparse_mask_out(&dense, &mask)
        .expect("write sparse mask output");

    assert_eq!(
        scalar_float(&selected.to_dense().expect("selected dense")),
        5.0
    );
    assert_eq!(scalar_float(&output.to_dense().expect("output dense")), 5.0);
    assert_eq!(scalar_float(&dense), 10.0);
    assert_eq!(scalar_float(&mask_source), 2.0);
}

#[test]
fn sparse_boundaries_reject_invalid_inputs_without_outputs() {
    let dense = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create dense");
    let invalid_dimension = required_tensor_error(
        dense.to_sparse_dimensions(3),
        "oversized sparse rank unexpectedly succeeded",
    );
    let invalid_block = required_tensor_error(
        dense.to_sparse_bsr(&[0, 1]),
        "zero block extent unexpectedly succeeded",
    );
    let mut integer = 91;
    let mut boolean = true;
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_sparse_dim(ptr::null_mut(), &mut integer),
            ffi::terlan_pytorch_sparse_dense_dim(ptr::null_mut(), &mut integer),
            ffi::terlan_pytorch_sparse_is_coalesced(ptr::null_mut(), &mut boolean),
        ]
    };

    assert_ne!(invalid_dimension.status, 0);
    assert_ne!(invalid_block.status, 0);
    assert_eq!(statuses, [7110; 3]);
    assert_eq!(integer, 91);
    assert!(boolean);
}

#[test]
fn sparse_copy_and_resize_families_preserve_their_mutation_contracts() {
    let source_dense =
        Tensor::from_floats(&[1.0, 0.0, 0.0, 2.0], &[2, 2]).expect("create source dense");
    let zero_dense =
        Tensor::from_floats(&[0.0, 0.0, 0.0, 0.0], &[2, 2]).expect("create zero dense");
    let template_dense = Tensor::from_floats(&[0.0; 12], &[3, 4]).expect("create template dense");
    let source = source_dense
        .to_sparse_dimensions(2)
        .expect("create sparse source");
    let destination = zero_dense
        .to_sparse_dimensions(2)
        .expect("create sparse destination");
    let template = template_dense
        .to_sparse_dimensions(2)
        .expect("create sparse template");
    let copied = destination
        .copy_sparse_to_sparse(&source, false)
        .expect("functional sparse copy");
    let mut copied_output = zero_dense
        .to_sparse_dimensions(2)
        .expect("create sparse copy output");
    let mut copied_in_place = zero_dense
        .to_sparse_dimensions(2)
        .expect("create sparse copy destination");
    copied_output
        .copy_sparse_to_sparse_out(&destination, &source, false)
        .expect("output sparse copy");
    copied_in_place
        .copy_sparse_to_sparse_in_place(&source, false)
        .expect("in-place sparse copy");

    let resized = destination
        .resize_as_sparse(&template)
        .expect("functional template resize");
    let mut resized_output = destination
        .to_dense()
        .expect("dense resize output seed")
        .to_sparse_dimensions(2)
        .expect("sparse resize output seed");
    let mut resized_in_place = destination
        .to_dense()
        .expect("dense resize destination seed")
        .to_sparse_dimensions(2)
        .expect("sparse resize destination seed");
    resized_output
        .resize_as_sparse_out(&destination, &template)
        .expect("output template resize");
    resized_in_place
        .resize_as_sparse_in_place(&template)
        .expect("in-place template resize");

    let explicit = destination
        .sparse_resize(&[3, 4], 2, 0)
        .expect("functional explicit resize");
    let mut explicit_output = destination
        .to_dense()
        .expect("dense explicit output seed")
        .to_sparse_dimensions(2)
        .expect("sparse explicit output seed");
    let mut explicit_in_place = destination
        .to_dense()
        .expect("dense explicit destination seed")
        .to_sparse_dimensions(2)
        .expect("sparse explicit destination seed");
    explicit_output
        .sparse_resize_out(&destination, &[3, 4], 2, 0)
        .expect("output explicit resize");
    explicit_in_place
        .sparse_resize_in_place(&[3, 4], 2, 0)
        .expect("in-place explicit resize");

    let cleared = source
        .sparse_resize_and_clear(&[3, 4], 2, 0)
        .expect("functional resize and clear");
    let mut cleared_output = source
        .to_dense()
        .expect("dense clear output seed")
        .to_sparse_dimensions(2)
        .expect("sparse clear output seed");
    let mut cleared_in_place = source
        .to_dense()
        .expect("dense clear destination seed")
        .to_sparse_dimensions(2)
        .expect("sparse clear destination seed");
    cleared_output
        .sparse_resize_and_clear_out(&source, &[3, 4], 2, 0)
        .expect("output resize and clear");
    cleared_in_place
        .sparse_resize_and_clear_in_place(&[3, 4], 2, 0)
        .expect("in-place resize and clear");

    assert_eq!(scalar_float(&copied.to_dense().expect("copied dense")), 3.0);
    assert_eq!(
        scalar_float(&copied_output.to_dense().expect("copy output dense")),
        3.0
    );
    assert_eq!(
        scalar_float(&copied_in_place.to_dense().expect("copy destination dense")),
        3.0
    );
    for tensor in [
        &resized,
        &resized_output,
        &resized_in_place,
        &explicit,
        &explicit_output,
        &explicit_in_place,
        &cleared,
        &cleared_output,
        &cleared_in_place,
    ] {
        assert_eq!(tensor.shape().expect("resized shape"), vec![3, 4]);
    }
    assert_eq!(
        cleared
            .values()
            .expect("cleared values")
            .shape()
            .expect("cleared value shape"),
        vec![0]
    );
    assert_eq!(
        cleared_output
            .values()
            .expect("output cleared values")
            .shape()
            .expect("output cleared value shape"),
        vec![0]
    );
    assert_eq!(
        cleared_in_place
            .values()
            .expect("in-place cleared values")
            .shape()
            .expect("in-place cleared value shape"),
        vec![0]
    );
    assert_eq!(source.shape().expect("source shape"), vec![2, 2]);
    assert_eq!(destination.shape().expect("destination shape"), vec![2, 2]);
}

#[test]
fn configurable_dense_and_sparse_conversions_cover_optional_arguments() {
    let dense = Tensor::from_floats(&[1.0, 0.0, 0.0, 2.0], &[2, 2]).expect("create dense");
    let coo = dense
        .to_sparse_configured(1, true, &[], false, 0, false)
        .expect("generic COO conversion");
    let default_coo = dense
        .to_sparse_configured(0, false, &[], false, 0, false)
        .expect("default generic sparse conversion");
    let csr = dense
        .to_sparse_configured(2, true, &[], false, 0, false)
        .expect("generic CSR conversion");
    let bsr = dense
        .to_sparse_configured(5, true, &[1, 1], true, 0, true)
        .expect("generic BSR conversion");
    let inherited = coo
        .to_dense_configured(0, false, false, false)
        .expect("dense conversion with absent options");
    let typed = required_tensor_error(
        coo.to_dense_configured(7, true, false, false),
        "explicit dense dtype unexpectedly succeeded",
    );
    let masked = coo
        .to_dense_configured(0, false, true, true)
        .expect("dense conversion with gradient mask option");
    let incompatible_dtype = required_tensor_error(
        coo.to_dense_configured(6, true, false, false),
        "incompatible dense dtype unexpectedly succeeded",
    );
    let bad_layout = required_tensor_error(
        dense.to_sparse_configured(99, true, &[], false, 0, false),
        "invalid sparse layout unexpectedly succeeded",
    );
    let bad_block = required_tensor_error(
        dense.to_sparse_configured(5, true, &[], true, 0, false),
        "empty present block size unexpectedly succeeded",
    );

    assert_eq!(coo.sparse_dim().expect("COO sparse rank"), 2);
    assert_eq!(
        default_coo.sparse_dim().expect("default COO sparse rank"),
        2
    );
    assert_eq!(
        csr.crow_indices()
            .expect("CSR row pointers")
            .shape()
            .expect("CSR pointer shape"),
        vec![3]
    );
    assert_eq!(
        bsr.values()
            .expect("BSR values")
            .shape()
            .expect("BSR value shape"),
        vec![2, 1, 1]
    );
    assert_eq!(inherited.dtype().expect("inherited dtype"), 7);
    assert_eq!(masked.dtype().expect("masked dtype"), 7);
    assert_eq!(scalar_float(&inherited), 3.0);
    assert_eq!(scalar_float(&masked), 3.0);
    assert_eq!(typed.status, 7111);
    assert_eq!(incompatible_dtype.status, 7111);
    assert_eq!(bad_layout.status, 7110);
    assert_ne!(bad_block.status, 0);
}

#[test]
fn sparse_constructor_families_cover_inferred_and_explicit_sizes() {
    let empty = Tensor::coo_empty_configured(&[2, 3], 7, true, 1, true, 0, -1, true, false, true)
        .expect("create configured empty COO");
    let mut empty_output =
        Tensor::from_floats(&[1.0], &[1]).expect("create empty COO output storage");
    empty_output
        .coo_empty_out(&[3, 2])
        .expect("write empty COO output");

    let coordinates = Tensor::from_ints(&[0, 1, 0, 1], &[2, 2]).expect("create COO coordinates");
    let values = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create COO values");
    let inferred_coo = Tensor::coo_from_components_configured(
        &coordinates,
        &values,
        &[],
        false,
        0,
        false,
        0,
        false,
        0,
        -1,
        false,
        false,
        false,
        true,
        true,
    )
    .expect("create inferred COO");
    let sized_coo = Tensor::coo_from_components_configured(
        &coordinates,
        &values,
        &[2, 2],
        true,
        7,
        true,
        1,
        true,
        0,
        -1,
        true,
        false,
        true,
        true,
        true,
    )
    .expect("create explicitly sized COO");

    let compressed = Tensor::from_ints(&[0, 1, 2], &[3]).expect("create compressed indices");
    let plain = Tensor::from_ints(&[0, 1], &[2]).expect("create plain indices");
    let block_values = Tensor::from_floats(&[1.0, 2.0], &[2, 1, 1])
        .expect("create block values")
        .to_dtype(6)
        .expect("convert block values to default sparse dtype");
    let csr_values = Tensor::from_floats(&[1.0, 2.0], &[2])
        .expect("create scalar values")
        .to_dtype(6)
        .expect("convert scalar values to default sparse dtype");

    let mut constructed = Vec::new();
    for (kind, tensor_values) in [
        (2, &csr_values),
        (4, &csr_values),
        (5, &block_values),
        (6, &block_values),
    ] {
        constructed.push(
            Tensor::compressed_from_components_configured(
                kind,
                &compressed,
                &plain,
                tensor_values,
                &[],
                false,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
            )
            .expect("create inferred compressed tensor"),
        );
        constructed.push(
            Tensor::compressed_from_components_configured(
                kind,
                &compressed,
                &plain,
                tensor_values,
                &[2, 2],
                true,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
            )
            .expect("create explicitly sized compressed tensor"),
        );
    }
    for has_size in [false, true] {
        constructed.push(
            Tensor::compressed_from_components_configured(
                0,
                &compressed,
                &plain,
                &csr_values,
                if has_size { &[2, 2][..] } else { &[][..] },
                has_size,
                6,
                true,
                2,
                true,
                0,
                -1,
                true,
                false,
                true,
            )
            .expect("create generic compressed tensor"),
        );
    }

    assert_eq!(empty.shape().expect("empty COO shape"), vec![2, 3]);
    assert_eq!(empty.dtype().expect("empty COO dtype"), 7);
    assert_eq!(
        empty
            .values()
            .expect("empty COO values")
            .shape()
            .expect("empty value shape"),
        vec![0]
    );
    assert_eq!(
        empty_output.shape().expect("empty output shape"),
        vec![3, 2]
    );
    assert_eq!(
        inferred_coo.shape().expect("inferred COO shape"),
        vec![2, 2]
    );
    assert_eq!(sized_coo.shape().expect("sized COO shape"), vec![2, 2]);
    assert_eq!(
        scalar_float(&inferred_coo.to_dense().expect("inferred COO dense")),
        3.0
    );
    assert_eq!(
        scalar_float(&sized_coo.to_dense().expect("sized COO dense")),
        3.0
    );
    assert_eq!(constructed.len(), 10);
    for tensor in &constructed {
        assert_eq!(tensor.shape().expect("compressed shape"), vec![2, 2]);
        assert_eq!(
            scalar_float(&tensor.to_dense().expect("compressed dense")),
            3.0
        );
    }
    assert_eq!(
        coordinates.shape().expect("coordinate source shape"),
        vec![2, 2]
    );
    assert_eq!(values.shape().expect("value source shape"), vec![2]);
}

#[test]
fn sparse_constructors_reject_malformed_options_and_raw_inputs() {
    let invalid_dtype = required_tensor_error(
        Tensor::coo_empty_configured(&[2, 2], 99, true, 1, true, 0, -1, true, false, false),
        "unsupported sparse dtype unexpectedly succeeded",
    );
    let invalid_layout = required_tensor_error(
        Tensor::coo_empty_configured(&[2, 2], 7, true, 99, true, 0, -1, true, false, false),
        "unsupported sparse layout unexpectedly succeeded",
    );
    let invalid_device = required_tensor_error(
        Tensor::coo_empty_configured(&[2, 2], 7, true, 1, true, 99, 0, true, false, false),
        "unsupported sparse device unexpectedly succeeded",
    );
    let compressed = Tensor::from_ints(&[0, 1, 2], &[3]).expect("create compressed indices");
    let plain = Tensor::from_ints(&[0, 1], &[2]).expect("create plain indices");
    let values = Tensor::from_floats(&[1.0, 2.0], &[2]).expect("create sparse values");
    let invalid_kind = required_tensor_error(
        Tensor::compressed_from_components_configured(
            99,
            &compressed,
            &plain,
            &values,
            &[2, 2],
            true,
            0,
            false,
            0,
            false,
            0,
            -1,
            false,
            false,
            false,
        ),
        "unsupported compressed constructor unexpectedly succeeded",
    );
    let missing_layout = required_tensor_error(
        Tensor::compressed_from_components_configured(
            0,
            &compressed,
            &plain,
            &values,
            &[2, 2],
            true,
            0,
            false,
            0,
            false,
            0,
            -1,
            false,
            false,
            false,
        ),
        "generic compressed constructor without a layout unexpectedly succeeded",
    );

    let mut output = ptr::null_mut();
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_sparse_coo_empty(
                ptr::null(),
                1,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
                &mut output,
            ),
            ffi::terlan_pytorch_sparse_coo_empty(
                ptr::null(),
                -1,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
                &mut output,
            ),
            ffi::terlan_pytorch_sparse_coo_empty(
                ptr::null(),
                0,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
                ptr::null_mut(),
            ),
            ffi::terlan_pytorch_sparse_coo_empty_out(ptr::null_mut(), ptr::null(), 0),
            ffi::terlan_pytorch_sparse_coo_from_components(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
                false,
                false,
                &mut output,
            ),
            ffi::terlan_pytorch_sparse_compressed_from_components(
                2,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                0,
                false,
                0,
                false,
                0,
                false,
                0,
                -1,
                false,
                false,
                false,
                &mut output,
            ),
        ]
    };

    assert_eq!(invalid_dtype.status, 7110);
    assert_eq!(invalid_layout.status, 7110);
    assert_eq!(invalid_device.status, 7110);
    assert_eq!(invalid_kind.status, 7110);
    assert_eq!(missing_layout.status, 7111);
    assert_eq!(statuses, [7110; 6]);
    assert!(output.is_null());
}

#[test]
fn sampled_addmm_and_dense_backward_match_sparse_libtorch_semantics() {
    let mask_dense =
        Tensor::from_floats(&[1.0, 0.0, 0.0, 2.0], &[2, 2]).expect("create mask dense");
    let mask = mask_dense.to_sparse_csr().expect("create CSR mask");
    let left =
        Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("create left matrix");
    let right =
        Tensor::from_floats(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).expect("create right matrix");
    let default_result = mask
        .sampled_addmm_numeric(&left, &right, 1.0, true, 1.0, true)
        .expect("sampled addmm with integer defaults");
    let scaled_result = mask
        .sampled_addmm_numeric(&left, &right, 0.5, false, 2.0, false)
        .expect("sampled addmm with floating scalars");
    let mut output = mask_dense.to_sparse_csr().expect("create sampled output");
    output
        .sampled_addmm_numeric_out(&mask, &left, &right, 0.5, false, 2.0, false)
        .expect("write sampled addmm output");

    let gradient = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create gradient");
    let coo_input = mask_dense
        .to_sparse_dimensions(2)
        .expect("create COO input");
    let inherited_backward = gradient
        .to_dense_backward_configured(&coo_input, false, false)
        .expect("dense backward with absent mask option");
    let masked_backward = gradient
        .to_dense_backward_configured(&coo_input, true, true)
        .expect("dense backward with explicit mask");
    let unmasked_backward = gradient
        .to_dense_backward_configured(&coo_input, false, true)
        .expect("dense backward with disabled mask");
    let bad_left = Tensor::from_floats(&[1.0, 2.0], &[1, 2]).expect("create bad left");
    let invalid_shape = required_tensor_error(
        mask.sampled_addmm_numeric(&bad_left, &right, 1.0, true, 1.0, true),
        "incompatible sampled matrices unexpectedly succeeded",
    );

    assert_eq!(
        scalar_float(&default_result.to_dense().expect("default sampled dense")),
        18.0
    );
    assert_eq!(
        scalar_float(&scaled_result.to_dense().expect("scaled sampled dense")),
        31.5
    );
    assert_eq!(
        scalar_float(&output.to_dense().expect("output sampled dense")),
        31.5
    );
    assert_eq!(
        scalar_float(
            &inherited_backward
                .to_dense()
                .expect("inherited backward dense")
        ),
        5.0
    );
    assert_eq!(
        scalar_float(&masked_backward.to_dense().expect("masked backward dense")),
        5.0
    );
    assert_eq!(
        scalar_float(
            &unmasked_backward
                .to_dense()
                .expect("unmasked backward dense")
        ),
        10.0
    );
    assert_ne!(invalid_shape.status, 0);
    assert_eq!(scalar_float(&mask_dense), 3.0);
    assert_eq!(scalar_float(&left), 21.0);
    assert_eq!(scalar_float(&right), 4.0);

    let mut raw_output = ptr::null_mut();
    let null_statuses = unsafe {
        [
            ffi::terlan_pytorch_sparse_sampled_addmm(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                1.0,
                true,
                1.0,
                true,
                &mut raw_output,
            ),
            ffi::terlan_pytorch_sparse_sampled_addmm_out(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                1.0,
                true,
                1.0,
                true,
            ),
            ffi::terlan_pytorch_sparse_to_dense_backward(
                ptr::null_mut(),
                ptr::null_mut(),
                false,
                false,
                &mut raw_output,
            ),
        ]
    };
    assert_eq!(null_statuses, [7110; 3]);
    assert!(raw_output.is_null());
}
