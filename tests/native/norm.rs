use std::ptr;

use terlan_pytorch_cxx::{ffi, CAbiError, Tensor};

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 0.001
}

fn required_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

fn required_unit_error(result: Result<(), CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(()) => panic!("{context}"),
        Err(error) => error,
    }
}

#[test]
fn direct_norm_family_matches_libtorch_and_preserves_sources() {
    let vector = Tensor::from_floats(&[3.0, 4.0], &[2]).expect("create vector");
    let zero = Tensor::from_floats(&[0.0, 0.0], &[2]).expect("create zero vector");
    let matrix = Tensor::from_floats(&[3.0, 4.0, 0.0, 10.0], &[2, 2]).expect("create matrix");
    let condition_matrix =
        Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create condition matrix");

    let scalar = vector.norm(2.0).expect("scalar norm");
    let distance = vector.dist(&zero, 2.0).expect("distance");
    let rows = matrix
        .norm_dimensions(2.0, &[1], false)
        .expect("dimension norm");
    let rows_total = rows.sum().expect("sum dimension norms");
    let excluded = matrix
        .norm_except_dimension(2, 0)
        .expect("excluded-dimension norm");
    let excluded_total = excluded.sum().expect("sum excluded norms");
    let bounded = matrix.renorm(2.0, 0, 5.0).expect("renorm");
    let bounded_total = bounded.sum().expect("sum renorm result");
    let default_condition = condition_matrix
        .linalg_cond_default()
        .expect("default condition number");
    let integer_condition = condition_matrix
        .linalg_cond_int(1)
        .expect("integer condition number");
    let floating_condition = condition_matrix
        .linalg_cond_float(2.0)
        .expect("floating condition number");

    assert!(close(scalar.item_float().expect("scalar value"), 5.0));
    assert!(close(distance.item_float().expect("distance value"), 5.0));
    assert!(close(
        rows_total.item_float().expect("row norm total"),
        15.0
    ));
    assert!(close(
        excluded_total.item_float().expect("excluded norm total"),
        15.0
    ));
    assert!(close(
        bounded_total.item_float().expect("renorm total"),
        12.0
    ));
    assert!(close(
        default_condition.item_float().expect("default condition"),
        14.933_034
    ));
    assert!(close(
        integer_condition.item_float().expect("integer condition"),
        21.0
    ));
    assert!(close(
        floating_condition.item_float().expect("floating condition"),
        14.933_034
    ));
    assert_eq!(vector.shape().expect("vector shape"), vec![2]);
    assert_eq!(zero.shape().expect("zero shape"), vec![2]);
    assert_eq!(matrix.shape().expect("matrix shape"), vec![2, 2]);
    assert_eq!(rows.shape().expect("row norm shape"), vec![2]);
    assert_eq!(excluded.shape().expect("excluded norm shape"), vec![2, 1]);
    assert_eq!(bounded.shape().expect("renorm shape"), vec![2, 2]);
    assert_eq!(
        condition_matrix.shape().expect("condition matrix shape"),
        vec![2, 2]
    );
}

#[test]
fn direct_norm_family_rejects_null_handles_at_the_c_boundary() {
    let mut out = ptr::null_mut();
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_norm_scalar(ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_dist_scalar(ptr::null_mut(), ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_norm_dimensions(
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                &mut out,
            ),
            ffi::terlan_pytorch_norm_except_dimension(ptr::null_mut(), 2, 0, &mut out),
            ffi::terlan_pytorch_renorm(ptr::null_mut(), 2.0, 0, 5.0, &mut out),
            ffi::terlan_pytorch_linalg_cond_default(ptr::null_mut(), &mut out),
            ffi::terlan_pytorch_linalg_cond_int(ptr::null_mut(), 1, &mut out),
            ffi::terlan_pytorch_linalg_cond_float(ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_linalg_matrix_norm_int(ptr::null_mut(), 1, &mut out),
            ffi::terlan_pytorch_linalg_matrix_norm_float(ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_linalg_matrix_norm_int_configured(
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_matrix_norm_float_configured(
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_norm_default(ptr::null_mut(), &mut out),
            ffi::terlan_pytorch_linalg_norm_int(ptr::null_mut(), 1, &mut out),
            ffi::terlan_pytorch_linalg_norm_float(ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_linalg_norm_int_configured(
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_norm_float_configured(
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_vector_norm_int(ptr::null_mut(), 1, &mut out),
            ffi::terlan_pytorch_linalg_vector_norm_float(ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_linalg_vector_norm_int_configured(
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_vector_norm_float_configured(
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_powsum_int(ptr::null_mut(), 1, &mut out),
            ffi::terlan_pytorch_linalg_powsum_float(ptr::null_mut(), 2.0, &mut out),
            ffi::terlan_pytorch_linalg_powsum_int_configured(
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_powsum_float_configured(
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
                &mut out,
            ),
            ffi::terlan_pytorch_linalg_cond_default_out(ptr::null_mut(), ptr::null_mut()),
            ffi::terlan_pytorch_linalg_cond_int_out(ptr::null_mut(), ptr::null_mut(), 1),
            ffi::terlan_pytorch_linalg_cond_float_out(ptr::null_mut(), ptr::null_mut(), 2.0),
            ffi::terlan_pytorch_linalg_matrix_norm_int_out(ptr::null_mut(), ptr::null_mut(), 1),
            ffi::terlan_pytorch_linalg_matrix_norm_float_out(ptr::null_mut(), ptr::null_mut(), 2.0),
            ffi::terlan_pytorch_linalg_matrix_norm_int_configured_out(
                ptr::null_mut(),
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
            ),
            ffi::terlan_pytorch_linalg_matrix_norm_float_configured_out(
                ptr::null_mut(),
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
            ),
            ffi::terlan_pytorch_linalg_norm_default_out(ptr::null_mut(), ptr::null_mut()),
            ffi::terlan_pytorch_linalg_norm_int_out(ptr::null_mut(), ptr::null_mut(), 1),
            ffi::terlan_pytorch_linalg_norm_float_out(ptr::null_mut(), ptr::null_mut(), 2.0),
            ffi::terlan_pytorch_linalg_norm_int_configured_out(
                ptr::null_mut(),
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
            ),
            ffi::terlan_pytorch_linalg_norm_float_configured_out(
                ptr::null_mut(),
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
            ),
            ffi::terlan_pytorch_linalg_vector_norm_int_out(ptr::null_mut(), ptr::null_mut(), 1),
            ffi::terlan_pytorch_linalg_vector_norm_float_out(ptr::null_mut(), ptr::null_mut(), 2.0),
            ffi::terlan_pytorch_linalg_vector_norm_int_configured_out(
                ptr::null_mut(),
                ptr::null_mut(),
                1,
                ptr::null(),
                0,
                false,
                6,
            ),
            ffi::terlan_pytorch_linalg_vector_norm_float_configured_out(
                ptr::null_mut(),
                ptr::null_mut(),
                2.0,
                ptr::null(),
                0,
                false,
                6,
            ),
        ]
    };

    assert_eq!(statuses, [7094; 41]);
    assert!(out.is_null());
}

#[test]
fn direct_norm_family_contains_libtorch_dimension_errors() {
    let matrix = Tensor::from_floats(&[3.0, 4.0, 0.0, 10.0], &[2, 2]).expect("create matrix");

    let dimension_error = required_error(
        matrix.norm_dimensions(2.0, &[7], false),
        "out-of-range norm dimension was accepted",
    );
    let excluded_error = required_error(
        matrix.norm_except_dimension(2, 7),
        "out-of-range excluded dimension was accepted",
    );
    let renorm_error = required_error(
        matrix.renorm(2.0, 7, 5.0),
        "out-of-range renorm dimension was accepted",
    );
    let condition_error = required_error(
        matrix.linalg_cond_int(0),
        "invalid zero condition-number order was accepted",
    );
    let matrix_norm_error = required_error(
        matrix.linalg_matrix_norm_float_configured(2.0, &[7, 1], false, 6),
        "out-of-range matrix-norm dimension was accepted",
    );
    let norm_error = required_error(
        matrix.linalg_norm_float_configured(2.0, &[7], false, 6),
        "out-of-range general-norm dimension was accepted",
    );
    let vector_norm_error = required_error(
        matrix.linalg_vector_norm_float_configured(2.0, &[7], false, 6),
        "out-of-range vector-norm dimension was accepted",
    );
    let powsum_error = required_error(
        matrix.linalg_powsum_float_configured(2.0, &[7], false, 6),
        "out-of-range power-sum dimension was accepted",
    );
    let dtype_error = required_error(
        matrix.linalg_vector_norm_float_configured(2.0, &[0], false, 999),
        "invalid scalar dtype was accepted",
    );

    assert_eq!(dimension_error.status, 7095);
    assert_eq!(excluded_error.status, 7095);
    assert_eq!(renorm_error.status, 7095);
    assert_eq!(condition_error.status, 7095);
    assert_eq!(matrix_norm_error.status, 7095);
    assert_eq!(norm_error.status, 7095);
    assert_eq!(vector_norm_error.status, 7095);
    assert_eq!(powsum_error.status, 7095);
    assert_eq!(dtype_error.status, 7095);
    assert_eq!(matrix.shape().expect("source remains valid"), vec![2, 2]);
}

#[test]
fn direct_scalar_out_family_contains_libtorch_errors() {
    let matrix =
        Tensor::from_floats(&[3.0, 4.0, 0.0, 10.0], &[2, 2]).expect("create source matrix");
    let mut output = Tensor::from_floats(&[], &[0]).expect("create output tensor");

    let dimension_error = required_unit_error(
        output.linalg_matrix_norm_float_configured_out(&matrix, 2.0, &[7, 1], false, 7),
        "out-of-range matrix-norm output dimension was accepted",
    );
    let dtype_error = required_unit_error(
        output.linalg_norm_float_configured_out(&matrix, 2.0, &[0, 1], false, 999),
        "invalid general-norm output dtype was accepted",
    );
    let vector_error = required_unit_error(
        output.linalg_vector_norm_float_configured_out(&matrix, 2.0, &[7], false, 7),
        "out-of-range vector-norm output dimension was accepted",
    );
    let condition_error = required_unit_error(
        output.linalg_cond_int_out(&matrix, 0),
        "invalid condition-number output order was accepted",
    );

    assert_eq!(dimension_error.status, 7095);
    assert_eq!(dtype_error.status, 7095);
    assert_eq!(vector_error.status, 7095);
    assert_eq!(condition_error.status, 7095);
    assert_eq!(output.shape().expect("output remains valid"), vec![0]);
    assert_eq!(matrix.shape().expect("source remains valid"), vec![2, 2]);
}
