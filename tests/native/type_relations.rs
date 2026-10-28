use std::ptr;

use terlan_pytorch_cxx::{
    can_cast_native, ffi, promote_types_native, result_type_scalar_scalar_native, CAbiError, Tensor,
};

/// Creates one typed tensor on the selected device.
fn tensor(
    values: &[f64],
    shape: &[i64],
    dtype: i64,
    device_type: i64,
    device_index: i64,
) -> Tensor {
    Tensor::from_floats(values, shape)
        .expect("create type-relation tensor")
        .to_dtype(dtype)
        .expect("convert type-relation tensor")
        .to_device(device_type, device_index)
        .expect("place type-relation tensor")
}

/// Extracts one expected C ABI error without discarding an unexpected value.
fn bool_error(result: Result<bool, CAbiError>, context: &str) -> CAbiError {
    match result {
        Ok(value) => panic!("{context}: returned {value}"),
        Err(error) => error,
    }
}

/// Verifies tensor comparisons and all result-type overloads on one device.
fn verify_tensor_queries(device_type: i64, device_index: i64) {
    let left = tensor(&[1.0, 2.0], &[2], 6, device_type, device_index);
    let exact = tensor(&[1.0, 2.0], &[2], 6, device_type, device_index);
    let near = tensor(&[1.0, 2.0001], &[2], 6, device_type, device_index);
    let matrix = tensor(&[1.0, 2.0], &[1, 2], 6, device_type, device_index);
    let integer = tensor(&[1.0, 2.0], &[2], 3, device_type, device_index);
    let double = tensor(&[1.0, 2.0], &[2], 7, device_type, device_index);

    assert!(left.equal_native(&exact).expect("query exact equality"));
    assert!(!left.equal_native(&near).expect("query unequal values"));
    assert!(left
        .is_same_size_native(&near)
        .expect("query matching shapes"));
    assert!(!left
        .is_same_size_native(&matrix)
        .expect("query mismatching shapes"));
    assert!(!left
        .allclose_native(&near, 1e-5, 1e-8, false)
        .expect("query default tolerance"));
    assert!(left
        .allclose_native(&near, 1e-3, 0.0, false)
        .expect("query configured tolerance"));

    let nan_left = tensor(&[f64::NAN], &[1], 7, device_type, device_index);
    let nan_right = tensor(&[f64::NAN], &[1], 7, device_type, device_index);
    assert!(!nan_left
        .allclose_native(&nan_right, 1e-5, 1e-8, false)
        .expect("query unequal NaNs"));
    assert!(nan_left
        .allclose_native(&nan_right, 1e-5, 1e-8, true)
        .expect("query equal NaNs"));

    assert_eq!(integer.result_type_tensor_native(&left).unwrap(), 6);
    assert_eq!(integer.result_type_scalar_native(false, 0, 1.5).unwrap(), 6);
    assert_eq!(
        double
            .result_type_scalar_tensor_native(false, 0, 1.5)
            .unwrap(),
        7
    );

    assert!(!left.is_complex_native().unwrap());
    assert!(!left.is_conj_native().unwrap());
    assert!(!left.is_distributed_native().unwrap());
    assert!(left.is_floating_point_native().unwrap());
    assert!(!left.is_inference_native().unwrap());
    assert!(left.is_leaf_native().unwrap());
    assert!(!left.is_neg_native().unwrap());
    assert!(left.is_signed_native().unwrap());
    assert!(!left.retains_grad_native().unwrap());
    assert_eq!(left.get_device_native().unwrap(), device_index);
    assert_eq!(left.output_nr_native().unwrap(), 0);
    assert_eq!(matrix.size_dimension_native(-1).unwrap(), 2);
    assert_eq!(matrix.size_dimension_native(0).unwrap(), 1);
    assert_eq!(matrix.stride_dimension_native(-1).unwrap(), 1);
    assert_eq!(matrix.stride_dimension_native(0).unwrap(), 2);
    assert!(matrix.size_dimension_native(2).is_err());
    assert!(matrix.stride_dimension_native(-3).is_err());

    let alias = left.alias().unwrap();
    assert!(left.is_set_to_native(&alias).unwrap());
    assert!(!left.is_set_to_native(&exact).unwrap());
    let truth = tensor(&[1.0], &[], 11, device_type, device_index);
    let falsehood = tensor(&[0.0], &[], 11, device_type, device_index);
    assert!(truth.is_nonzero_native().unwrap());
    assert!(!falsehood.is_nonzero_native().unwrap());
    assert!(!truth.is_signed_native().unwrap());
    assert!(left.is_nonzero_native().is_err());

    let imaginary = tensor(&[0.0, 0.0], &[2], 6, device_type, device_index);
    let complex = left.complex(&imaginary).unwrap();
    let conjugated = complex.conj().unwrap();
    assert!(complex.is_complex_native().unwrap());
    assert!(!complex.is_conj_native().unwrap());
    assert!(conjugated.is_complex_native().unwrap());
    assert!(conjugated.is_conj_native().unwrap());

    let leaf = tensor(&[1.0, 2.0], &[2], 7, device_type, device_index)
        .require_grad()
        .unwrap();
    let mut retained = leaf.multiply_scalar(2.0).unwrap();
    assert!(!retained.is_leaf_native().unwrap());
    assert!(!retained.retains_grad_native().unwrap());
    retained.retain_grad_native().unwrap();
    assert!(retained.retains_grad_native().unwrap());
    retained.sum().unwrap().backward().unwrap();
    assert!(retained.gradient().is_ok());
}

#[test]
fn cpu_matches_pinned_type_relation_semantics() {
    verify_tensor_queries(0, -1);
    assert!(can_cast_native(3, 4).unwrap());
    assert!(can_cast_native(4, 6).unwrap());
    assert!(!can_cast_native(6, 4).unwrap());
    assert!(!can_cast_native(9, 6).unwrap());
    assert!(can_cast_native(6, 9).unwrap());
    assert_eq!(promote_types_native(3, 6).unwrap(), 6);
    assert_eq!(promote_types_native(5, 6).unwrap(), 6);
    assert_eq!(promote_types_native(9, 7).unwrap(), 10);
    assert_eq!(promote_types_native(0, 1).unwrap(), 2);
    assert_eq!(
        result_type_scalar_scalar_native(true, 1, 0.0, false, 0, 1.5).unwrap(),
        6
    );
}

#[test]
fn invalid_native_arguments_are_rejected_without_writing_outputs() {
    assert_eq!(
        bool_error(can_cast_native(-1, 6), "negative dtype succeeded").status,
        7240
    );
    assert_eq!(
        bool_error(can_cast_native(6, 46), "sentinel dtype succeeded").status,
        7240
    );
    assert_eq!(promote_types_native(99, 6).unwrap_err().status, 7240);

    let mut boolean = true;
    let mut dtype = 99_i32;
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_allclose(
                ptr::null_mut(),
                ptr::null_mut(),
                1e-5,
                1e-8,
                false,
                &mut boolean,
            ),
            ffi::terlan_pytorch_equal(ptr::null_mut(), ptr::null_mut(), &mut boolean),
            ffi::terlan_pytorch_is_same_size(ptr::null_mut(), ptr::null_mut(), &mut boolean),
            ffi::terlan_pytorch_can_cast(6, 7, ptr::null_mut()),
            ffi::terlan_pytorch_promote_types(6, 7, ptr::null_mut()),
            ffi::terlan_pytorch_result_type_tensor(ptr::null_mut(), ptr::null_mut(), &mut dtype),
            ffi::terlan_pytorch_result_type_scalar(ptr::null_mut(), true, 1, 0.0, &mut dtype),
            ffi::terlan_pytorch_result_type_scalar_tensor(
                ptr::null_mut(),
                true,
                1,
                0.0,
                &mut dtype,
            ),
            ffi::terlan_pytorch_result_type_scalar_scalar(
                true,
                1,
                0.0,
                false,
                0,
                1.0,
                ptr::null_mut(),
            ),
            ffi::terlan_pytorch_is_complex(ptr::null_mut(), &mut boolean),
            ffi::terlan_pytorch_is_set_to(ptr::null_mut(), ptr::null_mut(), &mut boolean),
            ffi::terlan_pytorch_get_device(ptr::null_mut(), std::ptr::null_mut()),
            ffi::terlan_pytorch_size_dimension(ptr::null_mut(), 0, std::ptr::null_mut()),
            ffi::terlan_pytorch_retain_grad(ptr::null_mut()),
        ]
    };
    assert_eq!(statuses, [7240; 14]);
    assert!(boolean);
    assert_eq!(dtype, 99);
}

#[test]
fn cuda_matches_type_relation_semantics_when_required() {
    let probe = Tensor::from_float(1.0)
        .expect("create CUDA probe")
        .to_device(1, 0);
    match probe {
        Ok(_) => verify_tensor_queries(1, 0),
        Err(error) => {
            if std::env::var_os("TERLAN_PYTORCH_REQUIRE_CUDA").is_some() {
                panic!("required CUDA placement failed: {error:?}");
            }
        }
    }
}
