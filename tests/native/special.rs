use std::ptr;

use terlan_pytorch_cxx::{ffi, CAbiError, Tensor};

fn required_tensor_error(result: Result<Tensor, CAbiError>, context: &str) -> CAbiError {
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
fn direct_special_scalar_family_rejects_null_handles() {
    let mut out = ptr::null_mut();
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_special_chebyshev_t_order_int(ptr::null_mut(), 2, &mut out),
            ffi::terlan_pytorch_special_chebyshev_t_order_float(
                ptr::null_mut(),
                2.0,
                &mut out,
            ),
            ffi::terlan_pytorch_special_chebyshev_t_input_int(0, ptr::null_mut(), &mut out),
            ffi::terlan_pytorch_special_chebyshev_t_input_float(
                0.5,
                ptr::null_mut(),
                &mut out,
            ),
            ffi::terlan_pytorch_special_chebyshev_t_order_int_out(
                ptr::null_mut(),
                ptr::null_mut(),
                2,
            ),
            ffi::terlan_pytorch_special_chebyshev_t_order_float_out(
                ptr::null_mut(),
                ptr::null_mut(),
                2.0,
            ),
            ffi::terlan_pytorch_special_chebyshev_t_input_int_out(
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            ),
            ffi::terlan_pytorch_special_chebyshev_t_input_float_out(
                ptr::null_mut(),
                0.5,
                ptr::null_mut(),
            ),
        ]
    };

    assert_eq!(statuses, [7101; 8]);
    assert!(out.is_null());
}

#[test]
fn generalized_special_scalar_family_rejects_null_handles() {
    let mut out = ptr::null_mut();
    let statuses = unsafe {
        [
            ffi::terlan_pytorch_special_scalar_right_int(ptr::null_mut(), 0, 2, &mut out),
            ffi::terlan_pytorch_special_scalar_right_float(
                ptr::null_mut(),
                0,
                2.0,
                &mut out,
            ),
            ffi::terlan_pytorch_special_scalar_left_int(0, 2, ptr::null_mut(), &mut out),
            ffi::terlan_pytorch_special_scalar_left_float(
                0,
                2.0,
                ptr::null_mut(),
                &mut out,
            ),
            ffi::terlan_pytorch_special_scalar_right_int_out(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                2,
            ),
            ffi::terlan_pytorch_special_scalar_right_float_out(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                2.0,
            ),
            ffi::terlan_pytorch_special_scalar_left_int_out(
                ptr::null_mut(),
                0,
                2,
                ptr::null_mut(),
            ),
            ffi::terlan_pytorch_special_scalar_left_float_out(
                ptr::null_mut(),
                0,
                2.0,
                ptr::null_mut(),
            ),
        ]
    };

    assert_eq!(statuses, [7101; 8]);
    assert!(out.is_null());
}

#[test]
fn generalized_special_scalar_family_rejects_unknown_operations() {
    let source = Tensor::from_floats(&[0.25, 0.5], &[2]).expect("create source tensor");
    let mut output = Tensor::from_floats(&[7.0, 9.0], &[2]).expect("create output tensor");
    let invalid_operation = 99;

    let errors = [
        required_tensor_error(
            source.scalar_right_int(invalid_operation, 2),
            "integer right dispatch unexpectedly succeeded",
        ),
        required_tensor_error(
            source.scalar_right_float(invalid_operation, 2.0),
            "floating right dispatch unexpectedly succeeded",
        ),
        required_tensor_error(
            Tensor::scalar_left_int(invalid_operation, 2, &source),
            "integer left dispatch unexpectedly succeeded",
        ),
        required_tensor_error(
            Tensor::scalar_left_float(invalid_operation, 2.0, &source),
            "floating left dispatch unexpectedly succeeded",
        ),
        required_unit_error(
            output.scalar_right_int_out(&source, invalid_operation, 2),
            "integer right output dispatch unexpectedly succeeded",
        ),
        required_unit_error(
            output.scalar_right_float_out(&source, invalid_operation, 2.0),
            "floating right output dispatch unexpectedly succeeded",
        ),
        required_unit_error(
            output.scalar_left_int_out(invalid_operation, 2, &source),
            "integer left output dispatch unexpectedly succeeded",
        ),
        required_unit_error(
            output.scalar_left_float_out(invalid_operation, 2.0, &source),
            "floating left output dispatch unexpectedly succeeded",
        ),
    ];

    assert!(errors.iter().all(|error| error.status == 7101));
    assert_eq!(source.shape().expect("source shape"), vec![2]);
    assert_eq!(output.shape().expect("output shape"), vec![2]);
    assert_eq!(output.sum().expect("sum output").item_float().expect("sum value"), 16.0);
}
