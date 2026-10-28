use terlan_pytorch_cxx::ffi;

#[test]
fn generated_cxx_executes_dtype_and_scalar_result_relations() {
    let can_cast = ffi::terlan_exception_pytorch_typerelationsnative_can_cast_native(3, 4);
    let can_cast = can_cast.as_ref().expect("contained can_cast result");
    assert!(can_cast.is_ok());
    assert!(can_cast.bool_value());

    let promoted = ffi::terlan_exception_pytorch_typerelationsnative_promote_types_native(3, 6);
    let promoted = promoted.as_ref().expect("contained promote_types result");
    assert!(promoted.is_ok());
    assert_eq!(promoted.value(), 6);

    let result_type =
        ffi::terlan_exception_pytorch_typerelationsnative_result_type_scalar_scalar_native(
            true, 1, 0.0, false, 0, 1.5,
        );
    let result_type = result_type
        .as_ref()
        .expect("contained scalar/scalar result_type");
    assert!(result_type.is_ok());
    assert_eq!(result_type.value(), 6);
}

#[test]
fn generated_cxx_rejects_unselected_dtype_sentinels() {
    for value in [-1, 46, 47, i64::MAX] {
        let can_cast =
            ffi::terlan_exception_pytorch_typerelationsnative_can_cast_native(value, 6);
        let can_cast = can_cast.as_ref().expect("contained invalid can_cast result");
        assert!(!can_cast.is_ok());
        assert_eq!(can_cast.code().to_string_lossy(), "pytorch_operation_failed");

        let promoted =
            ffi::terlan_exception_pytorch_typerelationsnative_promote_types_native(6, value);
        let promoted = promoted
            .as_ref()
            .expect("contained invalid promote_types result");
        assert!(!promoted.is_ok());
        assert_eq!(promoted.code().to_string_lossy(), "pytorch_operation_failed");
    }
}
