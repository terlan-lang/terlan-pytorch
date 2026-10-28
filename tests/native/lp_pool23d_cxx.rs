use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn tensor(values: &[f64], shape: &[i64]) -> UniquePtr<ffi::Tensor> {
    let cpu = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, shape);
    assert!(!cpu.is_null(), "tensor construction returned null");
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        ffi::terlan_owned_value_pytorch_tensor_to_device(
            cpu.as_ref().expect("constructed tensor"),
            1,
            0,
        )
    } else {
        cpu
    }
}

fn assert_allclose(label: &str, actual: &ffi::Tensor, values: &[f64], shape: &[i64]) {
    let expected = tensor(values, shape);
    let result = ffi::terlan_exception_pytorch_typerelationsnative_allclose_native(
        actual,
        expected.as_ref().expect("expected tensor"),
        1e-6,
        1e-6,
        false,
    );
    let result = result.as_ref().expect("contained allclose result");
    assert!(result.is_ok(), "allclose failed: {}", result.message());
    assert!(result.bool_value(), "{label}: Tensor values differ");
}

#[test]
fn lp_pool2d_and_3d_match_lib_torch_frontend() {
    let input2d = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[1, 1, 4, 4],
    );
    let result2d = ffi::terlan_owned_value_pytorch_functionalcompletion_lp_pool2d_configured(
        input2d.as_ref().expect("2d input"),
        true,
        2,
        0.0,
        2,
        false,
        2,
        false,
    );
    assert_allclose(
        "lp_pool2d default stride",
        result2d.as_ref().expect("2d result"),
        &[8.124038696289062, 11.747340202331543, 23.366641998291016, 27.312999725341797],
        &[1, 1, 2, 2],
    );

    let input3d = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[1, 1, 2, 2, 4],
    );
    let result3d = ffi::terlan_owned_value_pytorch_functionalcompletion_lp_pool3d_configured(
        input3d.as_ref().expect("3d input"),
        false,
        0,
        2.0,
        2,
        true,
        1,
        false,
    );
    assert_allclose(
        "lp_pool3d explicit stride",
        result3d.as_ref().expect("3d result"),
        &[24.73863410949707, 27.20294189453125, 29.73213768005371],
        &[1, 1, 1, 1, 3],
    );
}
