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

fn shape(tensor: &ffi::Tensor) -> Vec<i64> {
    ffi::terlan_collection_pytorch_tensor_shape(tensor)
        .as_ref()
        .expect("tensor shape projection")
        .iter()
        .copied()
        .collect()
}

#[test]
fn interpolate_matches_size_and_scale_forms() {
    let input = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let nearest = ffi::terlan_owned_value_pytorch_functionalcompletion_interpolate_configured(
        input.as_ref().expect("input"),
        true,
        &[4, 4],
        false,
        &[],
        "nearest",
        false,
        false,
        false,
        false,
        false,
    );
    assert_eq!(shape(nearest.as_ref().expect("nearest result")), [1, 1, 4, 4]);
    let bilinear = ffi::terlan_owned_value_pytorch_functionalcompletion_interpolate_configured(
        input.as_ref().expect("input"),
        false,
        &[],
        true,
        &[2.0, 2.0],
        "bilinear",
        true,
        false,
        false,
        false,
        false,
    );
    assert_eq!(shape(bilinear.as_ref().expect("bilinear result")), [1, 1, 4, 4]);
}
