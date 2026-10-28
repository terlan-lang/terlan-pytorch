use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn assert_ok(value: UniquePtr<ffi::TerlanExceptionEnvelope>) {
    let value = value.as_ref().expect("exception envelope");
    assert!(
        value.is_ok(),
        "CUDA record_stream failed: {}",
        value.message()
    );
}

#[test]
fn cuda_record_stream_uses_libtorch_stream_identity() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() != Ok("cuda") {
        return;
    }

    let host = ffi::terlan_owned_value_pytorch_tensor_from_floats(&[1.0, 2.0], &[2]);
    let mut device =
        ffi::terlan_owned_value_pytorch_tensor_to_device(host.as_ref().expect("host tensor"), 1, 0);
    assert_eq!(
        ffi::tensor_device_type(device.as_ref().expect("CUDA tensor"))
            .expect("device type query must not throw"),
        1
    );
    assert_eq!(
        ffi::tensor_device_index(device.as_ref().expect("CUDA tensor"))
            .expect("device index query must not throw"),
        0
    );

    let stream = ffi::cuda_stream_create(0, false).expect("CUDA stream construction must not throw");
    assert!(!stream.is_null(), "CUDA stream construction returned null");
    assert_ok(ffi::terlan_mutation_pytorch_cuda_record_stream(
        device.pin_mut(),
        stream.as_ref().expect("CUDA stream"),
    ));
    ffi::cuda_stream_synchronize(stream.as_ref().expect("CUDA stream"))
        .expect("CUDA stream synchronization must not throw");
}
