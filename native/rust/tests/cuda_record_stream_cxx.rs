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
fn cuda_stream_reports_unavailable_without_visible_devices() {
    const CHILD: &str = "TERLAN_PYTORCH_TEST_NO_CUDA_CHILD";
    if std::env::var_os(CHILD).is_some() {
        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").unwrap(), "");
        for high_priority in [false, true] {
            let stream = ffi::cuda_stream_create(0, high_priority)
                .expect("no visible devices should report unavailable, not an invalid index");
            assert!(stream.is_null());
        }
        assert!(ffi::cuda_stream_create(-1, false).is_err());
        return;
    }

    // CUDA caches device discovery, so exercise the unavailable backend in a
    // fresh process without changing the environment of concurrent tests.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "cuda_stream_reports_unavailable_without_visible_devices",
        ])
        .env(CHILD, "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .expect("run isolated CUDA visibility regression");
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cuda_stream_rejects_out_of_range_device() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() != Ok("cuda") {
        return;
    }

    let error = match ffi::cuda_stream_create(i64::MAX, false) {
        Ok(_) => panic!("out-of-range CUDA device unexpectedly created a stream"),
        Err(error) => error,
    };
    assert!(
        error.what().contains("invalid CUDA device index"),
        "unexpected CUDA device rejection: {}",
        error.what()
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

    let stream =
        ffi::cuda_stream_create(0, false).expect("CUDA stream construction must not throw");
    assert!(!stream.is_null(), "CUDA stream construction returned null");
    assert_ok(ffi::terlan_mutation_pytorch_cuda_record_stream(
        device.pin_mut(),
        stream.as_ref().expect("CUDA stream"),
    ));
    ffi::cuda_stream_synchronize(stream.as_ref().expect("CUDA stream"))
        .expect("CUDA stream synchronization must not throw");
}
