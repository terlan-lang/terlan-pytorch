use terlan_pytorch_cxx::Tensor;

unsafe extern "C" {
    fn terlan_pytorch_test_packet_deleter_count() -> i64;
    fn terlan_pytorch_test_reset_packet_deleter_count();
    fn terlan_pytorch_test_export_packet_delete_count() -> i64;
    fn terlan_pytorch_test_reset_export_packet_delete_count();
}

fn reset_export_packet_delete_count() {
    // SAFETY: the package-private audit function only resets thread-local test state.
    unsafe { terlan_pytorch_test_reset_export_packet_delete_count() };
}

fn export_packet_delete_count() -> i64 {
    // SAFETY: the package-private audit function only reads thread-local test state.
    unsafe { terlan_pytorch_test_export_packet_delete_count() }
}

fn reset_packet_deleter_count() {
    // SAFETY: the package-private audit function only resets thread-local test state.
    unsafe { terlan_pytorch_test_reset_packet_deleter_count() };
}

fn packet_deleter_count() -> i64 {
    // SAFETY: the package-private audit function only reads thread-local test state.
    unsafe { terlan_pytorch_test_packet_deleter_count() }
}

fn float64_packet(values: &[f64], shape: &[u64], strides: &[u64]) -> Vec<u8> {
    assert_eq!(shape.len(), strides.len());
    let mut packet = Vec::new();
    packet.extend_from_slice(b"TNXP");
    packet.extend_from_slice(&[
        1,
        if cfg!(target_endian = "little") { 1 } else { 2 },
        2,
        64,
        1,
        0,
        1,
        0,
    ]);
    packet.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for dimension in shape {
        packet.extend_from_slice(&dimension.to_le_bytes());
    }
    for stride in strides {
        packet.extend_from_slice(&stride.to_le_bytes());
    }
    for value in values {
        packet.extend_from_slice(&value.to_ne_bytes());
    }
    packet
}

#[test]
fn float64_packet_constructs_an_owned_tensor() {
    let packet = float64_packet(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 1]);
    let tensor = Tensor::from_array(&packet).expect("import tensor packet");

    assert_eq!(tensor.shape().expect("copy shape"), vec![2, 2]);
}

#[test]
fn packet_manager_deleter_runs_once_on_success_and_rejection() {
    reset_packet_deleter_count();
    let packet = float64_packet(&[1.0, 2.0], &[2], &[1]);
    let tensor = Tensor::from_packet(&packet).expect("import owned packet");
    assert_eq!(packet_deleter_count(), 1);
    drop(tensor);
    assert_eq!(packet_deleter_count(), 1);

    reset_packet_deleter_count();
    let unsupported = scalar_packet(99, 8, &[1, 2]);
    assert!(Tensor::from_packet(&unsupported).is_err());
    assert_eq!(packet_deleter_count(), 1);
}

#[test]
fn zero_stride_packet_materializes_an_owned_broadcast_tensor() {
    let packet = float64_packet(&[1.0, 2.0, 3.0], &[4, 3], &[0, 1]);
    let tensor = Tensor::from_array(&packet).expect("import broadcast tensor packet");

    assert_eq!(tensor.shape().expect("copy shape"), vec![4, 3]);
    assert_eq!(tensor.numel().expect("logical elements"), 12);
}

#[test]
fn truncated_float64_packet_is_rejected_before_native_data_access() {
    let mut packet = float64_packet(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 1]);
    packet.pop();

    let error = match Tensor::from_array(&packet) {
        Ok(_) => panic!("truncated packet was accepted"),
        Err(error) => error,
    };
    assert_eq!(error.operation, "pytorch.tensor.import_tensor_packet");
    assert_eq!(error.status, 7013);
}

#[test]
fn malformed_packet_header_and_trailing_bytes_are_rejected() {
    let mut bad_magic = float64_packet(&[1.0], &[1], &[1]);
    bad_magic[0] = b'X';
    assert!(Tensor::from_array(&bad_magic).is_err());

    let mut trailing = float64_packet(&[1.0], &[1], &[1]);
    trailing.push(0);
    assert!(Tensor::from_array(&trailing).is_err());
}

#[test]
fn unsupported_packet_dtype_is_rejected() {
    let packet = scalar_packet(99, 8, &[1, 2]);
    assert!(Tensor::from_array(&packet).is_err());
}

fn scalar_packet(code: u8, bits: u8, values: &[u8]) -> Vec<u8> {
    tensor_packet(code, bits, values, &[2])
}

fn tensor_packet(code: u8, bits: u8, values: &[u8], shape: &[i64]) -> Vec<u8> {
    let mut packet = Vec::new();
    packet.extend_from_slice(b"TNXP");
    packet.extend_from_slice(&[
        1,
        if cfg!(target_endian = "little") { 1 } else { 2 },
        code,
        bits,
        1,
        0,
        1,
        0,
    ]);
    packet.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for dimension in shape {
        packet.extend_from_slice(&dimension.to_le_bytes());
    }
    let mut stride = 1i64;
    let mut strides = vec![0i64; shape.len()];
    for index in (0..shape.len()).rev() {
        strides[index] = stride;
        stride *= shape[index];
    }
    for stride in strides {
        packet.extend_from_slice(&stride.to_le_bytes());
    }
    packet.extend_from_slice(values);
    packet
}

#[test]
fn packet_import_supports_cuda_and_image_scalar_types() {
    let cases = [
        (2, 32, [1.0f32.to_ne_bytes(), 2.0f32.to_ne_bytes()].concat(), 6),
        (0, 32, [1i32.to_ne_bytes(), 2i32.to_ne_bytes()].concat(), 3),
        (1, 8, vec![1u8, 2u8], 0),
        (6, 8, vec![0u8, 1u8], 11),
    ];
    for (code, bits, values, expected_dtype) in cases {
        let tensor = Tensor::from_array(&scalar_packet(code, bits, &values)).unwrap_or_else(
            |error| panic!("import dtype code={code} bits={bits}: {error:?}"),
        );
        assert_eq!(tensor.shape().expect("copy shape"), vec![2]);
        assert_eq!(tensor.dtype().expect("read dtype"), expected_dtype);
    }
}

#[test]
fn packet_export_round_trips_every_shared_dtype_and_owns_its_bytes() {
    let cases = [
        (2, 32, [1.0f32.to_ne_bytes(), 2.0f32.to_ne_bytes(), 3.0f32.to_ne_bytes(), 4.0f32.to_ne_bytes()].concat()),
        (2, 64, [1.0f64.to_ne_bytes(), 2.0f64.to_ne_bytes(), 3.0f64.to_ne_bytes(), 4.0f64.to_ne_bytes()].concat()),
        (0, 32, [1i32.to_ne_bytes(), 2i32.to_ne_bytes(), 3i32.to_ne_bytes(), 4i32.to_ne_bytes()].concat()),
        (0, 64, [1i64.to_ne_bytes(), 2i64.to_ne_bytes(), 3i64.to_ne_bytes(), 4i64.to_ne_bytes()].concat()),
        (1, 8, vec![1u8, 2u8, 3u8, 4u8]),
        (6, 8, vec![0u8, 1u8, 1u8, 0u8]),
    ];
    for (code, bits, values) in cases {
        let expected = tensor_packet(code, bits, &values, &[2, 2]);
        let tensor = Tensor::from_packet(&expected).expect("import shared dtype");
        reset_export_packet_delete_count();
        let exported = tensor.to_packet().expect("export owned packet");
        assert_eq!(export_packet_delete_count(), 1);
        assert_eq!(exported, expected);
        drop(tensor);
        let first = Tensor::from_packet(&exported).expect("first independent import");
        let second = Tensor::from_packet(&exported).expect("reusable copied packet");
        assert_eq!(first.shape().expect("first shape"), vec![2, 2]);
        assert_eq!(second.shape().expect("second shape"), vec![2, 2]);
    }
}

#[test]
fn packet_export_materializes_noncontiguous_tensors_in_logical_order() {
    let tensor = Tensor::from_floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("source tensor");
    let transposed = tensor.transpose(0, 1).expect("noncontiguous transpose");
    let packet = transposed.to_packet().expect("contiguous packet snapshot");
    let expected_values = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]
        .into_iter()
        .flat_map(f64::to_ne_bytes)
        .collect::<Vec<_>>();
    assert_eq!(packet, tensor_packet(2, 64, &expected_values, &[3, 2]));
}
