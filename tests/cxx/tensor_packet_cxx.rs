use terlan_pytorch_cxx::ffi;

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
fn generated_cxx_round_trips_an_owned_contiguous_packet() {
    let packet = float64_packet(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 1]);
    let tensor = ffi::import_tensor_packet(&packet).expect("CXX exception");
    let tensor = tensor.as_ref().expect("valid packet must produce a tensor");
    let exported = ffi::export_tensor_packet(tensor).expect("CXX exception");
    let exported = exported
        .as_ref()
        .expect("valid tensor must produce an owned byte vector");
    assert_eq!(exported.as_slice(), packet);
}

#[test]
fn generated_cxx_accepts_a_zero_stride_broadcast_layout() {
    let packet = float64_packet(&[1.0, 2.0, 3.0], &[4, 3], &[0, 1]);
    assert!(
        !ffi::import_tensor_packet(&packet).expect("zero-stride packet must not throw")
            .is_null()
    );
}

#[test]
fn generated_cxx_contains_malformed_packet_failures() {
    let mut bad_magic = float64_packet(&[1.0], &[1], &[1]);
    bad_magic[0] = b'X';
    assert!(
        ffi::import_tensor_packet(&bad_magic).expect("malformed packet must be rejected without throwing")
            .is_null()
    );

    let mut truncated = float64_packet(&[1.0], &[1], &[1]);
    truncated.pop();
    assert!(
        ffi::import_tensor_packet(&truncated).expect("truncated packet must be rejected without throwing")
            .is_null()
    );

    let mut trailing = float64_packet(&[1.0], &[1], &[1]);
    trailing.push(0);
    assert!(
        ffi::import_tensor_packet(&trailing).expect("trailing bytes must be rejected without throwing")
            .is_null()
    );
}
