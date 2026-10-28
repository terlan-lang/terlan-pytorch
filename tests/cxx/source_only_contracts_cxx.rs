use terlan_pytorch_cxx::ffi;

#[test]
fn generated_cxx_executes_source_only_public_contracts() {
    let input = ffi::terlan_owned_value_pytorch_tensor_from_floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let input_ref = input.as_ref().expect("source tensor");

    let converted = ffi::terlan_owned_value_pytorch_distance_chalf(input_ref);
    assert!(!converted.is_null());

    let scalar = ffi::terlan_owned_value_pytorch_scalartensorcompletion_scalar_tensor(true, 7, 0.0);
    assert!(!scalar.is_null());

    let mut relu_input =
        ffi::terlan_owned_value_pytorch_tensor_from_floats(&[-1.0, 0.0, 1.0, -2.0], &[2, 2]);
    let rrelu = ffi::terlan_mutation_pytorch_randomcompletion_rrelu_in_place(
        relu_input.pin_mut(),
        0.125,
        0.3333333333333333,
        false,
    );
    assert!(rrelu.as_ref().expect("rrelu result").is_ok());

    let chunks = ffi::terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources(
        input_ref, 2, 0,
    );
    let chunks_ref = chunks.as_ref().expect("unsafe_chunk result");
    assert!(
        ffi::terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_is_ok(
            chunks_ref,
        )
    );
    assert_eq!(
        ffi::terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_len(
            chunks_ref,
        ),
        2
    );
    assert!(
        !ffi::terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_element(
            chunks_ref, 0,
        )
        .is_null()
    );
}
