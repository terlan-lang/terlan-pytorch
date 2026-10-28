use cxx::UniquePtr;
use terlan_pytorch_cxx::ffi;

fn cpu_tensor(values: &[f64], dimensions: &[i64]) -> UniquePtr<ffi::Tensor> {
    let value = ffi::terlan_owned_value_pytorch_tensor_from_floats(values, dimensions);
    assert!(!value.is_null(), "tensor construction returned null");
    ffi::terlan_owned_value_pytorch_tensor_to_dtype(value.as_ref().expect("constructed tensor"), 6)
}

fn quantize(value: &ffi::Tensor) -> UniquePtr<ffi::Tensor> {
    let result = ffi::terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor(
        value, 0.1, 0, 12,
    );
    assert!(!result.is_null(), "quantization returned null");
    result
}

fn shape(value: &ffi::Tensor) -> Vec<i64> {
    ffi::terlan_collection_pytorch_tensor_shape(value)
        .as_ref()
        .expect("shape projection")
        .iter()
        .copied()
        .collect()
}

fn assert_ok(value: UniquePtr<ffi::TerlanExceptionEnvelope>) {
    let value = value.as_ref().expect("exception envelope");
    assert!(value.is_ok(), "native mutation failed: {}", value.message());
}

#[test]
fn quantized_pooling_and_batch_norm_match_libtorch_cpu() {
    if std::env::var("TERLAN_PYTORCH_TEST_DEVICE").as_deref() == Ok("cuda") {
        return;
    }

    let source1d = cpu_tensor(&[0.0, 1.0, 2.0, 3.0], &[1, 1, 4]);
    let source2d = cpu_tensor(
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
        &[1, 1, 4, 4],
    );
    let source3d = cpu_tensor(
        &(0..64).map(|value| value as f64).collect::<Vec<_>>(),
        &[1, 1, 4, 4, 4],
    );
    let q1d = quantize(source1d.as_ref().expect("1d source"));
    let q2d = quantize(source2d.as_ref().expect("2d source"));
    let q3d = quantize(source3d.as_ref().expect("3d source"));

    let pool1d = ffi::terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool1d(
        q1d.as_ref().expect("1d quantized"),
        &[2],
        &[2],
        &[0],
        &[1],
        false,
    );
    let pool2d = ffi::terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool2d(
        q2d.as_ref().expect("2d quantized"),
        &[2, 2],
        &[2, 2],
        &[0, 0],
        &[1, 1],
        false,
    );
    let pool3d = ffi::terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool3d(
        q3d.as_ref().expect("3d quantized"),
        &[2, 2, 2],
        &[2, 2, 2],
        &[0, 0, 0],
        &[1, 1, 1],
        false,
    );
    assert_eq!(shape(pool1d.as_ref().expect("1d pool")), [1, 1, 2]);
    assert_eq!(shape(pool2d.as_ref().expect("2d pool")), [1, 1, 2, 2]);
    assert_eq!(shape(pool3d.as_ref().expect("3d pool")), [1, 1, 2, 2, 2]);

    let mut pool1d_out = ffi::terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool1d(
        q1d.as_ref().expect("1d quantized"),
        &[2],
        &[2],
        &[0],
        &[1],
        false,
    );
    let mut pool2d_out = ffi::terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool2d(
        q2d.as_ref().expect("2d quantized"),
        &[2, 2],
        &[2, 2],
        &[0, 0],
        &[1, 1],
        false,
    );
    let mut pool3d_out = ffi::terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool3d(
        q3d.as_ref().expect("3d quantized"),
        &[2, 2, 2],
        &[2, 2, 2],
        &[0, 0, 0],
        &[1, 1, 1],
        false,
    );
    assert_ok(
        ffi::terlan_mutation_pytorch_quantizedpooling_quantized_max_pool1d_out(
            pool1d_out.pin_mut(),
            q1d.as_ref().expect("1d quantized"),
            &[2],
            &[2],
            &[0],
            &[1],
            false,
        ),
    );
    assert_ok(
        ffi::terlan_mutation_pytorch_quantizedpooling_quantized_max_pool2d_out(
            pool2d_out.pin_mut(),
            q2d.as_ref().expect("2d quantized"),
            &[2, 2],
            &[2, 2],
            &[0, 0],
            &[1, 1],
            false,
        ),
    );
    assert_ok(
        ffi::terlan_mutation_pytorch_quantizedpooling_quantized_max_pool3d_out(
            pool3d_out.pin_mut(),
            q3d.as_ref().expect("3d quantized"),
            &[2, 2, 2],
            &[2, 2, 2],
            &[0, 0, 0],
            &[1, 1, 1],
            false,
        ),
    );

    let qbn_source = cpu_tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 2, 2]);
    let qbn_input = quantize(qbn_source.as_ref().expect("batch norm source"));
    let weight = cpu_tensor(&[1.0, 1.0], &[2]);
    let bias = cpu_tensor(&[0.0, 0.0], &[2]);
    let mean = cpu_tensor(&[0.0, 0.0], &[2]);
    let var = cpu_tensor(&[1.0, 1.0], &[2]);
    let qbn = ffi::terlan_owned_value_pytorch_quantizednormalization_quantized_batch_norm(
        qbn_input.as_ref().expect("batch norm input"),
        true,
        weight.as_ref().expect("weight"),
        true,
        bias.as_ref().expect("bias"),
        mean.as_ref().expect("mean"),
        var.as_ref().expect("variance"),
        0.00001,
        0.1,
        0,
    );
    assert_eq!(shape(qbn.as_ref().expect("batch norm result")), [2, 2, 2]);
    let mut qbn_out = ffi::terlan_owned_value_pytorch_quantizednormalization_quantized_batch_norm(
        qbn_input.as_ref().expect("batch norm input"),
        true,
        weight.as_ref().expect("weight"),
        true,
        bias.as_ref().expect("bias"),
        mean.as_ref().expect("mean"),
        var.as_ref().expect("variance"),
        0.00001,
        0.1,
        0,
    );
    assert_ok(
        ffi::terlan_mutation_pytorch_quantizednormalization_quantized_batch_norm_out(
            qbn_out.pin_mut(),
            qbn_input.as_ref().expect("batch norm input"),
            true,
            weight.as_ref().expect("weight"),
            true,
            bias.as_ref().expect("bias"),
            mean.as_ref().expect("mean"),
            var.as_ref().expect("variance"),
            0.00001,
            0.1,
            0,
        ),
    );
    assert_eq!(
        shape(qbn_out.as_ref().expect("batch norm output")),
        [2, 2, 2]
    );
}
