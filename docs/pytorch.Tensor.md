# pytorch.Tensor

CPU tensor lifecycle and scalar metadata through LibTorch's stable C ABI.

## `from_int`

Creates a scalar int64 tensor.

- C symbol: `aoti_torch_scalar_to_tensor_int64`
- NativeBoundary operation: `pytorch.tensor.from_int`
- Ownership: `opaque_handle`

## `from_float`

Creates a scalar float64 tensor.

- C symbol: `aoti_torch_scalar_to_tensor_float64`
- NativeBoundary operation: `pytorch.tensor.from_float`
- Ownership: `opaque_handle`

## `from_bool`

Creates a scalar bool tensor.

- C symbol: `aoti_torch_scalar_to_tensor_bool`
- NativeBoundary operation: `pytorch.tensor.from_bool`
- Ownership: `opaque_handle`

## `from_floats`

Copies row-major float64 values into an independently owned CPU tensor with the requested shape.

- C symbol: `terlan_pytorch_tensor_from_f64`
- NativeBoundary operation: `pytorch.tensor.from_floats`
- Ownership: `opaque_handle`

## `from_ints`

Copies row-major int64 values into an independently owned CPU tensor with the requested shape.

- C symbol: `terlan_pytorch_tensor_from_i64`
- NativeBoundary operation: `pytorch.tensor.from_ints`
- Ownership: `opaque_handle`

## `full_float`

Creates a CPU float64 tensor with the requested shape and repeated value.

- C symbol: `aoti_torch_aten_full`
- NativeBoundary operation: `pytorch.tensor.full_float`
- Ownership: `opaque_handle`

## `item`

Reads the scalar int64 item.

- C symbol: `aoti_torch_item_int64`
- NativeBoundary operation: `pytorch.tensor.item`
- Ownership: `borrowed_handle`

## `item_float`

Reads the scalar float64 item.

- C symbol: `aoti_torch_item_float64`
- NativeBoundary operation: `pytorch.tensor.item_float`
- Ownership: `borrowed_handle`

## `item_bool`

Reads the scalar bool item.

- C symbol: `aoti_torch_item_bool`
- NativeBoundary operation: `pytorch.tensor.item_bool`
- Ownership: `borrowed_handle`

## `dim`

Returns the tensor rank.

- C symbol: `aoti_torch_get_dim`
- NativeBoundary operation: `pytorch.tensor.dim`
- Ownership: `borrowed_handle`

## `numel`

Returns the tensor element count.

- C symbol: `aoti_torch_get_numel`
- NativeBoundary operation: `pytorch.tensor.numel`
- Ownership: `borrowed_handle`

## `shape`

Immediately copies LibTorch's borrowed size array into an owned Terlan list.

- C symbol: `aoti_torch_get_sizes`
- NativeBoundary operation: `pytorch.tensor.shape`
- Ownership: `value`

## `dtype`

Returns LibTorch's runtime dtype identifier.

- C symbol: `aoti_torch_get_dtype`
- NativeBoundary operation: `pytorch.tensor.dtype`
- Ownership: `borrowed_handle`

## `device_type`

Returns LibTorch's runtime device type identifier.

- C symbol: `aoti_torch_get_device_type`
- NativeBoundary operation: `pytorch.tensor.device_type`
- Ownership: `borrowed_handle`

## `device_index`

Returns the device index.

- C symbol: `aoti_torch_get_device_index`
- NativeBoundary operation: `pytorch.tensor.device_index`
- Ownership: `borrowed_handle`

## `clone`

Clones a tensor through LibTorch's versioned StableIValue dispatcher.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.clone`
- Ownership: `opaque_handle`

## `unsqueeze`

Inserts a dimension through a signed StableIValue dispatcher argument.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.unsqueeze`
- Ownership: `opaque_handle`

## `matmul`

Multiplies two tensors through independently owned StableIValue handle inputs.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.matmul`
- Ownership: `opaque_handle`

## `transpose`

Transposes two dimensions through a named overload and mixed StableIValue stack.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.transpose`
- Ownership: `opaque_handle`

## `multiply`

Multiplies two tensors through the exact named Tensor dispatcher overload.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.multiply`
- Ownership: `opaque_handle`

## `add`

Adds a scaled tensor with CPU broadcasting through LibTorch's generated stable C shim.

- C symbol: `aoti_torch_cpu_add_Tensor`
- NativeBoundary operation: `pytorch.tensor.add`
- Ownership: `opaque_handle`

## `subtract`

Subtracts a scaled tensor through LibTorch's generated stable C shim.

- C symbol: `aoti_torch_aten_subtract_Tensor`
- NativeBoundary operation: `pytorch.tensor.subtract`
- Ownership: `opaque_handle`

## `relu`

Applies the exact default aten::relu operation through the StableIValue dispatcher.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.relu`
- Ownership: `opaque_handle`

## `gelu`

Applies exact aten::gelu with the fixed approximate=none schema default through an owned stable string.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.gelu`
- Ownership: `opaque_handle`

## `softmax`

Normalizes logits through the exact aten::softmax.int dispatcher overload with inherited dtype.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.softmax`
- Ownership: `opaque_handle`

## `argmax`

Returns int64 maximum-value indices through the exact aten::argmax schema, including its owned optional dimension and boolean keep-dimension arguments.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.argmax`
- Ownership: `opaque_handle`

## `embedding`

Looks up int64 indices in a rank-2 embedding table through exact aten::embedding.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.embedding`
- Ownership: `opaque_handle`

## `batch_norm`

Applies exact aten::batch_norm with explicit affine parameters and running statistics.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.batch_norm`
- Ownership: `opaque_handle`

## `layer_norm`

Applies exact aten::layer_norm over an owned normalized-shape list with explicit affine parameters.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.layer_norm`
- Ownership: `opaque_handle`

## `mse_loss`

Computes elementwise squared error through exact aten::mse_loss; reduction is 0 for none, 1 for mean, or 2 for sum.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.mse_loss`
- Ownership: `opaque_handle`

## `mse_loss_gradient`

Computes the exact aten::mse_loss_backward gradient with respect to predicted values.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.mse_loss_gradient`
- Ownership: `opaque_handle`

## `cross_entropy_loss`

Computes classification loss through exact aten::cross_entropy_loss with no class-weight tensor.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.cross_entropy_loss`
- Ownership: `opaque_handle`

## `narrow`

Returns an independently owned narrowed tensor view through LibTorch's generated stable C shim.

- C symbol: `aoti_torch_aten_narrow`
- NativeBoundary operation: `pytorch.tensor.narrow`
- Ownership: `opaque_handle`

## `flatten`

Flattens an inclusive dimension range through the exact named StableIValue dispatcher overload.

- C symbol: `torch_call_dispatcher`
- NativeBoundary operation: `pytorch.tensor.flatten`
- Ownership: `opaque_handle`

## `amax`

Reduces over copied dimensions through LibTorch's generated stable C amax shim.

- C symbol: `aoti_torch_aten_amax`
- NativeBoundary operation: `pytorch.tensor.amax`
- Ownership: `opaque_handle`

## `conv2d`

Applies NCHW 2D convolution with required bias through LibTorch's generated CPU stable C shim.

- C symbol: `terlan_pytorch_conv2d`
- NativeBoundary operation: `pytorch.tensor.conv2d`
- Ownership: `opaque_handle`

## `adaptive_avg_pool2d`

Reduces NCHW spatial dimensions to an exact output size through LibTorch's generated CPU adaptive-average-pooling shim.

- C symbol: `aoti_torch_cpu__adaptive_avg_pool2d`
- NativeBoundary operation: `pytorch.tensor.adaptive_avg_pool2d`
- Ownership: `opaque_handle`

## `load_torchscript`

Loads a serialized TorchScript archive into a reusable thread-confined model resource.

- C symbol: `terlan_pytorch_load_torchscript`
- NativeBoundary operation: `pytorch.torchscript.load`
- Ownership: `opaque_handle`

## `torchscript_forward`

Runs a loaded single-input TorchScript model and returns an independently owned tensor.

- C symbol: `terlan_pytorch_torchscript_forward`
- NativeBoundary operation: `pytorch.torchscript.forward`
- Ownership: `opaque_handle`

## `torchscript_forward2`

Runs a loaded two-positional-tensor TorchScript model and returns an independently owned tensor.

- C symbol: `terlan_pytorch_torchscript_forward2`
- NativeBoundary operation: `pytorch.torchscript.forward2`
- Ownership: `opaque_handle`

## `torchscript_forward_pair`

Runs a loaded model whose forward result is exactly a two-tensor tuple and owns that result container.

- C symbol: `terlan_pytorch_torchscript_forward_pair`
- NativeBoundary operation: `pytorch.torchscript.forward_pair`
- Ownership: `opaque_handle`

## `torchscript_forward_float`

Runs a loaded model whose forward result is exactly a TorchScript float and copies it into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_float`
- NativeBoundary operation: `pytorch.torchscript.forward_float`
- Ownership: `value`

## `torchscript_forward_int`

Runs a loaded model whose forward result is exactly a TorchScript integer and copies it into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_int`
- NativeBoundary operation: `pytorch.torchscript.forward_int`
- Ownership: `value`

## `torchscript_forward_bool`

Runs a loaded model whose forward result is exactly a TorchScript boolean and copies it into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_bool`
- NativeBoundary operation: `pytorch.torchscript.forward_bool`
- Ownership: `value`

## `torchscript_forward_string`

Runs a loaded model whose forward result is exactly a TorchScript string and immediately copies its owned UTF-8 bytes into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_string`
- NativeBoundary operation: `pytorch.torchscript.forward_string`
- Ownership: `value`

## `torchscript_forward_int_list`

Runs a loaded model whose forward result is exactly a TorchScript integer list and immediately copies its owned elements into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_int_list`
- NativeBoundary operation: `pytorch.torchscript.forward_int_list`
- Ownership: `value`

## `torchscript_forward_float_list`

Runs a loaded model whose forward result is exactly a TorchScript floating-point list and immediately copies its owned elements into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_float_list`
- NativeBoundary operation: `pytorch.torchscript.forward_float_list`
- Ownership: `value`

## `torchscript_forward_bool_list`

Runs a loaded model whose forward result is exactly a TorchScript boolean list, validates its portable byte encoding, and copies it into Terlan.

- C symbol: `terlan_pytorch_torchscript_forward_bool_list`
- NativeBoundary operation: `pytorch.torchscript.forward_bool_list`
- Ownership: `value`

## `torchscript_forward_string_list`

Runs a loaded model whose forward result is exactly a TorchScript string list and copies each length-delimited UTF-8 value into Terlan before native destruction.

- C symbol: `terlan_pytorch_torchscript_forward_string_list`
- NativeBoundary operation: `pytorch.torchscript.forward_string_list`
- Ownership: `value`

## `tensor_pair_first`

Returns an independently owned handle for the first tensor in a model-output pair.

- C symbol: `terlan_pytorch_tensor_pair_first`
- NativeBoundary operation: `pytorch.torchscript.tensor_pair_first`
- Ownership: `opaque_handle`

## `tensor_pair_second`

Returns an independently owned handle for the second tensor in a model-output pair.

- C symbol: `terlan_pytorch_tensor_pair_second`
- NativeBoundary operation: `pytorch.torchscript.tensor_pair_second`
- Ownership: `opaque_handle`

## `run_torchscript`

Loads, runs, and disposes a TorchScript module as a stateless convenience call.

- C symbol: `terlan_pytorch_run_torchscript`
- NativeBoundary operation: `pytorch.tensor.run_torchscript`
- Ownership: `opaque_handle`

## `dispose_model`

Releases the loaded TorchScript model resource.

- C symbol: `terlan_pytorch_delete_torchscript_model`
- NativeBoundary operation: `pytorch.torchscript.dispose`
- Ownership: `dispose_handle`

## `dispose_tensor_pair`

Releases the owned two-tensor model-output container.

- C symbol: `terlan_pytorch_delete_tensor_pair`
- NativeBoundary operation: `pytorch.torchscript.dispose_tensor_pair`
- Ownership: `dispose_handle`

## `dispose`

Releases the owned tensor handle.

- C symbol: `aoti_torch_delete_tensor_object`
- NativeBoundary operation: `pytorch.tensor.dispose`
- Ownership: `dispose_handle`

## `abi_version`

Returns the LibTorch ABI version exported by the runtime.

- C symbol: `aoti_torch_abi_version`
- NativeBoundary operation: `pytorch.abi_version`
- Ownership: `value`
