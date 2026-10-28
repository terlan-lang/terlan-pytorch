# pytorch.Tensor

Generated CXX-owned LibTorch tensor resources and migration support.

## `ones`

Creates a CPU tensor filled with ones.

- C++ symbol: `ones`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.ones`
- Ownership: `owned_handle`

## `ones_dtype`

Creates a CPU tensor of a reviewed LibTorch dtype filled with ones.

- C++ symbol: `ones`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.ones_dtype`
- Ownership: `owned_handle`

## `ones_layout`

Creates a tensor with reviewed dtype and layout values.

- C++ symbol: `ones`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.ones_layout`
- Ownership: `owned_handle`

## `ones_device`

Creates a tensor with reviewed dtype, layout, and parsed device values.

- C++ symbol: `ones`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.ones_device`
- Ownership: `owned_handle`

## `ones_options`

Creates a tensor with all optional LibTorch factory controls supplied.

- C++ symbol: `ones`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.ones_options`
- Ownership: `owned_handle`

## `item`

Returns the single tensor element as a scalar.

- C++ symbol: `item`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.item`
- Ownership: `owned_handle`

## `mul_int`

Returns the elementwise product with an integer scalar.

- C++ symbol: `mul`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.mul_int`
- Ownership: `owned_handle`

## `scalar_to_int`

Converts a scalar to an integer with contained errors.

- C++ symbol: `to`
- NativeBoundary operation: `pytorch.cxx_migration.scalar.to_int`
- Ownership: `value`

## `broadcast_tensors`

Broadcasts tensors to a common shape with generated list-input and list-output ownership.

- C++ symbol: `broadcast_tensors`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.broadcast_tensors`
- Ownership: `owned_handle`

## `meshgrid`

Creates coordinate grids using LibTorch's default indexing.

- C++ symbol: `meshgrid`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.meshgrid`
- Ownership: `owned_handle`

## `meshgrid_indexing`

Creates coordinate grids using explicit xy or ij indexing.

- C++ symbol: `meshgrid`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.meshgrid_indexing`
- Ownership: `owned_handle`

## `scalar_to_float`

Converts a scalar to a double-precision Float through generated exception containment.

- C++ symbol: `to`
- NativeBoundary operation: `pytorch.cxx_migration.scalar.to_float`
- Ownership: `value`

## `dtype`

Returns the tensor dtype as a reviewed symbolic value.

- C++ symbol: `scalar_type`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.dtype`
- Ownership: `borrowed_handle`

## `layout`

Returns the tensor layout as a reviewed symbolic value.

- C++ symbol: `layout`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.layout`
- Ownership: `borrowed_handle`

## `device`

Returns the tensor device as its canonical LibTorch spelling.

- C++ symbol: `device`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.device`
- Ownership: `borrowed_handle`

## `is_defined`

Returns whether the tensor has a defined LibTorch implementation.

- C++ symbol: `defined`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_defined`
- Ownership: `borrowed_handle`

## `is_cpu`

Returns whether the tensor uses the CPU backend.

- C++ symbol: `is_cpu`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_cpu`
- Ownership: `borrowed_handle`

## `is_cuda`

Returns whether the tensor uses the CUDA backend.

- C++ symbol: `is_cuda`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_cuda`
- Ownership: `borrowed_handle`

## `requires_grad`

Returns whether autograd records operations on the tensor.

- C++ symbol: `requires_grad`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.requires_grad`
- Ownership: `borrowed_handle`

## `is_leaf`

Returns whether the tensor is an autograd leaf.

- C++ symbol: `is_leaf`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_leaf`
- Ownership: `borrowed_handle`

## `dispose_scalar`

Drops the uniquely owned LibTorch scalar.

- NativeBoundary operation: `pytorch.cxx_migration.scalar.dispose`
- Ownership: `dispose_handle`

## `abs`

Returns the elementwise absolute-value tensor.

- C++ symbol: `abs`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.abs`
- Ownership: `owned_handle`

## `detach`

Returns a tensor detached from the autograd graph.

- C++ symbol: `detach`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.detach`
- Ownership: `owned_handle`

## `ceil`

Returns the elementwise ceiling.

- C++ symbol: `ceil`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.ceil`
- Ownership: `owned_handle`

## `floor`

Returns the elementwise floor.

- C++ symbol: `floor`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.floor`
- Ownership: `owned_handle`

## `reciprocal`

Returns the elementwise reciprocal.

- C++ symbol: `reciprocal`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.reciprocal`
- Ownership: `owned_handle`

## `round`

Returns the elementwise rounded value.

- C++ symbol: `round`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.round`
- Ownership: `owned_handle`

## `sign`

Returns the elementwise sign.

- C++ symbol: `sign`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.sign`
- Ownership: `owned_handle`

## `square`

Returns the elementwise square.

- C++ symbol: `square`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.square`
- Ownership: `owned_handle`

## `trunc`

Returns the elementwise truncation.

- C++ symbol: `trunc`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.trunc`
- Ownership: `owned_handle`

## `has_storage`

Returns whether the tensor has storage.

- C++ symbol: `has_storage`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.has_storage`
- Ownership: `borrowed_handle`

## `is_complex`

Returns whether the tensor has a complex dtype.

- C++ symbol: `is_complex`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_complex`
- Ownership: `borrowed_handle`

## `is_floating_point`

Returns whether the tensor has a floating-point dtype.

- C++ symbol: `is_floating_point`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_floating_point`
- Ownership: `borrowed_handle`

## `is_quantized`

Returns whether the tensor uses a quantized backend.

- C++ symbol: `is_quantized`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_quantized`
- Ownership: `borrowed_handle`

## `is_signed`

Returns whether the tensor has a signed dtype.

- C++ symbol: `is_signed`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_signed`
- Ownership: `borrowed_handle`

## `is_sparse`

Returns whether the tensor uses a sparse backend.

- C++ symbol: `is_sparse`
- NativeBoundary operation: `pytorch.cxx_migration.tensor.is_sparse`
- Ownership: `borrowed_handle`

## `from_array`

Compatibility alias that validates and copies a package-neutral TNXP value into independently owned LibTorch storage.

- C++ symbol: `import_tensor_packet`
- NativeBoundary operation: `pytorch.tensor.import_tensor_packet`
- Ownership: `owned_handle`

## `from_packet`

Validates and copies a package-neutral TNXP value into independently owned LibTorch storage.

- C++ symbol: `import_tensor_packet`
- NativeBoundary operation: `pytorch.tensor.import_packet`
- Ownership: `owned_handle`

## `to_packet`

Copies a contiguous CPU snapshot into a package-neutral TNXP value.

- C++ symbol: `export_tensor_packet`
- NativeBoundary operation: `pytorch.tensor.export_packet`
- Ownership: `value`

## `shape`

Copies the tensor shape from LibTorch's borrowed size view.

- C++ symbol: `sizes`
- NativeBoundary operation: `pytorch.tensor.shape`
- Ownership: `borrowed_handle`

## `strides`

Copies the tensor strides from LibTorch's borrowed stride view.

- C++ symbol: `strides`
- NativeBoundary operation: `pytorch.tensor.strides`
- Ownership: `borrowed_handle`

## `is_contiguous`

Returns whether the tensor is contiguous in LibTorch's default memory format.

- C++ symbol: `is_contiguous`
- NativeBoundary operation: `pytorch.tensor.is_contiguous`
- Ownership: `borrowed_handle`

## `is_contiguous_memory_format_native`

Private memory-format contiguity leaf for the public adapter.

- C++ symbol: `is_contiguous`
- NativeBoundary operation: `pytorch.tensor.is_contiguous_memory_format_native`
- Ownership: `borrowed_handle`

## `is_contiguous_memory_format`

Returns whether the tensor is contiguous in the requested LibTorch memory format.

- Implementation: generated Terlan composition

## `clone`

Returns a copy using the default memory-format policy.

- C++ symbol: `clone`
- NativeBoundary operation: `pytorch.tensor.clone`
- Ownership: `owned_handle`

## `relu`

Returns the elementwise rectified tensor.

- C++ symbol: `relu`
- NativeBoundary operation: `pytorch.tensor.relu`
- Ownership: `owned_handle`

## `requires_grad_in_place`

Updates leaf autograd participation through exact aten::requires_grad_.

- C++ symbol: `requires_grad_`
- NativeBoundary operation: `pytorch.tensor.requires_grad_in_place`
- Ownership: `mutable_handle`

## `multiply`

Returns the elementwise product with another tensor.

- C++ symbol: `mul`
- NativeBoundary operation: `pytorch.tensor.multiply`
- Ownership: `owned_handle`

## `multiply_scalar`

Returns the elementwise product with a floating-point scalar.

- C++ symbol: `mul`
- NativeBoundary operation: `pytorch.tensor.multiply_scalar`
- Ownership: `owned_handle`

## `negate`

Returns the elementwise negated tensor.

- C++ symbol: `neg`
- NativeBoundary operation: `pytorch.tensor.negate`
- Ownership: `owned_handle`

## `dim`

Returns the tensor rank through generated exception containment.

- C++ symbol: `dim`
- NativeBoundary operation: `pytorch.tensor.dim`
- Ownership: `borrowed_handle`

## `numel`

Returns the tensor element count through generated exception containment.

- C++ symbol: `numel`
- NativeBoundary operation: `pytorch.tensor.numel`
- Ownership: `borrowed_handle`

## `dispose`

Drops the uniquely owned LibTorch tensor.

- NativeBoundary operation: `pytorch.tensor.dispose`
- Ownership: `dispose_handle`

## `unsqueeze`

Generated direct ATen Tensor adapter.

- C++ symbol: `unsqueeze`
- NativeBoundary operation: `pytorch.tensor.unsqueeze`
- Ownership: `owned_handle`

## `matmul`

Generated direct ATen Tensor adapter.

- C++ symbol: `matmul`
- NativeBoundary operation: `pytorch.tensor.matmul`
- Ownership: `owned_handle`

## `transpose`

Generated direct ATen Tensor adapter.

- C++ symbol: `transpose`
- NativeBoundary operation: `pytorch.tensor.transpose`
- Ownership: `owned_handle`

## `exp`

Generated direct ATen Tensor adapter.

- C++ symbol: `exp`
- NativeBoundary operation: `pytorch.tensor.exp`
- Ownership: `owned_handle`

## `sqrt`

Generated direct ATen Tensor adapter.

- C++ symbol: `sqrt`
- NativeBoundary operation: `pytorch.tensor.sqrt`
- Ownership: `owned_handle`

## `log`

Generated direct ATen Tensor adapter.

- C++ symbol: `log`
- NativeBoundary operation: `pytorch.tensor.log`
- Ownership: `owned_handle`

## `minimum`

Generated direct ATen Tensor adapter.

- C++ symbol: `minimum`
- NativeBoundary operation: `pytorch.tensor.minimum`
- Ownership: `owned_handle`

## `gather`

Generated direct ATen Tensor adapter.

- C++ symbol: `gather`
- NativeBoundary operation: `pytorch.tensor.gather`
- Ownership: `owned_handle`

## `gather_out`

Generated direct ATen Tensor adapter.

- C++ symbol: `gather_out`
- NativeBoundary operation: `pytorch.tensor.gather_out`
- Ownership: `mutable_handle`

## `mean`

Generated direct ATen Tensor adapter.

- C++ symbol: `mean`
- NativeBoundary operation: `pytorch.tensor.mean`
- Ownership: `owned_handle`

## `sum`

Generated direct ATen Tensor adapter.

- C++ symbol: `sum`
- NativeBoundary operation: `pytorch.tensor.sum`
- Ownership: `owned_handle`

## `mean_dimensions`

Generated direct ATen Tensor adapter.

- C++ symbol: `mean`
- NativeBoundary operation: `pytorch.tensor.mean_dimensions`
- Ownership: `owned_handle`

## `sum_dimensions`

Generated direct ATen Tensor adapter.

- C++ symbol: `sum`
- NativeBoundary operation: `pytorch.tensor.sum_dimensions`
- Ownership: `owned_handle`

## `gelu`

Generated direct ATen Tensor adapter.

- C++ symbol: `gelu`
- NativeBoundary operation: `pytorch.tensor.gelu`
- Ownership: `owned_handle`

## `softmax`

Generated direct ATen Tensor adapter.

- C++ symbol: `softmax`
- NativeBoundary operation: `pytorch.tensor.softmax`
- Ownership: `owned_handle`

## `argmax`

Generated direct ATen Tensor adapter.

- C++ symbol: `argmax`
- NativeBoundary operation: `pytorch.tensor.argmax`
- Ownership: `owned_handle`

## `embedding`

Generated direct ATen Tensor adapter.

- C++ symbol: `embedding`
- NativeBoundary operation: `pytorch.tensor.embedding`
- Ownership: `owned_handle`

## `embedding_out`

Generated direct ATen Tensor adapter.

- C++ symbol: `embedding_out`
- NativeBoundary operation: `pytorch.tensor.embedding_out`
- Ownership: `mutable_handle`

## `embedding_renorm`

Generated direct ATen Tensor adapter.

- C++ symbol: `embedding_renorm`
- NativeBoundary operation: `pytorch.tensor.embedding_renorm`
- Ownership: `owned_handle`

## `embedding_renorm_in_place`

Generated direct ATen Tensor adapter.

- C++ symbol: `embedding_renorm_`
- NativeBoundary operation: `pytorch.tensor.embedding_renorm_in_place`
- Ownership: `mutable_handle`

## `embedding_renorm_out`

Generated direct ATen Tensor adapter.

- C++ symbol: `embedding_renorm_out`
- NativeBoundary operation: `pytorch.tensor.embedding_renorm_out`
- Ownership: `mutable_handle`

## `batch_norm`

Generated direct ATen Tensor adapter.

- C++ symbol: `batch_norm`
- NativeBoundary operation: `pytorch.tensor.batch_norm`
- Ownership: `owned_handle`

## `layer_norm`

Generated direct ATen Tensor adapter.

- C++ symbol: `layer_norm`
- NativeBoundary operation: `pytorch.tensor.layer_norm`
- Ownership: `owned_handle`

## `mse_loss`

Generated direct ATen Tensor adapter.

- C++ symbol: `mse_loss`
- NativeBoundary operation: `pytorch.tensor.mse_loss`
- Ownership: `owned_handle`

## `narrow`

Generated direct ATen Tensor adapter.

- C++ symbol: `narrow`
- NativeBoundary operation: `pytorch.tensor.narrow`
- Ownership: `owned_handle`

## `flatten`

Generated direct ATen Tensor adapter.

- C++ symbol: `flatten`
- NativeBoundary operation: `pytorch.tensor.flatten`
- Ownership: `owned_handle`

## `amax`

Generated direct ATen Tensor adapter.

- C++ symbol: `amax`
- NativeBoundary operation: `pytorch.tensor.amax`
- Ownership: `owned_handle`

## `adaptive_avg_pool2d`

Generated direct ATen Tensor adapter.

- C++ symbol: `adaptive_avg_pool2d`
- NativeBoundary operation: `pytorch.tensor.adaptive_avg_pool2d`
- Ownership: `owned_handle`

## `copy`

Generated direct ATen Tensor adapter.

- C++ symbol: `copy`
- NativeBoundary operation: `pytorch.tensor.copy`
- Ownership: `owned_handle`

## `copy_out`

Generated direct ATen Tensor adapter.

- C++ symbol: `copy_out`
- NativeBoundary operation: `pytorch.tensor.copy_out`
- Ownership: `mutable_handle`

## `from_int`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_from_int`
- NativeBoundary operation: `pytorch.tensor.from_int`
- Ownership: `owned_handle`

## `from_float`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_from_float`
- NativeBoundary operation: `pytorch.tensor.from_float`
- Ownership: `owned_handle`

## `from_bool`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_from_bool`
- NativeBoundary operation: `pytorch.tensor.from_bool`
- Ownership: `owned_handle`

## `from_floats`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_from_floats`
- NativeBoundary operation: `pytorch.tensor.from_floats`
- Ownership: `owned_handle`

## `from_ints`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_from_ints`
- NativeBoundary operation: `pytorch.tensor.from_ints`
- Ownership: `owned_handle`

## `full_float`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_full_float`
- NativeBoundary operation: `pytorch.tensor.full_float`
- Ownership: `owned_handle`

## `item`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_item_int`
- NativeBoundary operation: `pytorch.tensor.item`
- Ownership: `value`

## `item_float`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_item_float`
- NativeBoundary operation: `pytorch.tensor.item_float`
- Ownership: `value`

## `item_bool`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_item_bool`
- NativeBoundary operation: `pytorch.tensor.item_bool`
- Ownership: `value`

## `storage_offset`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_storage_offset`
- NativeBoundary operation: `pytorch.tensor.storage_offset`
- Ownership: `value`

## `dtype`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_dtype`
- NativeBoundary operation: `pytorch.tensor.dtype`
- Ownership: `value`

## `device_type`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_device_type`
- NativeBoundary operation: `pytorch.tensor.device_type`
- Ownership: `value`

## `device_index`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_device_index`
- NativeBoundary operation: `pytorch.tensor.device_index`
- Ownership: `value`

## `to_device`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_to_device`
- NativeBoundary operation: `pytorch.tensor.to_device`
- Ownership: `owned_handle`

## `to_dtype`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_to_dtype`
- NativeBoundary operation: `pytorch.tensor.to_dtype`
- Ownership: `owned_handle`

## `to_other`

Generated CXX tensor-to-other conversion adapter.

- C++ symbol: `tensor_to_other`
- NativeBoundary operation: `pytorch.tensor.to_other`
- Ownership: `owned_handle`

## `to_device_dtype`

Generated CXX tensor device/dtype conversion adapter.

- C++ symbol: `tensor_to_device_dtype`
- NativeBoundary operation: `pytorch.tensor.to_device_dtype`
- Ownership: `owned_handle`

## `to_dtype_layout`

Generated CXX tensor dtype/layout conversion adapter.

- C++ symbol: `tensor_to_dtype_layout`
- NativeBoundary operation: `pytorch.tensor.to_dtype_layout`
- Ownership: `owned_handle`

## `nested_from_tensor_list`

Constructs a nested tensor from component tensors.

- C++ symbol: `tensor_nested_from_tensor_list`
- NativeBoundary operation: `pytorch.tensor.nested_from_tensor_list`
- Ownership: `owned_handle`

## `nested_to_padded_tensor`

Exact generated CXX adapter for aten::nested_to_padded_tensor.

- C++ symbol: `tensor_nested_to_padded_tensor_exact`
- NativeBoundary operation: `pytorch.tensor.nested_to_padded_tensor`
- Ownership: `owned_handle`

## `to_padded_tensor`

Converts a nested tensor to padded dense storage.

- C++ symbol: `tensor_to_padded_tensor`
- NativeBoundary operation: `pytorch.tensor.to_padded_tensor`
- Ownership: `owned_handle`

## `to_padded_tensor_out`

Writes a nested tensor's padded dense representation.

- C++ symbol: `tensor_to_padded_tensor_out`
- NativeBoundary operation: `pytorch.tensor.to_padded_tensor_out`
- Ownership: `mutable_handle`

## `from_file`

Maps file-backed storage into a tensor.

- C++ symbol: `tensor_from_file`
- NativeBoundary operation: `pytorch.tensor.from_file`
- Ownership: `owned_handle`

## `from_file_out`

Maps file-backed storage into retained output.

- C++ symbol: `tensor_from_file_out`
- NativeBoundary operation: `pytorch.tensor.from_file_out`
- Ownership: `mutable_handle`

## `full_typed_native`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_full_typed`
- NativeBoundary operation: `pytorch.tensor.full_typed`
- Ownership: `owned_handle`

## `empty_typed_native`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_empty_typed`
- NativeBoundary operation: `pytorch.tensor.empty_typed`
- Ownership: `owned_handle`

## `arange_typed_native`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_arange_typed`
- NativeBoundary operation: `pytorch.tensor.arange_typed`
- Ownership: `owned_handle`

## `linspace_typed_native`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_linspace_typed`
- NativeBoundary operation: `pytorch.tensor.linspace_typed`
- Ownership: `owned_handle`

## `logspace_typed_native`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_logspace_typed`
- NativeBoundary operation: `pytorch.tensor.logspace_typed`
- Ownership: `owned_handle`

## `eye_typed_native`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_eye_typed`
- NativeBoundary operation: `pytorch.tensor.eye_typed`
- Ownership: `owned_handle`

## `add`

Generated CXX Tensor operation adapter.

- C++ symbol: `add`
- NativeBoundary operation: `pytorch.tensor.add`
- Ownership: `owned_handle`

## `subtract`

Generated CXX Tensor operation adapter.

- C++ symbol: `sub`
- NativeBoundary operation: `pytorch.tensor.subtract`
- Ownership: `owned_handle`

## `divide`

Generated CXX Tensor operation adapter.

- C++ symbol: `div`
- NativeBoundary operation: `pytorch.tensor.divide`
- Ownership: `owned_handle`

## `add_scalar`

Generated CXX Tensor operation adapter.

- C++ symbol: `add`
- NativeBoundary operation: `pytorch.tensor.add_scalar`
- Ownership: `owned_handle`

## `gelu_out`

Generated CXX Tensor operation adapter.

- C++ symbol: `gelu_out`
- NativeBoundary operation: `pytorch.tensor.gelu_out`
- Ownership: `mutable_handle`

## `gelu_in_place`

Generated CXX Tensor operation adapter.

- C++ symbol: `gelu_`
- NativeBoundary operation: `pytorch.tensor.gelu_in_place`
- Ownership: `mutable_handle`

## `copy_in_place`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_copy_in_place`
- NativeBoundary operation: `pytorch.tensor.copy_in_place`
- Ownership: `mutable_handle`

## `require_grad`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_require_grad`
- NativeBoundary operation: `pytorch.tensor.require_grad`
- Ownership: `owned_handle`

## `backward`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_backward`
- NativeBoundary operation: `pytorch.tensor.backward`
- Ownership: `borrowed_handle`

## `backward_with_gradient`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_backward_with_gradient`
- NativeBoundary operation: `pytorch.tensor.backward_with_gradient`
- Ownership: `borrowed_handle`

## `backward_many_token`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_backward_many_token`
- NativeBoundary operation: `pytorch.tensor.backward_many_token`
- Ownership: `owned_handle`

## `autograd_grad_many`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_autograd_grad_many`
- NativeBoundary operation: `pytorch.tensor.autograd_grad_many`
- Ownership: `owned_handle`

## `autograd_grad`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_autograd_grad`
- NativeBoundary operation: `pytorch.tensor.autograd_grad`
- Ownership: `owned_handle`

## `gradient`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_gradient`
- NativeBoundary operation: `pytorch.tensor.gradient`
- Ownership: `owned_handle`

## `gradient_with_graph`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_gradient_with_graph`
- NativeBoundary operation: `pytorch.tensor.gradient_with_graph`
- Ownership: `owned_handle`

## `sgd_step`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_sgd_step`
- NativeBoundary operation: `pytorch.tensor.sgd_step`
- Ownership: `owned_handle`

## `categorical_sample`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_categorical_sample`
- NativeBoundary operation: `pytorch.tensor.categorical_sample`
- Ownership: `owned_handle`

## `mse_loss_gradient`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_mse_loss_gradient`
- NativeBoundary operation: `pytorch.tensor.mse_loss_gradient`
- Ownership: `owned_handle`

## `cross_entropy_loss`

Generated CXX Tensor operation adapter.

- C++ symbol: `tensor_cross_entropy_loss`
- NativeBoundary operation: `pytorch.tensor.cross_entropy_loss`
- Ownership: `owned_handle`

## `clamp`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_clamp`
- NativeBoundary operation: `pytorch.tensor.clamp`
- Ownership: `owned_handle`

## `conv2d`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_conv2d`
- NativeBoundary operation: `pytorch.tensor.conv2d`
- Ownership: `owned_handle`

## `embedding_bag`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_embedding_bag`
- NativeBoundary operation: `pytorch.tensor.embedding_bag`
- Ownership: `opaque_handle`

## `embedding_bag_public`

Public-compatible embedding_bag form returning only the embedding output while releasing LibTorch auxiliary tensors.

- Implementation: generated Terlan composition

## `embedding_bag_weighted`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_embedding_bag_weighted`
- NativeBoundary operation: `pytorch.tensor.embedding_bag_weighted`
- Ownership: `opaque_handle`

## `embedding_bag_padding`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_embedding_bag_padding`
- NativeBoundary operation: `pytorch.tensor.embedding_bag_padding`
- Ownership: `opaque_handle`

## `embedding_bag_padding_weighted`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_embedding_bag_padding_weighted`
- NativeBoundary operation: `pytorch.tensor.embedding_bag_padding_weighted`
- Ownership: `opaque_handle`

## `abi_version`

Generated CXX Tensor compatibility adapter.

- C++ symbol: `tensor_abi_version`
- NativeBoundary operation: `pytorch.abi_version`
- Ownership: `value`

## `load_torchscript`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_load_torchscript`
- NativeBoundary operation: `pytorch.torchscript.load`
- Ownership: `opaque_handle`

## `torchscript_forward`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward`
- NativeBoundary operation: `pytorch.torchscript.forward`
- Ownership: `value`

## `torchscript_forward2`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward2`
- NativeBoundary operation: `pytorch.torchscript.forward2`
- Ownership: `value`

## `torchscript_forward_pair`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_pair`
- NativeBoundary operation: `pytorch.torchscript.forward_pair`
- Ownership: `opaque_handle`

## `torchscript_forward_float`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_float`
- NativeBoundary operation: `pytorch.torchscript.forward_float`
- Ownership: `value`

## `torchscript_forward_int`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_int`
- NativeBoundary operation: `pytorch.torchscript.forward_int`
- Ownership: `value`

## `torchscript_forward_bool`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_bool`
- NativeBoundary operation: `pytorch.torchscript.forward_bool`
- Ownership: `value`

## `torchscript_forward_string`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_string`
- NativeBoundary operation: `pytorch.torchscript.forward_string`
- Ownership: `value`

## `torchscript_forward_int_list`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_int_list`
- NativeBoundary operation: `pytorch.torchscript.forward_int_list`
- Ownership: `value`

## `torchscript_forward_float_list`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_float_list`
- NativeBoundary operation: `pytorch.torchscript.forward_float_list`
- Ownership: `value`

## `torchscript_forward_bool_list`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_bool_list`
- NativeBoundary operation: `pytorch.torchscript.forward_bool_list`
- Ownership: `value`

## `torchscript_forward_string_list`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_string_list`
- NativeBoundary operation: `pytorch.torchscript.forward_string_list`
- Ownership: `value`

## `torchscript_forward_optional_tensor`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_optional_tensor`
- NativeBoundary operation: `pytorch.torchscript.forward_optional_tensor`
- Ownership: `value`

## `torchscript_forward_tensor_list`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_torchscript_forward_tensor_list`
- NativeBoundary operation: `pytorch.torchscript.forward_tensor_list`
- Ownership: `owned_handle`

## `tensor_pair_first`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_pair_first`
- NativeBoundary operation: `pytorch.torchscript.tensor_pair_first`
- Ownership: `value`

## `tensor_pair_second`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_pair_second`
- NativeBoundary operation: `pytorch.torchscript.tensor_pair_second`
- Ownership: `value`

## `run_torchscript`

Generated CXX TorchScript compatibility adapter.

- C++ symbol: `tensor_run_torchscript`
- NativeBoundary operation: `pytorch.tensor.run_torchscript`
- Ownership: `value`

## `dispose_model`

Drops a TorchScript module.

- NativeBoundary operation: `pytorch.torchscript.dispose`
- Ownership: `dispose_handle`

## `dispose_tensor_pair`

Drops a tensor pair.

- NativeBoundary operation: `pytorch.torchscript.dispose_tensor_pair`
- Ownership: `dispose_handle`

## `tanh`

Returns the elementwise hyperbolic tangent.

- C++ symbol: `tanh`
- NativeBoundary operation: `pytorch.tensor.tanh_alias`
- Ownership: `owned_handle`

## `softmin`

Computes softmin by applying softmax to the negated input.

- Implementation: generated Terlan composition

## `softsign`

Computes softsign as input divided by one plus its absolute value.

- Implementation: generated Terlan composition

## `tanhshrink`

Computes tanhshrink as input minus its hyperbolic tangent.

- Implementation: generated Terlan composition

## `is_pinned_cxx`

Private exact CXX leaf for aten::is_pinned.

- C++ symbol: `is_pinned`
- NativeBoundary operation: `pytorch.tensor.is_pinned_cxx`
- Ownership: `value`

## `is_pinned`

Public option-normalizing composition for aten::is_pinned.

- Implementation: generated Terlan composition

## `resize_cxx`

Private exact CXX leaf for aten::resize.

- C++ symbol: `resize`
- NativeBoundary operation: `pytorch.tensor.resize_cxx`
- Ownership: `owned_handle`

## `resize`

Resizes a tensor to the requested shape and optional memory format.

- Implementation: generated Terlan composition

## `resize_out_cxx`

Private exact CXX leaf for aten::resize.out.

- C++ symbol: `resize_out`
- NativeBoundary operation: `pytorch.tensor.resize_out_cxx`
- Ownership: `mutable_handle`

## `resize_out`

Resizes the retained output tensor to the requested shape.

- Implementation: generated Terlan composition

## `resize_in_place_cxx`

Private exact CXX leaf for aten::resize_.

- C++ symbol: `resize_`
- NativeBoundary operation: `pytorch.tensor.resize_in_place_cxx`
- Ownership: `mutable_handle`

## `resize_in_place`

Resizes the retained tensor in place.

- Implementation: generated Terlan composition

## `empty_strided_cxx`

Private exact CXX leaf for aten::empty_strided.

- C++ symbol: `empty_strided`
- NativeBoundary operation: `pytorch.tensor.empty_strided_cxx`
- Ownership: `owned_handle`

## `empty_strided`

Creates an uninitialized tensor with explicit shape and strides.

- Implementation: generated Terlan composition

## `empty_strided_out_cxx`

Private exact CXX leaf for aten::empty_strided.out.

- C++ symbol: `empty_strided_out`
- NativeBoundary operation: `pytorch.tensor.empty_strided_out_cxx`
- Ownership: `mutable_handle`

## `empty_strided_out`

Reshapes retained output storage with explicit strides.

- Implementation: generated Terlan composition

## `empty_permuted_cxx`

Private exact CXX leaf for aten::empty_permuted.

- C++ symbol: `empty_permuted`
- NativeBoundary operation: `pytorch.tensor.empty_permuted_cxx`
- Ownership: `owned_handle`

## `empty_permuted`

Creates a tensor with explicit physical dimension order.

- Implementation: generated Terlan composition

## `empty_permuted_out_cxx`

Private exact CXX leaf for aten::empty_permuted.out.

- C++ symbol: `empty_permuted_out`
- NativeBoundary operation: `pytorch.tensor.empty_permuted_out_cxx`
- Ownership: `mutable_handle`

## `empty_permuted_out`

Writes explicit physical dimension order into retained output.

- Implementation: generated Terlan composition
