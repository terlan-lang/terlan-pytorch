#pragma once

#include "libtorch_bindings.hpp"
#include "rust/cxx.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace at::native {

class TerlanPytorchTensorTensorResourceListInput final {
 public:
  bool push(const at::Tensor& value) noexcept;
  const std::vector<at::Tensor>& values() const noexcept;

 private:
  std::vector<at::Tensor> values_;
};
std::unique_ptr<TerlanPytorchTensorTensorResourceListInput> terlan_resource_list_pytorch_tensor_tensor_new() noexcept;
bool terlan_resource_list_pytorch_tensor_tensor_push(TerlanPytorchTensorTensorResourceListInput& values, const at::Tensor& value) noexcept;

std::unique_ptr<at::native::CudaStream> terlan_resource_pytorch_cuda_stream_copy(const at::native::CudaStream& value) noexcept;
std::unique_ptr<at::native::Generator> terlan_resource_pytorch_randomcompletion_generator_copy(const at::native::Generator& value) noexcept;
std::unique_ptr<at::native::StorageHandle> terlan_resource_pytorch_storagenative_storagehandle_copy(const at::native::StorageHandle& value) noexcept;
std::unique_ptr<at::Tensor> terlan_resource_pytorch_tensor_tensor_copy(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones(rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_dtype(rust::Slice<const std::int64_t> size, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_layout(rust::Slice<const std::int64_t> size, std::int64_t dtype, std::int64_t layout) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_device(rust::Slice<const std::int64_t> size, std::int64_t dtype, std::int64_t layout, rust::Str device) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_options(rust::Slice<const std::int64_t> size, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept;
std::unique_ptr<c10::Scalar> terlan_owned_value_pytorch_tensor_item(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mul_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_abs(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_detach(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ceil(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_floor(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_reciprocal(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_round(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sign(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_square(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_trunc(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_clone(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_relu(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_multiply(const at::Tensor& value, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_multiply_scalar(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_negate(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_unsqueeze(const Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_matmul(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_transpose(const Tensor & self, int64_t dim0, int64_t dim1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_exp(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sqrt(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_log(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_minimum(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mean(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sum(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mean_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sum_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gelu(const Tensor & self, rust::Str approximate) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_softmax(const Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_argmax(const Tensor & self, std::int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_embedding_renorm(const Tensor & self, const Tensor & indices, double max_norm, double norm_type) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_batch_norm(const Tensor & input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean, const at::Tensor& running_var, bool training, double momentum, double eps, bool cudnn_enabled) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_layer_norm(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor& weight, const at::Tensor& bias, double eps, bool cudnn_enable) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_amax(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_adaptive_avg_pool2d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_copy(const Tensor & self, const Tensor & src, bool non_blocking) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_int(std::int64_t arg0) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_float(double arg0) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_bool(bool arg0) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_floats(rust::Slice<const double> arg0, rust::Slice<const std::int64_t> arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_ints(rust::Slice<const std::int64_t> arg0, rust::Slice<const std::int64_t> arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_full_float(rust::Slice<const std::int64_t> arg0, double arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_device(const at::Tensor & arg0, std::int64_t arg1, std::int64_t arg2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_dtype(const at::Tensor & arg0, std::int64_t arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_other(const at::Tensor & tensor, const at::Tensor & other, bool non_blocking) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_device_dtype(const at::Tensor & tensor, int64_t device_type, int64_t device_index, int64_t dtype, bool non_blocking, bool copy) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_dtype_layout(const at::Tensor & tensor, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool non_blocking, bool copy, bool has_memory_format, int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_nested_from_tensor_list(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_nested_to_padded_tensor(const at::Tensor & tensor, double padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_padded_tensor(const at::Tensor & tensor, double padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_file(rust::Str filename, bool has_shared, bool shared, bool has_size, int64_t size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, int64_t has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_full_typed_native(rust::Slice<const std::int64_t> arg0, double arg1, std::int64_t arg2, std::int64_t arg3, std::int64_t arg4) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_empty_typed_native(rust::Slice<const std::int64_t> arg0, std::int64_t arg1, std::int64_t arg2, std::int64_t arg3) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_arange_typed_native(double arg0, double arg1, double arg2, std::int64_t arg3, std::int64_t arg4, std::int64_t arg5, bool arg6) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_linspace_typed_native(double arg0, double arg1, std::int64_t arg2, std::int64_t arg3, std::int64_t arg4, std::int64_t arg5, bool arg6) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_logspace_typed_native(double arg0, double arg1, std::int64_t arg2, double arg3, std::int64_t arg4, std::int64_t arg5, std::int64_t arg6, bool arg7) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_eye_typed_native(std::int64_t arg0, std::int64_t arg1, std::int64_t arg2, std::int64_t arg3, std::int64_t arg4, bool arg5) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_add(const Tensor & self, const Tensor & other, double alpha) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_subtract(const Tensor & self, const Tensor & other, double alpha) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_divide(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_add_scalar(const Tensor & self, double other, double alpha) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_require_grad(const at::Tensor & arg0) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_backward_many_token(const TerlanPytorchTensorTensorResourceListInput& outputs, const TerlanPytorchTensorTensorResourceListInput& grad_outputs, bool retain_graph, bool create_graph) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_autograd_grad(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, bool arg4) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gradient(const at::Tensor & arg0) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gradient_with_graph(const at::Tensor & arg0) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sgd_step(const at::Tensor & arg0, double arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_categorical_sample(const at::Tensor & arg0, std::int64_t arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mse_loss_gradient(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, std::int64_t arg3) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_cross_entropy_loss(const at::Tensor & arg0, const at::Tensor & arg1, std::int64_t arg2, std::int64_t arg3, double arg4) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_clamp(const at::Tensor & arg0, double arg1, double arg2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_conv2d(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, rust::Slice<const std::int64_t> arg3, rust::Slice<const std::int64_t> arg4, rust::Slice<const std::int64_t> arg5, std::int64_t arg6) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_tensor_torchscript_forward_pair(const TorchScriptModel & arg0, const at::Tensor & arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_tanh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_resize_cxx(const Tensor & self, rust::Slice<const std::int64_t> size, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_empty_strided_cxx(rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_empty_permuted_cxx(rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> physical_layout, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_block_diag(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_cartesian_prod(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_column_stack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_dstack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_hstack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_row_stack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_vstack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_cat(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_stack(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_concat(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_concatenate(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distance_cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distance_chalf(const at::Tensor & input) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distance_pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_alpha_dropout(const Tensor & input, double p, bool train) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_dropout(const Tensor & input, double p, bool train) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_feature_dropout(const Tensor & input, double p, bool train) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_native_dropout_backward(const Tensor & grad_output, const Tensor & mask, double scale) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & buffer) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dropoutcompletion_feature_alpha_dropout(const Tensor & input, double p, bool train) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_glucompletion_glu_jvp(const Tensor & glu, const Tensor & x, const Tensor & dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_glucompletion_glu_backward_jvp(const Tensor & grad_x, const Tensor & grad_glu, const Tensor & x, const Tensor & dgrad_glu, const Tensor & dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_adaptive_avg_pool1d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool1d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool1d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & indices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & indices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode, const Tensor & indices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode, const Tensor & indices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_unpool2d(const Tensor & self, const Tensor & indices, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_unpool3d(const Tensor & self, const Tensor & indices, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d_divisor(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d_divisor(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d_backward_divisor(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d_backward_divisor(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_affine_grid_generator(const Tensor & theta, rust::Slice<const std::int64_t> size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_affine_grid_generator_backward(const Tensor & grad, rust::Slice<const std::int64_t> size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_diagonal_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t offset, int64_t dim1, int64_t dim2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_select_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_slice_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_unfold_backward(const Tensor & grad_in, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t size, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_reflection_pad3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_cumulative_trapezoid_dx_int(const Tensor & y, std::int64_t dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_cumulative_trapezoid_dx_float(const Tensor & y, double dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_cumulative_trapezoid_x(const Tensor & y, const Tensor & x, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapezoid_dx_int(const Tensor & y, std::int64_t dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapezoid_dx_float(const Tensor & y, double dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapezoid_x(const Tensor & y, const Tensor & x, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapz_dx(const Tensor & y, double dx, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapz_x(const Tensor & y, const Tensor & x, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_kl_div(const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nantonumnative_nan_to_num_configured(const Tensor & self, bool has_nan, double nan, bool has_posinf, double posinf, bool has_neginf, double neginf) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_copy(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_scatter(const Tensor & self, const Tensor & src, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_offset(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_copy_offset(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_scatter_offset(const Tensor & self, const Tensor & src, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexfillnative_index_fill_scalar(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexfillnative_index_fill_tensor(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lerpnative_lerp_scalar(const Tensor & self, const Tensor & end, bool weight_is_int, std::int64_t weight_int, double weight_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lerpnative_lerp_tensor(const Tensor & self, const Tensor & end, const Tensor & weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_wherenative_where_scalar(const Tensor & condition, bool self_is_int, std::int64_t self_int, double self_float, bool other_is_int, std::int64_t other_int, double other_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_wherenative_where_scalar_other(const Tensor & condition, const Tensor & self, bool other_is_int, std::int64_t other_int, double other_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_wherenative_where_scalar_self(const Tensor & condition, bool self_is_int, std::int64_t self_int, double self_float, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexmutationnative_index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexmutationnative_index_select_backward(const Tensor & grad, rust::Slice<const std::int64_t> self_sizes, int64_t dim, const Tensor & index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fillnative_fill_scalar(const Tensor & self, bool value_is_int, std::int64_t value_int, double value_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fillnative_masked_fill_scalar(const Tensor & self, const Tensor & mask, bool value_is_int, std::int64_t value_int, double value_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_int(const Tensor & self, std::int64_t lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_float(const Tensor & self, double lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_backward_int(const Tensor & grad_out, const Tensor & self, std::int64_t lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_backward_float(const Tensor & grad_out, const Tensor & self, double lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_int(const Tensor & self, std::int64_t lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_float(const Tensor & self, double lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_backward_float(const Tensor & grad_output, const Tensor & self, double lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_int(const Tensor & self, std::int64_t beta, std::int64_t threshold) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_float(const Tensor & self, double beta, double threshold) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t beta, std::int64_t threshold) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_backward_float(const Tensor & grad_output, const Tensor & self, double beta, double threshold) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_int(const Tensor & self, std::int64_t threshold, std::int64_t value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_float(const Tensor & self, double threshold, double value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t threshold) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_backward_float(const Tensor & grad_output, const Tensor & self, double threshold) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_celu_int(const Tensor & self, std::int64_t alpha) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_celu_float(const Tensor & self, double alpha) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_int(const Tensor & self, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_float(const Tensor & self, double alpha, double scale, double input_scale) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_backward_int(const Tensor & grad_output, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale, bool is_result, const Tensor & self_or_result) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_backward_float(const Tensor & grad_output, double alpha, double scale, double input_scale, bool is_result, const Tensor & self_or_result) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_int(const Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_float(const Tensor & self, double min_val, double max_val) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_backward_float(const Tensor & grad_output, const Tensor & self, double min_val, double max_val) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_int(const Tensor & self, std::int64_t negative_slope) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_float(const Tensor & self, double negative_slope) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t negative_slope, bool self_is_result) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_backward_float(const Tensor & grad_output, const Tensor & self, double negative_slope, bool self_is_result) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_huber_loss(const Tensor & self, const Tensor & target, int64_t reduction, double delta) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_huber_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index, const at::Tensor & total_weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index, const at::Tensor & total_weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_nll_loss_nd(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction, double beta) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_narrow_view(const Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_narrow_tensor(const Tensor & self, int64_t dim, const Tensor & start, int64_t length) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_unfold(const at::Tensor& value, int64_t dimension, int64_t size, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_trace_backward(const Tensor & grad, rust::Slice<const std::int64_t> sizes) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_zero(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_im2col(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> dilation, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> stride) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_col2im(const Tensor & self, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> dilation, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> stride) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_tensordot(const Tensor & self, const Tensor & other, rust::Slice<const std::int64_t> dims_self, rust::Slice<const std::int64_t> dims_other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_argsort_stable(const Tensor & self, bool stable, int64_t dim, bool descending) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_bucketize(const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_diagonal_copy(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_diagonal_scatter(const Tensor & self, const Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_select(const Tensor & self, int64_t dim, const Tensor & index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_isin(const Tensor & elements, const Tensor & test_elements, bool assume_unique, bool invert) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_masked_fill_tensor(const Tensor & self, const Tensor & mask, const Tensor & value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_put(const Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_select_scatter(const Tensor & self, const Tensor & src, int64_t dim, int64_t index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_unfold_copy(const Tensor & self, int64_t dimension, int64_t size, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_where_tensor(const Tensor & condition, const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_tensor(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_put(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present, const at::Tensor & values, bool accumulate) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_cumprod_backward(const Tensor & grad, const Tensor & input, int64_t dim, const Tensor & output) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_fake_quantize_per_channel_affine_cachemask_backward(const Tensor & grad, const Tensor & mask) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_fake_quantize_per_tensor_affine_cachemask_backward(const Tensor & grad, const Tensor & mask) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_gather_backward(const Tensor & grad, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_gelu_backward(const Tensor & grad_output, const Tensor & self, rust::Str approximate) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_logit_backward(const Tensor & grad_output, const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_logit_backward_epsilon(const Tensor & grad_output, const Tensor & self, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_masked_select_backward(const Tensor & grad, const Tensor & input, const Tensor & mask) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_bmm(const Tensor & self, const Tensor & mat2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_complex(const Tensor & real, const Tensor & imag) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_divide_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_expand_as(const at::Tensor& value, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_fill_tensor(const Tensor & self, const Tensor & value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_ger(const Tensor & self, const Tensor & vec2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_hardsigmoid_backward(const Tensor & grad_output, const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_hardswish_backward(const Tensor & grad_output, const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_infinitely_differentiable_gelu_backward(const Tensor & grad, const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_linalg_householder_product(const Tensor & input, const Tensor & tau) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_linalg_matmul(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_matrix_exp_backward(const Tensor & self, const Tensor & grad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_mish_backward(const Tensor & grad_output, const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_mm(const Tensor & self, const Tensor & mat2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_multiply_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_mv(const Tensor & self, const Tensor & vec) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_orgqr(const Tensor & self, const Tensor & input2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_outer(const Tensor & self, const Tensor & vec2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_polar(const Tensor & abs, const Tensor & angle) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_prelu(const Tensor & self, const Tensor & weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_reshape_as(const at::Tensor& value, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_sigmoid_backward(const Tensor & grad_output, const Tensor & output) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_silu_backward(const Tensor & grad_output, const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_t(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_u(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_v(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_w(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_gammainc(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_gammaincc(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_hermite_polynomial_h(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_hermite_polynomial_he(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_laguerre_polynomial_l(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_legendre_polynomial_p(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_t(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_u(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_v(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_w(const Tensor & x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_tanh_backward(const Tensor & grad_output, const Tensor & output) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_type_as(const at::Tensor& value, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_view_as(const at::Tensor& value, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_linear_no_bias(const Tensor & input, const Tensor & weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_linear_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_log_softmax_default_dtype(const Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_log_softmax_dtype(const Tensor & self, int64_t dim, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_no_affine(const Tensor & input, int64_t num_groups, double eps, bool cudnn_enabled) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_weight(const Tensor & input, int64_t num_groups, const at::Tensor& weight, double eps, bool cudnn_enabled) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_bias(const Tensor & input, int64_t num_groups, const at::Tensor& bias, double eps, bool cudnn_enabled) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_affine(const Tensor & input, int64_t num_groups, const at::Tensor& weight, const at::Tensor& bias, double eps, bool cudnn_enabled) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_basic(const Tensor & self, int64_t n, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_prepend(const Tensor & self, int64_t n, int64_t dim, const at::Tensor& prepend) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_append(const Tensor & self, int64_t n, int64_t dim, const at::Tensor& append) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_both(const Tensor & self, int64_t n, int64_t dim, const at::Tensor& prepend, const at::Tensor& append) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_masked_scatter_backward(const Tensor & grad_output, const Tensor & mask, rust::Slice<const std::int64_t> sizes) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_normalize_norm(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_normalize_clamp_min(const Tensor & self, double min) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_normalize_divide(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_local_response_norm_configured(const at::Tensor & input, int64_t size, double alpha, double beta, double k) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_multilabel_soft_margin_loss_configured(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, bool has_size_average, bool size_average, bool has_reduce, bool reduce, rust::Str reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_lp_pool1d_configured(const at::Tensor & input, bool norm_type_is_int, int64_t norm_type_int, double norm_type_float, int64_t kernel_size, bool has_stride, int64_t stride, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_max_unpool1d_configured(const at::Tensor & input, const at::Tensor & indices, int64_t kernel_size, bool has_stride, int64_t stride, int64_t padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_lp_pool2d_configured(const at::Tensor & input, bool norm_type_is_int, int64_t norm_type_int, double norm_type_float, int64_t kernel_size, bool has_stride, int64_t stride, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_lp_pool3d_configured(const at::Tensor & input, bool norm_type_is_int, int64_t norm_type_int, double norm_type_float, int64_t kernel_size, bool has_stride, int64_t stride, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_gumbel_softmax_configured(const at::Tensor & logits, double tau, bool hard, double eps, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_gaussian_nll_loss_configured(const at::Tensor & input, const at::Tensor & target, bool var_is_tensor, const at::Tensor & var_tensor, double var_scalar, bool full, double eps, rust::Str reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_interpolate_configured(const at::Tensor & input, bool has_size, rust::Slice<const std::int64_t> size, bool has_scale, rust::Slice<const double> scale_factor, rust::Str mode, bool has_align_corners, bool align_corners, bool has_recompute, bool recompute_scale_factor, bool antialias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingnative_constant_pad_configured(const Tensor & self, rust::Slice<const std::int64_t> pad, bool value_is_int, std::int64_t integer_value, double floating_value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingnative_pad_without_value(const Tensor & self, rust::Slice<const std::int64_t> pad, rust::Str mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingnative_pad_with_value(const Tensor & self, rust::Slice<const std::int64_t> pad, rust::Str mode, double value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy(const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_backward_weight(const Tensor & grad_output, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_positive_weight(const Tensor & self, const Tensor & target, const at::Tensor& pos_weight, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_both_weights(const Tensor & self, const Tensor & target, const at::Tensor& weight, const at::Tensor& pos_weight, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_int_mode(const Tensor & self, std::int64_t other, rust::Str rounding_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_float_mode(const Tensor & self, double other, rust::Str rounding_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_tensor_mode(const Tensor & self, const Tensor & other, rust::Str rounding_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_int_mode(const Tensor & self, std::int64_t other, rust::Str rounding_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_float_mode(const Tensor & self, double other, rust::Str rounding_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_tensor_mode(const Tensor & self, const Tensor & other, rust::Str rounding_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_true_divide_right_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_true_divide_right_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_all(const Tensor & self, int64_t dim, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_from(const Tensor & self, int64_t dim, std::int64_t start, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_until(const Tensor & self, int64_t dim, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_bounds(const Tensor & self, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_all(const Tensor & self, int64_t dim, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_from(const Tensor & self, int64_t dim, std::int64_t start, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_until(const Tensor & self, int64_t dim, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_bounds(const Tensor & self, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_all(const Tensor & self, const Tensor & src, int64_t dim, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_from(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_until(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_bounds(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fft(const Tensor & self, int64_t dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_hfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifft(const Tensor & self, int64_t dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ihfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_irfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_hfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ihfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_irfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_hfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ihfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_irfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftfreq(int64_t n, double d) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfftfreq(int64_t n, double d) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftshift(const Tensor & self, rust::Slice<const std::int64_t> dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftshift_all(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifftshift(const Tensor & self, rust::Slice<const std::int64_t> dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifftshift_all(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_selection_nonzero_static(const at::Tensor & tensor, std::int64_t size, std::int64_t fill_value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_selection_value_selecting_reduction_backward_exact(const at::Tensor & gradient, std::int64_t dimension, const at::Tensor & indices, rust::Slice<const std::int64_t> sizes, bool keep_dimension) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorlisttransforms_flatten_dense_tensors(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorlisttransforms_pad_sequence_native(const TerlanPytorchTensorTensorResourceListInput& sequences, bool batch_first, double padding_value, rust::Str padding_side) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorlisttransforms_einsum_native(rust::Str equation, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cdist_default_compute(const Tensor & x1, const Tensor & x2, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cdist_compute(const Tensor & x1, const Tensor & x2, double p, std::int64_t compute_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_chain_matmul_native(const TerlanPytorchTensorTensorResourceListInput& matrices) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_contiguous_native(const at::Tensor& value, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cross_default_dimension(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cross_dimension(const Tensor & self, const Tensor & other, std::int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_bilinear_no_bias(const Tensor & input1, const Tensor & input2, const Tensor & weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_bilinear_with_bias(const Tensor & input1, const Tensor & input2, const Tensor & weight, const at::Tensor& bias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_cummaxmin_backward(const Tensor & grad, const Tensor & input, const Tensor & indices, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_set_empty(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_set_source_tensor(const Tensor & self, const Tensor & source) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_adjoint(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_alias(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_alias_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_atleast_1d(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_atleast_2d(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_atleast_3d(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_detach(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_detach_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_imag(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_lift(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_lift_fresh(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_lift_fresh_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_positive(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_ravel(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_real(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_resolve_conj(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_resolve_neg(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_squeeze(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_squeeze_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_t(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_t_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_complex(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_complex_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_real(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_real_copy(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_conj(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_data(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_mH(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_mT(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_matrix_H(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_numpy_T(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_convolution(const at::Tensor & input, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool transposed, rust::Slice<const std::int64_t> output_padding, std::int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_padding_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_padding_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_padding_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_padding_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_padding_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_padding_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose1d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose2d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose3d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose1d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose2d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose3d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_default(const Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_default(const Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_int(const Tensor & self, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_int(const Tensor & self, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_float(const Tensor & self, double correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_float(const Tensor & self, double correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_dimensions_int(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_dimensions_int(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_dimensions_float(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_dimensions_float(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_indices_native(const Tensor & repeats) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_indices_output_size_native(const Tensor & repeats, std::int64_t output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_flat_native(const Tensor & self, const Tensor & repeats) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_flat_output_size_native(const Tensor & self, const Tensor & repeats, std::int64_t output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_dimension_native(const Tensor & self, const Tensor & repeats, std::int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_dimension_output_size_native(const Tensor & self, const Tensor & repeats, std::int64_t dim, std::int64_t output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_flat_native(const Tensor & self, int64_t repeats) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_flat_output_size_native(const Tensor & self, int64_t repeats, std::int64_t output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_dimension_native(const Tensor & self, int64_t repeats, std::int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_dimension_output_size_native(const Tensor & self, int64_t repeats, std::int64_t dim, std::int64_t output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addmm_dtype(const Tensor & self, const Tensor & mat1, const Tensor & mat2, std::int64_t out_dtype, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_baddbmm_dtype(const Tensor & self, const Tensor & batch1, const Tensor & batch2, std::int64_t out_dtype, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_add_tensor_cxx(const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_sub_tensor_cxx(const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_rsub_tensor_cxx(const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_add_scalar_cxx(const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_sub_scalar_cxx(const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_rsub_scalar_cxx(const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_scalar_min(const Tensor & self, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_scalar_max(const Tensor & self, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_scalar_min_max(const Tensor & self, bool min_is_int, std::int64_t min_int, double min_float, bool max_is_int, std::int64_t max_int, double max_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_tensor_min(const Tensor & self, const Tensor & min) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_tensor_max(const Tensor & self, const Tensor & max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_tensor_min_max(const Tensor & self, const at::Tensor& min, const at::Tensor& max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_bincount_weighted(const Tensor & self, const at::Tensor& weights, int64_t minlength) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_bincount_unweighted(const Tensor & self, int64_t minlength) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_histc_int(const Tensor & self, int64_t bins, std::int64_t min, std::int64_t max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_histc_float(const Tensor & self, int64_t bins, double min, double max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_index_reduce(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, rust::Str reduce, bool include_self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_bartlett(int64_t window_length, bool periodic, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_blackman(int64_t window_length, bool periodic, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_hamming(int64_t window_length, bool periodic, double alpha, double beta, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_hann(int64_t window_length, bool periodic, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_kaiser(int64_t window_length, bool periodic, double beta, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_arctan2(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_atan2(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_and_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_left_shift_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_or_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_right_shift_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_xor_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_clamp_max_tensor(const Tensor & self, const Tensor & max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_clamp_min_tensor(const Tensor & self, const Tensor & min) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_copysign_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_dot(const Tensor & self, const Tensor & tensor) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_eq_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_float_power_tensor(const Tensor & self, const Tensor & exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_floor_divide(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_fmax(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_fmin(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_fmod_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_gcd(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_ge_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_greater_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_greater_equal_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_gt_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_heaviside(const Tensor & self, const Tensor & values) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_hypot(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_igamma(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_igammac(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_inner(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_kron(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_lcm(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_ldexp_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_le_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_less_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_less_equal_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logaddexp(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logaddexp2(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logical_and(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logical_or(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logical_xor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_lt_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_masked_select(const Tensor & self, const Tensor & mask) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_max_other(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_maximum(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_min_other(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_ne_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_nextafter(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_not_equal_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_pow_tensor(const Tensor & self, const Tensor & exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_remainder_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_special_xlog1py(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_special_xlogy(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_special_zeta(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_take(const Tensor & self, const Tensor & index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_true_divide_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_vdot(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_xlogy_tensor(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_src(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_legacy_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_legacy_multiply(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_sum(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_prod(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_mean(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_amax(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_amin(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_int(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_float(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_int_add(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_int_multiply(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_float_add(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_float_multiply(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_arange_end(bool end_is_integer, std::int64_t end_integer, double end_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_arange_start_end(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_arange_start_end_step(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool step_is_integer, std::int64_t step_integer, double step_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_eye_square(int64_t n, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_eye_rect(int64_t n, int64_t m, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_scalar_scalar(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_tensor_tensor(const Tensor & start, const Tensor & end, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_tensor_scalar(const Tensor & start, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_scalar_tensor(bool start_is_integer, std::int64_t start_integer, double start_floating, const Tensor & end, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_scalar_scalar(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_tensor_tensor(const Tensor & start, const Tensor & end, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_tensor_scalar(const Tensor & start, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_scalar_tensor(bool start_is_integer, std::int64_t start_integer, double start_floating, const Tensor & end, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_range_start_end(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_range_start_end_step(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool step_is_integer, std::int64_t step_integer, double step_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_tril_indices_cxx(int64_t row, int64_t col, int64_t offset, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_triu_indices_cxx(int64_t row, int64_t col, int64_t offset, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_right_int(const Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_left_int(std::int64_t x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_right_float(const Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_left_float(double x, const Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_right_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_left_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_right_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_left_float(double self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_right_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_left_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_right_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_left_float(double self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_right_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_left_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_right_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_left_float(double self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_empty(rust::Slice<const std::int64_t> size, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_full(rust::Slice<const std::int64_t> size, bool fill_value_is_integer, std::int64_t fill_value_integer, double fill_value_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_ones(rust::Slice<const std::int64_t> size, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_zeros(rust::Slice<const std::int64_t> size, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_empty_like(const Tensor & self, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_full_like(const Tensor & self, bool fill_value_is_integer, std::int64_t fill_value_integer, double fill_value_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_ones_like(const Tensor & self, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_zeros_like(const Tensor & self, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_and_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_or_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_xor_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_left_shift_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_right_shift_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_remainder_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_pow_int(const Tensor & self, std::int64_t exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_float_power_int(const Tensor & self, std::int64_t exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_mul_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_xlogy_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_copysign_int(const Tensor & self, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_remainder_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_pow_float(const Tensor & self, double exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_float_power_float(const Tensor & self, double exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_mul_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_xlogy_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_fmod_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_copysign_float(const Tensor & self, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_and_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_or_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_xor_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_left_shift_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_right_shift_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_remainder_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_pow_int(std::int64_t self, const Tensor & exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_float_power_int(std::int64_t self, const Tensor & exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_xlogy_int(std::int64_t self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_remainder_float(double self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_pow_float(double self, const Tensor & exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_float_power_float(double self, const Tensor & exponent) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_xlogy_float(double self, const Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_eq_float(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ne_float(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_lt_float(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_le_float(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_gt_float(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ge_float(const at::Tensor& value, double other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_eq_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ne_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_lt_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_le_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_gt_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ge_int(const at::Tensor& value, std::int64_t other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_isin_tensor_scalar(const Tensor & elements, bool test_element_is_integer, std::int64_t test_element_integer, double test_element_floating, bool assume_unique, bool invert) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_isin_scalar_tensor(bool element_is_integer, std::int64_t element_integer, double element_floating, const Tensor & test_elements, bool assume_unique, bool invert) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_searchsorted_tensor(const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, bool has_side, rust::Str side, bool has_sorter, const at::Tensor& sorter) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_searchsorted_scalar(const Tensor & sorted_sequence, bool self_is_integer, std::int64_t self_integer, double self_floating, bool out_int32, bool right, bool has_side, rust::Str side, bool has_sorter, const at::Tensor& sorter) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_bucketize_scalar(bool self_is_integer, std::int64_t self_integer, double self_floating, const Tensor & boundaries, bool out_int32, bool right) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unarystoragenative_round_decimals(const at::Tensor & self, int64_t decimals) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_abs(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_ceil(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_floor(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_frac(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_reciprocal(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_round(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sign(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_square(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_trunc(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_acos(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_acosh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_all(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_any(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_argwhere(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_asin(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_asinh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_atan(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_atanh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_bitwise_not(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_cos(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_cosh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_deg2rad(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_erf(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_erfc(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_erfinv(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_exp2(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_expm1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_hardsigmoid(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_hardswish(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isfinite(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isinf(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isnan(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isneginf(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isposinf(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_log10(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_log1p(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_log2(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_logical_not(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_max(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_median(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_min(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_mish(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_nonzero(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_rad2deg(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_relu6(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_rsqrt(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_selu(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sigmoid(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_signbit(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_silu(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sin(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sinc(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sinh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_tan(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_tanh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_trace(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_channel_shuffle(const at::Tensor & self, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_diag(const at::Tensor & self, int64_t diagonal) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_diagflat(const at::Tensor & self, int64_t offset) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_glu(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_linalg_matrix_power(const at::Tensor & self, int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_linalg_tensorinv(const at::Tensor & self, int64_t ind) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_logcumsumexp(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_matrix_power(const at::Tensor & self, int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_moveaxis_int(const at::Tensor & self, int64_t source, int64_t destination) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_movedim_int(const at::Tensor & self, int64_t source, int64_t destination) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_mvlgamma(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_native_channel_shuffle(const at::Tensor & self, int64_t groups) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_one_hot(const at::Tensor & self, int64_t num_classes) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_pixel_shuffle(const at::Tensor & self, int64_t upscale_factor) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_pixel_unshuffle(const at::Tensor & self, int64_t downscale_factor) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_select_int(const at::Tensor & self, int64_t dim, int64_t index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_select_copy_int(const at::Tensor & self, int64_t dim, int64_t index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_special_multigammaln(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_dim(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_copy_dim(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_swapaxes(const at::Tensor & self, int64_t axis0, int64_t axis1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_swapdims(const at::Tensor & self, int64_t dim0, int64_t dim1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_transpose_copy_int(const at::Tensor & self, int64_t dim0, int64_t dim1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_tril(const at::Tensor & self, int64_t diagonal) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_triu(const at::Tensor & self, int64_t diagonal) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_unsqueeze_copy(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_adaptive_avg_pool3d(const at::Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_broadcast_to(const at::Tensor & self, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_count_nonzero_dim(const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_flip(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_permute(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_permute_copy(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reflection_pad1d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reflection_pad2d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reflection_pad3d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_repeat(const at::Tensor& value, rust::Slice<const std::int64_t> repeats) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_replication_pad1d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_replication_pad2d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_replication_pad3d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reshape(const at::Tensor & self, rust::Slice<const std::int64_t> shape) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_dims(const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_copy_dims(const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_sum_to_size(const at::Tensor& value, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_tile(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_view(const at::Tensor& value, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_view_copy(const at::Tensor & self, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_expand(const at::Tensor& value, rust::Slice<const std::int64_t> size, bool implicit) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_expand_copy(const at::Tensor & self, rust::Slice<const std::int64_t> size, bool implicit) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_moveaxis(const at::Tensor & self, rust::Slice<const std::int64_t> source, rust::Slice<const std::int64_t> destination) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_movedim(const at::Tensor & self, rust::Slice<const std::int64_t> source, rust::Slice<const std::int64_t> destination) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_narrow_copy(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_roll(const at::Tensor & self, rust::Slice<const std::int64_t> shifts, rust::Slice<const std::int64_t> dims) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_rot90(const at::Tensor & self, int64_t k, rust::Slice<const std::int64_t> dims) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_unflatten(const at::Tensor & self, int64_t dim, rust::Slice<const std::int64_t> sizes) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_absolute(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_angle(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arccos(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arccosh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arcsin(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arcsinh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arctan(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arctanh(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_conj_physical(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_corrcoef(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_det(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_digamma(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_fix(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_fliplr(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_flipud(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_i0(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_inverse(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_isreal(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_lgamma(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_linalg_eigvals(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_linalg_matrix_exp(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_log_sigmoid(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_logdet(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_matrix_exp(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_msort(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_nanmedian(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_negative(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_sgn(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_j0(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_j1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_y0(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_y1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_digamma(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_entr(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erf(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erfc(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erfcx(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erfinv(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_exp2(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_expit(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_expm1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_gammaln(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i0(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i0e(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i1e(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log1p(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log_ndtr(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_i0(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_i1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_k0(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_k1(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_ndtr(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_ndtri(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_psi(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_sinc(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_airy_ai(const at::Tensor & x) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_logsumexp(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_polygamma(const at::Tensor & self, int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_polygamma(const at::Tensor & self, int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_round(const at::Tensor & self, int64_t decimals) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_scaled_modified_bessel_k0(const at::Tensor & x) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_scaled_modified_bessel_k1(const at::Tensor & x) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_spherical_bessel_j0(const at::Tensor & x) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_logit(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_logit_epsilon(const at::Tensor & self, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_logit(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_logit_epsilon(const at::Tensor & self, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_softmax(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_softmax_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log_softmax(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log_softmax_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_order_int(const at::Tensor & x, std::int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_order_float(const at::Tensor & x, double n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_input_int(std::int64_t x, const at::Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_input_float(double x, const at::Tensor & n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_get_rng_state() noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rand_generator(const Generator & generator, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randn_generator(const Generator & generator, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randperm_generator(const Generator & generator, int64_t n, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_rand_like(const Generator & generator, const Tensor & input, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randn_like(const Generator & generator, const Tensor & input, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint(const Generator & generator, int64_t high, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_low(const Generator & generator, int64_t low, int64_t high, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_like(const Generator & generator, const Tensor & input, int64_t high, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_like_tensor(const Generator & generator, const Tensor & input, const Tensor & high, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_like_low(const Generator & generator, const Tensor & input, int64_t low, int64_t high, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rand_default(rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randn_default(rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_default(int64_t high, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_low_default(int64_t low, int64_t high, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rand_like_default(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randn_like_default(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_like_default(const at::Tensor & self, int64_t high) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_like_low_dtype_default(const at::Tensor & self, int64_t low, int64_t high) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_like_tensor_default(const at::Tensor & self, const at::Tensor & high) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_random_default(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_random_from(const at::Tensor & self, int64_t from, int64_t to) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_random_to(const at::Tensor & self, int64_t to) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rrelu_default(const at::Tensor & self, double lower, double upper, bool training) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rrelu_with_noise(const at::Tensor & self, const at::Tensor & noise, bool has_generator, const Generator & generator, double lower, double upper, bool training) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rrelu_with_noise_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, double lower, double upper, bool training, bool self_is_result) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spectralcompletion_stft_default(const at::Tensor & self, int64_t n_fft) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spectralcompletion_stft_center_default(const at::Tensor & self, int64_t n_fft, bool center, rust::Str pad_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spectralcompletion_istft_default(const at::Tensor & self, int64_t n_fft) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_segmentreducecompletion_segment_reduce_lengths(const at::Tensor & data, rust::Str reduce, const at::Tensor & lengths) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_attentioncompletion_scaled_dot_product_attention_default(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedallocationcompletion_empty_quantized(rust::Slice<const std::int64_t> size, const at::Tensor & qtensor) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_ctclosscompletion_ctc_loss_intlist(const at::Tensor & log_probs, const at::Tensor & targets, rust::Slice<const std::int64_t> input_lengths, rust::Slice<const std::int64_t> target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_ctclosscompletion_ctc_loss_tensor(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_covcompletion_cov(const at::Tensor & self, int64_t correction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cholesky(const at::Tensor & self, bool upper) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cholesky_inverse(const at::Tensor & self, bool upper) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_linalg_matrix_rank(const at::Tensor & self, double tol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_linalg_pinv(const at::Tensor & self, double rcond, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nuclear_norm(const at::Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_pdist(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_pinverse(const at::Tensor & self, double rcond) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_dist(const at::Tensor & self, const at::Tensor & other, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm_dimensions(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm_except_dimension(const at::Tensor & v, int64_t pow, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_renorm(const at::Tensor & self, double p, int64_t dim, double maxnorm) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_std(const at::Tensor & self, bool unbiased) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_var(const at::Tensor & self, bool unbiased) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_all_dim(const at::Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_any_dim(const at::Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argsort(const at::Tensor & self, int64_t dim, bool descending) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_combinations(const at::Tensor & self, int64_t r, bool with_replacement) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_amin(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argmax_all(const Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argmin(const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argmin_all(const at::Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_count_nonzero(const at::Tensor & self, std::int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_count_nonzero_all(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_all_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_any_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_logsumexp(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_prod(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_prod_dim(const at::Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumsum(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumprod(const at::Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumsum_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumprod_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_std_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_var_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm_dimensions_to_dtype(const at::Tensor & arg0, std::int64_t arg1) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nanmean_native(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nanmean_dimensions_native(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nanmean_dimensions_dtype_native(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t dtype, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nansum_native(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nansum_dimensions_native(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nansum_dimensions_dtype_native(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t dtype, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_frobenius_norm_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nuclear_norm_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_int_pair(const Tensor & self, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_float_pair(const Tensor & self, double correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_dimensions_int_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_dimensions_float_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_int_pair(const Tensor & self, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_float_pair(const Tensor & self, double correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_dimensions_int_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept;
std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_dimensions_float_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_dimensions(const Tensor & self, int64_t sparse_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_dense(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_coalesce(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_indices(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_values(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_crow_indices(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_col_indices(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_ccol_indices(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_row_indices(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csr(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csr_configured(const Tensor & self, std::int64_t dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csc(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csc_configured(const Tensor & self, std::int64_t dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsr(const Tensor & self, rust::Slice<const std::int64_t> blocksize) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsr_configured(const Tensor & self, rust::Slice<const std::int64_t> blocksize, std::int64_t dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsc(const Tensor & self, rust::Slice<const std::int64_t> blocksize) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsc_configured(const Tensor & self, rust::Slice<const std::int64_t> blocksize, std::int64_t dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_copy_sparse_to_sparse(const Tensor & self, const Tensor & src, bool non_blocking) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_resize_as_sparse(const Tensor & self, const Tensor & the_template) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sparse_resize(const Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sparse_resize_and_clear(const Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_crow_indices_copy(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_col_indices_copy(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_ccol_indices_copy(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_row_indices_copy(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_indices_copy(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_values_copy(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sparse_mask(const at::Tensor & self, const at::Tensor & mask) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_dense_configured(const at::Tensor & self, int64_t dtype, bool has_dtype, bool masked_grad, bool has_masked_grad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_configured(const at::Tensor & self, int64_t layout, bool has_layout, rust::Slice<const std::int64_t> block_size, bool has_block_size, int64_t dense_dim, bool has_dense_dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_coo_empty_configured(rust::Slice<const std::int64_t> size, int64_t dtype, bool has_dtype, int64_t layout, bool has_layout, int64_t device_type, int64_t device_index, bool has_device, bool pin_memory, bool has_pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_coo_from_components_configured(const at::Tensor & indices, const at::Tensor & values, rust::Slice<const std::int64_t> size, bool has_size, int64_t dtype, bool has_dtype, int64_t layout, bool has_layout, int64_t device_type, int64_t device_index, bool has_device, bool pin_memory, bool has_pin_memory, bool is_coalesced, bool has_is_coalesced) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_compressed_from_components_configured(int64_t kind, const at::Tensor & compressed, const at::Tensor & plain, const at::Tensor & values, rust::Slice<const std::int64_t> size, bool has_size, int64_t dtype, bool has_dtype, int64_t layout, bool has_layout, int64_t device_type, int64_t device_index, bool has_device, bool pin_memory, bool has_pin_memory) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sampled_addmm_numeric(const at::Tensor & mask, const at::Tensor & left, const at::Tensor & right, double beta, bool beta_is_integer, double alpha, bool alpha_is_integer) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_dense_backward_configured(const at::Tensor & gradient, const at::Tensor & input, bool masked_grad, bool has_masked_grad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cholesky(const Tensor & self, bool upper) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cross(const Tensor & self, const Tensor & other, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_det(const Tensor & A) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_inv(const Tensor & A) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_ldl_solve(const Tensor & LD, const Tensor & pivots, const Tensor & B, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_lu_solve(const Tensor & LU, const Tensor & pivots, const Tensor & B, bool left, bool adjoint) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tolerance(const at::Tensor & self, const at::Tensor & tol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_rcond(const at::Tensor & self, const at::Tensor & rcond, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_solve(const Tensor & A, const Tensor & B, bool left) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_solve_triangular(const Tensor & self, const Tensor & B, bool upper, bool left, bool unitriangular) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vecdot(const Tensor & x, const Tensor & y, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_string(const at::Tensor & self, rust::Str p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_default(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_int(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_float(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_int(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_float(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_default(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_int(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_float(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_int(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_float(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_int(const at::Tensor & self, int64_t p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_float(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_eigvalsh(const Tensor & self, rust::Str UPLO) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_default(const at::Tensor & self, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tolerances(const at::Tensor & self, double atol, double rtol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_default(const at::Tensor & self, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_absolute(const at::Tensor & self, const at::Tensor & atol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_relative(const at::Tensor & self, const at::Tensor & rtol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_tolerances(const at::Tensor & self, const at::Tensor & atol, const at::Tensor & rtol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_default(const at::Tensor & self, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tolerances(const at::Tensor & self, double atol, double rtol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_default(const at::Tensor & self, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_absolute(const at::Tensor & self, const at::Tensor & atol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_relative(const at::Tensor & self, const at::Tensor & rtol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_tolerances(const at::Tensor & self, const at::Tensor & atol, const at::Tensor & rtol, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_svdvals(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vander(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vander_columns(const at::Tensor & self, int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_string(const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_string_dtype(const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_string(const at::Tensor & self, rust::Str p, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_string_dimensions(const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_tensorsolve(const at::Tensor & self, const at::Tensor & other) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_tensorsolve_dimensions(const at::Tensor & self, const at::Tensor & other, rust::Slice<const std::int64_t> dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_multi_dot(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_diagonal(const Tensor & A, int64_t offset, int64_t dim1, int64_t dim2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_losscompletion_multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_losscompletion_multilabel_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, std::int64_t reduction, const at::Tensor & is_target) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_tensor_qparams(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_fake_quantize_per_channel_affine(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_channel(const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor_tensor_qparams(const Tensor & self, const Tensor & scale, const Tensor & zero_point, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor_dynamic(const Tensor & self, std::int64_t dtype, bool reduce_range) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_int_repr(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_dequantize(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_q_per_channel_scales(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_q_per_channel_zero_points(const at::Tensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_exponential(const at::Tensor & self, double lambd) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_poisson(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_binomial(const at::Tensor & count, const at::Tensor & prob) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_cauchy(const at::Tensor & self, double median, double sigma) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_geometric(const at::Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_log_normal(const at::Tensor & self, double mean, double std) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_uniform(const at::Tensor & self, double from, double to) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_functional(const at::Tensor & self, double mean, double std) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_tensor_tensor(const at::Tensor & mean, const at::Tensor & std) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_tensor_float(const at::Tensor & mean, double std) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_float_tensor(double mean, const at::Tensor & std) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_float_float(double mean, double std, rust::Slice<const std::int64_t> size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_bernoulli(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_bernoulli_probability(const Tensor & self, double p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_bernoulli_tensor(const Tensor & self, const Tensor & p) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_multinomial(const Tensor & self, int64_t num_samples, bool replacement) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_hashcompletion_hash_tensor(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_resizeascompletion_resize_as(const Tensor & self, const Tensor & the_template, bool has_memory_format, std::int64_t memory_format) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_all(const Tensor & self, const Tensor & src, int64_t dim, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_from(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_until(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_bounds(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_default(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_weight(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor& weight) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_eps(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_weight_eps(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor& weight, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nlllosscompletion_nll_loss_default(const Tensor & self, const Tensor & target, int64_t reduction, int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nlllosscompletion_nll_loss_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction, int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_multimarginlosscompletion_multi_margin_loss_default(const Tensor & self, const Tensor & target, double p, double margin, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_multimarginlosscompletion_multi_margin_loss_weight(const Tensor & self, const Tensor & target, double p, double margin, const at::Tensor& weight, int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_multimarginlosscompletion_multi_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, double p, double margin, bool has_weight, const at::Tensor & weight, std::int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebilinear2dcompletion_upsample_bilinear2d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebicubic2dcompletion_upsample_bicubic2d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nllloss2dcompletion_nll_loss2d_default(const Tensor & self, const Tensor & target, int64_t reduction, int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nllloss2dcompletion_nll_loss2d_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction, int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_viewdtypecompletion_view_dtype(const at::Tensor & self, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_viewcopydtypecompletion_view_copy_dtype(const at::Tensor & self, std::int64_t dtype) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantilecompletion_quantile_tensor(const at::Tensor & self, const at::Tensor & q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantilecompletion_quantile_scalar(const at::Tensor & self, double q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nanquantilecompletion_nanquantile_tensor(const at::Tensor & self, const at::Tensor & q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nanquantilecompletion_nanquantile_scalar(const at::Tensor & self, double q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randpermcompletion_randperm_default(int64_t n) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_instancenormcompletion_instance_norm_default(const Tensor & input, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplelinear1dcompletion_upsample_linear1d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplenearest1dcompletion_upsample_nearest1d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplenearest2dcompletion_upsample_nearest2d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplenearest3dcompletion_upsample_nearest3d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_nearest1d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_nearest2d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_nearest3d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_linear1d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_trilinear3d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_bilinear2d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_bicubic2d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsampletrilinear3dcompletion_upsample_trilinear3d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparsematmulcompletion_hspmm_default(const at::Tensor & mat1, const at::Tensor & mat2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparsematmulcompletion_smm_default(const at::Tensor & self, const at::Tensor & mat2) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparsematmulcompletion_sspaddmm_default(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, double beta, double alpha) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_bicubic2d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_bilinear2d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_linear1d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_nearest1d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_nearest2d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_nearest3d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_trilinear3d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept;
std::unique_ptr<at::native::HistogramddResult> terlan_owned_value_pytorch_histogramddnative_histogramdd_tensor_bins_result(const at::Tensor & input, const TerlanPytorchTensorTensorResourceListInput& bins, bool has_range, rust::Slice<const double> range, bool has_weight, const at::Tensor & weight, bool density) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_histogramddnative_histogramdd_result_hist(const HistogramddResult & result) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_storagenative_storage_set(const at::Tensor & tensor, const StorageHandle & source) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_storagenative_storage_set_offset(const at::Tensor & tensor, const StorageHandle & source, std::int64_t storage_offset, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_recurrent_rnn_tanh_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_recurrent_rnn_relu_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_recurrent_gru_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_thnn_conv2d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv3d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_dilated2d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_dilated3d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_transpose3d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, rust::Slice<const std::int64_t> dilation) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationinternals_fused_moving_avg_obs_fake_quant(const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, const at::Tensor & running_min, const at::Tensor & running_max, const at::Tensor & scale, const at::Tensor & zero_point, double averaging_const, std::int64_t quant_min, std::int64_t quant_max, std::int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool1d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizednormalization_quantized_batch_norm(const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, std::int64_t output_zero_point) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scalartensorcompletion_scalar_tensor(bool scalar_is_integer, int64_t scalar_integer, double scalar_floating) noexcept;
class TerlanOwnedTuplePytorchTensorEmbeddingBag final {
 public:
  explicit TerlanOwnedTuplePytorchTensorEmbeddingBag(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2, std::unique_ptr<at::Tensor> value_3) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)), value_3_(std::move(value_3)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }
  std::unique_ptr<at::Tensor> take_3() noexcept { return std::move(value_3_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
  std::unique_ptr<at::Tensor> value_3_;
};
std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBag> terlan_owned_tuple_pytorch_tensor_embedding_bag(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, bool arg6) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept;

class TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted final {
 public:
  explicit TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2, std::unique_ptr<at::Tensor> value_3) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)), value_3_(std::move(value_3)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }
  std::unique_ptr<at::Tensor> take_3() noexcept { return std::move(value_3_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
  std::unique_ptr<at::Tensor> value_3_;
};
std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, const at::Tensor & arg6, bool arg7) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept;

class TerlanOwnedTuplePytorchTensorEmbeddingBagPadding final {
 public:
  explicit TerlanOwnedTuplePytorchTensorEmbeddingBagPadding(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2, std::unique_ptr<at::Tensor> value_3) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)), value_3_(std::move(value_3)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }
  std::unique_ptr<at::Tensor> take_3() noexcept { return std::move(value_3_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
  std::unique_ptr<at::Tensor> value_3_;
};
std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBagPadding> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, bool arg6, std::int64_t arg7) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept;

class TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted final {
 public:
  explicit TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2, std::unique_ptr<at::Tensor> value_3) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)), value_3_(std::move(value_3)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }
  std::unique_ptr<at::Tensor> take_3() noexcept { return std::move(value_3_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
  std::unique_ptr<at::Tensor> value_3_;
};
std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, const at::Tensor & arg6, bool arg7, std::int64_t arg8) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeDropout final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeDropout(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeDropout> terlan_owned_tuple_pytorch_regularization_native_dropout(const at::Tensor & input, double p, bool train) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_dropout_take_0(TerlanOwnedTuplePytorchRegularizationNativeDropout& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_dropout_take_1(TerlanOwnedTuplePytorchRegularizationNativeDropout& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeLayerNorm final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeLayerNorm(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeLayerNorm> terlan_owned_tuple_pytorch_regularization_native_layer_norm(const at::Tensor & input, rust::Slice<const std::int64_t> normalized_shape, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_take_0(TerlanOwnedTuplePytorchRegularizationNativeLayerNorm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_take_1(TerlanOwnedTuplePytorchRegularizationNativeLayerNorm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_take_2(TerlanOwnedTuplePytorchRegularizationNativeLayerNorm& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeGroupNorm final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeGroupNorm(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeGroupNorm> terlan_owned_tuple_pytorch_regularization_native_group_norm(const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, std::int64_t n, std::int64_t c, std::int64_t hxw, std::int64_t group, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_take_0(TerlanOwnedTuplePytorchRegularizationNativeGroupNorm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_take_1(TerlanOwnedTuplePytorchRegularizationNativeGroupNorm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_take_2(TerlanOwnedTuplePytorchRegularizationNativeGroupNorm& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, bool has_weight, const at::Tensor & weight, std::int64_t n, std::int64_t c, std::int64_t hxw, std::int64_t group, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward_take_0(TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward_take_1(TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward_take_2(TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward_take_0(TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward_take_1(TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward_take_2(TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeBatchNorm final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeBatchNorm(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeBatchNorm> terlan_owned_tuple_pytorch_regularization_native_batch_norm(const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, bool training, double momentum, double eps) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_take_0(TerlanOwnedTuplePytorchRegularizationNativeBatchNorm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_take_1(TerlanOwnedTuplePytorchRegularizationNativeBatchNorm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_take_2(TerlanOwnedTuplePytorchRegularizationNativeBatchNorm& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, bool has_save_mean, const at::Tensor & save_mean, bool has_save_invstd, const at::Tensor & save_invstd, bool train, double eps, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward_take_0(TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward_take_1(TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward_take_2(TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationLogSigmoidForward final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationLogSigmoidForward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationLogSigmoidForward> terlan_owned_tuple_pytorch_regularization_log_sigmoid_forward(const at::Tensor & input) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_log_sigmoid_forward_take_0(TerlanOwnedTuplePytorchRegularizationLogSigmoidForward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_log_sigmoid_forward_take_1(TerlanOwnedTuplePytorchRegularizationLogSigmoidForward& value) noexcept;

class TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats final {
 public:
  explicit TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats> terlan_owned_tuple_pytorch_regularization_batch_norm_update_stats(const at::Tensor & input, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, double momentum) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_batch_norm_update_stats_take_0(TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_batch_norm_update_stats_take_1(TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats& value) noexcept;

class TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool1d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool1d_take_0(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool1d_take_1(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d& value) noexcept;

class TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool2d_take_0(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool2d_take_1(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d& value) noexcept;

class TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool3d_take_0(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool3d_take_1(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d& value) noexcept;

class TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices> terlan_owned_tuple_pytorch_pooling_max_pool1d_with_indices(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool1d_with_indices_take_0(TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool1d_with_indices_take_1(TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices& value) noexcept;

class TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices> terlan_owned_tuple_pytorch_pooling_max_pool2d_with_indices(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool2d_with_indices_take_0(TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool2d_with_indices_take_1(TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices& value) noexcept;

class TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices> terlan_owned_tuple_pytorch_pooling_max_pool3d_with_indices(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool3d_with_indices_take_0(TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool3d_with_indices_take_1(TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices& value) noexcept;

class TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d> terlan_owned_tuple_pytorch_pooling_fractional_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & random_samples) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool2d_take_0(TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool2d_take_1(TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d& value) noexcept;

class TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d final {
 public:
  explicit TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d> terlan_owned_tuple_pytorch_pooling_fractional_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & random_samples) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool3d_take_0(TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool3d_take_1(TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d& value) noexcept;

class TerlanOwnedTuplePytorchSpatialGridSampler2dBackward final {
 public:
  explicit TerlanOwnedTuplePytorchSpatialGridSampler2dBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSpatialGridSampler2dBackward> terlan_owned_tuple_pytorch_spatial_grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, std::int64_t interpolation_mode, std::int64_t padding_mode, bool align_corners, bool output_mask_input, bool output_mask_grid) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_2d_backward_take_0(TerlanOwnedTuplePytorchSpatialGridSampler2dBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_2d_backward_take_1(TerlanOwnedTuplePytorchSpatialGridSampler2dBackward& value) noexcept;

class TerlanOwnedTuplePytorchSpatialGridSampler3dBackward final {
 public:
  explicit TerlanOwnedTuplePytorchSpatialGridSampler3dBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSpatialGridSampler3dBackward> terlan_owned_tuple_pytorch_spatial_grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, std::int64_t interpolation_mode, std::int64_t padding_mode, bool align_corners, bool output_mask_input, bool output_mask_grid) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_3d_backward_take_0(TerlanOwnedTuplePytorchSpatialGridSampler3dBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_3d_backward_take_1(TerlanOwnedTuplePytorchSpatialGridSampler3dBackward& value) noexcept;

class TerlanOwnedTuplePytorchLossNllLossForward final {
 public:
  explicit TerlanOwnedTuplePytorchLossNllLossForward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLossNllLossForward> terlan_owned_tuple_pytorch_loss_nll_loss_forward(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss_forward_take_0(TerlanOwnedTuplePytorchLossNllLossForward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss_forward_take_1(TerlanOwnedTuplePytorchLossNllLossForward& value) noexcept;

class TerlanOwnedTuplePytorchLossNllLoss2dForward final {
 public:
  explicit TerlanOwnedTuplePytorchLossNllLoss2dForward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLossNllLoss2dForward> terlan_owned_tuple_pytorch_loss_nll_loss2d_forward(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss2d_forward_take_0(TerlanOwnedTuplePytorchLossNllLoss2dForward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss2d_forward_take_1(TerlanOwnedTuplePytorchLossNllLoss2dForward& value) noexcept;

class TerlanOwnedTuplePytorchSelectionCummax final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionCummax(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionCummax> terlan_owned_tuple_pytorch_selection_cummax(const Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummax_take_0(TerlanOwnedTuplePytorchSelectionCummax& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummax_take_1(TerlanOwnedTuplePytorchSelectionCummax& value) noexcept;

class TerlanOwnedTuplePytorchSelectionCummin final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionCummin(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionCummin> terlan_owned_tuple_pytorch_selection_cummin(const Tensor & self, int64_t dim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummin_take_0(TerlanOwnedTuplePytorchSelectionCummin& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummin_take_1(TerlanOwnedTuplePytorchSelectionCummin& value) noexcept;

class TerlanOwnedTuplePytorchSelectionKthvalue final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionKthvalue(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionKthvalue> terlan_owned_tuple_pytorch_selection_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_kthvalue_take_0(TerlanOwnedTuplePytorchSelectionKthvalue& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_kthvalue_take_1(TerlanOwnedTuplePytorchSelectionKthvalue& value) noexcept;

class TerlanOwnedTuplePytorchSelectionMaxDim final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionMaxDim(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionMaxDim> terlan_owned_tuple_pytorch_selection_max_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_max_dim_take_0(TerlanOwnedTuplePytorchSelectionMaxDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_max_dim_take_1(TerlanOwnedTuplePytorchSelectionMaxDim& value) noexcept;

class TerlanOwnedTuplePytorchSelectionMedianDim final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionMedianDim(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionMedianDim> terlan_owned_tuple_pytorch_selection_median_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_median_dim_take_0(TerlanOwnedTuplePytorchSelectionMedianDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_median_dim_take_1(TerlanOwnedTuplePytorchSelectionMedianDim& value) noexcept;

class TerlanOwnedTuplePytorchSelectionMinDim final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionMinDim(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionMinDim> terlan_owned_tuple_pytorch_selection_min_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_min_dim_take_0(TerlanOwnedTuplePytorchSelectionMinDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_min_dim_take_1(TerlanOwnedTuplePytorchSelectionMinDim& value) noexcept;

class TerlanOwnedTuplePytorchSelectionMode final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionMode(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionMode> terlan_owned_tuple_pytorch_selection_mode(const Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_mode_take_0(TerlanOwnedTuplePytorchSelectionMode& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_mode_take_1(TerlanOwnedTuplePytorchSelectionMode& value) noexcept;

class TerlanOwnedTuplePytorchSelectionNanmedianDim final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionNanmedianDim(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionNanmedianDim> terlan_owned_tuple_pytorch_selection_nanmedian_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_nanmedian_dim_take_0(TerlanOwnedTuplePytorchSelectionNanmedianDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_nanmedian_dim_take_1(TerlanOwnedTuplePytorchSelectionNanmedianDim& value) noexcept;

class TerlanOwnedTuplePytorchSelectionSort final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionSort(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionSort> terlan_owned_tuple_pytorch_selection_sort(const Tensor & self, int64_t dim, bool descending) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_sort_take_0(TerlanOwnedTuplePytorchSelectionSort& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_sort_take_1(TerlanOwnedTuplePytorchSelectionSort& value) noexcept;

class TerlanOwnedTuplePytorchSelectionTopk final {
 public:
  explicit TerlanOwnedTuplePytorchSelectionTopk(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSelectionTopk> terlan_owned_tuple_pytorch_selection_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_topk_take_0(TerlanOwnedTuplePytorchSelectionTopk& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_topk_take_1(TerlanOwnedTuplePytorchSelectionTopk& value) noexcept;

class TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward final {
 public:
  explicit TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward_take_0(TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward_take_1(TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward_take_2(TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward& value) noexcept;

class TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward final {
 public:
  explicit TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool has_bias_sizes, rust::Slice<const std::int64_t> bias_sizes, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool transposed, rust::Slice<const std::int64_t> output_padding, std::int64_t groups, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward_take_0(TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward_take_1(TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward_take_2(TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward& value) noexcept;

class TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault final {
 public:
  explicit TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_default(const Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_default_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_default_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault& value) noexcept;

class TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault final {
 public:
  explicit TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_default(const Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_default_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_default_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault& value) noexcept;

class TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions final {
 public:
  explicit TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_dimensions_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_dimensions_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions& value) noexcept;

class TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions final {
 public:
  explicit TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_dimensions_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_dimensions_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions& value) noexcept;

class TerlanOwnedTuplePytorchUnaryFrexp final {
 public:
  explicit TerlanOwnedTuplePytorchUnaryFrexp(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchUnaryFrexp> terlan_owned_tuple_pytorch_unary_frexp(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_unary_frexp_take_0(TerlanOwnedTuplePytorchUnaryFrexp& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_unary_frexp_take_1(TerlanOwnedTuplePytorchUnaryFrexp& value) noexcept;

class TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional final {
 public:
  explicit TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional> terlan_owned_tuple_pytorch_randomcompletion_rrelu_with_noise_functional(const at::Tensor & self, const at::Tensor & noise, bool has_generator, const Generator & generator, double lower, double upper, bool training) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_randomcompletion_rrelu_with_noise_functional_take_0(TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_randomcompletion_rrelu_with_noise_functional_take_1(TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional& value) noexcept;

class TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt final {
 public:
  explicit TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct(const at::Tensor & self, int64_t bins, bool has_range, rust::Slice<const double> range, bool has_weight, const at::Tensor & weight, bool density) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct_take_0(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct_take_1(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt& value) noexcept;

class TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor final {
 public:
  explicit TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor(const at::Tensor & self, const at::Tensor & bins, bool has_weight, const at::Tensor & weight, bool density) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor_take_0(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor_take_1(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor& value) noexcept;

class TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive final {
 public:
  explicit TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive(const at::Tensor & self, bool return_inverse, bool return_counts) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive_take_0(TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive_take_1(TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive_take_2(TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive& value) noexcept;

class TerlanOwnedTuplePytorchUniqueCompletionUniqueDim final {
 public:
  explicit TerlanOwnedTuplePytorchUniqueCompletionUniqueDim(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchUniqueCompletionUniqueDim> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_take_0(TerlanOwnedTuplePytorchUniqueCompletionUniqueDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_take_1(TerlanOwnedTuplePytorchUniqueCompletionUniqueDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_take_2(TerlanOwnedTuplePytorchUniqueCompletionUniqueDim& value) noexcept;

class TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive final {
 public:
  explicit TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive(const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive_take_0(TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive_take_1(TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive_take_2(TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive& value) noexcept;

class TerlanOwnedTuplePytorchSvdCompletionSvdDefault final {
 public:
  explicit TerlanOwnedTuplePytorchSvdCompletionSvdDefault(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchSvdCompletionSvdDefault> terlan_owned_tuple_pytorch_svdcompletion_svd_default(const at::Tensor & self, bool some, bool compute_uv) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_svdcompletion_svd_default_take_0(TerlanOwnedTuplePytorchSvdCompletionSvdDefault& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_svdcompletion_svd_default_take_1(TerlanOwnedTuplePytorchSvdCompletionSvdDefault& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_svdcompletion_svd_default_take_2(TerlanOwnedTuplePytorchSvdCompletionSvdDefault& value) noexcept;

class TerlanOwnedTuplePytorchStatisticsAminmax final {
 public:
  explicit TerlanOwnedTuplePytorchStatisticsAminmax(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchStatisticsAminmax> terlan_owned_tuple_pytorch_statistics_aminmax(const at::Tensor & self, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_take_0(TerlanOwnedTuplePytorchStatisticsAminmax& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_take_1(TerlanOwnedTuplePytorchStatisticsAminmax& value) noexcept;

class TerlanOwnedTuplePytorchStatisticsAminmaxDim final {
 public:
  explicit TerlanOwnedTuplePytorchStatisticsAminmaxDim(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchStatisticsAminmaxDim> terlan_owned_tuple_pytorch_statistics_aminmax_dim(const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_dim_take_0(TerlanOwnedTuplePytorchStatisticsAminmaxDim& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_dim_take_1(TerlanOwnedTuplePytorchStatisticsAminmaxDim& value) noexcept;

class TerlanOwnedTuplePytorchStatisticsStdMean final {
 public:
  explicit TerlanOwnedTuplePytorchStatisticsStdMean(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchStatisticsStdMean> terlan_owned_tuple_pytorch_statistics_std_mean(const at::Tensor & self, bool unbiased) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_take_0(TerlanOwnedTuplePytorchStatisticsStdMean& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_take_1(TerlanOwnedTuplePytorchStatisticsStdMean& value) noexcept;

class TerlanOwnedTuplePytorchStatisticsVarMean final {
 public:
  explicit TerlanOwnedTuplePytorchStatisticsVarMean(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchStatisticsVarMean> terlan_owned_tuple_pytorch_statistics_var_mean(const at::Tensor & self, bool unbiased) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_take_0(TerlanOwnedTuplePytorchStatisticsVarMean& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_take_1(TerlanOwnedTuplePytorchStatisticsVarMean& value) noexcept;

class TerlanOwnedTuplePytorchStatisticsStdMeanDimensions final {
 public:
  explicit TerlanOwnedTuplePytorchStatisticsStdMeanDimensions(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchStatisticsStdMeanDimensions> terlan_owned_tuple_pytorch_statistics_std_mean_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_dimensions_take_0(TerlanOwnedTuplePytorchStatisticsStdMeanDimensions& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_dimensions_take_1(TerlanOwnedTuplePytorchStatisticsStdMeanDimensions& value) noexcept;

class TerlanOwnedTuplePytorchStatisticsVarMeanDimensions final {
 public:
  explicit TerlanOwnedTuplePytorchStatisticsVarMeanDimensions(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchStatisticsVarMeanDimensions> terlan_owned_tuple_pytorch_statistics_var_mean_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_dimensions_take_0(TerlanOwnedTuplePytorchStatisticsVarMeanDimensions& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_dimensions_take_1(TerlanOwnedTuplePytorchStatisticsVarMeanDimensions& value) noexcept;

class TerlanOwnedTuplePytorchLinalgGeqrf final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgGeqrf(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgGeqrf> terlan_owned_tuple_pytorch_linalg_geqrf(const at::Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_geqrf_take_0(TerlanOwnedTuplePytorchLinalgGeqrf& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_geqrf_take_1(TerlanOwnedTuplePytorchLinalgGeqrf& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx> terlan_owned_tuple_pytorch_linalg_linalg_cholesky_ex(const Tensor & self, bool upper, bool check_errors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_cholesky_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_cholesky_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgEig final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgEig(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgEig> terlan_owned_tuple_pytorch_linalg_linalg_eig(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eig_take_0(TerlanOwnedTuplePytorchLinalgLinalgEig& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eig_take_1(TerlanOwnedTuplePytorchLinalgLinalgEig& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgEigh final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgEigh(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgEigh> terlan_owned_tuple_pytorch_linalg_linalg_eigh(const Tensor & self, rust::Str UPLO) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eigh_take_0(TerlanOwnedTuplePytorchLinalgLinalgEigh& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eigh_take_1(TerlanOwnedTuplePytorchLinalgLinalgEigh& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgInvEx final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgInvEx(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgInvEx> terlan_owned_tuple_pytorch_linalg_linalg_inv_ex(const Tensor & A, bool check_errors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_inv_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgInvEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_inv_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgInvEx& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLdlFactor final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLdlFactor(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLdlFactor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor(const Tensor & self, bool hermitian) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_take_0(TerlanOwnedTuplePytorchLinalgLinalgLdlFactor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_take_1(TerlanOwnedTuplePytorchLinalgLinalgLdlFactor& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex(const Tensor & self, bool hermitian, bool check_errors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex_take_2(TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLu final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLu(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLu> terlan_owned_tuple_pytorch_linalg_linalg_lu(const Tensor & A, bool pivot) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_take_0(TerlanOwnedTuplePytorchLinalgLinalgLu& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_take_1(TerlanOwnedTuplePytorchLinalgLinalgLu& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_take_2(TerlanOwnedTuplePytorchLinalgLinalgLu& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLuFactor final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLuFactor(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLuFactor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor(const Tensor & A, bool pivot) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_take_0(TerlanOwnedTuplePytorchLinalgLinalgLuFactor& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_take_1(TerlanOwnedTuplePytorchLinalgLinalgLuFactor& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex(const Tensor & A, bool pivot, bool check_errors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex_take_2(TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgQr final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgQr(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgQr> terlan_owned_tuple_pytorch_linalg_linalg_qr(const Tensor & A, rust::Str mode) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_qr_take_0(TerlanOwnedTuplePytorchLinalgLinalgQr& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_qr_take_1(TerlanOwnedTuplePytorchLinalgLinalgQr& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgSlogdet final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgSlogdet(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgSlogdet> terlan_owned_tuple_pytorch_linalg_linalg_slogdet(const Tensor & A) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_slogdet_take_0(TerlanOwnedTuplePytorchLinalgLinalgSlogdet& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_slogdet_take_1(TerlanOwnedTuplePytorchLinalgLinalgSlogdet& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgSolveEx final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgSolveEx(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgSolveEx> terlan_owned_tuple_pytorch_linalg_linalg_solve_ex(const Tensor & A, const Tensor & B, bool left, bool check_errors) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_solve_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgSolveEx& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_solve_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgSolveEx& value) noexcept;

class TerlanOwnedTuplePytorchLinalgQr final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgQr(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgQr> terlan_owned_tuple_pytorch_linalg_qr(const Tensor & self, bool some) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_qr_take_0(TerlanOwnedTuplePytorchLinalgQr& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_qr_take_1(TerlanOwnedTuplePytorchLinalgQr& value) noexcept;

class TerlanOwnedTuplePytorchLinalgSlogdet final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgSlogdet(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgSlogdet> terlan_owned_tuple_pytorch_linalg_slogdet(const Tensor & self) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_slogdet_take_0(TerlanOwnedTuplePytorchLinalgSlogdet& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_slogdet_take_1(TerlanOwnedTuplePytorchLinalgSlogdet& value) noexcept;

class TerlanOwnedTuplePytorchLinalgTriangularSolve final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgTriangularSolve(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgTriangularSolve> terlan_owned_tuple_pytorch_linalg_triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_triangular_solve_take_0(TerlanOwnedTuplePytorchLinalgTriangularSolve& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_triangular_solve_take_1(TerlanOwnedTuplePytorchLinalgTriangularSolve& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLstsq final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLstsq(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2, std::unique_ptr<at::Tensor> value_3) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)), value_3_(std::move(value_3)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }
  std::unique_ptr<at::Tensor> take_3() noexcept { return std::move(value_3_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
  std::unique_ptr<at::Tensor> value_3_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLstsq> terlan_owned_tuple_pytorch_linalg_linalg_lstsq(const at::Tensor & a, const at::Tensor & b) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_0(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_1(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_2(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_3(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2, std::unique_ptr<at::Tensor> value_3) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)), value_3_(std::move(value_3)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }
  std::unique_ptr<at::Tensor> take_3() noexcept { return std::move(value_3_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
  std::unique_ptr<at::Tensor> value_3_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured(const at::Tensor & a, const at::Tensor & b, double rcond, rust::Str driver) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_0(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_1(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_2(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_3(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept;

class TerlanOwnedTuplePytorchLinalgLinalgSvd final {
 public:
  explicit TerlanOwnedTuplePytorchLinalgLinalgSvd(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgSvd> terlan_owned_tuple_pytorch_linalg_linalg_svd(const at::Tensor & self, bool full) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_svd_take_0(TerlanOwnedTuplePytorchLinalgLinalgSvd& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_svd_take_1(TerlanOwnedTuplePytorchLinalgLinalgSvd& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_svd_take_2(TerlanOwnedTuplePytorchLinalgLinalgSvd& value) noexcept;

class TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward final {
 public:
  explicit TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward> terlan_owned_tuple_pytorch_losscompletion_multilabel_margin_loss_forward(const at::Tensor & input, const at::Tensor & target, std::int64_t reduction) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_losscompletion_multilabel_margin_loss_forward_take_0(TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_losscompletion_multilabel_margin_loss_forward_take_1(TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward& value) noexcept;

class TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask final {
 public:
  explicit TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask_take_0(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask_take_1(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask& value) noexcept;

class TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask final {
 public:
  explicit TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask_take_0(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask_take_1(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask& value) noexcept;

class TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault final {
 public:
  explicit TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default(const at::Tensor & lu_data, const at::Tensor & lu_pivots, bool unpack_data, bool unpack_pivots) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default_take_0(TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default_take_1(TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default_take_2(TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentLstmCell final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentLstmCell(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentLstmCell> terlan_owned_tuple_pytorch_recurrent_lstm_cell(const at::Tensor & input, const TerlanPytorchTensorTensorResourceListInput& hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_cell_take_0(TerlanOwnedTuplePytorchRecurrentLstmCell& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_cell_take_1(TerlanOwnedTuplePytorchRecurrentLstmCell& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentRnnTanh final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentRnnTanh(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentRnnTanh> terlan_owned_tuple_pytorch_recurrent_rnn_tanh(const at::Tensor & input, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_tanh_take_0(TerlanOwnedTuplePytorchRecurrentRnnTanh& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_tanh_take_1(TerlanOwnedTuplePytorchRecurrentRnnTanh& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentRnnRelu final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentRnnRelu(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentRnnRelu> terlan_owned_tuple_pytorch_recurrent_rnn_relu(const at::Tensor & input, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_relu_take_0(TerlanOwnedTuplePytorchRecurrentRnnRelu& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_relu_take_1(TerlanOwnedTuplePytorchRecurrentRnnRelu& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentGru final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentGru(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentGru> terlan_owned_tuple_pytorch_recurrent_gru(const at::Tensor & input, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_gru_take_0(TerlanOwnedTuplePytorchRecurrentGru& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_gru_take_1(TerlanOwnedTuplePytorchRecurrentGru& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentLstm final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentLstm(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentLstm> terlan_owned_tuple_pytorch_recurrent_lstm(const at::Tensor & input, const TerlanPytorchTensorTensorResourceListInput& hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_take_0(TerlanOwnedTuplePytorchRecurrentLstm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_take_1(TerlanOwnedTuplePytorchRecurrentLstm& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_take_2(TerlanOwnedTuplePytorchRecurrentLstm& value) noexcept;

class TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized final {
 public:
  explicit TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized> terlan_owned_tuple_pytorch_quantizationinternals_choose_qparams_optimized(const Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationinternals_choose_qparams_optimized_take_0(TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationinternals_choose_qparams_optimized_take_1(TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentDataRnnTanhData final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentDataRnnTanhData(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataRnnTanhData> terlan_owned_tuple_pytorch_recurrentdata_rnn_tanh_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_tanh_data_take_0(TerlanOwnedTuplePytorchRecurrentDataRnnTanhData& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_tanh_data_take_1(TerlanOwnedTuplePytorchRecurrentDataRnnTanhData& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentDataRnnReluData final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentDataRnnReluData(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataRnnReluData> terlan_owned_tuple_pytorch_recurrentdata_rnn_relu_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_relu_data_take_0(TerlanOwnedTuplePytorchRecurrentDataRnnReluData& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_relu_data_take_1(TerlanOwnedTuplePytorchRecurrentDataRnnReluData& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentDataGruData final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentDataGruData(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataGruData> terlan_owned_tuple_pytorch_recurrentdata_gru_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_gru_data_take_0(TerlanOwnedTuplePytorchRecurrentDataGruData& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_gru_data_take_1(TerlanOwnedTuplePytorchRecurrentDataGruData& value) noexcept;

class TerlanOwnedTuplePytorchRecurrentDataLstmData final {
 public:
  explicit TerlanOwnedTuplePytorchRecurrentDataLstmData(std::unique_ptr<at::Tensor> value_0, std::unique_ptr<at::Tensor> value_1, std::unique_ptr<at::Tensor> value_2) noexcept : value_0_(std::move(value_0)), value_1_(std::move(value_1)), value_2_(std::move(value_2)) {}
  std::unique_ptr<at::Tensor> take_0() noexcept { return std::move(value_0_); }
  std::unique_ptr<at::Tensor> take_1() noexcept { return std::move(value_1_); }
  std::unique_ptr<at::Tensor> take_2() noexcept { return std::move(value_2_); }

 private:
  std::unique_ptr<at::Tensor> value_0_;
  std::unique_ptr<at::Tensor> value_1_;
  std::unique_ptr<at::Tensor> value_2_;
};
std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataLstmData> terlan_owned_tuple_pytorch_recurrentdata_lstm_data(const at::Tensor & data, const at::Tensor & batch_sizes, const TerlanPytorchTensorTensorResourceListInput& hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_lstm_data_take_0(TerlanOwnedTuplePytorchRecurrentDataLstmData& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_lstm_data_take_1(TerlanOwnedTuplePytorchRecurrentDataLstmData& value) noexcept;
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_lstm_data_take_2(TerlanOwnedTuplePytorchRecurrentDataLstmData& value) noexcept;


}  // namespace at::native
