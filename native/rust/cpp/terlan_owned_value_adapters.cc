#include "include/terlan_owned_value_adapters.hpp"

#include <string>
#include <utility>

namespace at::native {

bool TerlanPytorchTensorTensorResourceListInput::push(const at::Tensor& value) noexcept {
  try { values_.push_back(value); return true; } catch (...) { return false; }
}
const std::vector<at::Tensor>& TerlanPytorchTensorTensorResourceListInput::values() const noexcept { return values_; }
std::unique_ptr<TerlanPytorchTensorTensorResourceListInput> terlan_resource_list_pytorch_tensor_tensor_new() noexcept {
  try { return std::make_unique<TerlanPytorchTensorTensorResourceListInput>(); } catch (...) { return nullptr; }
}
bool terlan_resource_list_pytorch_tensor_tensor_push(TerlanPytorchTensorTensorResourceListInput& values, const at::Tensor& value) noexcept { return values.push(value); }

std::unique_ptr<at::native::CudaStream> terlan_resource_pytorch_cuda_stream_copy(const at::native::CudaStream& value) noexcept {
  try { return std::make_unique<at::native::CudaStream>(value); } catch (...) { return nullptr; }
}

std::unique_ptr<at::native::Generator> terlan_resource_pytorch_randomcompletion_generator_copy(const at::native::Generator& value) noexcept {
  try { return std::make_unique<at::native::Generator>(value); } catch (...) { return nullptr; }
}

std::unique_ptr<at::native::StorageHandle> terlan_resource_pytorch_storagenative_storagehandle_copy(const at::native::StorageHandle& value) noexcept {
  try { return std::make_unique<at::native::StorageHandle>(value); } catch (...) { return nullptr; }
}

std::unique_ptr<at::Tensor> terlan_resource_pytorch_tensor_tensor_copy(const at::Tensor& value) noexcept {
  try { return std::make_unique<at::Tensor>(value); } catch (...) { return nullptr; }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones(rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(ones(at::IntArrayRef(size.data(), size.size()), std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_dtype(rust::Slice<const std::int64_t> size, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(ones(at::IntArrayRef(size.data(), size.size()), ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype)), std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_layout(rust::Slice<const std::int64_t> size, std::int64_t dtype, std::int64_t layout) noexcept {
  try {
    return std::make_unique<at::Tensor>(ones(at::IntArrayRef(size.data(), size.size()), ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<at::Layout>(static_cast<c10::Layout>(layout)), std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_device(rust::Slice<const std::int64_t> size, std::int64_t dtype, std::int64_t layout, rust::Str device) noexcept {
  try {
    return std::make_unique<at::Tensor>(ones(at::IntArrayRef(size.data(), size.size()), ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<at::Layout>(static_cast<c10::Layout>(layout)), ::std::optional<at::Device>(c10::Device(std::string(device.data(), device.size()))), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ones_options(rust::Slice<const std::int64_t> size, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(ones(at::IntArrayRef(size.data(), size.size()), ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<at::Layout>(static_cast<c10::Layout>(layout)), ::std::optional<at::Device>(c10::Device(std::string(device.data(), device.size()))), ::std::optional<bool>(pin_memory)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<c10::Scalar> terlan_owned_value_pytorch_tensor_item(const at::Tensor& value) noexcept {
  try {
    using TerlanSelectedOverload = Scalar (at::Tensor::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::item);
    return std::make_unique<c10::Scalar>((value.*terlan_selected_overload)());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mul_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::mul);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_abs(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.abs());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_detach(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.detach());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_ceil(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.ceil());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_floor(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.floor());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_reciprocal(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.reciprocal());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_round(const at::Tensor& value) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::round);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sign(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.sign());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_square(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.square());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_trunc(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.trunc());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_clone(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.clone(std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_relu(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.relu());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_multiply(const at::Tensor& value, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::mul);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_multiply_scalar(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::mul);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_negate(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.neg());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_unsqueeze(const Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::unsqueeze(self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_matmul(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::matmul(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_transpose(const Tensor & self, int64_t dim0, int64_t dim1) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::transpose(self, dim0, dim1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_exp(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::exp(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sqrt(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sqrt(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_log(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_minimum(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::minimum(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::gather(self, dim, index, sparse_grad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mean(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mean);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sum(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sum);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mean_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, bool, ::std::optional<ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mean);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sum_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, bool, ::std::optional<ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sum);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gelu(const Tensor & self, rust::Str approximate) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::gelu(self, c10::string_view(approximate.data(), approximate.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_softmax(const Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softmax(self, dim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_argmax(const Tensor & self, std::int64_t dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::argmax(self, ::std::optional<int64_t>(dim), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_embedding_renorm(const Tensor & self, const Tensor & indices, double max_norm, double norm_type) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::embedding_renorm(self, indices, max_norm, norm_type));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_batch_norm(const Tensor & input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean, const at::Tensor& running_var, bool training, double momentum, double eps, bool cudnn_enabled) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::batch_norm(input, std::optional<at::Tensor>(weight), std::optional<at::Tensor>(bias), std::optional<at::Tensor>(running_mean), std::optional<at::Tensor>(running_var), training, momentum, eps, cudnn_enabled));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_layer_norm(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor& weight, const at::Tensor& bias, double eps, bool cudnn_enable) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::layer_norm(input, IntArrayRef(normalized_shape.data(), normalized_shape.size()), std::optional<at::Tensor>(weight), std::optional<at::Tensor>(bias), eps, cudnn_enable));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::mse_loss(self, target, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, int64_t, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::narrow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, start, length));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::flatten(self, start_dim, end_dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_amax(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::amax(self, IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_adaptive_avg_pool2d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::adaptive_avg_pool2d(self, IntArrayRef(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_copy(const Tensor & self, const Tensor & src, bool non_blocking) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::copy(self, src, non_blocking));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_int(std::int64_t arg0) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_from_int(arg0));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_float(double arg0) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_from_float(arg0));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_bool(bool arg0) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_from_bool(arg0));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_floats(rust::Slice<const double> arg0, rust::Slice<const std::int64_t> arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_from_floats(arg0, c10::ArrayRef<std::int64_t>(arg1.data(), arg1.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_ints(rust::Slice<const std::int64_t> arg0, rust::Slice<const std::int64_t> arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_from_ints(c10::ArrayRef<std::int64_t>(arg0.data(), arg0.size()), c10::ArrayRef<std::int64_t>(arg1.data(), arg1.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_full_float(rust::Slice<const std::int64_t> arg0, double arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_full_float(c10::ArrayRef<std::int64_t>(arg0.data(), arg0.size()), arg1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_device(const at::Tensor & arg0, std::int64_t arg1, std::int64_t arg2) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_device(arg0, arg1, arg2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_dtype(const at::Tensor & arg0, std::int64_t arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_dtype(arg0, arg1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_other(const at::Tensor & tensor, const at::Tensor & other, bool non_blocking) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_other(tensor, other, non_blocking));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_device_dtype(const at::Tensor & tensor, int64_t device_type, int64_t device_index, int64_t dtype, bool non_blocking, bool copy) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_device_dtype(tensor, device_type, device_index, dtype, non_blocking, copy));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_dtype_layout(const at::Tensor & tensor, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool non_blocking, bool copy, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_dtype_layout(tensor, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory, non_blocking, copy, has_memory_format, memory_format));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_nested_from_tensor_list(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_nested_from_tensor_list(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_nested_to_padded_tensor(const at::Tensor & tensor, double padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_nested_to_padded_tensor_exact(tensor, padding, has_output_size, c10::ArrayRef<std::int64_t>(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_to_padded_tensor(const at::Tensor & tensor, double padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_padded_tensor(tensor, padding, has_output_size, c10::ArrayRef<int64_t>(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_from_file(rust::Str filename, bool has_shared, bool shared, bool has_size, int64_t size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, int64_t has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_from_file(std::string_view(filename.data(), filename.size()), has_shared, shared, has_size, size, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_full_typed_native(rust::Slice<const std::int64_t> arg0, double arg1, std::int64_t arg2, std::int64_t arg3, std::int64_t arg4) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_full_typed(c10::ArrayRef<std::int64_t>(arg0.data(), arg0.size()), arg1, arg2, arg3, arg4));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_empty_typed_native(rust::Slice<const std::int64_t> arg0, std::int64_t arg1, std::int64_t arg2, std::int64_t arg3) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_empty_typed(c10::ArrayRef<std::int64_t>(arg0.data(), arg0.size()), arg1, arg2, arg3));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_arange_typed_native(double arg0, double arg1, double arg2, std::int64_t arg3, std::int64_t arg4, std::int64_t arg5, bool arg6) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_arange_typed(arg0, arg1, arg2, arg3, arg4, arg5, arg6));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_linspace_typed_native(double arg0, double arg1, std::int64_t arg2, std::int64_t arg3, std::int64_t arg4, std::int64_t arg5, bool arg6) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_linspace_typed(arg0, arg1, arg2, arg3, arg4, arg5, arg6));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_logspace_typed_native(double arg0, double arg1, std::int64_t arg2, double arg3, std::int64_t arg4, std::int64_t arg5, std::int64_t arg6, bool arg7) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_logspace_typed(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_eye_typed_native(std::int64_t arg0, std::int64_t arg1, std::int64_t arg2, std::int64_t arg3, std::int64_t arg4, bool arg5) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_eye_typed(arg0, arg1, arg2, arg3, arg4, arg5));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_add(const Tensor & self, const Tensor & other, double alpha) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::add);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, c10::Scalar(alpha)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_subtract(const Tensor & self, const Tensor & other, double alpha) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sub);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, c10::Scalar(alpha)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_divide(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_add_scalar(const Tensor & self, double other, double alpha) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::add);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other), c10::Scalar(alpha)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_require_grad(const at::Tensor & arg0) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_require_grad(arg0));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_backward_many_token(const TerlanPytorchTensorTensorResourceListInput& outputs, const TerlanPytorchTensorTensorResourceListInput& grad_outputs, bool retain_graph, bool create_graph) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_backward_many_token(outputs.values(), grad_outputs.values(), retain_graph, create_graph));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_autograd_grad(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, bool arg4) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_autograd_grad(arg0, arg1, arg2, arg3, arg4));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gradient(const at::Tensor & arg0) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_gradient(arg0));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_gradient_with_graph(const at::Tensor & arg0) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_gradient_with_graph(arg0));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_sgd_step(const at::Tensor & arg0, double arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_sgd_step(arg0, arg1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_categorical_sample(const at::Tensor & arg0, std::int64_t arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_categorical_sample(arg0, arg1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_mse_loss_gradient(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, std::int64_t arg3) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_mse_loss_gradient(arg0, arg1, arg2, arg3));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_cross_entropy_loss(const at::Tensor & arg0, const at::Tensor & arg1, std::int64_t arg2, std::int64_t arg3, double arg4) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_cross_entropy_loss(arg0, arg1, arg2, arg3, arg4));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_clamp(const at::Tensor & arg0, double arg1, double arg2) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_clamp(arg0, arg1, arg2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_conv2d(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, rust::Slice<const std::int64_t> arg3, rust::Slice<const std::int64_t> arg4, rust::Slice<const std::int64_t> arg5, std::int64_t arg6) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_conv2d(arg0, arg1, arg2, c10::ArrayRef<std::int64_t>(arg3.data(), arg3.size()), c10::ArrayRef<std::int64_t>(arg4.data(), arg4.size()), c10::ArrayRef<std::int64_t>(arg5.data(), arg5.size()), arg6));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_tensor_torchscript_forward_pair(const TorchScriptModel & arg0, const at::Tensor & arg1) noexcept {
  try {
    return std::make_unique<TensorPair>(tensor_torchscript_forward_pair(arg0, arg1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_tanh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tanh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_resize_cxx(const Tensor & self, rust::Slice<const std::int64_t> size, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::resize(self, IntArrayRef(size.data(), size.size()), has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_empty_strided_cxx(rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(IntArrayRef, IntArrayRef, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::empty_strided);
    return std::make_unique<at::Tensor>(terlan_selected_overload(IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensor_empty_permuted_cxx(rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> physical_layout, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(IntArrayRef, IntArrayRef, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::empty_permuted);
    return std::make_unique<at::Tensor>(terlan_selected_overload(IntArrayRef(size.data(), size.size()), IntArrayRef(physical_layout.data(), physical_layout.size()), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_block_diag(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::block_diag(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_cartesian_prod(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cartesian_prod(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_column_stack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::column_stack(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_dstack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::dstack(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_hstack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hstack(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_row_stack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::row_stack(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_vstack(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::vstack(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_cat(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cat(tensors.values(), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_stack(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::stack(tensors.values(), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_concat(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::concat(tensors.values(), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorjoining_concatenate(const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::concatenate(tensors.values(), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distance_cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cosine_similarity(x1, x2, dim, eps));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distance_chalf(const at::Tensor & input) noexcept {
  try {
    return std::make_unique<at::Tensor>(chalf_exact(input));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distance_pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pairwise_distance(x1, x2, p, eps, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_alpha_dropout(const Tensor & input, double p, bool train) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::alpha_dropout(input, p, train));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_dropout(const Tensor & input, double p, bool train) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::dropout(input, p, train));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_feature_dropout(const Tensor & input, double p, bool train) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::feature_dropout(input, p, train));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_native_dropout_backward(const Tensor & grad_output, const Tensor & mask, double scale) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::native_dropout_backward(grad_output, mask, scale));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_regularization_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & buffer) noexcept {
  try {
    return std::make_unique<at::Tensor>(log_sigmoid_backward_exact(grad_output, input, buffer));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dropoutcompletion_feature_alpha_dropout(const Tensor & input, double p, bool train) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::feature_alpha_dropout(input, p, train));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_glucompletion_glu_jvp(const Tensor & glu, const Tensor & x, const Tensor & dx, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::glu_jvp(glu, x, dx, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_glucompletion_glu_backward_jvp(const Tensor & grad_x, const Tensor & grad_glu, const Tensor & x, const Tensor & dgrad_glu, const Tensor & dx, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_adaptive_avg_pool1d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::adaptive_avg_pool1d(self, IntArrayRef(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool1d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool1d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool1d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_pool1d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool2d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool3d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool2d_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool3d_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_pool2d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_pool3d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::adaptive_max_pool2d_backward(grad_output, self, indices));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::adaptive_max_pool3d_backward(grad_output, self, indices));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & indices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fractional_max_pool2d_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), indices));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & indices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fractional_max_pool3d_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), indices));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode, const Tensor & indices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_pool2d_with_indices_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode, indices));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode, const Tensor & indices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_pool3d_with_indices_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode, indices));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_unpool2d(const Tensor & self, const Tensor & indices, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_unpool2d(self, indices, IntArrayRef(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_max_unpool3d(const Tensor & self, const Tensor & indices, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::max_unpool3d(self, indices, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d_divisor(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool2d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d_divisor(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool3d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool2d_backward_divisor(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool2d_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_pooling_avg_pool3d_backward_divisor(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::avg_pool3d_backward(grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_affine_grid_generator(const Tensor & theta, rust::Slice<const std::int64_t> size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::affine_grid_generator(theta, IntArrayRef(size.data(), size.size()), align_corners));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_affine_grid_generator_backward(const Tensor & grad, rust::Slice<const std::int64_t> size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::affine_grid_generator_backward(grad, IntArrayRef(size.data(), size.size()), align_corners));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spatial_grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diagonal(self, offset, dim1, dim2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_diagonal_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diagonal_backward(grad_output, IntArrayRef(input_sizes.data(), input_sizes.size()), offset, dim1, dim2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_select_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::select_backward(grad_output, IntArrayRef(input_sizes.data(), input_sizes.size()), dim, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_slice_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_backward(grad_output, IntArrayRef(input_sizes.data(), input_sizes.size()), dim, start, end, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapegradient_unfold_backward(const Tensor & grad_in, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t size, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::unfold_backward(grad_in, IntArrayRef(input_sizes.data(), input_sizes.size()), dim, size, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reflection_pad1d_backward(grad_output, self, IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reflection_pad2d_backward(grad_output, self, IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_reflection_pad3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reflection_pad3d_backward(grad_output, self, IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::replication_pad1d_backward(grad_output, self, IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::replication_pad2d_backward(grad_output, self, IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingbackwardcompletion_replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::replication_pad3d_backward(grad_output, self, IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_cumulative_trapezoid_dx_int(const Tensor & y, std::int64_t dx, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::cumulative_trapezoid);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, c10::Scalar(dx), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_cumulative_trapezoid_dx_float(const Tensor & y, double dx, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::cumulative_trapezoid);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, c10::Scalar(dx), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_cumulative_trapezoid_x(const Tensor & y, const Tensor & x, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::cumulative_trapezoid);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, x, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapezoid_dx_int(const Tensor & y, std::int64_t dx, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::trapezoid);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, c10::Scalar(dx), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapezoid_dx_float(const Tensor & y, double dx, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::trapezoid);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, c10::Scalar(dx), dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapezoid_x(const Tensor & y, const Tensor & x, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::trapezoid);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, x, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapz_dx(const Tensor & y, double dx, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, double, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::trapz);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, dx, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_trapz_x(const Tensor & y, const Tensor & x, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::trapz);
    return std::make_unique<at::Tensor>(terlan_selected_overload(y, x, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_integrationcompletion_kl_div(const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::kl_div(self, target, reduction, log_target));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nantonumnative_nan_to_num_configured(const Tensor & self, bool has_nan, double nan, bool has_posinf, double posinf, bool has_neginf, double neginf) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nan_to_num(self, has_nan ? ::std::optional<double>(nan) : std::nullopt, has_posinf ? ::std::optional<double>(posinf) : std::nullopt, has_neginf ? ::std::optional<double>(neginf) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::as_strided(self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_copy(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::as_strided_copy(self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_scatter(const Tensor & self, const Tensor & src, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::as_strided_scatter(self, src, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_offset(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::as_strided(self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), ::std::optional<int64_t>(storage_offset)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_copy_offset(const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::as_strided_copy(self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), ::std::optional<int64_t>(storage_offset)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_strided_as_strided_scatter_offset(const Tensor & self, const Tensor & src, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::as_strided_scatter(self, src, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), ::std::optional<int64_t>(storage_offset)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexfillnative_index_fill_scalar(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::index_fill);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexfillnative_index_fill_tensor(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::index_fill);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lerpnative_lerp_scalar(const Tensor & self, const Tensor & end, bool weight_is_int, std::int64_t weight_int, double weight_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lerp);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, end, weight_is_int ? c10::Scalar(weight_int) : c10::Scalar(weight_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lerpnative_lerp_tensor(const Tensor & self, const Tensor & end, const Tensor & weight) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lerp);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, end, weight));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_wherenative_where_scalar(const Tensor & condition, bool self_is_int, std::int64_t self_int, double self_float, bool other_is_int, std::int64_t other_int, double other_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::where);
    return std::make_unique<at::Tensor>(terlan_selected_overload(condition, self_is_int ? c10::Scalar(self_int) : c10::Scalar(self_float), other_is_int ? c10::Scalar(other_int) : c10::Scalar(other_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_wherenative_where_scalar_other(const Tensor & condition, const Tensor & self, bool other_is_int, std::int64_t other_int, double other_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::where);
    return std::make_unique<at::Tensor>(terlan_selected_overload(condition, self, other_is_int ? c10::Scalar(other_int) : c10::Scalar(other_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_wherenative_where_scalar_self(const Tensor & condition, bool self_is_int, std::int64_t self_int, double self_float, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::where);
    return std::make_unique<at::Tensor>(terlan_selected_overload(condition, self_is_int ? c10::Scalar(self_int) : c10::Scalar(self_float), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexmutationnative_index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::index_add(self, dim, index, source, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexmutationnative_index_select_backward(const Tensor & grad, rust::Slice<const std::int64_t> self_sizes, int64_t dim, const Tensor & index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::index_select_backward(grad, IntArrayRef(self_sizes.data(), self_sizes.size()), dim, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fillnative_fill_scalar(const Tensor & self, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fill);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fillnative_masked_fill_scalar(const Tensor & self, const Tensor & mask, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::masked_fill);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, mask, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_int(const Tensor & self, std::int64_t lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardshrink(self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_float(const Tensor & self, double lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardshrink(self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_backward_int(const Tensor & grad_out, const Tensor & self, std::int64_t lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardshrink_backward(grad_out, self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_hardshrink_backward_float(const Tensor & grad_out, const Tensor & self, double lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardshrink_backward(grad_out, self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_int(const Tensor & self, std::int64_t lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softshrink(self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_float(const Tensor & self, double lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softshrink(self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softshrink_backward(grad_output, self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softshrink_backward_float(const Tensor & grad_output, const Tensor & self, double lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softshrink_backward(grad_output, self, c10::Scalar(lambd)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_int(const Tensor & self, std::int64_t beta, std::int64_t threshold) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softplus(self, c10::Scalar(beta), c10::Scalar(threshold)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_float(const Tensor & self, double beta, double threshold) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softplus(self, c10::Scalar(beta), c10::Scalar(threshold)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t beta, std::int64_t threshold) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softplus_backward(grad_output, self, c10::Scalar(beta), c10::Scalar(threshold)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_softplus_backward_float(const Tensor & grad_output, const Tensor & self, double beta, double threshold) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::softplus_backward(grad_output, self, c10::Scalar(beta), c10::Scalar(threshold)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_int(const Tensor & self, std::int64_t threshold, std::int64_t value) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::threshold(self, c10::Scalar(threshold), c10::Scalar(value)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_float(const Tensor & self, double threshold, double value) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::threshold(self, c10::Scalar(threshold), c10::Scalar(value)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t threshold) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::threshold_backward(grad_output, self, c10::Scalar(threshold)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_activationcompletion_threshold_backward_float(const Tensor & grad_output, const Tensor & self, double threshold) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::threshold_backward(grad_output, self, c10::Scalar(threshold)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_celu_int(const Tensor & self, std::int64_t alpha) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::celu(self, c10::Scalar(alpha)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_celu_float(const Tensor & self, double alpha) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::celu(self, c10::Scalar(alpha)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_int(const Tensor & self, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::elu(self, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_float(const Tensor & self, double alpha, double scale, double input_scale) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::elu(self, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_backward_int(const Tensor & grad_output, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale, bool is_result, const Tensor & self_or_result) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::elu_backward(grad_output, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale), is_result, self_or_result));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_elu_backward_float(const Tensor & grad_output, double alpha, double scale, double input_scale, bool is_result, const Tensor & self_or_result) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::elu_backward(grad_output, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale), is_result, self_or_result));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_int(const Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardtanh(self, c10::Scalar(min_val), c10::Scalar(max_val)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_float(const Tensor & self, double min_val, double max_val) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardtanh(self, c10::Scalar(min_val), c10::Scalar(max_val)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardtanh_backward(grad_output, self, c10::Scalar(min_val), c10::Scalar(max_val)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_hardtanh_backward_float(const Tensor & grad_output, const Tensor & self, double min_val, double max_val) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardtanh_backward(grad_output, self, c10::Scalar(min_val), c10::Scalar(max_val)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_int(const Tensor & self, std::int64_t negative_slope) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::leaky_relu(self, c10::Scalar(negative_slope)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_float(const Tensor & self, double negative_slope) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::leaky_relu(self, c10::Scalar(negative_slope)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_backward_int(const Tensor & grad_output, const Tensor & self, std::int64_t negative_slope, bool self_is_result) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::leaky_relu_backward(grad_output, self, c10::Scalar(negative_slope), self_is_result));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_extendedactivationcompletion_leaky_relu_backward_float(const Tensor & grad_output, const Tensor & self, double negative_slope, bool self_is_result) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::leaky_relu_backward(grad_output, self, c10::Scalar(negative_slope), self_is_result));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cosine_embedding_loss(input1, input2, target, margin, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hinge_embedding_loss(self, target, margin, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_huber_loss(const Tensor & self, const Tensor & target, int64_t reduction, double delta) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::huber_loss(self, target, reduction, delta));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_huber_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::huber_loss_backward(grad_output, self, target, reduction, delta));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::l1_loss(self, target, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_nll_loss_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index, const at::Tensor & total_weight) noexcept {
  try {
    return std::make_unique<at::Tensor>(nll_loss_backward_exact(grad_output, input, target, has_weight, weight, reduction, ignore_index, total_weight));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index, const at::Tensor & total_weight) noexcept {
  try {
    return std::make_unique<at::Tensor>(nll_loss2d_backward_exact(grad_output, input, target, has_weight, weight, reduction, ignore_index, total_weight));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_nll_loss_nd(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept {
  try {
    return std::make_unique<at::Tensor>(nll_loss_nd_exact(input, target, has_weight, weight, reduction, ignore_index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::margin_ranking_loss(input1, input2, target, margin, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::poisson_nll_loss(input, target, log_input, full, eps, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction, double beta) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::smooth_l1_loss(self, target, reduction, beta));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::smooth_l1_loss_backward(grad_output, self, target, reduction, beta));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::soft_margin_loss(self, target, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_loss_triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_narrow_view(const Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, int64_t, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::narrow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, start, length));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_narrow_tensor(const Tensor & self, int64_t dim, const Tensor & start, int64_t length) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::narrow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, start, length));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_unfold(const at::Tensor& value, int64_t dimension, int64_t size, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.unfold(dimension, size, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_trace_backward(const Tensor & grad, rust::Slice<const std::int64_t> sizes) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::trace_backward(grad, IntArrayRef(sizes.data(), sizes.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_zero(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::zero(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_im2col(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> dilation, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::im2col(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(dilation.data(), dilation.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(stride.data(), stride.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_col2im(const Tensor & self, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> dilation, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::col2im(self, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(dilation.data(), dilation.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(stride.data(), stride.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shapecompletion_tensordot(const Tensor & self, const Tensor & other, rust::Slice<const std::int64_t> dims_self, rust::Slice<const std::int64_t> dims_other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tensordot(self, other, IntArrayRef(dims_self.data(), dims_self.size()), IntArrayRef(dims_other.data(), dims_other.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_argsort_stable(const Tensor & self, bool stable, int64_t dim, bool descending) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, bool, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::argsort);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, stable, dim, descending));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_bucketize(const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bucketize);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, boundaries, out_int32, right));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diag_embed(self, offset, dim1, dim2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_diagonal_copy(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diagonal_copy(self, offset, dim1, dim2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_diagonal_scatter(const Tensor & self, const Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diagonal_scatter(self, src, offset, dim1, dim2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::index_copy(self, dim, index, source));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_select(const Tensor & self, int64_t dim, const Tensor & index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::index_select(self, dim, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_isin(const Tensor & elements, const Tensor & test_elements, bool assume_unique, bool invert) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::isin);
    return std::make_unique<at::Tensor>(terlan_selected_overload(elements, test_elements, assume_unique, invert));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_masked_fill_tensor(const Tensor & self, const Tensor & mask, const Tensor & value) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::masked_fill);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, mask, value));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::masked_scatter(self, mask, source));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_put(const Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::put(self, index, source, accumulate));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, src));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_add(self, dim, index, src));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_select_scatter(const Tensor & self, const Tensor & src, int64_t dim, int64_t index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::select_scatter(self, src, dim, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_unfold_copy(const Tensor & self, int64_t dimension, int64_t size, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::unfold_copy(self, dimension, size, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_where_tensor(const Tensor & condition, const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::where);
    return std::make_unique<at::Tensor>(terlan_selected_overload(condition, self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_tensor(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_index(tensor, indices.values(), c10::ArrayRef<std::int64_t>(present.data(), present.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_indexing_index_put(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present, const at::Tensor & values, bool accumulate) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_index_put(tensor, indices.values(), c10::ArrayRef<std::int64_t>(present.data(), present.size()), values, accumulate));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_cumprod_backward(const Tensor & grad, const Tensor & input, int64_t dim, const Tensor & output) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cumprod_backward(grad, input, dim, output));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_fake_quantize_per_channel_affine_cachemask_backward(const Tensor & grad, const Tensor & mask) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fake_quantize_per_channel_affine_cachemask_backward(grad, mask));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_fake_quantize_per_tensor_affine_cachemask_backward(const Tensor & grad, const Tensor & mask) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fake_quantize_per_tensor_affine_cachemask_backward(grad, mask));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_gather_backward(const Tensor & grad, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::gather_backward(grad, self, dim, index, sparse_grad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_gelu_backward(const Tensor & grad_output, const Tensor & self, rust::Str approximate) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::gelu_backward(grad_output, self, c10::string_view(approximate.data(), approximate.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_logit_backward(const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logit_backward(grad_output, self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_logit_backward_epsilon(const Tensor & grad_output, const Tensor & self, double eps) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logit_backward(grad_output, self, ::std::optional<double>(eps)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::glu_backward(grad_output, self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_masked_select_backward(const Tensor & grad, const Tensor & input, const Tensor & mask) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::masked_select_backward(grad, input, mask));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_gradients_soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::soft_margin_loss_backward(grad_output, self, target, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_bmm(const Tensor & self, const Tensor & mat2) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bmm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, mat2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_complex(const Tensor & real, const Tensor & imag) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::complex(real, imag));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_divide_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_expand_as(const at::Tensor& value, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.expand_as(other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_fill_tensor(const Tensor & self, const Tensor & value) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fill);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, value));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_ger(const Tensor & self, const Tensor & vec2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::ger(self, vec2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_hardsigmoid_backward(const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardsigmoid_backward(grad_output, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_hardswish_backward(const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardswish_backward(grad_output, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_infinitely_differentiable_gelu_backward(const Tensor & grad, const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::infinitely_differentiable_gelu_backward(grad, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_linalg_householder_product(const Tensor & input, const Tensor & tau) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_householder_product(input, tau));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_linalg_matmul(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_matmul(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_matrix_exp_backward(const Tensor & self, const Tensor & grad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::matrix_exp_backward(self, grad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_mish_backward(const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::mish_backward(grad_output, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_mm(const Tensor & self, const Tensor & mat2) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, mat2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_multiply_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::multiply);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_mv(const Tensor & self, const Tensor & vec) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::mv(self, vec));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_orgqr(const Tensor & self, const Tensor & input2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::orgqr(self, input2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_outer(const Tensor & self, const Tensor & vec2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::outer(self, vec2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_polar(const Tensor & abs, const Tensor & angle) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::polar(abs, angle));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_prelu(const Tensor & self, const Tensor & weight) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::prelu(self, weight));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_reshape_as(const at::Tensor& value, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.reshape_as(other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_sigmoid_backward(const Tensor & grad_output, const Tensor & output) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sigmoid_backward(grad_output, output));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_silu_backward(const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::silu_backward(grad_output, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_t(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_u(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_v(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_chebyshev_polynomial_w(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_gammainc(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_gammainc(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_gammaincc(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_gammaincc(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_hermite_polynomial_h(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_hermite_polynomial_he(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_laguerre_polynomial_l(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_legendre_polynomial_p(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_t(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_u(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_v(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_special_shifted_chebyshev_polynomial_w(const Tensor & x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_tanh_backward(const Tensor & grad_output, const Tensor & output) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tanh_backward(grad_output, output));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_type_as(const at::Tensor& value, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.type_as(other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_dense_view_as(const at::Tensor& value, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.view_as(other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_linear_no_bias(const Tensor & input, const Tensor & weight) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linear(input, weight, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_linear_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linear(input, weight, std::optional<at::Tensor>(bias)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_log_softmax_default_dtype(const Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log_softmax(self, dim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_log_softmax_dtype(const Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log_softmax(self, dim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_no_affine(const Tensor & input, int64_t num_groups, double eps, bool cudnn_enabled) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::group_norm(input, num_groups, std::nullopt, std::nullopt, eps, cudnn_enabled));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_weight(const Tensor & input, int64_t num_groups, const at::Tensor& weight, double eps, bool cudnn_enabled) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::group_norm(input, num_groups, std::optional<at::Tensor>(weight), std::nullopt, eps, cudnn_enabled));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_bias(const Tensor & input, int64_t num_groups, const at::Tensor& bias, double eps, bool cudnn_enabled) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::group_norm(input, num_groups, std::nullopt, std::optional<at::Tensor>(bias), eps, cudnn_enabled));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_group_norm_affine(const Tensor & input, int64_t num_groups, const at::Tensor& weight, const at::Tensor& bias, double eps, bool cudnn_enabled) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::group_norm(input, num_groups, std::optional<at::Tensor>(weight), std::optional<at::Tensor>(bias), eps, cudnn_enabled));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_basic(const Tensor & self, int64_t n, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diff(self, n, dim, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_prepend(const Tensor & self, int64_t n, int64_t dim, const at::Tensor& prepend) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diff(self, n, dim, std::optional<at::Tensor>(prepend), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_append(const Tensor & self, int64_t n, int64_t dim, const at::Tensor& append) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diff(self, n, dim, std::nullopt, std::optional<at::Tensor>(append)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_diff_both(const Tensor & self, int64_t n, int64_t dim, const at::Tensor& prepend, const at::Tensor& append) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diff(self, n, dim, std::optional<at::Tensor>(prepend), std::optional<at::Tensor>(append)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_masked_scatter_backward(const Tensor & grad_output, const Tensor & mask, rust::Slice<const std::int64_t> sizes) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::masked_scatter_backward(grad_output, mask, IntArrayRef(sizes.data(), sizes.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_normalize_norm(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const ::std::optional<at::Scalar> &, at::IntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::norm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::optional<c10::Scalar>(c10::Scalar(p)), at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_normalize_clamp_min(const Tensor & self, double min) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_min);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(min)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_normalize_divide(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_local_response_norm_configured(const at::Tensor & input, int64_t size, double alpha, double beta, double k) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_local_response_norm(input, size, alpha, beta, k));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_multilabel_soft_margin_loss_configured(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, bool has_size_average, bool size_average, bool has_reduce, bool reduce, rust::Str reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_multilabel_soft_margin_loss_configured(input, target, has_weight, weight, has_size_average, size_average, has_reduce, reduce, std::string_view(reduction.data(), reduction.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_lp_pool1d_configured(const at::Tensor & input, bool norm_type_is_int, int64_t norm_type_int, double norm_type_float, int64_t kernel_size, bool has_stride, int64_t stride, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_lp_pool1d_configured(input, norm_type_is_int, norm_type_int, norm_type_float, kernel_size, has_stride, stride, ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_max_unpool1d_configured(const at::Tensor & input, const at::Tensor & indices, int64_t kernel_size, bool has_stride, int64_t stride, int64_t padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_max_unpool1d_configured(input, indices, kernel_size, has_stride, stride, padding, has_output_size, c10::ArrayRef<int64_t>(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_lp_pool2d_configured(const at::Tensor & input, bool norm_type_is_int, int64_t norm_type_int, double norm_type_float, int64_t kernel_size, bool has_stride, int64_t stride, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_lp_pool2d_configured(input, norm_type_is_int, norm_type_int, norm_type_float, kernel_size, has_stride, stride, ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_lp_pool3d_configured(const at::Tensor & input, bool norm_type_is_int, int64_t norm_type_int, double norm_type_float, int64_t kernel_size, bool has_stride, int64_t stride, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_lp_pool3d_configured(input, norm_type_is_int, norm_type_int, norm_type_float, kernel_size, has_stride, stride, ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_gumbel_softmax_configured(const at::Tensor & logits, double tau, bool hard, double eps, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_gumbel_softmax_configured(logits, tau, hard, eps, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_gaussian_nll_loss_configured(const at::Tensor & input, const at::Tensor & target, bool var_is_tensor, const at::Tensor & var_tensor, double var_scalar, bool full, double eps, rust::Str reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_gaussian_nll_loss_configured(input, target, var_is_tensor, var_tensor, var_scalar, full, eps, std::string_view(reduction.data(), reduction.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_functionalcompletion_interpolate_configured(const at::Tensor & input, bool has_size, rust::Slice<const std::int64_t> size, bool has_scale, rust::Slice<const double> scale_factor, rust::Str mode, bool has_align_corners, bool align_corners, bool has_recompute, bool recompute_scale_factor, bool antialias) noexcept {
  try {
    return std::make_unique<at::Tensor>(functional_interpolate_configured(input, has_size, c10::ArrayRef<int64_t>(size.data(), size.size()), has_scale, scale_factor, std::string_view(mode.data(), mode.size()), has_align_corners, align_corners, has_recompute, recompute_scale_factor, antialias));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingnative_constant_pad_configured(const Tensor & self, rust::Slice<const std::int64_t> pad, bool value_is_int, std::int64_t integer_value, double floating_value) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::constant_pad_nd(self, IntArrayRef(pad.data(), pad.size()), value_is_int ? c10::Scalar(integer_value) : c10::Scalar(floating_value)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingnative_pad_without_value(const Tensor & self, rust::Slice<const std::int64_t> pad, rust::Str mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pad(self, IntArrayRef(pad.data(), pad.size()), c10::string_view(mode.data(), mode.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_paddingnative_pad_with_value(const Tensor & self, rust::Slice<const std::int64_t> pad, rust::Str mode, double value) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pad(self, IntArrayRef(pad.data(), pad.size()), c10::string_view(mode.data(), mode.size()), ::std::optional<double>(value)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy(const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy(self, target, std::nullopt, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy(self, target, std::optional<at::Tensor>(weight), reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy_backward(grad_output, self, target, std::nullopt, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_backward_weight(const Tensor & grad_output, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy_backward(grad_output, self, target, std::optional<at::Tensor>(weight), reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy_with_logits(self, target, std::nullopt, std::nullopt, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy_with_logits(self, target, std::optional<at::Tensor>(weight), std::nullopt, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_positive_weight(const Tensor & self, const Tensor & target, const at::Tensor& pos_weight, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy_with_logits(self, target, std::nullopt, std::optional<at::Tensor>(pos_weight), reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_both_weights(const Tensor & self, const Tensor & target, const at::Tensor& weight, const at::Tensor& pos_weight, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binary_cross_entropy_with_logits(self, target, std::optional<at::Tensor>(weight), std::optional<at::Tensor>(pos_weight), reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_int_mode(const Tensor & self, std::int64_t other, rust::Str rounding_mode) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_right_float_mode(const Tensor & self, double other, rust::Str rounding_mode) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_div_tensor_mode(const Tensor & self, const Tensor & other, rust::Str rounding_mode) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_int_mode(const Tensor & self, std::int64_t other, rust::Str rounding_mode) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_right_float_mode(const Tensor & self, double other, rust::Str rounding_mode) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_divide_tensor_mode(const Tensor & self, const Tensor & other, rust::Str rounding_mode) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_true_divide_right_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::true_divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_division_true_divide_right_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::true_divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_all(const Tensor & self, int64_t dim, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice(self, dim, std::nullopt, std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_from(const Tensor & self, int64_t dim, std::int64_t start, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice(self, dim, ::std::optional<int64_t>(start), std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_until(const Tensor & self, int64_t dim, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice(self, dim, std::nullopt, ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_bounds(const Tensor & self, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice(self, dim, ::std::optional<int64_t>(start), ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_all(const Tensor & self, int64_t dim, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_copy(self, dim, std::nullopt, std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_from(const Tensor & self, int64_t dim, std::int64_t start, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_copy(self, dim, ::std::optional<int64_t>(start), std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_until(const Tensor & self, int64_t dim, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_copy(self, dim, std::nullopt, ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_copy_bounds(const Tensor & self, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_copy(self, dim, ::std::optional<int64_t>(start), ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_all(const Tensor & self, const Tensor & src, int64_t dim, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_scatter(self, src, dim, std::nullopt, std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_from(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_scatter(self, src, dim, ::std::optional<int64_t>(start), std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_until(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_scatter(self, src, dim, std::nullopt, ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_slicenative_slice_scatter_bounds(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_scatter(self, src, dim, ::std::optional<int64_t>(start), ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fft(const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_fft(self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_hfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_hfft(self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifft(const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ifft(self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ihfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ihfft(self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_irfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_irfft(self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfft(const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_rfft(self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_fft2(self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_hfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_hfft2(self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ifft2(self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ihfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ihfft2(self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_irfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_irfft2(self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfft2(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_rfft2(self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_fftn(self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_hfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_hfftn(self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ifftn(self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ihfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ihfftn(self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_irfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_irfftn(self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfftn(const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_rfftn(self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftfreq(int64_t n, double d) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, double, TensorOptions);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fft_fftfreq);
    return std::make_unique<at::Tensor>(terlan_selected_overload(n, d, {}));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_rfftfreq(int64_t n, double d) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, double, TensorOptions);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fft_rfftfreq);
    return std::make_unique<at::Tensor>(terlan_selected_overload(n, d, {}));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftshift(const Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_fftshift(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_fftshift_all(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_fftshift(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifftshift(const Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ifftshift(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size()))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fft_ifftshift_all(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fft_ifftshift(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_selection_nonzero_static(const at::Tensor & tensor, std::int64_t size, std::int64_t fill_value) noexcept {
  try {
    return std::make_unique<at::Tensor>(nonzero_static_exact(tensor, size, fill_value));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_selection_value_selecting_reduction_backward_exact(const at::Tensor & gradient, std::int64_t dimension, const at::Tensor & indices, rust::Slice<const std::int64_t> sizes, bool keep_dimension) noexcept {
  try {
    return std::make_unique<at::Tensor>(value_selecting_reduction_backward_exact(gradient, dimension, indices, c10::ArrayRef<std::int64_t>(sizes.data(), sizes.size()), keep_dimension));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorlisttransforms_flatten_dense_tensors(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::flatten_dense_tensors(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorlisttransforms_pad_sequence_native(const TerlanPytorchTensorTensorResourceListInput& sequences, bool batch_first, double padding_value, rust::Str padding_side) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pad_sequence(sequences.values(), batch_first, padding_value, c10::string_view(padding_side.data(), padding_side.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_tensorlisttransforms_einsum_native(rust::Str equation, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::einsum(c10::string_view(equation.data(), equation.size()), tensors.values(), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cdist_default_compute(const Tensor & x1, const Tensor & x2, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cdist(x1, x2, p, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cdist_compute(const Tensor & x1, const Tensor & x2, double p, std::int64_t compute_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cdist(x1, x2, p, ::std::optional<int64_t>(compute_mode)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_chain_matmul_native(const TerlanPytorchTensorTensorResourceListInput& matrices) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::chain_matmul(matrices.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_contiguous_native(const at::Tensor& value, std::int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.contiguous(static_cast<c10::MemoryFormat>(memory_format)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cross_default_dimension(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cross(self, other, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_numericalcompletionnative_cross_dimension(const Tensor & self, const Tensor & other, std::int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cross(self, other, ::std::optional<int64_t>(dim)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_bilinear_no_bias(const Tensor & input1, const Tensor & input2, const Tensor & weight) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bilinear(input1, input2, weight, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_bilinear_with_bias(const Tensor & input1, const Tensor & input2, const Tensor & weight, const at::Tensor& bias) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bilinear(input1, input2, weight, std::optional<at::Tensor>(bias)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_tbc(self, weight, bias, pad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_cummaxmin_backward(const Tensor & grad, const Tensor & input, const Tensor & indices, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cummaxmin_backward(grad, input, indices, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_set_empty(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::set);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linearstoragecompletion_set_source_tensor(const Tensor & self, const Tensor & source) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::set);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, source));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_adjoint(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::adjoint(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_alias(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::alias(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_alias_copy(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::alias_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_atleast_1d(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::atleast_1d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_atleast_2d(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::atleast_2d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_atleast_3d(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::atleast_3d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_detach(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::detach(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_detach_copy(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::detach_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_imag(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::imag(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_lift(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::lift(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_lift_fresh(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::lift_fresh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_lift_fresh_copy(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::lift_fresh_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_positive(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::positive(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_ravel(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::ravel(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_real(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::real(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_resolve_conj(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::resolve_conj(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_resolve_neg(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::resolve_neg(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_squeeze(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_squeeze_copy(const Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze_copy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_t(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::t(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_t_copy(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::t_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_complex(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::view_as_complex(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_complex_copy(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::view_as_complex_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_real(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::view_as_real(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_view_as_real_copy(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::view_as_real_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_conj(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.conj());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_data(const at::Tensor& value) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::data);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_mH(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.mH());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_mT(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.mT());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_matrix_H(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.matrix_H());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_view_numpy_T(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.numpy_T());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_convolution(const at::Tensor & input, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool transposed, rust::Slice<const std::int64_t> output_padding, std::int64_t groups) noexcept {
  try {
    return std::make_unique<at::Tensor>(convolution_exact(input, weight, has_bias, bias, c10::ArrayRef<std::int64_t>(stride.data(), stride.size()), c10::ArrayRef<std::int64_t>(padding.data(), padding.size()), c10::ArrayRef<std::int64_t>(dilation.data(), dilation.size()), transposed, c10::ArrayRef<std::int64_t>(output_padding.data(), output_padding.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv1d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv2d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv3d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv1d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv2d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv3d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_padding_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv1d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), c10::string_view(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_padding_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv2d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), c10::string_view(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_padding_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv3d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), c10::string_view(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv1d_padding_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv1d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), c10::string_view(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv2d_padding_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv2d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), c10::string_view(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv3d_padding_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Str padding, rust::Slice<const std::int64_t> dilation, int64_t groups) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const ::std::optional<Tensor> &, IntArrayRef, c10::string_view, IntArrayRef, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::conv3d);
    return std::make_unique<at::Tensor>(terlan_selected_overload(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), c10::string_view(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose1d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_transpose1d(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(output_padding.data(), output_padding.size()), groups, IntArrayRef(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose2d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_transpose2d(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(output_padding.data(), output_padding.size()), groups, IntArrayRef(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose3d_no_bias(const Tensor & input, const Tensor & weight, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_transpose3d(input, weight, std::nullopt, IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(output_padding.data(), output_padding.size()), groups, IntArrayRef(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose1d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_transpose1d(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(output_padding.data(), output_padding.size()), groups, IntArrayRef(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose2d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_transpose2d(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(output_padding.data(), output_padding.size()), groups, IntArrayRef(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_convolutionnative_conv_transpose3d_with_bias(const Tensor & input, const Tensor & weight, const at::Tensor& bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, int64_t groups, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conv_transpose3d(input, weight, std::optional<at::Tensor>(bias), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(output_padding.data(), output_padding.size()), groups, IntArrayRef(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_default(const Tensor & self, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt, std::nullopt, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_default(const Tensor & self, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt, std::nullopt, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_int(const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_int(const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_float(const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_float(const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_dimensions_int(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_dimensions_int(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_std_correction_dimensions_float(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_correctionstatisticscompletion_var_correction_dimensions_float(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_indices_native(const Tensor & repeats) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(repeats, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_indices_output_size_native(const Tensor & repeats, std::int64_t output_size) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(repeats, ::std::optional<int64_t>(output_size)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_flat_native(const Tensor & self, const Tensor & repeats) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_flat_output_size_native(const Tensor & self, const Tensor & repeats, std::int64_t output_size) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, std::nullopt, ::std::optional<int64_t>(output_size)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_dimension_native(const Tensor & self, const Tensor & repeats, std::int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, ::std::optional<int64_t>(dim), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_tensor_dimension_output_size_native(const Tensor & self, const Tensor & repeats, std::int64_t dim, std::int64_t output_size) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, ::std::optional<int64_t>(dim), ::std::optional<int64_t>(output_size)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_flat_native(const Tensor & self, int64_t repeats) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_flat_output_size_native(const Tensor & self, int64_t repeats, std::int64_t output_size) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, std::nullopt, ::std::optional<int64_t>(output_size)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_dimension_native(const Tensor & self, int64_t repeats, std::int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, ::std::optional<int64_t>(dim), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_repeatinterleavenative_repeat_scalar_dimension_output_size_native(const Tensor & self, int64_t repeats, std::int64_t dim, std::int64_t output_size) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, ::std::optional<int64_t>, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::repeat_interleave);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, repeats, ::std::optional<int64_t>(dim), ::std::optional<int64_t>(output_size)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::addbmm(self, batch1, batch2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::addcdiv(self, tensor1, tensor2, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::addcmul(self, tensor1, tensor2, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::addmm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, mat1, mat2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addmm_dtype(const Tensor & self, const Tensor & mat1, const Tensor & mat2, std::int64_t out_dtype, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &, ScalarType, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::addmm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, mat1, mat2, static_cast<c10::ScalarType>(out_dtype), beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::addmv(self, mat, vec, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::addr(self, vec1, vec2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::baddbmm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, batch1, batch2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_fusedarithmeticnative_fused_baddbmm_dtype(const Tensor & self, const Tensor & batch1, const Tensor & batch2, std::int64_t out_dtype, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Tensor &, ScalarType, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::baddbmm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, batch1, batch2, static_cast<c10::ScalarType>(out_dtype), beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_add_tensor_cxx(const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::add);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_sub_tensor_cxx(const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sub);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_rsub_tensor_cxx(const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::rsub);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_add_scalar_cxx(const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::add);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_sub_scalar_cxx(const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sub);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scaledarithmeticnative_scaled_rsub_scalar_cxx(const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::rsub);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_scalar_min(const Tensor & self, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_min);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, bound_is_int ? c10::Scalar(bound_int) : c10::Scalar(bound_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_scalar_max(const Tensor & self, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_max);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, bound_is_int ? c10::Scalar(bound_int) : c10::Scalar(bound_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_scalar_min_max(const Tensor & self, bool min_is_int, std::int64_t min_int, double min_float, bool max_is_int, std::int64_t max_int, double max_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const ::std::optional<Scalar> &, const ::std::optional<Scalar> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::optional<c10::Scalar>(min_is_int ? c10::Scalar(min_int) : c10::Scalar(min_float)), std::optional<c10::Scalar>(max_is_int ? c10::Scalar(max_int) : c10::Scalar(max_float))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_tensor_min(const Tensor & self, const Tensor & min) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_min);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, min));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_tensor_max(const Tensor & self, const Tensor & max) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_max);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, max));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_clampnative_clamp_tensor_min_max(const Tensor & self, const at::Tensor& min, const at::Tensor& max) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const ::std::optional<Tensor> &, const ::std::optional<Tensor> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::optional<at::Tensor>(min), std::optional<at::Tensor>(max)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_bincount_weighted(const Tensor & self, const at::Tensor& weights, int64_t minlength) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bincount(self, std::optional<at::Tensor>(weights), minlength));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_bincount_unweighted(const Tensor & self, int64_t minlength) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bincount(self, std::nullopt, minlength));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_histc_int(const Tensor & self, int64_t bins, std::int64_t min, std::int64_t max) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::histc(self, bins, c10::Scalar(min), c10::Scalar(max)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_histc_float(const Tensor & self, int64_t bins, double min, double max) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::histc(self, bins, c10::Scalar(min), c10::Scalar(max)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_aggregationnative_aggregation_index_reduce(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, rust::Str reduce, bool include_self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::index_reduce(self, dim, index, source, c10::string_view(reduce.data(), reduce.size()), include_self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_bartlett(int64_t window_length, bool periodic, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, bool, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bartlett_window);
    return std::make_unique<at::Tensor>(terlan_selected_overload(window_length, periodic, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<Layout>(static_cast<c10::Layout>(layout)), ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))), ::std::optional<bool>(pin_memory)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_blackman(int64_t window_length, bool periodic, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, bool, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::blackman_window);
    return std::make_unique<at::Tensor>(terlan_selected_overload(window_length, periodic, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<Layout>(static_cast<c10::Layout>(layout)), ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))), ::std::optional<bool>(pin_memory)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_hamming(int64_t window_length, bool periodic, double alpha, double beta, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, bool, double, double, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::hamming_window);
    return std::make_unique<at::Tensor>(terlan_selected_overload(window_length, periodic, alpha, beta, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<Layout>(static_cast<c10::Layout>(layout)), ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))), ::std::optional<bool>(pin_memory)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_hann(int64_t window_length, bool periodic, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, bool, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::hann_window);
    return std::make_unique<at::Tensor>(terlan_selected_overload(window_length, periodic, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<Layout>(static_cast<c10::Layout>(layout)), ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))), ::std::optional<bool>(pin_memory)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_windownative_window_kaiser(int64_t window_length, bool periodic, double beta, std::int64_t dtype, std::int64_t layout, rust::Str device, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, bool, double, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::kaiser_window);
    return std::make_unique<at::Tensor>(terlan_selected_overload(window_length, periodic, beta, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)), ::std::optional<Layout>(static_cast<c10::Layout>(layout)), ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))), ::std::optional<bool>(pin_memory)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_arctan2(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arctan2(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_atan2(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::atan2(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_and_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_and);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_left_shift_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_left_shift);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_or_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_or);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_right_shift_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_right_shift);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_bitwise_xor_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_xor);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_clamp_max_tensor(const Tensor & self, const Tensor & max) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_max);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, max));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_clamp_min_tensor(const Tensor & self, const Tensor & min) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_min);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, min));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_copysign_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::copysign);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_dot(const Tensor & self, const Tensor & tensor) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::dot(self, tensor));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_eq_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eq);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_float_power_tensor(const Tensor & self, const Tensor & exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, exponent));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_floor_divide(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::floor_divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_fmax(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fmax(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_fmin(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fmin(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_fmod_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fmod);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_gcd(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::gcd(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_ge_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ge);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_greater_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::greater);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_greater_equal_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::greater_equal);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_gt_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::gt);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_heaviside(const Tensor & self, const Tensor & values) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::heaviside(self, values));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_hypot(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hypot(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_igamma(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::igamma(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_igammac(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::igammac(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_inner(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::inner(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isclose(self, other, rtol, atol, equal_nan));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_kron(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::kron(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_lcm(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::lcm(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_ldexp_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::ldexp(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_le_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::le);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_less_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::less);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_less_equal_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::less_equal);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logaddexp(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logaddexp(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logaddexp2(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logaddexp2(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logical_and(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logical_and(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logical_or(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logical_or(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_logical_xor(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logical_xor(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_lt_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lt);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_masked_select(const Tensor & self, const Tensor & mask) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::masked_select(self, mask));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_max_other(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::max);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_maximum(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::maximum(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_min_other(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::min);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_ne_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ne);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_nextafter(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nextafter(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_not_equal_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::not_equal);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_pow_tensor(const Tensor & self, const Tensor & exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, exponent));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_remainder_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_special_xlog1py(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_special_xlogy(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_special_zeta(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_take(const Tensor & self, const Tensor & index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::take(self, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_true_divide_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::true_divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_vdot(const Tensor & self, const Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::vdot(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binary_xlogy_tensor(const Tensor & self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_src(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, src));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_legacy_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, src, c10::string_view(reduce.data(), reduce.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_legacy_multiply(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, src, c10::string_view(reduce.data(), reduce.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_add(self, dim, index, src));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_sum(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_reduce(self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_prod(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_reduce(self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_mean(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_reduce(self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_amax(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_reduce(self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_tensor_modern_amin(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::scatter_reduce(self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_int(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_float(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_int_add(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_int_multiply(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_float_add(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scatternative_scatter_scalar_reduce_float_multiply(const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_arange_end(bool end_is_integer, std::int64_t end_integer, double end_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::arange);
    return std::make_unique<at::Tensor>(terlan_selected_overload(end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_arange_start_end(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::arange);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_arange_start_end_step(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool step_is_integer, std::int64_t step_integer, double step_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Scalar &, const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::arange);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), step_is_integer ? c10::Scalar(step_integer) : c10::Scalar(step_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_eye_square(int64_t n, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eye);
    return std::make_unique<at::Tensor>(terlan_selected_overload(n, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_eye_rect(int64_t n, int64_t m, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eye);
    return std::make_unique<at::Tensor>(terlan_selected_overload(n, m, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_scalar_scalar(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Scalar &, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_tensor_tensor(const Tensor & start, const Tensor & end, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start, end, steps, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_tensor_scalar(const Tensor & start, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start, end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_linspace_scalar_tensor(bool start_is_integer, std::int64_t start_integer, double start_floating, const Tensor & end, int64_t steps, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end, steps, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_scalar_scalar(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Scalar &, int64_t, double, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps, base, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_tensor_tensor(const Tensor & start, const Tensor & end, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, int64_t, double, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start, end, steps, base, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_tensor_scalar(const Tensor & start, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, int64_t, double, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start, end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps, base, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_logspace_scalar_tensor(bool start_is_integer, std::int64_t start_integer, double start_floating, const Tensor & end, int64_t steps, double base, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &, int64_t, double, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end, steps, base, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_range_start_end(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::range);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_range_range_start_end_step(bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool step_is_integer, std::int64_t step_integer, double step_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Scalar &, const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::range);
    return std::make_unique<at::Tensor>(terlan_selected_overload(start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), step_is_integer ? c10::Scalar(step_integer) : c10::Scalar(step_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_tril_indices_cxx(int64_t row, int64_t col, int64_t offset, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, int64_t, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::tril_indices);
    return std::make_unique<at::Tensor>(terlan_selected_overload(row, col, offset, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rangenative_triu_indices_cxx(int64_t row, int64_t col, int64_t offset, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, int64_t, int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::triu_indices);
    return std::make_unique<at::Tensor>(terlan_selected_overload(row, col, offset, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_h_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_hermite_polynomial_he_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_laguerre_polynomial_l_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_legendre_polynomial_p_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_right_int(const Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_left_int(std::int64_t x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_right_float(const Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_left_float(double x, const Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_right_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_left_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_right_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlog1py_left_float(double self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_right_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_left_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_right_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_xlogy_left_float(double self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_right_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_left_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_right_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_specialscalarnative_ss_zeta_left_float(double self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_empty(rust::Slice<const std::int64_t> size, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(IntArrayRef, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>, ::std::optional<MemoryFormat>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::empty);
    return std::make_unique<at::Tensor>(terlan_selected_overload(IntArrayRef(size.data(), size.size()), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_full(rust::Slice<const std::int64_t> size, bool fill_value_is_integer, std::int64_t fill_value_integer, double fill_value_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(IntArrayRef, const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::full);
    return std::make_unique<at::Tensor>(terlan_selected_overload(IntArrayRef(size.data(), size.size()), fill_value_is_integer ? c10::Scalar(fill_value_integer) : c10::Scalar(fill_value_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_ones(rust::Slice<const std::int64_t> size, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(IntArrayRef, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ones);
    return std::make_unique<at::Tensor>(terlan_selected_overload(IntArrayRef(size.data(), size.size()), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_zeros(rust::Slice<const std::int64_t> size, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(IntArrayRef, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::zeros);
    return std::make_unique<at::Tensor>(terlan_selected_overload(IntArrayRef(size.data(), size.size()), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_empty_like(const Tensor & self, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>, ::std::optional<MemoryFormat>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::empty_like);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_full_like(const Tensor & self, bool fill_value_is_integer, std::int64_t fill_value_integer, double fill_value_floating, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>, ::std::optional<MemoryFormat>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::full_like);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, fill_value_is_integer ? c10::Scalar(fill_value_integer) : c10::Scalar(fill_value_floating), has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_ones_like(const Tensor & self, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>, ::std::optional<MemoryFormat>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ones_like);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_creationcompletionnative_creation_zeros_like(const Tensor & self, bool has_dtype, std::int64_t dtype, bool has_layout, std::int64_t layout, bool has_device, rust::Str device, bool has_pin_memory, bool pin_memory, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>, ::std::optional<MemoryFormat>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::zeros_like);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, has_dtype ? ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)) : std::nullopt, has_layout ? ::std::optional<Layout>(static_cast<c10::Layout>(layout)) : std::nullopt, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt, has_pin_memory ? ::std::optional<bool>(pin_memory) : std::nullopt, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_and_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_and);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_or_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_or);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_xor_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_xor);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_left_shift_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_left_shift);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_bitwise_right_shift_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_right_shift);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_remainder_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_pow_int(const Tensor & self, std::int64_t exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(exponent)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_float_power_int(const Tensor & self, std::int64_t exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(exponent)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_mul_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mul);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::floor_divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_xlogy_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_copysign_int(const Tensor & self, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::copysign);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_remainder_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_pow_float(const Tensor & self, double exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(exponent)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_float_power_float(const Tensor & self, double exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(exponent)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_mul_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mul);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::floor_divide);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_xlogy_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_fmod_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fmod);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_copysign_float(const Tensor & self, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::copysign);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_and_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_and);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_or_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_or);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_xor_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_xor);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_left_shift_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_left_shift);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_bitwise_right_shift_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_right_shift);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_remainder_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_pow_int(std::int64_t self, const Tensor & exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), exponent));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_float_power_int(std::int64_t self, const Tensor & exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), exponent));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_xlogy_int(std::int64_t self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_remainder_float(double self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_pow_float(double self, const Tensor & exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), exponent));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_float_power_float(double self, const Tensor & exponent) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), exponent));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_left_xlogy_float(double self, const Tensor & other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(self), other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_eq_float(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::eq);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ne_float(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ne);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_lt_float(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::lt);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_le_float(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::le);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_gt_float(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::gt);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ge_float(const at::Tensor& value, double other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ge);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_eq_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::eq);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ne_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ne);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_lt_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::lt);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_le_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::le);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_gt_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::gt);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_binaryscalarnative_binary_scalar_right_ge_int(const at::Tensor& value, std::int64_t other) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ge);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(c10::Scalar(other)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_isin_tensor_scalar(const Tensor & elements, bool test_element_is_integer, std::int64_t test_element_integer, double test_element_floating, bool assume_unique, bool invert) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::isin);
    return std::make_unique<at::Tensor>(terlan_selected_overload(elements, test_element_is_integer ? c10::Scalar(test_element_integer) : c10::Scalar(test_element_floating), assume_unique, invert));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_isin_scalar_tensor(bool element_is_integer, std::int64_t element_integer, double element_floating, const Tensor & test_elements, bool assume_unique, bool invert) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::isin);
    return std::make_unique<at::Tensor>(terlan_selected_overload(element_is_integer ? c10::Scalar(element_integer) : c10::Scalar(element_floating), test_elements, assume_unique, invert));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_searchsorted_tensor(const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, bool has_side, rust::Str side, bool has_sorter, const at::Tensor& sorter) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Tensor &, bool, bool, ::std::optional<c10::string_view>, const ::std::optional<Tensor> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::searchsorted);
    return std::make_unique<at::Tensor>(terlan_selected_overload(sorted_sequence, self, out_int32, right, has_side ? ::std::optional<c10::string_view>(std::basic_string_view<char>(side.data(), side.size())) : std::nullopt, has_sorter ? std::optional<at::Tensor>(sorter) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_searchsorted_scalar(const Tensor & sorted_sequence, bool self_is_integer, std::int64_t self_integer, double self_floating, bool out_int32, bool right, bool has_side, rust::Str side, bool has_sorter, const at::Tensor& sorter) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, const Scalar &, bool, bool, ::std::optional<c10::string_view>, const ::std::optional<Tensor> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::searchsorted);
    return std::make_unique<at::Tensor>(terlan_selected_overload(sorted_sequence, self_is_integer ? c10::Scalar(self_integer) : c10::Scalar(self_floating), out_int32, right, has_side ? ::std::optional<c10::string_view>(std::basic_string_view<char>(side.data(), side.size())) : std::nullopt, has_sorter ? std::optional<at::Tensor>(sorter) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_lookupnative_lookup_bucketize_scalar(bool self_is_integer, std::int64_t self_integer, double self_floating, const Tensor & boundaries, bool out_int32, bool right) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Scalar &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bucketize);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self_is_integer ? c10::Scalar(self_integer) : c10::Scalar(self_floating), boundaries, out_int32, right));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unarystoragenative_round_decimals(const at::Tensor & self, int64_t decimals) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::round);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, decimals));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_abs(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::abs(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_ceil(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::ceil(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_floor(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::floor(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_frac(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::frac(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_reciprocal(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reciprocal(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_round(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::round);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sign(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sign(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_square(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::square(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_trunc(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::trunc(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_acos(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::acos(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_acosh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::acosh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_all(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::all);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_any(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::any);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_argwhere(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::argwhere(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_asin(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::asin(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_asinh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::asinh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_atan(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::atan(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_atanh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::atanh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_bitwise_not(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bitwise_not(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_cos(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cos(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_cosh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cosh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_deg2rad(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::deg2rad(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_erf(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::erf(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_erfc(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::erfc(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_erfinv(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::erfinv(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_exp2(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::exp2(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_expm1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::expm1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_hardsigmoid(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardsigmoid(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_hardswish(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hardswish(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isfinite(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isfinite(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isinf(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isinf(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isnan(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isnan(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isneginf(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isneginf(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_isposinf(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isposinf(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_log10(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log10(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_log1p(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log1p(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_log2(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log2(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_logical_not(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logical_not(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_max(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::max);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_median(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::median);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_min(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::min);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_mish(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::mish(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_nonzero(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nonzero(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_rad2deg(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rad2deg(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_relu6(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::relu6(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_rsqrt(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rsqrt(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_selu(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::selu(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sigmoid(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sigmoid(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_signbit(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::signbit(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_silu(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::silu(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sin(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sin(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sinc(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sinc(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_sinh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sinh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_tan(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tan(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_tanh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tanh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_unary_trace(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::trace(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_channel_shuffle(const at::Tensor & self, int64_t groups) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::channel_shuffle(self, groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_diag(const at::Tensor & self, int64_t diagonal) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diag(self, diagonal));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_diagflat(const at::Tensor & self, int64_t offset) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::diagflat(self, offset));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_glu(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::glu(self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_linalg_matrix_power(const at::Tensor & self, int64_t n) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_matrix_power(self, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_linalg_tensorinv(const at::Tensor & self, int64_t ind) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_tensorinv(self, ind));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_logcumsumexp(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logcumsumexp(self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_matrix_power(const at::Tensor & self, int64_t n) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::matrix_power(self, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_moveaxis_int(const at::Tensor & self, int64_t source, int64_t destination) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::moveaxis);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, source, destination));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_movedim_int(const at::Tensor & self, int64_t source, int64_t destination) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::movedim);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, source, destination));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_mvlgamma(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::mvlgamma(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_native_channel_shuffle(const at::Tensor & self, int64_t groups) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::native_channel_shuffle(self, groups));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_one_hot(const at::Tensor & self, int64_t num_classes) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::one_hot(self, num_classes));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_pixel_shuffle(const at::Tensor & self, int64_t upscale_factor) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pixel_shuffle(self, upscale_factor));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_pixel_unshuffle(const at::Tensor & self, int64_t downscale_factor) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pixel_unshuffle(self, downscale_factor));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_select_int(const at::Tensor & self, int64_t dim, int64_t index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::select(self, dim, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_select_copy_int(const at::Tensor & self, int64_t dim, int64_t index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::select_copy(self, dim, index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_special_multigammaln(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_multigammaln(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_dim(const at::Tensor & self, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_copy_dim(const at::Tensor & self, int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze_copy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_swapaxes(const at::Tensor & self, int64_t axis0, int64_t axis1) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::swapaxes(self, axis0, axis1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_swapdims(const at::Tensor & self, int64_t dim0, int64_t dim1) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::swapdims(self, dim0, dim1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_transpose_copy_int(const at::Tensor & self, int64_t dim0, int64_t dim1) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::transpose_copy(self, dim0, dim1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_tril(const at::Tensor & self, int64_t diagonal) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tril(self, diagonal));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_triu(const at::Tensor & self, int64_t diagonal) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::triu(self, diagonal));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_unsqueeze_copy(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::unsqueeze_copy(self, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_adaptive_avg_pool3d(const at::Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::adaptive_avg_pool3d(self, at::IntArrayRef(output_size.data(), output_size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_broadcast_to(const at::Tensor & self, rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::broadcast_to(self, at::IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_count_nonzero_dim(const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::count_nonzero);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(dim.data(), dim.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_flip(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::flip(self, at::IntArrayRef(dims.data(), dims.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_permute(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::permute(self, at::IntArrayRef(dims.data(), dims.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_permute_copy(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::permute_copy(self, at::IntArrayRef(dims.data(), dims.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reflection_pad1d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reflection_pad1d(self, at::IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reflection_pad2d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reflection_pad2d(self, at::IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reflection_pad3d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reflection_pad3d(self, at::IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_repeat(const at::Tensor& value, rust::Slice<const std::int64_t> repeats) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.repeat(IntArrayRef(repeats.data(), repeats.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_replication_pad1d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::replication_pad1d(self, at::IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_replication_pad2d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::replication_pad2d(self, at::IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_replication_pad3d(const at::Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::replication_pad3d(self, at::IntArrayRef(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_reshape(const at::Tensor & self, rust::Slice<const std::int64_t> shape) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::reshape(self, at::IntArrayRef(shape.data(), shape.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_dims(const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(dim.data(), dim.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_squeeze_copy_dims(const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze_copy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(dim.data(), dim.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_sum_to_size(const at::Tensor& value, rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.sum_to_size(IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_tile(const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::tile(self, at::IntArrayRef(dims.data(), dims.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_view(const at::Tensor& value, rust::Slice<const std::int64_t> size) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (at::Tensor::*)(IntArrayRef) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::view);
    return std::make_unique<at::Tensor>((value.*terlan_selected_overload)(IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_view_copy(const at::Tensor & self, rust::Slice<const std::int64_t> size) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::view_copy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_expand(const at::Tensor& value, rust::Slice<const std::int64_t> size, bool implicit) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.expand(IntArrayRef(size.data(), size.size()), implicit));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_expand_copy(const at::Tensor & self, rust::Slice<const std::int64_t> size, bool implicit) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::expand_copy(self, at::IntArrayRef(size.data(), size.size()), implicit));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_moveaxis(const at::Tensor & self, rust::Slice<const std::int64_t> source, rust::Slice<const std::int64_t> destination) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::moveaxis);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(source.data(), source.size()), at::IntArrayRef(destination.data(), destination.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_movedim(const at::Tensor & self, rust::Slice<const std::int64_t> source, rust::Slice<const std::int64_t> destination) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::movedim);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(source.data(), source.size()), at::IntArrayRef(destination.data(), destination.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_narrow_copy(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::narrow_copy(self, dim, start, length));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_roll(const at::Tensor & self, rust::Slice<const std::int64_t> shifts, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::roll(self, at::IntArrayRef(shifts.data(), shifts.size()), at::IntArrayRef(dims.data(), dims.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_rot90(const at::Tensor & self, int64_t k, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rot90(self, k, at::IntArrayRef(dims.data(), dims.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_shape_unflatten(const at::Tensor & self, int64_t dim, rust::Slice<const std::int64_t> sizes) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::unflatten(self, dim, at::IntArrayRef(sizes.data(), sizes.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_absolute(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::absolute(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_angle(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::angle(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arccos(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arccos(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arccosh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arccosh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arcsin(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arcsin(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arcsinh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arcsinh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arctan(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arctan(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_arctanh(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::arctanh(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_conj_physical(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::conj_physical(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_corrcoef(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::corrcoef(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_det(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::det(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_digamma(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::digamma(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_fix(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fix(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_fliplr(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fliplr(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_flipud(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::flipud(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_i0(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::i0(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_inverse(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::inverse(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_isreal(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::isreal(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_lgamma(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::lgamma(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_linalg_eigvals(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_eigvals(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_linalg_matrix_exp(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_matrix_exp(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_log_sigmoid(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log_sigmoid(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_logdet(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logdet(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_matrix_exp(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::matrix_exp(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_msort(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::msort(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_nanmedian(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanmedian);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_negative(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::negative(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_sgn(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::sgn(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_j0(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_bessel_j0(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_j1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_bessel_j1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_y0(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_bessel_y0(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_bessel_y1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_bessel_y1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_digamma(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_digamma(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_entr(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_entr(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erf(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_erf(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erfc(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_erfc(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erfcx(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_erfcx(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_erfinv(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_erfinv(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_exp2(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_exp2(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_expit(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_expit(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_expm1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_expm1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_gammaln(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_gammaln(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i0(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_i0(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i0e(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_i0e(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_i1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_i1e(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_i1e(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log1p(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_log1p(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log_ndtr(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_log_ndtr(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_i0(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_modified_bessel_i0(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_i1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_modified_bessel_i1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_k0(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_modified_bessel_k0(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_modified_bessel_k1(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_modified_bessel_k1(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_ndtr(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_ndtr(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_ndtri(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_ndtri(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_psi(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_psi(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_sinc(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_sinc(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_airy_ai(const at::Tensor & x) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_airy_ai(x));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_logsumexp(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_logsumexp(self, at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_polygamma(const at::Tensor & self, int64_t n) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_polygamma(n, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_polygamma(const at::Tensor & self, int64_t n) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::polygamma(n, self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_round(const at::Tensor & self, int64_t decimals) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_round(self, decimals));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_scaled_modified_bessel_k0(const at::Tensor & x) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_scaled_modified_bessel_k0(x));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_scaled_modified_bessel_k1(const at::Tensor & x) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_scaled_modified_bessel_k1(x));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_spherical_bessel_j0(const at::Tensor & x) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_spherical_bessel_j0(x));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_logit(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_logit(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_logit_epsilon(const at::Tensor & self, double eps) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_logit(self, ::std::optional<double>(eps)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_logit(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logit(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_logit_epsilon(const at::Tensor & self, double eps) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logit(self, ::std::optional<double>(eps)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_softmax(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_softmax(self, dim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_softmax_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_softmax(self, dim, ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log_softmax(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_log_softmax(self, dim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_log_softmax_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::special_log_softmax(self, dim, ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_order_int(const at::Tensor & x, std::int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const at::Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_order_float(const at::Tensor & x, double n) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const at::Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(x, c10::Scalar(n)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_input_int(std::int64_t x, const at::Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Scalar &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_special_special_chebyshev_polynomial_t_input_float(double x, const at::Tensor & n) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Scalar &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t);
    return std::make_unique<at::Tensor>(terlan_selected_overload(c10::Scalar(x), n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_get_rng_state() noexcept {
  try {
    return std::make_unique<at::Tensor>(get_rng_state_exact());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rand_generator(const Generator & generator, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_rand(generator, c10::ArrayRef<std::int64_t>(size.data(), size.size()), has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randn_generator(const Generator & generator, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randn(generator, c10::ArrayRef<std::int64_t>(size.data(), size.size()), has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randperm_generator(const Generator & generator, int64_t n, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randperm(generator, n, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_rand_like(const Generator & generator, const Tensor & input, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_rand_like(generator, input, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory, has_memory_format, memory_format));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randn_like(const Generator & generator, const Tensor & input, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randn_like(generator, input, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory, has_memory_format, memory_format));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint(const Generator & generator, int64_t high, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randint(generator, high, c10::ArrayRef<std::int64_t>(size.data(), size.size()), has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_low(const Generator & generator, int64_t low, int64_t high, rust::Slice<const std::int64_t> size, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randint_low(generator, low, high, c10::ArrayRef<std::int64_t>(size.data(), size.size()), has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_like(const Generator & generator, const Tensor & input, int64_t high, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randint_like(generator, input, high, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory, has_memory_format, memory_format));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_like_tensor(const Generator & generator, const Tensor & input, const Tensor & high, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randint_like_tensor(generator, input, high, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory, has_memory_format, memory_format));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_generator_randint_like_low(const Generator & generator, const Tensor & input, int64_t low, int64_t high, bool has_dtype, int64_t dtype, bool has_layout, int64_t layout, bool has_device, int64_t device_type, int64_t device_index, bool has_pin_memory, bool pin_memory, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(generator_randint_like_low(generator, input, low, high, has_dtype, dtype, has_layout, layout, has_device, device_type, device_index, has_pin_memory, pin_memory, has_memory_format, memory_format));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rand_default(rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(rand_default(at::IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randn_default(rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(randn_default(at::IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_default(int64_t high, rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(randint_default(high, at::IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_low_default(int64_t low, int64_t high, rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(randint_low_default(low, high, at::IntArrayRef(size.data(), size.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rand_like_default(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(rand_like_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randn_like_default(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(randn_like_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_like_default(const at::Tensor & self, int64_t high) noexcept {
  try {
    return std::make_unique<at::Tensor>(randint_like_default(self, high));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_like_low_dtype_default(const at::Tensor & self, int64_t low, int64_t high) noexcept {
  try {
    return std::make_unique<at::Tensor>(randint_like_low_dtype_default(self, low, high));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_randint_like_tensor_default(const at::Tensor & self, const at::Tensor & high) noexcept {
  try {
    return std::make_unique<at::Tensor>(randint_like_tensor_default(self, high));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_random_default(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(random_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_random_from(const at::Tensor & self, int64_t from, int64_t to) noexcept {
  try {
    return std::make_unique<at::Tensor>(random_from(self, from, to));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_random_to(const at::Tensor & self, int64_t to) noexcept {
  try {
    return std::make_unique<at::Tensor>(random_to(self, to));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rrelu_default(const at::Tensor & self, double lower, double upper, bool training) noexcept {
  try {
    return std::make_unique<at::Tensor>(rrelu_default(self, lower, upper, training));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rrelu_with_noise(const at::Tensor & self, const at::Tensor & noise, bool has_generator, const Generator & generator, double lower, double upper, bool training) noexcept {
  try {
    return std::make_unique<at::Tensor>(rrelu_with_noise_exact(self, noise, has_generator, generator, lower, upper, training));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randomcompletion_rrelu_with_noise_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, double lower, double upper, bool training, bool self_is_result) noexcept {
  try {
    return std::make_unique<at::Tensor>(rrelu_with_noise_backward_exact(grad_output, self, noise, lower, upper, training, self_is_result));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spectralcompletion_stft_default(const at::Tensor & self, int64_t n_fft) noexcept {
  try {
    return std::make_unique<at::Tensor>(stft_default(self, n_fft));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spectralcompletion_stft_center_default(const at::Tensor & self, int64_t n_fft, bool center, rust::Str pad_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(stft_center_default(self, n_fft, center, std::string_view(pad_mode.data(), pad_mode.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_spectralcompletion_istft_default(const at::Tensor & self, int64_t n_fft) noexcept {
  try {
    return std::make_unique<at::Tensor>(istft_default(self, n_fft));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_segmentreducecompletion_segment_reduce_lengths(const at::Tensor & data, rust::Str reduce, const at::Tensor & lengths) noexcept {
  try {
    return std::make_unique<at::Tensor>(segment_reduce_lengths(data, std::string_view(reduce.data(), reduce.size()), lengths));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_attentioncompletion_scaled_dot_product_attention_default(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal) noexcept {
  try {
    return std::make_unique<at::Tensor>(scaled_dot_product_attention_default(query, key, value, dropout_p, is_causal));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedallocationcompletion_empty_quantized(rust::Slice<const std::int64_t> size, const at::Tensor & qtensor) noexcept {
  try {
    return std::make_unique<at::Tensor>(empty_quantized_default(at::IntArrayRef(size.data(), size.size()), qtensor));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_ctclosscompletion_ctc_loss_intlist(const at::Tensor & log_probs, const at::Tensor & targets, rust::Slice<const std::int64_t> input_lengths, rust::Slice<const std::int64_t> target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) noexcept {
  try {
    return std::make_unique<at::Tensor>(ctc_loss_intlist(log_probs, targets, at::IntArrayRef(input_lengths.data(), input_lengths.size()), at::IntArrayRef(target_lengths.data(), target_lengths.size()), blank, reduction, zero_infinity));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_ctclosscompletion_ctc_loss_tensor(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) noexcept {
  try {
    return std::make_unique<at::Tensor>(ctc_loss_tensor(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_covcompletion_cov(const at::Tensor & self, int64_t correction) noexcept {
  try {
    return std::make_unique<at::Tensor>(cov_default(self, correction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cholesky(const at::Tensor & self, bool upper) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cholesky(self, upper));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cholesky_inverse(const at::Tensor & self, bool upper) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cholesky_inverse(self, upper));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_linalg_matrix_rank(const at::Tensor & self, double tol, bool hermitian) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, double, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linalg_matrix_rank);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, tol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_linalg_pinv(const at::Tensor & self, double rcond, bool hermitian) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, double, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linalg_pinv);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, rcond, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nuclear_norm(const at::Tensor & self, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nuclear_norm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_pdist(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pdist(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_pinverse(const at::Tensor & self, double rcond) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::pinverse(self, rcond));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_dist(const at::Tensor & self, const at::Tensor & other, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::dist(self, other, c10::Scalar(p)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm(const at::Tensor & self, double p) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const at::Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::norm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, c10::Scalar(p)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm_dimensions(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const ::std::optional<at::Scalar> &, at::IntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::norm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::optional<c10::Scalar>(c10::Scalar(p)), at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm_except_dimension(const at::Tensor & v, int64_t pow, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::norm_except_dim(v, pow, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_renorm(const at::Tensor & self, double p, int64_t dim, double maxnorm) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::renorm(self, c10::Scalar(p), dim, c10::Scalar(maxnorm)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_std(const at::Tensor & self, bool unbiased) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, unbiased));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_var(const at::Tensor & self, bool unbiased) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, unbiased));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_all_dim(const at::Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::all);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_any_dim(const at::Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::any);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argsort(const at::Tensor & self, int64_t dim, bool descending) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::argsort);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, descending));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_combinations(const at::Tensor & self, int64_t r, bool with_replacement) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::combinations(self, r, with_replacement));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_amin(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::amin(self, at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argmax_all(const Tensor & self, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::argmax(self, std::nullopt, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argmin(const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::argmin(self, ::std::optional<int64_t>(dim), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_argmin_all(const at::Tensor & self, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::argmin(self, std::nullopt, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_count_nonzero(const at::Tensor & self, std::int64_t dim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::count_nonzero);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, ::std::optional<int64_t>(dim)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_count_nonzero_all(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::count_nonzero);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_all_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::OptionalIntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::all);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_any_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::OptionalIntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::any);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_logsumexp(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::logsumexp(self, at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_prod(const at::Tensor & self) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::prod);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_prod_dim(const at::Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, int64_t, bool, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::prod);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, dim, keepdim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumsum(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cumsum(self, dim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumprod(const at::Tensor & self, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cumprod(self, dim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumsum_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cumsum(self, dim, ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_cumprod_dtype(const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cumprod(self, dim, ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_std_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), unbiased, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_var_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), unbiased, keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_norm_dimensions_to_dtype(const at::Tensor & arg0, std::int64_t arg1) noexcept {
  try {
    return std::make_unique<at::Tensor>(tensor_to_dtype(arg0, arg1));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nanmean_native(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nanmean(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nanmean_dimensions_native(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nanmean(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nanmean_dimensions_dtype_native(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t dtype, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nanmean(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nansum_native(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nansum(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nansum_dimensions_native(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nansum(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nansum_dimensions_dtype_native(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t dtype, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nansum(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype))));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_frobenius_norm_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::frobenius_norm(self, at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_statistics_nuclear_norm_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, at::IntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nuclear_norm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, at::IntArrayRef(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_int_pair(const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_float_pair(const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_dimensions_int_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_std_mean_correction_dimensions_float_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_int_pair(const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_float_pair(const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_dimensions_int_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TensorPair> terlan_owned_value_pytorch_correctionstatisticspairnative_var_mean_correction_dimensions_float_pair(const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = TensorPair (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    return std::make_unique<TensorPair>(terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_dimensions(const Tensor & self, int64_t sparse_dim) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::compositeimplicitautograd::to_sparse);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, sparse_dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_dense(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_dense(self, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_coalesce(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::coalesce(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_indices(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(self.indices());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_values(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(self.values());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_crow_indices(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(self.crow_indices());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_col_indices(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(self.col_indices());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_ccol_indices(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(self.ccol_indices());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_row_indices(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(self.row_indices());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csr(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_csr(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csr_configured(const Tensor & self, std::int64_t dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_csr(self, ::std::optional<int64_t>(dense_dim)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csc(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_csc(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_csc_configured(const Tensor & self, std::int64_t dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_csc(self, ::std::optional<int64_t>(dense_dim)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsr(const Tensor & self, rust::Slice<const std::int64_t> blocksize) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_bsr(self, IntArrayRef(blocksize.data(), blocksize.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsr_configured(const Tensor & self, rust::Slice<const std::int64_t> blocksize, std::int64_t dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_bsr(self, IntArrayRef(blocksize.data(), blocksize.size()), ::std::optional<int64_t>(dense_dim)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsc(const Tensor & self, rust::Slice<const std::int64_t> blocksize) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_bsc(self, IntArrayRef(blocksize.data(), blocksize.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_bsc_configured(const Tensor & self, rust::Slice<const std::int64_t> blocksize, std::int64_t dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeimplicitautograd::to_sparse_bsc(self, IntArrayRef(blocksize.data(), blocksize.size()), ::std::optional<int64_t>(dense_dim)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_copy_sparse_to_sparse(const Tensor & self, const Tensor & src, bool non_blocking) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeexplicitautograd::copy_sparse_to_sparse(self, src, non_blocking));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_resize_as_sparse(const Tensor & self, const Tensor & the_template) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeexplicitautograd::resize_as_sparse(self, the_template));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sparse_resize(const Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeexplicitautograd::sparse_resize(self, IntArrayRef(size.data(), size.size()), sparse_dim, dense_dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sparse_resize_and_clear(const Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::compositeexplicitautograd::sparse_resize_and_clear(self, IntArrayRef(size.data(), size.size()), sparse_dim, dense_dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_crow_indices_copy(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_crow_indices_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_col_indices_copy(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_col_indices_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_ccol_indices_copy(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_ccol_indices_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_row_indices_copy(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_row_indices_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_indices_copy(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_indices_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_values_copy(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_values_copy(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sparse_mask(const at::Tensor & self, const at::Tensor & mask) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_mask(self, mask));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_dense_configured(const at::Tensor & self, int64_t dtype, bool has_dtype, bool masked_grad, bool has_masked_grad) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_to_dense_configured(self, dtype, has_dtype, masked_grad, has_masked_grad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_sparse_configured(const at::Tensor & self, int64_t layout, bool has_layout, rust::Slice<const std::int64_t> block_size, bool has_block_size, int64_t dense_dim, bool has_dense_dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_to_sparse_configured(self, layout, has_layout, c10::ArrayRef<int64_t>(block_size.data(), block_size.size()), has_block_size, dense_dim, has_dense_dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_coo_empty_configured(rust::Slice<const std::int64_t> size, int64_t dtype, bool has_dtype, int64_t layout, bool has_layout, int64_t device_type, int64_t device_index, bool has_device, bool pin_memory, bool has_pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_coo_empty_configured(c10::ArrayRef<int64_t>(size.data(), size.size()), dtype, has_dtype, layout, has_layout, device_type, device_index, has_device, pin_memory, has_pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_coo_from_components_configured(const at::Tensor & indices, const at::Tensor & values, rust::Slice<const std::int64_t> size, bool has_size, int64_t dtype, bool has_dtype, int64_t layout, bool has_layout, int64_t device_type, int64_t device_index, bool has_device, bool pin_memory, bool has_pin_memory, bool is_coalesced, bool has_is_coalesced) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_coo_from_components_configured(indices, values, c10::ArrayRef<int64_t>(size.data(), size.size()), has_size, dtype, has_dtype, layout, has_layout, device_type, device_index, has_device, pin_memory, has_pin_memory, is_coalesced, has_is_coalesced));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_compressed_from_components_configured(int64_t kind, const at::Tensor & compressed, const at::Tensor & plain, const at::Tensor & values, rust::Slice<const std::int64_t> size, bool has_size, int64_t dtype, bool has_dtype, int64_t layout, bool has_layout, int64_t device_type, int64_t device_index, bool has_device, bool pin_memory, bool has_pin_memory) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_compressed_from_components_configured(kind, compressed, plain, values, c10::ArrayRef<int64_t>(size.data(), size.size()), has_size, dtype, has_dtype, layout, has_layout, device_type, device_index, has_device, pin_memory, has_pin_memory));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_sampled_addmm_numeric(const at::Tensor & mask, const at::Tensor & left, const at::Tensor & right, double beta, bool beta_is_integer, double alpha, bool alpha_is_integer) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_sampled_addmm_numeric(mask, left, right, beta, beta_is_integer, alpha, alpha_is_integer));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparse_to_dense_backward_configured(const at::Tensor & gradient, const at::Tensor & input, bool masked_grad, bool has_masked_grad) noexcept {
  try {
    return std::make_unique<at::Tensor>(sparse_to_dense_backward_configured(gradient, input, masked_grad, has_masked_grad));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cholesky_solve(self, input2, upper));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cholesky(const Tensor & self, bool upper) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_cholesky(self, upper));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cross(const Tensor & self, const Tensor & other, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_cross(self, other, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_det(const Tensor & A) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_det(A));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_inv(const Tensor & A) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_inv(A));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_ldl_solve(const Tensor & LD, const Tensor & pivots, const Tensor & B, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_ldl_solve(LD, pivots, B, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_lu_solve(const Tensor & LU, const Tensor & pivots, const Tensor & B, bool left, bool adjoint) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_lu_solve(LU, pivots, B, left, adjoint));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tolerance(const at::Tensor & self, const at::Tensor & tol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_tolerance(self, tol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_rcond(const at::Tensor & self, const at::Tensor & rcond, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_rcond(self, rcond, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_solve(const Tensor & A, const Tensor & B, bool left) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_solve(A, B, left));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_solve_triangular(const Tensor & self, const Tensor & B, bool upper, bool left, bool unitriangular) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_solve_triangular(self, B, upper, left, unitriangular));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vecdot(const Tensor & x, const Tensor & y, int64_t dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_vecdot(x, y, dim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::lu_solve(self, LU_data, LU_pivots));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::ormqr(self, input2, input3, left, transpose));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_string(const at::Tensor & self, rust::Str p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_cond_string(self, std::string_view(p.data(), p.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_default(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_cond_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_int(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_cond_int(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_cond_float(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_cond_float(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_int(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_norm_int(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_float(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_norm_float(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_norm_int_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_norm_float_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_default(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_int(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_int(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_float(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_float(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_int_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_float_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_int(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_vector_norm_int(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_float(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_vector_norm_float(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_vector_norm_int_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vector_norm_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_vector_norm_float_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_int(const at::Tensor & self, int64_t p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_powsum_int(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_float(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_powsum_float(self, p));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_int_configured(const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_powsum_int_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_powsum_float_configured(const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_powsum_float_configured(self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_eigvalsh(const Tensor & self, rust::Str UPLO) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_eigvalsh(self, c10::string_view(UPLO.data(), UPLO.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_default(const at::Tensor & self, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_default(self, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tolerances(const at::Tensor & self, double atol, double rtol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_tolerances(self, atol, rtol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_default(const at::Tensor & self, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_tensor_default(self, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_absolute(const at::Tensor & self, const at::Tensor & atol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_tensor_absolute(self, atol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_relative(const at::Tensor & self, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_tensor_relative(self, rtol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_rank_tensor_tolerances(const at::Tensor & self, const at::Tensor & atol, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_rank_tensor_tolerances(self, atol, rtol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_default(const at::Tensor & self, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_default(self, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tolerances(const at::Tensor & self, double atol, double rtol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_tolerances(self, atol, rtol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_default(const at::Tensor & self, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_tensor_default(self, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_absolute(const at::Tensor & self, const at::Tensor & atol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_tensor_absolute(self, atol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_relative(const at::Tensor & self, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_tensor_relative(self, rtol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_pinv_tensor_tolerances(const at::Tensor & self, const at::Tensor & atol, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_pinv_tensor_tolerances(self, atol, rtol, hermitian));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_svdvals(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_svdvals_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vander(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_vander_default(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_vander_columns(const at::Tensor & self, int64_t n) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_vander_columns(self, n));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_string(const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_norm_string(self, std::string_view(p.data(), p.size()), c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_matrix_norm_string_dtype(const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_matrix_norm_string_dtype(self, std::string_view(p.data(), p.size()), c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_string(const at::Tensor & self, rust::Str p, bool keepdim) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_string(self, std::string_view(p.data(), p.size()), keepdim));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_norm_string_dimensions(const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_norm_string_dimensions(self, std::string_view(p.data(), p.size()), c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_tensorsolve(const at::Tensor & self, const at::Tensor & other) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_tensorsolve_default(self, other));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_tensorsolve_dimensions(const at::Tensor & self, const at::Tensor & other, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    return std::make_unique<at::Tensor>(linalg_tensorsolve_dimensions(self, other, c10::ArrayRef<int64_t>(dim.data(), dim.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_multi_dot(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_multi_dot(tensors.values()));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_linalg_linalg_diagonal(const Tensor & A, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::linalg_diagonal(A, offset, dim1, dim2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_losscompletion_multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::multilabel_margin_loss(self, target, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_losscompletion_multilabel_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, std::int64_t reduction, const at::Tensor & is_target) noexcept {
  try {
    return std::make_unique<at::Tensor>(multilabel_margin_loss_backward_exact(grad_output, input, target, reduction, is_target));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_tensor_qparams(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_fake_quantize_per_channel_affine(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_channel(const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantize_per_channel(self, scales, zero_points, axis, static_cast<c10::ScalarType>(dtype)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantize_per_tensor(self, scale, zero_point, static_cast<c10::ScalarType>(dtype)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor_tensor_qparams(const Tensor & self, const Tensor & scale, const Tensor & zero_point, std::int64_t dtype) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantize_per_tensor(self, scale, zero_point, static_cast<c10::ScalarType>(dtype)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_quantize_per_tensor_dynamic(const Tensor & self, std::int64_t dtype, bool reduce_range) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantize_per_tensor_dynamic(self, static_cast<c10::ScalarType>(dtype), reduce_range));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_int_repr(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::int_repr(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_dequantize(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::dequantize(self));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_q_per_channel_scales(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.q_per_channel_scales());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationcompletion_q_per_channel_zero_points(const at::Tensor& value) noexcept {
  try {
    return std::make_unique<at::Tensor>(value.q_per_channel_zero_points());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_exponential(const at::Tensor & self, double lambd) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::exponential(self, lambd, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_poisson(const at::Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::poisson(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_binomial(const at::Tensor & count, const at::Tensor & prob) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::binomial(count, prob, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_cauchy(const at::Tensor & self, double median, double sigma) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::cauchy(self, median, sigma, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_geometric(const at::Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::geometric(self, p, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_log_normal(const at::Tensor & self, double mean, double std) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::log_normal(self, mean, std, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_uniform(const at::Tensor & self, double from, double to) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::uniform(self, from, to, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_functional(const at::Tensor & self, double mean, double std) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::normal_functional(self, mean, std, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_tensor_tensor(const at::Tensor & mean, const at::Tensor & std) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::normal(mean, std, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_tensor_float(const at::Tensor & mean, double std) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::normal(mean, std, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_float_tensor(double mean, const at::Tensor & std) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::normal(mean, std, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_normal_float_float(double mean, double std, rust::Slice<const std::int64_t> size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::normal(mean, std, at::IntArrayRef(size.data(), size.size()), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_bernoulli(const Tensor & self) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bernoulli(self, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_bernoulli_probability(const Tensor & self, double p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bernoulli(self, p, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_bernoulli_tensor(const Tensor & self, const Tensor & p) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::bernoulli(self, p, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_distributioncompletion_multinomial(const Tensor & self, int64_t num_samples, bool replacement) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::multinomial(self, num_samples, replacement, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_hashcompletion_hash_tensor(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::hash_tensor(self, IntArrayRef(dim.data(), dim.size()), keepdim, mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_resizeascompletion_resize_as(const Tensor & self, const Tensor & the_template, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::resize_as(self, the_template, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_all(const Tensor & self, const Tensor & src, int64_t dim, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_inverse(self, src, dim, std::nullopt, std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_from(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_inverse(self, src, dim, ::std::optional<int64_t>(start), std::nullopt, step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_until(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_inverse(self, src, dim, std::nullopt, ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sliceinversecompletion_slice_inverse_bounds(const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::slice_inverse(self, src, dim, ::std::optional<int64_t>(start), ::std::optional<int64_t>(end), step));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_default(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rms_norm(input, IntArrayRef(normalized_shape.data(), normalized_shape.size()), std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_weight(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor& weight) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rms_norm(input, IntArrayRef(normalized_shape.data(), normalized_shape.size()), std::optional<at::Tensor>(weight), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_eps(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, double eps) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rms_norm(input, IntArrayRef(normalized_shape.data(), normalized_shape.size()), std::nullopt, ::std::optional<double>(eps)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_rmsnormcompletion_rms_norm_weight_eps(const Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor& weight, double eps) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::rms_norm(input, IntArrayRef(normalized_shape.data(), normalized_shape.size()), std::optional<at::Tensor>(weight), ::std::optional<double>(eps)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nlllosscompletion_nll_loss_default(const Tensor & self, const Tensor & target, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nll_loss(self, target, std::nullopt, reduction, ignore_index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nlllosscompletion_nll_loss_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nll_loss(self, target, std::optional<at::Tensor>(weight), reduction, ignore_index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_multimarginlosscompletion_multi_margin_loss_default(const Tensor & self, const Tensor & target, double p, double margin, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::multi_margin_loss(self, target, c10::Scalar(p), c10::Scalar(margin), std::nullopt, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_multimarginlosscompletion_multi_margin_loss_weight(const Tensor & self, const Tensor & target, double p, double margin, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::multi_margin_loss(self, target, c10::Scalar(p), c10::Scalar(margin), std::optional<at::Tensor>(weight), reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_multimarginlosscompletion_multi_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, double p, double margin, bool has_weight, const at::Tensor & weight, std::int64_t reduction) noexcept {
  try {
    return std::make_unique<at::Tensor>(multi_margin_loss_backward_exact(grad_output, input, target, p, margin, has_weight, weight, reduction));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebilinear2dcompletion_upsample_bilinear2d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_bilinear2d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebicubic2dcompletion_upsample_bicubic2d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_bicubic2d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nllloss2dcompletion_nll_loss2d_default(const Tensor & self, const Tensor & target, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nll_loss2d(self, target, std::nullopt, reduction, ignore_index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nllloss2dcompletion_nll_loss2d_weight(const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::nll_loss2d(self, target, std::optional<at::Tensor>(weight), reduction, ignore_index));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_viewdtypecompletion_view_dtype(const at::Tensor & self, std::int64_t dtype) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, ScalarType);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::compositeexplicitautograd::view);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, static_cast<c10::ScalarType>(dtype)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_viewcopydtypecompletion_view_copy_dtype(const at::Tensor & self, std::int64_t dtype) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, ScalarType);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::view_copy);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, static_cast<c10::ScalarType>(dtype)));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantilecompletion_quantile_tensor(const at::Tensor & self, const at::Tensor & q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::quantile);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantilecompletion_quantile_scalar(const at::Tensor & self, double q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, double, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::quantile);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nanquantilecompletion_nanquantile_tensor(const at::Tensor & self, const at::Tensor & q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanquantile);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_nanquantilecompletion_nanquantile_scalar(const at::Tensor & self, double q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    using TerlanSelectedOverload = at::Tensor (*)(const at::Tensor &, double, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanquantile);
    return std::make_unique<at::Tensor>(terlan_selected_overload(self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_randpermcompletion_randperm_default(int64_t n) noexcept {
  try {
    using TerlanSelectedOverload = Tensor (*)(int64_t, ::std::optional<ScalarType>, ::std::optional<Layout>, ::std::optional<Device>, ::std::optional<bool>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::randperm);
    return std::make_unique<at::Tensor>(terlan_selected_overload(n, std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_instancenormcompletion_instance_norm_default(const Tensor & input, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::instance_norm(input, std::nullopt, std::nullopt, std::nullopt, std::nullopt, use_input_stats, momentum, eps, cudnn_enabled));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplelinear1dcompletion_upsample_linear1d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_linear1d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplenearest1dcompletion_upsample_nearest1d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest1d(self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplenearest2dcompletion_upsample_nearest2d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest2d(self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplenearest3dcompletion_upsample_nearest3d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest3d(self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_nearest1d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest1d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_nearest2d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest2d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_nearest3d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest3d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_linear1d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_linear1d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_trilinear3d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_trilinear3d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_bilinear2d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_bilinear2d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplebackwardcompletion_upsample_bicubic2d_backward(const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_bicubic2d_backward(grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsampletrilinear3dcompletion_upsample_trilinear3d(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_trilinear3d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparsematmulcompletion_hspmm_default(const at::Tensor & mat1, const at::Tensor & mat2) noexcept {
  try {
    return std::make_unique<at::Tensor>(hspmm_default(mat1, mat2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparsematmulcompletion_smm_default(const at::Tensor & self, const at::Tensor & mat2) noexcept {
  try {
    return std::make_unique<at::Tensor>(smm_default(self, mat2));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_sparsematmulcompletion_sspaddmm_default(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, double beta, double alpha) noexcept {
  try {
    return std::make_unique<at::Tensor>(sspaddmm_default(self, mat1, mat2, beta, alpha));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_bicubic2d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_bicubic2d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_bilinear2d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_bilinear2d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_linear1d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_linear1d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_nearest1d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest1d(self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_nearest2d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest2d(self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_nearest3d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_nearest3d(self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_upsamplevectorcompletion_upsample_trilinear3d_vec(const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::upsample_trilinear3d(self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt, std::nullopt));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::native::HistogramddResult> terlan_owned_value_pytorch_histogramddnative_histogramdd_tensor_bins_result(const at::Tensor & input, const TerlanPytorchTensorTensorResourceListInput& bins, bool has_range, rust::Slice<const double> range, bool has_weight, const at::Tensor & weight, bool density) noexcept {
  try {
    return histogramdd_tensor_bins_result(input, bins.values(), has_range, range, has_weight, weight, density);
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_histogramddnative_histogramdd_result_hist(const HistogramddResult & result) noexcept {
  try {
    return std::make_unique<at::Tensor>(histogramdd_result_hist(result));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_storagenative_storage_set(const at::Tensor & tensor, const StorageHandle & source) noexcept {
  try {
    return std::make_unique<at::Tensor>(storage_set(tensor, source));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_storagenative_storage_set_offset(const at::Tensor & tensor, const StorageHandle & source, std::int64_t storage_offset, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    return std::make_unique<at::Tensor>(storage_set_offset(tensor, source, storage_offset, c10::ArrayRef<std::int64_t>(size.data(), size.size()), c10::ArrayRef<std::int64_t>(stride.data(), stride.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_recurrent_rnn_tanh_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept {
  try {
    return std::make_unique<at::Tensor>(rnn_tanh_cell_exact(input, hx, w_ih, w_hh, has_b_ih, b_ih, has_b_hh, b_hh));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_recurrent_rnn_relu_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept {
  try {
    return std::make_unique<at::Tensor>(rnn_relu_cell_exact(input, hx, w_ih, w_hh, has_b_ih, b_ih, has_b_hh, b_hh));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_recurrent_gru_cell(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept {
  try {
    return std::make_unique<at::Tensor>(gru_cell_exact(input, hx, w_ih, w_hh, has_b_ih, b_ih, has_b_hh, b_hh));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_thnn_conv2d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(thnn_conv2d_exact(input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv3d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    return std::make_unique<at::Tensor>(slow_conv3d_exact(input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_dilated2d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(slow_conv_dilated2d_exact(input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_dilated3d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(slow_conv_dilated3d_exact(input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(slow_conv_transpose2d_exact(input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(output_padding.data(), output_padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_legacyconvolution_slow_conv_transpose3d(const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    return std::make_unique<at::Tensor>(slow_conv_transpose3d_exact(input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(output_padding.data(), output_padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size())));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizationinternals_fused_moving_avg_obs_fake_quant(const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, const at::Tensor & running_min, const at::Tensor & running_max, const at::Tensor & scale, const at::Tensor & zero_point, double averaging_const, std::int64_t quant_min, std::int64_t quant_max, std::int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) noexcept {
  try {
    return std::make_unique<at::Tensor>(fused_moving_avg_obs_fake_quant_exact(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool1d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantized_max_pool1d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantized_max_pool2d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizedpooling_quantized_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    return std::make_unique<at::Tensor>(at::quantized_max_pool3d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_quantizednormalization_quantized_batch_norm(const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, std::int64_t output_zero_point) noexcept {
  try {
    return std::make_unique<at::Tensor>(quantized_batch_norm_exact(input, has_weight, weight, has_bias, bias, mean, var, eps, output_scale, output_zero_point));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<at::Tensor> terlan_owned_value_pytorch_scalartensorcompletion_scalar_tensor(bool scalar_is_integer, int64_t scalar_integer, double scalar_floating) noexcept {
  try {
    return std::make_unique<at::Tensor>(scalar_tensor_exact(scalar_is_integer, scalar_integer, scalar_floating));
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBag> terlan_owned_tuple_pytorch_tensor_embedding_bag(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, bool arg6) noexcept {
  try {
    auto result = at::native::tensor_embedding_bag(arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    return std::make_unique<TerlanOwnedTuplePytorchTensorEmbeddingBag>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))), std::make_unique<at::Tensor>(std::get<3>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept { return value.take_2(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBag& value) noexcept { return value.take_3(); }

std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, const at::Tensor & arg6, bool arg7) noexcept {
  try {
    auto result = at::native::tensor_embedding_bag_weighted(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    return std::make_unique<TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))), std::make_unique<at::Tensor>(std::get<3>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept { return value.take_2(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_weighted_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBagWeighted& value) noexcept { return value.take_3(); }

std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBagPadding> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, bool arg6, std::int64_t arg7) noexcept {
  try {
    auto result = at::native::tensor_embedding_bag_padding(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    return std::make_unique<TerlanOwnedTuplePytorchTensorEmbeddingBagPadding>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))), std::make_unique<at::Tensor>(std::get<3>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept { return value.take_2(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBagPadding& value) noexcept { return value.take_3(); }

std::unique_ptr<TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted(const at::Tensor & arg0, const at::Tensor & arg1, const at::Tensor & arg2, bool arg3, std::int64_t arg4, bool arg5, const at::Tensor & arg6, bool arg7, std::int64_t arg8) noexcept {
  try {
    auto result = at::native::tensor_embedding_bag_padding_weighted(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    return std::make_unique<TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))), std::make_unique<at::Tensor>(std::get<3>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_0(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_1(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_2(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept { return value.take_2(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_tensor_embedding_bag_padding_weighted_take_3(TerlanOwnedTuplePytorchTensorEmbeddingBagPaddingWeighted& value) noexcept { return value.take_3(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeDropout> terlan_owned_tuple_pytorch_regularization_native_dropout(const at::Tensor & input, double p, bool train) noexcept {
  try {
    auto result = at::native::native_dropout_exact(input, p, train);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeDropout>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_dropout_take_0(TerlanOwnedTuplePytorchRegularizationNativeDropout& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_dropout_take_1(TerlanOwnedTuplePytorchRegularizationNativeDropout& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeLayerNorm> terlan_owned_tuple_pytorch_regularization_native_layer_norm(const at::Tensor & input, rust::Slice<const std::int64_t> normalized_shape, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, double eps) noexcept {
  try {
    auto result = at::native::native_layer_norm_exact(input, c10::ArrayRef<std::int64_t>(normalized_shape.data(), normalized_shape.size()), has_weight, weight, has_bias, bias, eps);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeLayerNorm>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_take_0(TerlanOwnedTuplePytorchRegularizationNativeLayerNorm& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_take_1(TerlanOwnedTuplePytorchRegularizationNativeLayerNorm& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_take_2(TerlanOwnedTuplePytorchRegularizationNativeLayerNorm& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeGroupNorm> terlan_owned_tuple_pytorch_regularization_native_group_norm(const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, std::int64_t n, std::int64_t c, std::int64_t hxw, std::int64_t group, double eps) noexcept {
  try {
    auto result = at::native::native_group_norm_exact(input, has_weight, weight, has_bias, bias, n, c, hxw, group, eps);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeGroupNorm>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_take_0(TerlanOwnedTuplePytorchRegularizationNativeGroupNorm& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_take_1(TerlanOwnedTuplePytorchRegularizationNativeGroupNorm& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_take_2(TerlanOwnedTuplePytorchRegularizationNativeGroupNorm& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, bool has_weight, const at::Tensor & weight, std::int64_t n, std::int64_t c, std::int64_t hxw, std::int64_t group, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    auto result = at::native::native_group_norm_backward_exact(grad_out, input, mean, rstd, has_weight, weight, n, c, hxw, group, output_mask_input, output_mask_weight, output_mask_bias);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward_take_0(TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward_take_1(TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_group_norm_backward_take_2(TerlanOwnedTuplePytorchRegularizationNativeGroupNormBackward& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    auto result = at::native::native_layer_norm_backward_exact(grad_out, input, c10::ArrayRef<std::int64_t>(normalized_shape.data(), normalized_shape.size()), mean, rstd, has_weight, weight, has_bias, bias, output_mask_input, output_mask_weight, output_mask_bias);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward_take_0(TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward_take_1(TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_layer_norm_backward_take_2(TerlanOwnedTuplePytorchRegularizationNativeLayerNormBackward& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeBatchNorm> terlan_owned_tuple_pytorch_regularization_native_batch_norm(const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, bool training, double momentum, double eps) noexcept {
  try {
    auto result = at::native::native_batch_norm_exact(input, has_weight, weight, has_bias, bias, has_running_mean, running_mean, has_running_var, running_var, training, momentum, eps);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeBatchNorm>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_take_0(TerlanOwnedTuplePytorchRegularizationNativeBatchNorm& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_take_1(TerlanOwnedTuplePytorchRegularizationNativeBatchNorm& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_take_2(TerlanOwnedTuplePytorchRegularizationNativeBatchNorm& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, bool has_save_mean, const at::Tensor & save_mean, bool has_save_invstd, const at::Tensor & save_invstd, bool train, double eps, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    auto result = at::native::native_batch_norm_backward_exact(grad_out, input, has_weight, weight, has_running_mean, running_mean, has_running_var, running_var, has_save_mean, save_mean, has_save_invstd, save_invstd, train, eps, output_mask_input, output_mask_weight, output_mask_bias);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward_take_0(TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward_take_1(TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_native_batch_norm_backward_take_2(TerlanOwnedTuplePytorchRegularizationNativeBatchNormBackward& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationLogSigmoidForward> terlan_owned_tuple_pytorch_regularization_log_sigmoid_forward(const at::Tensor & input) noexcept {
  try {
    auto result = at::native::log_sigmoid_forward_exact(input);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationLogSigmoidForward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_log_sigmoid_forward_take_0(TerlanOwnedTuplePytorchRegularizationLogSigmoidForward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_log_sigmoid_forward_take_1(TerlanOwnedTuplePytorchRegularizationLogSigmoidForward& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats> terlan_owned_tuple_pytorch_regularization_batch_norm_update_stats(const at::Tensor & input, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, double momentum) noexcept {
  try {
    auto result = at::native::batch_norm_update_stats_exact(input, has_running_mean, running_mean, has_running_var, running_var, momentum);
    return std::make_unique<TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_batch_norm_update_stats_take_0(TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_regularization_batch_norm_update_stats_take_1(TerlanOwnedTuplePytorchRegularizationBatchNormUpdateStats& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool1d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    auto result = at::adaptive_max_pool1d(self, IntArrayRef(output_size.data(), output_size.size()));
    return std::make_unique<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool1d_take_0(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool1d_take_1(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool1d& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    auto result = at::adaptive_max_pool2d(self, IntArrayRef(output_size.data(), output_size.size()));
    return std::make_unique<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool2d_take_0(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool2d_take_1(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool2d& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    auto result = at::adaptive_max_pool3d(self, IntArrayRef(output_size.data(), output_size.size()));
    return std::make_unique<TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool3d_take_0(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_adaptive_max_pool3d_take_1(TerlanOwnedTuplePytorchPoolingAdaptiveMaxPool3d& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices> terlan_owned_tuple_pytorch_pooling_max_pool1d_with_indices(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    auto result = at::max_pool1d_with_indices(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
    return std::make_unique<TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool1d_with_indices_take_0(TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool1d_with_indices_take_1(TerlanOwnedTuplePytorchPoolingMaxPool1dWithIndices& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices> terlan_owned_tuple_pytorch_pooling_max_pool2d_with_indices(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    auto result = at::max_pool2d_with_indices(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
    return std::make_unique<TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool2d_with_indices_take_0(TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool2d_with_indices_take_1(TerlanOwnedTuplePytorchPoolingMaxPool2dWithIndices& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices> terlan_owned_tuple_pytorch_pooling_max_pool3d_with_indices(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    auto result = at::max_pool3d_with_indices(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
    return std::make_unique<TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool3d_with_indices_take_0(TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_max_pool3d_with_indices_take_1(TerlanOwnedTuplePytorchPoolingMaxPool3dWithIndices& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d> terlan_owned_tuple_pytorch_pooling_fractional_max_pool2d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & random_samples) noexcept {
  try {
    auto result = at::fractional_max_pool2d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), random_samples);
    return std::make_unique<TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool2d_take_0(TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool2d_take_1(TerlanOwnedTuplePytorchPoolingFractionalMaxPool2d& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d> terlan_owned_tuple_pytorch_pooling_fractional_max_pool3d(const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & random_samples) noexcept {
  try {
    auto result = at::fractional_max_pool3d(self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), random_samples);
    return std::make_unique<TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool3d_take_0(TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_pooling_fractional_max_pool3d_take_1(TerlanOwnedTuplePytorchPoolingFractionalMaxPool3d& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSpatialGridSampler2dBackward> terlan_owned_tuple_pytorch_spatial_grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, std::int64_t interpolation_mode, std::int64_t padding_mode, bool align_corners, bool output_mask_input, bool output_mask_grid) noexcept {
  try {
    auto result = at::native::grid_sampler_2d_backward_exact(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask_input, output_mask_grid);
    return std::make_unique<TerlanOwnedTuplePytorchSpatialGridSampler2dBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_2d_backward_take_0(TerlanOwnedTuplePytorchSpatialGridSampler2dBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_2d_backward_take_1(TerlanOwnedTuplePytorchSpatialGridSampler2dBackward& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSpatialGridSampler3dBackward> terlan_owned_tuple_pytorch_spatial_grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, std::int64_t interpolation_mode, std::int64_t padding_mode, bool align_corners, bool output_mask_input, bool output_mask_grid) noexcept {
  try {
    auto result = at::native::grid_sampler_3d_backward_exact(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask_input, output_mask_grid);
    return std::make_unique<TerlanOwnedTuplePytorchSpatialGridSampler3dBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_3d_backward_take_0(TerlanOwnedTuplePytorchSpatialGridSampler3dBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_spatial_grid_sampler_3d_backward_take_1(TerlanOwnedTuplePytorchSpatialGridSampler3dBackward& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLossNllLossForward> terlan_owned_tuple_pytorch_loss_nll_loss_forward(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept {
  try {
    auto result = at::native::nll_loss_forward_exact(input, target, has_weight, weight, reduction, ignore_index);
    return std::make_unique<TerlanOwnedTuplePytorchLossNllLossForward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss_forward_take_0(TerlanOwnedTuplePytorchLossNllLossForward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss_forward_take_1(TerlanOwnedTuplePytorchLossNllLossForward& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLossNllLoss2dForward> terlan_owned_tuple_pytorch_loss_nll_loss2d_forward(const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept {
  try {
    auto result = at::native::nll_loss2d_forward_exact(input, target, has_weight, weight, reduction, ignore_index);
    return std::make_unique<TerlanOwnedTuplePytorchLossNllLoss2dForward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss2d_forward_take_0(TerlanOwnedTuplePytorchLossNllLoss2dForward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_loss_nll_loss2d_forward_take_1(TerlanOwnedTuplePytorchLossNllLoss2dForward& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionCummax> terlan_owned_tuple_pytorch_selection_cummax(const Tensor & self, int64_t dim) noexcept {
  try {
    auto result = at::cummax(self, dim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionCummax>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummax_take_0(TerlanOwnedTuplePytorchSelectionCummax& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummax_take_1(TerlanOwnedTuplePytorchSelectionCummax& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionCummin> terlan_owned_tuple_pytorch_selection_cummin(const Tensor & self, int64_t dim) noexcept {
  try {
    auto result = at::cummin(self, dim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionCummin>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummin_take_0(TerlanOwnedTuplePytorchSelectionCummin& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_cummin_take_1(TerlanOwnedTuplePytorchSelectionCummin& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionKthvalue> terlan_owned_tuple_pytorch_selection_kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) noexcept {
  try {
    auto result = at::kthvalue(self, k, dim, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionKthvalue>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_kthvalue_take_0(TerlanOwnedTuplePytorchSelectionKthvalue& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_kthvalue_take_1(TerlanOwnedTuplePytorchSelectionKthvalue& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionMaxDim> terlan_owned_tuple_pytorch_selection_max_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::max);
    auto result = terlan_selected_overload(self, dim, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionMaxDim>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_max_dim_take_0(TerlanOwnedTuplePytorchSelectionMaxDim& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_max_dim_take_1(TerlanOwnedTuplePytorchSelectionMaxDim& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionMedianDim> terlan_owned_tuple_pytorch_selection_median_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::median);
    auto result = terlan_selected_overload(self, dim, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionMedianDim>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_median_dim_take_0(TerlanOwnedTuplePytorchSelectionMedianDim& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_median_dim_take_1(TerlanOwnedTuplePytorchSelectionMedianDim& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionMinDim> terlan_owned_tuple_pytorch_selection_min_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::min);
    auto result = terlan_selected_overload(self, dim, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionMinDim>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_min_dim_take_0(TerlanOwnedTuplePytorchSelectionMinDim& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_min_dim_take_1(TerlanOwnedTuplePytorchSelectionMinDim& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionMode> terlan_owned_tuple_pytorch_selection_mode(const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    auto result = at::mode(self, dim, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionMode>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_mode_take_0(TerlanOwnedTuplePytorchSelectionMode& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_mode_take_1(TerlanOwnedTuplePytorchSelectionMode& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionNanmedianDim> terlan_owned_tuple_pytorch_selection_nanmedian_dim(const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanmedian);
    auto result = terlan_selected_overload(self, dim, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionNanmedianDim>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_nanmedian_dim_take_0(TerlanOwnedTuplePytorchSelectionNanmedianDim& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_nanmedian_dim_take_1(TerlanOwnedTuplePytorchSelectionNanmedianDim& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionSort> terlan_owned_tuple_pytorch_selection_sort(const Tensor & self, int64_t dim, bool descending) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sort);
    auto result = terlan_selected_overload(self, dim, descending);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionSort>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_sort_take_0(TerlanOwnedTuplePytorchSelectionSort& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_sort_take_1(TerlanOwnedTuplePytorchSelectionSort& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchSelectionTopk> terlan_owned_tuple_pytorch_selection_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) noexcept {
  try {
    auto result = at::topk(self, k, dim, largest, sorted);
    return std::make_unique<TerlanOwnedTuplePytorchSelectionTopk>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_topk_take_0(TerlanOwnedTuplePytorchSelectionTopk& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_selection_topk_take_1(TerlanOwnedTuplePytorchSelectionTopk& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) noexcept {
  try {
    auto result = at::conv_tbc_backward(self, input, weight, bias, pad);
    return std::make_unique<TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward_take_0(TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward_take_1(TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linearstoragecompletion_conv_tbc_backward_take_2(TerlanOwnedTuplePytorchLinearStorageCompletionConvTbcBackward& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool has_bias_sizes, rust::Slice<const std::int64_t> bias_sizes, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool transposed, rust::Slice<const std::int64_t> output_padding, std::int64_t groups, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    auto result = at::native::convolution_backward_exact(grad_output, input, weight, has_bias_sizes, c10::ArrayRef<std::int64_t>(bias_sizes.data(), bias_sizes.size()), c10::ArrayRef<std::int64_t>(stride.data(), stride.size()), c10::ArrayRef<std::int64_t>(padding.data(), padding.size()), c10::ArrayRef<std::int64_t>(dilation.data(), dilation.size()), transposed, c10::ArrayRef<std::int64_t>(output_padding.data(), output_padding.size()), groups, output_mask_input, output_mask_weight, output_mask_bias);
    return std::make_unique<TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward_take_0(TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward_take_1(TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_convolutionnative_convolution_backward_take_2(TerlanOwnedTuplePytorchConvolutionNativeConvolutionBackward& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_default(const Tensor & self, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    auto result = terlan_selected_overload(self, std::nullopt, std::nullopt, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_default_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_default_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDefault& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_default(const Tensor & self, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    auto result = terlan_selected_overload(self, std::nullopt, std::nullopt, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_default_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_default_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDefault& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    auto result = terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_dimensions_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_std_mean_correction_dimensions_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionStdMeanCorrectionDimensions& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_dimensions(const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<Tensor, Tensor> (*)(const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    auto result = terlan_selected_overload(self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_dimensions_take_0(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_correctionstatisticscompletion_var_mean_correction_dimensions_take_1(TerlanOwnedTuplePytorchCorrectionStatisticsCompletionVarMeanCorrectionDimensions& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchUnaryFrexp> terlan_owned_tuple_pytorch_unary_frexp(const at::Tensor & self) noexcept {
  try {
    auto result = at::frexp(self);
    return std::make_unique<TerlanOwnedTuplePytorchUnaryFrexp>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_unary_frexp_take_0(TerlanOwnedTuplePytorchUnaryFrexp& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_unary_frexp_take_1(TerlanOwnedTuplePytorchUnaryFrexp& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional> terlan_owned_tuple_pytorch_randomcompletion_rrelu_with_noise_functional(const at::Tensor & self, const at::Tensor & noise, bool has_generator, const Generator & generator, double lower, double upper, bool training) noexcept {
  try {
    auto result = at::native::rrelu_with_noise_functional_exact(self, noise, has_generator, generator, lower, upper, training);
    return std::make_unique<TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_randomcompletion_rrelu_with_noise_functional_take_0(TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_randomcompletion_rrelu_with_noise_functional_take_1(TerlanOwnedTuplePytorchRandomCompletionRreluWithNoiseFunctional& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct(const at::Tensor & self, int64_t bins, bool has_range, rust::Slice<const double> range, bool has_weight, const at::Tensor & weight, bool density) noexcept {
  try {
    auto result = at::native::histogram_bin_ct_configured(self, bins, has_range, range, has_weight, weight, density);
    return std::make_unique<TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct_take_0(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bin_ct_take_1(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinCt& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor(const at::Tensor & self, const at::Tensor & bins, bool has_weight, const at::Tensor & weight, bool density) noexcept {
  try {
    auto result = at::native::histogram_bins_tensor_configured(self, bins, has_weight, weight, density);
    return std::make_unique<TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor_take_0(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_histogramcompletion_histogram_bins_tensor_take_1(TerlanOwnedTuplePytorchHistogramCompletionHistogramBinsTensor& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive(const at::Tensor & self, bool return_inverse, bool return_counts) noexcept {
  try {
    auto result = at::native::unique_consecutive_default(self, return_inverse, return_counts, std::nullopt);
    return std::make_unique<TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive_take_0(TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive_take_1(TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_consecutive_take_2(TerlanOwnedTuplePytorchUniqueCompletionUniqueConsecutive& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchUniqueCompletionUniqueDim> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) noexcept {
  try {
    auto result = at::native::unique_dim_default(self, dim, sorted, return_inverse, return_counts);
    return std::make_unique<TerlanOwnedTuplePytorchUniqueCompletionUniqueDim>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_take_0(TerlanOwnedTuplePytorchUniqueCompletionUniqueDim& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_take_1(TerlanOwnedTuplePytorchUniqueCompletionUniqueDim& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_take_2(TerlanOwnedTuplePytorchUniqueCompletionUniqueDim& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive(const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) noexcept {
  try {
    auto result = at::native::unique_dim_consecutive_default(self, dim, return_inverse, return_counts);
    return std::make_unique<TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive_take_0(TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive_take_1(TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_uniquecompletion_unique_dim_consecutive_take_2(TerlanOwnedTuplePytorchUniqueCompletionUniqueDimConsecutive& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchSvdCompletionSvdDefault> terlan_owned_tuple_pytorch_svdcompletion_svd_default(const at::Tensor & self, bool some, bool compute_uv) noexcept {
  try {
    auto result = at::native::svd_default(self, some, compute_uv);
    return std::make_unique<TerlanOwnedTuplePytorchSvdCompletionSvdDefault>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_svdcompletion_svd_default_take_0(TerlanOwnedTuplePytorchSvdCompletionSvdDefault& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_svdcompletion_svd_default_take_1(TerlanOwnedTuplePytorchSvdCompletionSvdDefault& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_svdcompletion_svd_default_take_2(TerlanOwnedTuplePytorchSvdCompletionSvdDefault& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchStatisticsAminmax> terlan_owned_tuple_pytorch_statistics_aminmax(const at::Tensor & self, bool keepdim) noexcept {
  try {
    auto result = at::aminmax(self, std::nullopt, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchStatisticsAminmax>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_take_0(TerlanOwnedTuplePytorchStatisticsAminmax& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_take_1(TerlanOwnedTuplePytorchStatisticsAminmax& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchStatisticsAminmaxDim> terlan_owned_tuple_pytorch_statistics_aminmax_dim(const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept {
  try {
    auto result = at::aminmax(self, ::std::optional<int64_t>(dim), keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchStatisticsAminmaxDim>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_dim_take_0(TerlanOwnedTuplePytorchStatisticsAminmaxDim& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_aminmax_dim_take_1(TerlanOwnedTuplePytorchStatisticsAminmaxDim& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchStatisticsStdMean> terlan_owned_tuple_pytorch_statistics_std_mean(const at::Tensor & self, bool unbiased) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    auto result = terlan_selected_overload(self, unbiased);
    return std::make_unique<TerlanOwnedTuplePytorchStatisticsStdMean>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_take_0(TerlanOwnedTuplePytorchStatisticsStdMean& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_take_1(TerlanOwnedTuplePytorchStatisticsStdMean& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchStatisticsVarMean> terlan_owned_tuple_pytorch_statistics_var_mean(const at::Tensor & self, bool unbiased) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    auto result = terlan_selected_overload(self, unbiased);
    return std::make_unique<TerlanOwnedTuplePytorchStatisticsVarMean>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_take_0(TerlanOwnedTuplePytorchStatisticsVarMean& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_take_1(TerlanOwnedTuplePytorchStatisticsVarMean& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchStatisticsStdMeanDimensions> terlan_owned_tuple_pytorch_statistics_std_mean_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_mean);
    auto result = terlan_selected_overload(self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), unbiased, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchStatisticsStdMeanDimensions>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_dimensions_take_0(TerlanOwnedTuplePytorchStatisticsStdMeanDimensions& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_std_mean_dimensions_take_1(TerlanOwnedTuplePytorchStatisticsStdMeanDimensions& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchStatisticsVarMeanDimensions> terlan_owned_tuple_pytorch_statistics_var_mean_dimensions(const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept {
  try {
    using TerlanSelectedOverload = ::std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_mean);
    auto result = terlan_selected_overload(self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), unbiased, keepdim);
    return std::make_unique<TerlanOwnedTuplePytorchStatisticsVarMeanDimensions>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_dimensions_take_0(TerlanOwnedTuplePytorchStatisticsVarMeanDimensions& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_statistics_var_mean_dimensions_take_1(TerlanOwnedTuplePytorchStatisticsVarMeanDimensions& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgGeqrf> terlan_owned_tuple_pytorch_linalg_geqrf(const at::Tensor & self) noexcept {
  try {
    auto result = at::native::terlan_geqrf(self);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgGeqrf>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_geqrf_take_0(TerlanOwnedTuplePytorchLinalgGeqrf& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_geqrf_take_1(TerlanOwnedTuplePytorchLinalgGeqrf& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx> terlan_owned_tuple_pytorch_linalg_linalg_cholesky_ex(const Tensor & self, bool upper, bool check_errors) noexcept {
  try {
    auto result = at::linalg_cholesky_ex(self, upper, check_errors);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_cholesky_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_cholesky_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgCholeskyEx& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgEig> terlan_owned_tuple_pytorch_linalg_linalg_eig(const Tensor & self) noexcept {
  try {
    auto result = at::linalg_eig(self);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgEig>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eig_take_0(TerlanOwnedTuplePytorchLinalgLinalgEig& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eig_take_1(TerlanOwnedTuplePytorchLinalgLinalgEig& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgEigh> terlan_owned_tuple_pytorch_linalg_linalg_eigh(const Tensor & self, rust::Str UPLO) noexcept {
  try {
    auto result = at::linalg_eigh(self, c10::string_view(UPLO.data(), UPLO.size()));
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgEigh>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eigh_take_0(TerlanOwnedTuplePytorchLinalgLinalgEigh& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_eigh_take_1(TerlanOwnedTuplePytorchLinalgLinalgEigh& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgInvEx> terlan_owned_tuple_pytorch_linalg_linalg_inv_ex(const Tensor & A, bool check_errors) noexcept {
  try {
    auto result = at::linalg_inv_ex(A, check_errors);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgInvEx>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_inv_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgInvEx& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_inv_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgInvEx& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLdlFactor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor(const Tensor & self, bool hermitian) noexcept {
  try {
    auto result = at::linalg_ldl_factor(self, hermitian);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLdlFactor>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_take_0(TerlanOwnedTuplePytorchLinalgLinalgLdlFactor& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_take_1(TerlanOwnedTuplePytorchLinalgLinalgLdlFactor& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex(const Tensor & self, bool hermitian, bool check_errors) noexcept {
  try {
    auto result = at::linalg_ldl_factor_ex(self, hermitian, check_errors);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_ldl_factor_ex_take_2(TerlanOwnedTuplePytorchLinalgLinalgLdlFactorEx& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLu> terlan_owned_tuple_pytorch_linalg_linalg_lu(const Tensor & A, bool pivot) noexcept {
  try {
    auto result = at::linalg_lu(A, pivot);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLu>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_take_0(TerlanOwnedTuplePytorchLinalgLinalgLu& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_take_1(TerlanOwnedTuplePytorchLinalgLinalgLu& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_take_2(TerlanOwnedTuplePytorchLinalgLinalgLu& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLuFactor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor(const Tensor & A, bool pivot) noexcept {
  try {
    auto result = at::linalg_lu_factor(A, pivot);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLuFactor>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_take_0(TerlanOwnedTuplePytorchLinalgLinalgLuFactor& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_take_1(TerlanOwnedTuplePytorchLinalgLinalgLuFactor& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex(const Tensor & A, bool pivot, bool check_errors) noexcept {
  try {
    auto result = at::linalg_lu_factor_ex(A, pivot, check_errors);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lu_factor_ex_take_2(TerlanOwnedTuplePytorchLinalgLinalgLuFactorEx& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgQr> terlan_owned_tuple_pytorch_linalg_linalg_qr(const Tensor & A, rust::Str mode) noexcept {
  try {
    auto result = at::linalg_qr(A, c10::string_view(mode.data(), mode.size()));
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgQr>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_qr_take_0(TerlanOwnedTuplePytorchLinalgLinalgQr& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_qr_take_1(TerlanOwnedTuplePytorchLinalgLinalgQr& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgSlogdet> terlan_owned_tuple_pytorch_linalg_linalg_slogdet(const Tensor & A) noexcept {
  try {
    auto result = at::linalg_slogdet(A);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgSlogdet>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_slogdet_take_0(TerlanOwnedTuplePytorchLinalgLinalgSlogdet& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_slogdet_take_1(TerlanOwnedTuplePytorchLinalgLinalgSlogdet& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgSolveEx> terlan_owned_tuple_pytorch_linalg_linalg_solve_ex(const Tensor & A, const Tensor & B, bool left, bool check_errors) noexcept {
  try {
    auto result = at::linalg_solve_ex(A, B, left, check_errors);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgSolveEx>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_solve_ex_take_0(TerlanOwnedTuplePytorchLinalgLinalgSolveEx& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_solve_ex_take_1(TerlanOwnedTuplePytorchLinalgLinalgSolveEx& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgQr> terlan_owned_tuple_pytorch_linalg_qr(const Tensor & self, bool some) noexcept {
  try {
    auto result = at::qr(self, some);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgQr>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_qr_take_0(TerlanOwnedTuplePytorchLinalgQr& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_qr_take_1(TerlanOwnedTuplePytorchLinalgQr& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgSlogdet> terlan_owned_tuple_pytorch_linalg_slogdet(const Tensor & self) noexcept {
  try {
    auto result = at::slogdet(self);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgSlogdet>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_slogdet_take_0(TerlanOwnedTuplePytorchLinalgSlogdet& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_slogdet_take_1(TerlanOwnedTuplePytorchLinalgSlogdet& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgTriangularSolve> terlan_owned_tuple_pytorch_linalg_triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) noexcept {
  try {
    auto result = at::triangular_solve(self, A, upper, transpose, unitriangular);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgTriangularSolve>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_triangular_solve_take_0(TerlanOwnedTuplePytorchLinalgTriangularSolve& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_triangular_solve_take_1(TerlanOwnedTuplePytorchLinalgTriangularSolve& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLstsq> terlan_owned_tuple_pytorch_linalg_linalg_lstsq(const at::Tensor & a, const at::Tensor & b) noexcept {
  try {
    auto result = at::native::linalg_lstsq_default(a, b);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLstsq>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))), std::make_unique<at::Tensor>(std::get<3>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_0(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_1(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_2(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept { return value.take_2(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_take_3(TerlanOwnedTuplePytorchLinalgLinalgLstsq& value) noexcept { return value.take_3(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured(const at::Tensor & a, const at::Tensor & b, double rcond, rust::Str driver) noexcept {
  try {
    auto result = at::native::linalg_lstsq_configured(a, b, rcond, std::string_view(driver.data(), driver.size()));
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))), std::make_unique<at::Tensor>(std::get<3>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_0(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_1(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_2(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept { return value.take_2(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_lstsq_configured_take_3(TerlanOwnedTuplePytorchLinalgLinalgLstsqConfigured& value) noexcept { return value.take_3(); }

std::unique_ptr<TerlanOwnedTuplePytorchLinalgLinalgSvd> terlan_owned_tuple_pytorch_linalg_linalg_svd(const at::Tensor & self, bool full) noexcept {
  try {
    auto result = at::native::linalg_svd_default(self, full);
    return std::make_unique<TerlanOwnedTuplePytorchLinalgLinalgSvd>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_svd_take_0(TerlanOwnedTuplePytorchLinalgLinalgSvd& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_svd_take_1(TerlanOwnedTuplePytorchLinalgLinalgSvd& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_linalg_linalg_svd_take_2(TerlanOwnedTuplePytorchLinalgLinalgSvd& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward> terlan_owned_tuple_pytorch_losscompletion_multilabel_margin_loss_forward(const at::Tensor & input, const at::Tensor & target, std::int64_t reduction) noexcept {
  try {
    auto result = at::native::multilabel_margin_loss_forward_exact(input, target, reduction);
    return std::make_unique<TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_losscompletion_multilabel_margin_loss_forward_take_0(TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_losscompletion_multilabel_margin_loss_forward_take_1(TerlanOwnedTuplePytorchLossCompletionMultilabelMarginLossForward& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    auto result = at::fake_quantize_per_tensor_affine_cachemask(self, scale, zero_point, quant_min, quant_max);
    return std::make_unique<TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask_take_0(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask_take_1(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerTensorAffineCachemask& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    auto result = at::fake_quantize_per_channel_affine_cachemask(self, scale, zero_point, axis, quant_min, quant_max);
    return std::make_unique<TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask_take_0(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask_take_1(TerlanOwnedTuplePytorchQuantizationCompletionFakeQuantizePerChannelAffineCachemask& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default(const at::Tensor & lu_data, const at::Tensor & lu_pivots, bool unpack_data, bool unpack_pivots) noexcept {
  try {
    auto result = at::native::lu_unpack_default(lu_data, lu_pivots, unpack_data, unpack_pivots);
    return std::make_unique<TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default_take_0(TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default_take_1(TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_luunpackcompletion_lu_unpack_default_take_2(TerlanOwnedTuplePytorchLuUnpackCompletionLuUnpackDefault& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentLstmCell> terlan_owned_tuple_pytorch_recurrent_lstm_cell(const at::Tensor & input, const TerlanPytorchTensorTensorResourceListInput& hx, const at::Tensor & w_ih, const at::Tensor & w_hh, bool has_b_ih, const at::Tensor & b_ih, bool has_b_hh, const at::Tensor & b_hh) noexcept {
  try {
    auto result = at::native::lstm_cell_exact(input, hx.values(), w_ih, w_hh, has_b_ih, b_ih, has_b_hh, b_hh);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentLstmCell>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_cell_take_0(TerlanOwnedTuplePytorchRecurrentLstmCell& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_cell_take_1(TerlanOwnedTuplePytorchRecurrentLstmCell& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentRnnTanh> terlan_owned_tuple_pytorch_recurrent_rnn_tanh(const at::Tensor & input, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept {
  try {
    auto result = at::native::rnn_tanh_exact(input, hx, params.values(), has_biases, num_layers, dropout, train, bidirectional, batch_first);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentRnnTanh>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_tanh_take_0(TerlanOwnedTuplePytorchRecurrentRnnTanh& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_tanh_take_1(TerlanOwnedTuplePytorchRecurrentRnnTanh& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentRnnRelu> terlan_owned_tuple_pytorch_recurrent_rnn_relu(const at::Tensor & input, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept {
  try {
    auto result = at::native::rnn_relu_exact(input, hx, params.values(), has_biases, num_layers, dropout, train, bidirectional, batch_first);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentRnnRelu>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_relu_take_0(TerlanOwnedTuplePytorchRecurrentRnnRelu& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_rnn_relu_take_1(TerlanOwnedTuplePytorchRecurrentRnnRelu& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentGru> terlan_owned_tuple_pytorch_recurrent_gru(const at::Tensor & input, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept {
  try {
    auto result = at::native::gru_exact(input, hx, params.values(), has_biases, num_layers, dropout, train, bidirectional, batch_first);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentGru>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_gru_take_0(TerlanOwnedTuplePytorchRecurrentGru& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_gru_take_1(TerlanOwnedTuplePytorchRecurrentGru& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentLstm> terlan_owned_tuple_pytorch_recurrent_lstm(const at::Tensor & input, const TerlanPytorchTensorTensorResourceListInput& hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) noexcept {
  try {
    auto result = at::native::lstm_exact(input, hx.values(), params.values(), has_biases, num_layers, dropout, train, bidirectional, batch_first);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentLstm>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_take_0(TerlanOwnedTuplePytorchRecurrentLstm& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_take_1(TerlanOwnedTuplePytorchRecurrentLstm& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrent_lstm_take_2(TerlanOwnedTuplePytorchRecurrentLstm& value) noexcept { return value.take_2(); }

std::unique_ptr<TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized> terlan_owned_tuple_pytorch_quantizationinternals_choose_qparams_optimized(const Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) noexcept {
  try {
    auto result = at::choose_qparams_optimized(input, numel, n_bins, ratio, bit_width);
    return std::make_unique<TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationinternals_choose_qparams_optimized_take_0(TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_quantizationinternals_choose_qparams_optimized_take_1(TerlanOwnedTuplePytorchQuantizationInternalsChooseQparamsOptimized& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataRnnTanhData> terlan_owned_tuple_pytorch_recurrentdata_rnn_tanh_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept {
  try {
    auto result = at::native::rnn_tanh_data_exact(data, batch_sizes, hx, params.values(), has_biases, num_layers, dropout, train, bidirectional);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentDataRnnTanhData>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_tanh_data_take_0(TerlanOwnedTuplePytorchRecurrentDataRnnTanhData& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_tanh_data_take_1(TerlanOwnedTuplePytorchRecurrentDataRnnTanhData& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataRnnReluData> terlan_owned_tuple_pytorch_recurrentdata_rnn_relu_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept {
  try {
    auto result = at::native::rnn_relu_data_exact(data, batch_sizes, hx, params.values(), has_biases, num_layers, dropout, train, bidirectional);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentDataRnnReluData>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_relu_data_take_0(TerlanOwnedTuplePytorchRecurrentDataRnnReluData& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_rnn_relu_data_take_1(TerlanOwnedTuplePytorchRecurrentDataRnnReluData& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataGruData> terlan_owned_tuple_pytorch_recurrentdata_gru_data(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept {
  try {
    auto result = at::native::gru_data_exact(data, batch_sizes, hx, params.values(), has_biases, num_layers, dropout, train, bidirectional);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentDataGruData>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_gru_data_take_0(TerlanOwnedTuplePytorchRecurrentDataGruData& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_gru_data_take_1(TerlanOwnedTuplePytorchRecurrentDataGruData& value) noexcept { return value.take_1(); }

std::unique_ptr<TerlanOwnedTuplePytorchRecurrentDataLstmData> terlan_owned_tuple_pytorch_recurrentdata_lstm_data(const at::Tensor & data, const at::Tensor & batch_sizes, const TerlanPytorchTensorTensorResourceListInput& hx, const TerlanPytorchTensorTensorResourceListInput& params, bool has_biases, std::int64_t num_layers, double dropout, bool train, bool bidirectional) noexcept {
  try {
    auto result = at::native::lstm_data_exact(data, batch_sizes, hx.values(), params.values(), has_biases, num_layers, dropout, train, bidirectional);
    return std::make_unique<TerlanOwnedTuplePytorchRecurrentDataLstmData>(std::make_unique<at::Tensor>(std::get<0>(std::move(result))), std::make_unique<at::Tensor>(std::get<1>(std::move(result))), std::make_unique<at::Tensor>(std::get<2>(std::move(result))));
  } catch (...) {
    return nullptr;
  }
}
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_lstm_data_take_0(TerlanOwnedTuplePytorchRecurrentDataLstmData& value) noexcept { return value.take_0(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_lstm_data_take_1(TerlanOwnedTuplePytorchRecurrentDataLstmData& value) noexcept { return value.take_1(); }
std::unique_ptr<at::Tensor> terlan_owned_tuple_pytorch_recurrentdata_lstm_data_take_2(TerlanOwnedTuplePytorchRecurrentDataLstmData& value) noexcept { return value.take_2(); }

}  // namespace at::native
