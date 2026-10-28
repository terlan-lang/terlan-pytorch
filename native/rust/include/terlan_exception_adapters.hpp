#pragma once

#include "libtorch_bindings.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace at::native {

class TerlanExceptionEnvelope final {
 public:
  TerlanExceptionEnvelope(bool ok, std::int64_t value, double float_value, bool bool_value, std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::int64_t value() const noexcept;
  double float_value() const noexcept;
  bool bool_value() const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::int64_t value_;
  double float_value_;
  bool bool_value_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_allclose_native(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_equal_native(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_same_size_native(const Tensor & self, const Tensor & other) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_complex_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_conj_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_distributed_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_floating_point_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_inference_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_leaf_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_neg_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_nonzero_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_signed_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_retains_grad_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_set_to_native(const at::Tensor& value, const Tensor & tensor) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_get_device_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_output_nr_native(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_size_dimension_native(const at::Tensor& value, int64_t dim) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_stride_dimension_native(const at::Tensor& value, int64_t dim) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_can_cast_native(std::int64_t from_, std::int64_t to) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_promote_types_native(std::int64_t type1, std::int64_t type2) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_tensor_native(const Tensor & tensor, const Tensor & other) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_scalar_native(const Tensor & tensor, bool scalar_is_integer, std::int64_t scalar_integer, double scalar_floating) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_scalar_tensor_native(const Tensor & tensor, bool scalar_is_integer, std::int64_t scalar_integer, double scalar_floating) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_scalar_scalar_native(bool first_is_integer, std::int64_t first_integer, double first_floating, bool second_is_integer, std::int64_t second_integer, double second_floating) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_scalar_to_int(const c10::Scalar& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_scalar_to_float(const c10::Scalar& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_defined(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_cpu(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_cuda(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_requires_grad(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_leaf(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_has_storage(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_complex(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_floating_point(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_quantized(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_signed(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_sparse(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_contiguous(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_contiguous_memory_format_native(const at::Tensor& value, std::int64_t memory_format) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_dim(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_numel(const at::Tensor& value) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_pinned_cxx(const Tensor & self, bool has_device, rust::Str device) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_sparse_sparse_dim(const Tensor & self) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_sparse_dense_dim(const Tensor & self) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_sparse_is_coalesced(const Tensor & self) noexcept;
std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_cuda_device_index(const CudaStream & stream) noexcept;

}  // namespace at::native
