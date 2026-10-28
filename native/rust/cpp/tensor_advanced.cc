#include "tensor_advanced.hpp"

#include <optional>
#include <stdexcept>

#include <ATen/ops/embedding_bag.h>
#include <ATen/ops/conv2d.h>

namespace at::native {

at::Tensor tensor_clamp(const at::Tensor& tensor, double minimum, double maximum) {
  if (minimum > maximum) throw std::invalid_argument("clamp minimum exceeds maximum");
  return at::clamp(tensor, minimum, maximum);
}

at::Tensor tensor_conv2d(const at::Tensor& input, const at::Tensor& weight,
                         const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
                         c10::ArrayRef<std::int64_t> padding,
                         c10::ArrayRef<std::int64_t> dilation,
                         std::int64_t groups) {
  return at::conv2d(input, weight, std::optional<at::Tensor>(bias), stride,
                    padding, dilation, groups);
}

EmbeddingBagResult tensor_embedding_bag(const at::Tensor& weight,
                                         const at::Tensor& indices,
                                         const at::Tensor& offsets,
                                         bool scale_gradient_by_frequency,
                                         std::int64_t mode, bool sparse,
                                         bool include_last_offset) {
  return at::embedding_bag(weight, indices, offsets, scale_gradient_by_frequency,
                           mode, sparse, std::nullopt, include_last_offset);
}

EmbeddingBagResult tensor_embedding_bag_weighted(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_gradient_by_frequency,
    std::int64_t mode, bool sparse, const at::Tensor& per_sample_weights,
    bool include_last_offset) {
  return at::embedding_bag(weight, indices, offsets, scale_gradient_by_frequency,
                           mode, sparse, std::optional<at::Tensor>(per_sample_weights),
                           include_last_offset);
}

EmbeddingBagResult tensor_embedding_bag_padding(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_gradient_by_frequency,
    std::int64_t mode, bool sparse, bool include_last_offset,
    std::int64_t padding_index) {
  return at::embedding_bag(weight, indices, offsets, scale_gradient_by_frequency,
                           mode, sparse, std::nullopt, include_last_offset,
                           std::optional<std::int64_t>(padding_index));
}

EmbeddingBagResult tensor_embedding_bag_padding_weighted(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_gradient_by_frequency,
    std::int64_t mode, bool sparse, const at::Tensor& per_sample_weights,
    bool include_last_offset, std::int64_t padding_index) {
  return at::embedding_bag(weight, indices, offsets, scale_gradient_by_frequency,
                           mode, sparse, std::optional<at::Tensor>(per_sample_weights),
                           include_last_offset,
                           std::optional<std::int64_t>(padding_index));
}

std::int64_t tensor_abi_version() { return 0x020d000000000000LL; }

}  // namespace at::native
