#pragma once

#include <cstdint>
#include <tuple>

#include <ATen/ATen.h>

namespace at::native {

using EmbeddingBagResult = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

at::Tensor tensor_clamp(const at::Tensor& tensor, double minimum, double maximum);
at::Tensor tensor_conv2d(const at::Tensor& input, const at::Tensor& weight,
                         const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
                         c10::ArrayRef<std::int64_t> padding,
                         c10::ArrayRef<std::int64_t> dilation,
                         std::int64_t groups);
EmbeddingBagResult tensor_embedding_bag(const at::Tensor& weight,
                                         const at::Tensor& indices,
                                         const at::Tensor& offsets,
                                         bool scale_gradient_by_frequency,
                                         std::int64_t mode, bool sparse,
                                         bool include_last_offset);
EmbeddingBagResult tensor_embedding_bag_weighted(const at::Tensor& weight,
                                                 const at::Tensor& indices,
                                                 const at::Tensor& offsets,
                                                 bool scale_gradient_by_frequency,
                                                 std::int64_t mode, bool sparse,
                                                 const at::Tensor& per_sample_weights,
                                                 bool include_last_offset);
EmbeddingBagResult tensor_embedding_bag_padding(const at::Tensor& weight,
                                                const at::Tensor& indices,
                                                const at::Tensor& offsets,
                                                bool scale_gradient_by_frequency,
                                                std::int64_t mode, bool sparse,
                                                bool include_last_offset,
                                                std::int64_t padding_index);
EmbeddingBagResult tensor_embedding_bag_padding_weighted(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_gradient_by_frequency,
    std::int64_t mode, bool sparse, const at::Tensor& per_sample_weights,
    bool include_last_offset, std::int64_t padding_index);
std::int64_t tensor_abi_version();

}  // namespace at::native
