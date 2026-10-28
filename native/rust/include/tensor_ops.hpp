#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/histogramdd.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_batch_norm_backward.h>
#include <ATen/ops/log_sigmoid_backward.h>
#include <ATen/ops/log_sigmoid_forward.h>
#include <ATen/ops/nll_loss_backward.h>
#include <ATen/ops/nll_loss_forward.h>
#include <ATen/ops/nll_loss2d_backward.h>
#include <ATen/ops/nll_loss2d_forward.h>
#include <ATen/ops/nll_loss_nd.h>
#include <ATen/ops/rrelu_with_noise_backward.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/multi_margin_loss_backward.h>
#include <ATen/ops/multilabel_margin_loss_backward.h>
#include <ATen/ops/multilabel_margin_loss_forward.h>
#include <ATen/ops/nested_to_padded_tensor.h>
#include <ATen/ops/batch_norm_update_stats.h>
#include <ATen/ops/lstm_cell.h>
#include <ATen/ops/rnn_tanh_cell.h>
#include <ATen/ops/rnn_relu_cell.h>
#include <ATen/ops/gru_cell.h>
#include <ATen/ops/rnn_tanh.h>
#include <ATen/ops/rnn_relu.h>
#include <ATen/ops/gru.h>
#include <ATen/ops/lstm.h>
#include <ATen/ops/quantized_batch_norm.h>
#include <ATen/ops/quantized_max_pool1d.h>
#include <ATen/ops/quantized_max_pool2d.h>
#include <ATen/ops/quantized_max_pool3d.h>
#include <ATen/ops/thnn_conv2d.h>
#include <ATen/ops/slow_conv3d.h>
#include <ATen/ops/slow_conv_dilated2d.h>
#include <ATen/ops/slow_conv_dilated3d.h>
#include <ATen/ops/slow_conv_transpose2d.h>
#include <ATen/ops/slow_conv_transpose3d.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/grid_sampler_2d_backward.h>
#include <ATen/ops/grid_sampler_3d_backward.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/native_group_norm_backward.h>
#include <ATen/ops/nonzero_static.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/split_copy.h>
#include <ATen/ops/split_with_sizes_copy.h>
#include <ATen/ops/unbind_copy.h>
#include <ATen/ops/unsafe_split.h>
#include <ATen/ops/unsafe_split_with_sizes.h>
#include <ATen/ops/value_selecting_reduction_backward.h>
#include <ATen/ops/is_vulkan_available.h>
#include <ATen/ops/cudnn_is_acceptable.h>
#include <ATen/ops/sym_is_contiguous.h>

#if __has_include("rust/cxx.h")
#include "rust/cxx.h"
#else
namespace rust { inline namespace cxxbridge1 { template <typename T> class Slice; } }
#endif

namespace at::native {

bool is_vulkan_available_exact();
bool cudnn_is_acceptable_exact(const at::Tensor& input);
bool sym_is_contiguous_exact(const at::Tensor& input, std::int64_t memory_format);
bool deterministic_algorithms_enabled_exact();
bool deterministic_algorithms_warn_only_exact();
std::int64_t deterministic_debug_mode_exact();
void set_deterministic_algorithms_exact(bool enabled, bool warn_only);
void set_deterministic_debug_mode_exact(std::int64_t mode);

class Generator {
 public:
  explicit Generator(at::Generator value) : value_(std::move(value)) {}
  at::Generator value_;
};

class StorageHandle {
 public:
  explicit StorageHandle(c10::Storage value) : value_(std::move(value)) {}
  c10::Storage value_;
};

class HistogramddResult {
 public:
  explicit HistogramddResult(std::tuple<at::Tensor, std::vector<at::Tensor>> value)
      : value_(std::move(value)) {}
  at::Tensor hist() const { return std::get<0>(value_); }
  std::vector<at::Tensor> bin_edges() const { return std::get<1>(value_); }

 private:
  std::tuple<at::Tensor, std::vector<at::Tensor>> value_;
};

std::unique_ptr<Generator> generator_create(std::int64_t seed);
std::unique_ptr<Generator> manual_seed_exact(std::int64_t seed);
std::int64_t initial_seed_exact();
std::int64_t seed_exact();
at::Tensor get_rng_state_exact();
void set_rng_state_exact(const at::Tensor& state);
std::unique_ptr<StorageHandle> storage_from_tensor(const at::Tensor& tensor);
bool storage_is_shared(const at::Tensor& tensor);
void storage_share_memory(at::Tensor& tensor);
at::Tensor nonzero_static_exact(const at::Tensor& tensor,
                                std::int64_t size,
                                std::int64_t fill_value);
void nonzero_static_exact_out(at::Tensor& output,
                              const at::Tensor& tensor,
                              std::int64_t size,
                              std::int64_t fill_value);
at::Tensor value_selecting_reduction_backward_exact(
    const at::Tensor& gradient, std::int64_t dimension,
    const at::Tensor& indices, c10::ArrayRef<std::int64_t> sizes,
    bool keep_dimension);
at::Tensor storage_set(const at::Tensor& tensor, const StorageHandle& source);
at::Tensor& storage_set_out(at::Tensor& output, const at::Tensor& tensor,
                            const StorageHandle& source);
at::Tensor storage_set_offset(const at::Tensor& tensor,
                              const StorageHandle& source,
                              std::int64_t storage_offset,
                              c10::ArrayRef<std::int64_t> size,
                              c10::ArrayRef<std::int64_t> stride);
at::Tensor& storage_set_offset_out(at::Tensor& output, const at::Tensor& tensor,
                                   const StorageHandle& source,
                                   std::int64_t storage_offset,
                                   c10::ArrayRef<std::int64_t> size,
                                   c10::ArrayRef<std::int64_t> stride);
at::Tensor& storage_set_in_place(at::Tensor& tensor,
                                 const StorageHandle& source);
at::Tensor& storage_set_in_place_offset(
    at::Tensor& tensor, const StorageHandle& source,
    std::int64_t storage_offset, c10::ArrayRef<std::int64_t> size,
    c10::ArrayRef<std::int64_t> stride);
std::unique_ptr<HistogramddResult> histogramdd_int_result(
    const at::Tensor& input, std::int64_t bins, bool has_range,
    rust::Slice<const double> range, bool has_weight, const at::Tensor& weight,
    bool density);
std::unique_ptr<HistogramddResult> histogramdd_bins_result(
    const at::Tensor& input, rust::Slice<const std::int64_t> bins, bool has_range,
    rust::Slice<const double> range, bool has_weight, const at::Tensor& weight,
    bool density);
std::unique_ptr<HistogramddResult> histogramdd_tensor_bins_result(
    const at::Tensor& input, c10::ArrayRef<at::Tensor> bins, bool has_range,
    rust::Slice<const double> range, bool has_weight, const at::Tensor& weight,
    bool density);
at::Tensor histogramdd_result_hist(const HistogramddResult& result);
std::vector<at::Tensor> histogramdd_result_bin_edges(
    const HistogramddResult& result);
at::Tensor generator_rand(const Generator& generator,
                          c10::ArrayRef<std::int64_t> size,
                          bool has_dtype, std::int64_t dtype,
                          bool has_layout, std::int64_t layout,
                          bool has_device, std::int64_t device_type,
                          std::int64_t device_index,
                          bool has_pin_memory, bool pin_memory);
at::Tensor& generator_rand_out(at::Tensor& output,
                               c10::ArrayRef<std::int64_t> size,
                               const Generator& generator);
at::Tensor generator_randn(const Generator& generator,
                           c10::ArrayRef<std::int64_t> size,
                           bool has_dtype, std::int64_t dtype,
                           bool has_layout, std::int64_t layout,
                           bool has_device, std::int64_t device_type,
                           std::int64_t device_index,
                           bool has_pin_memory, bool pin_memory);
at::Tensor& generator_randn_out(at::Tensor& output,
                                c10::ArrayRef<std::int64_t> size,
                                const Generator& generator);
at::Tensor generator_randperm(const Generator& generator,
                              std::int64_t n,
                              bool has_dtype, std::int64_t dtype,
                              bool has_layout, std::int64_t layout,
                              bool has_device, std::int64_t device_type,
                              std::int64_t device_index,
                              bool has_pin_memory, bool pin_memory);
at::Tensor& generator_randperm_out(at::Tensor& output,
                                   std::int64_t n,
                                   const Generator& generator);
at::Tensor generator_rand_like(const Generator& generator,
                               const at::Tensor& input,
                               bool has_dtype, std::int64_t dtype,
                               bool has_layout, std::int64_t layout,
                               bool has_device, std::int64_t device_type,
                               std::int64_t device_index,
                               bool has_pin_memory, bool pin_memory,
                               bool has_memory_format, std::int64_t memory_format);
at::Tensor& generator_rand_like_out(at::Tensor& output,
                                    const at::Tensor& input,
                                    const Generator& generator,
                                    bool has_memory_format,
                                    std::int64_t memory_format);
at::Tensor generator_randn_like(const Generator& generator,
                                const at::Tensor& input,
                                bool has_dtype, std::int64_t dtype,
                                bool has_layout, std::int64_t layout,
                                bool has_device, std::int64_t device_type,
                                std::int64_t device_index,
                                bool has_pin_memory, bool pin_memory,
                                bool has_memory_format, std::int64_t memory_format);
at::Tensor& generator_randn_like_out(at::Tensor& output,
                                     const at::Tensor& input,
                                     const Generator& generator,
                                     bool has_memory_format,
                                     std::int64_t memory_format);
at::Tensor generator_randint(const Generator& generator,
                             std::int64_t high,
                             c10::ArrayRef<std::int64_t> size,
                             bool has_dtype, std::int64_t dtype,
                             bool has_layout, std::int64_t layout,
                             bool has_device, std::int64_t device_type,
                             std::int64_t device_index,
                             bool has_pin_memory, bool pin_memory);
at::Tensor& generator_randint_out(at::Tensor& output,
                                  std::int64_t high,
                                  c10::ArrayRef<std::int64_t> size,
                                  const Generator& generator);
at::Tensor generator_randint_low(const Generator& generator,
                                 std::int64_t low, std::int64_t high,
                                 c10::ArrayRef<std::int64_t> size,
                                 bool has_dtype, std::int64_t dtype,
                                 bool has_layout, std::int64_t layout,
                                 bool has_device, std::int64_t device_type,
                                 std::int64_t device_index,
                                 bool has_pin_memory, bool pin_memory);
at::Tensor& generator_randint_low_out(at::Tensor& output,
                                      std::int64_t low, std::int64_t high,
                                      c10::ArrayRef<std::int64_t> size,
                                      const Generator& generator);
at::Tensor generator_randint_like(const Generator& generator,
                                  const at::Tensor& input,
                                  std::int64_t high,
                                  bool has_dtype, std::int64_t dtype,
                                  bool has_layout, std::int64_t layout,
                                  bool has_device, std::int64_t device_type,
                                  std::int64_t device_index,
                                  bool has_pin_memory, bool pin_memory,
                                  bool has_memory_format, std::int64_t memory_format);
at::Tensor& generator_randint_like_out(at::Tensor& output,
                                       const at::Tensor& input,
                                       std::int64_t high,
                                       const Generator& generator,
                                       bool has_memory_format,
                                       std::int64_t memory_format);
at::Tensor generator_randint_like_tensor(const Generator& generator,
                                         const at::Tensor& input,
                                         const at::Tensor& high,
                                         bool has_dtype, std::int64_t dtype,
                                         bool has_layout, std::int64_t layout,
                                         bool has_device, std::int64_t device_type,
                                         std::int64_t device_index,
                                         bool has_pin_memory, bool pin_memory,
                                         bool has_memory_format, std::int64_t memory_format);
at::Tensor& generator_randint_like_tensor_out(at::Tensor& output,
                                              const at::Tensor& input,
                                              const at::Tensor& high,
                                              const Generator& generator,
                                              bool has_memory_format,
                                              std::int64_t memory_format);
at::Tensor generator_randint_like_low(const Generator& generator,
                                      const at::Tensor& input,
                                      std::int64_t low, std::int64_t high,
                                      bool has_dtype, std::int64_t dtype,
                                      bool has_layout, std::int64_t layout,
                                      bool has_device, std::int64_t device_type,
                                      std::int64_t device_index,
                                      bool has_pin_memory, bool pin_memory,
                                      bool has_memory_format, std::int64_t memory_format);
at::Tensor& generator_randint_like_low_out(at::Tensor& output,
                                           const at::Tensor& input,
                                           std::int64_t low, std::int64_t high,
                                           const Generator& generator,
                                           bool has_memory_format,
                                           std::int64_t memory_format);

// Tensor-list output adapters preserve the upstream out= contracts while
// keeping the CXX bridge entirely in typed C++ collections.
void dequantize_tensors_out(c10::ArrayRef<at::Tensor> tensors,
                            c10::ArrayRef<at::Tensor> out);
void quantize_per_tensor_tensors_out(c10::ArrayRef<at::Tensor> tensors,
                                     const at::Tensor& scales,
                                     const at::Tensor& zero_points,
                                     std::int64_t dtype,
                                     c10::ArrayRef<at::Tensor> out);
void split_copy_tensor_out(const at::Tensor& input,
                           std::int64_t split_size,
                           std::int64_t dimension,
                           c10::ArrayRef<at::Tensor> out);
void split_with_sizes_copy_out(const at::Tensor& input,
                               c10::ArrayRef<std::int64_t> split_sizes,
                               std::int64_t dimension,
                               c10::ArrayRef<at::Tensor> out);
void unbind_copy_int_out(const at::Tensor& input,
                         std::int64_t dimension,
                         c10::ArrayRef<at::Tensor> out);
void unsafe_split_tensor_out(const at::Tensor& input,
                             std::int64_t split_size,
                             std::int64_t dimension,
                             c10::ArrayRef<at::Tensor> out);
void unsafe_split_with_sizes_out(const at::Tensor& input,
                                 c10::ArrayRef<std::int64_t> split_sizes,
                                 std::int64_t dimension,
                                 c10::ArrayRef<at::Tensor> out);

void tensor_copy_in_place(at::Tensor& output, const at::Tensor& input,
                          bool non_blocking);
std::tuple<at::Tensor, at::Tensor> native_dropout_exact(
    const at::Tensor& input, double p, bool train);
void native_dropout_exact_out(at::Tensor& output, at::Tensor& mask,
                              const at::Tensor& input, double p,
                              bool train);
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_exact(
    const at::Tensor& input, c10::ArrayRef<std::int64_t> normalized_shape,
    bool has_weight, const at::Tensor& weight, bool has_bias,
    const at::Tensor& bias, double eps);
void native_layer_norm_exact_out(
    at::Tensor& output, at::Tensor& mean, at::Tensor& rstd,
    const at::Tensor& input, c10::ArrayRef<std::int64_t> normalized_shape,
    bool has_weight, const at::Tensor& weight, bool has_bias,
    const at::Tensor& bias, double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_exact(
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, double eps);
void native_group_norm_exact_out(
    at::Tensor& output, at::Tensor& mean, at::Tensor& rstd,
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_exact(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& mean, const at::Tensor& rstd, bool has_weight,
    const at::Tensor& weight, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias);
void native_group_norm_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& mean, const at::Tensor& rstd, bool has_weight,
    const at::Tensor& weight, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_layer_norm_backward_exact(
    const at::Tensor& grad_out, const at::Tensor& input,
    c10::ArrayRef<std::int64_t> normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias);
void native_layer_norm_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_out, const at::Tensor& input,
    c10::ArrayRef<std::int64_t> normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias);
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_exact(
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool training, double momentum,
    double eps);
void native_batch_norm_exact_out(
    at::Tensor& output, at::Tensor& save_mean, at::Tensor& save_invstd,
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool training, double momentum,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm_backward_exact(
    const at::Tensor& grad_out, const at::Tensor& input, bool has_weight,
    const at::Tensor& weight, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool has_save_mean,
    const at::Tensor& save_mean, bool has_save_invstd,
    const at::Tensor& save_invstd, bool train, double eps,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias);
void native_batch_norm_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_out, const at::Tensor& input, bool has_weight,
    const at::Tensor& weight, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool has_save_mean,
    const at::Tensor& save_mean, bool has_save_invstd,
    const at::Tensor& save_invstd, bool train, double eps,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias);
std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward_exact(
    const at::Tensor& input);
void log_sigmoid_forward_exact_out(at::Tensor& output, at::Tensor& buffer,
                                   const at::Tensor& input);
at::Tensor log_sigmoid_backward_exact(const at::Tensor& grad_output,
                                      const at::Tensor& input,
                                      const at::Tensor& buffer);
void log_sigmoid_backward_exact_out(at::Tensor& grad_input,
                                    const at::Tensor& grad_output,
                                    const at::Tensor& input,
                                    const at::Tensor& buffer);
std::tuple<at::Tensor, at::Tensor> nll_loss_forward_exact(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index);
void nll_loss_forward_exact_out(
    at::Tensor& output, at::Tensor& total_weight, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index);
at::Tensor nll_loss_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index,
    const at::Tensor& total_weight);
void nll_loss_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index, const at::Tensor& total_weight);
std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward_exact(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index);
void nll_loss2d_forward_exact_out(
    at::Tensor& output, at::Tensor& total_weight, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index);
at::Tensor nll_loss2d_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index,
    const at::Tensor& total_weight);
void nll_loss2d_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index, const at::Tensor& total_weight);
at::Tensor nll_loss_nd_exact(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index);
at::Tensor rrelu_with_noise_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, double lower, double upper, bool training,
    bool self_is_result);
std::tuple<at::Tensor, at::Tensor> rrelu_with_noise_functional_exact(
    const at::Tensor& self, const at::Tensor& noise, bool has_generator,
    const Generator& generator, double lower, double upper, bool training);
at::Tensor rrelu_with_noise_exact(
    const at::Tensor& self, const at::Tensor& noise, bool has_generator,
    const Generator& generator, double lower, double upper, bool training);
void rrelu_with_noise_exact_out(
    at::Tensor& output, const at::Tensor& self, const at::Tensor& noise,
    bool has_generator, const Generator& generator, double lower, double upper,
    bool training);
void rrelu_with_noise_backward_exact_out(
    at::Tensor& output, const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, double lower, double upper, bool training,
    bool self_is_result);
at::Tensor multi_margin_loss_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, double p, double margin, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction);
void multi_margin_loss_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target, double p, double margin,
    bool has_weight, const at::Tensor& weight, std::int64_t reduction);
std::tuple<at::Tensor, at::Tensor> batch_norm_update_stats_exact(
    const at::Tensor& input, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, double momentum);
void batch_norm_update_stats_exact_out(
    at::Tensor& out_mean, at::Tensor& out_var, const at::Tensor& input,
    bool has_running_mean, const at::Tensor& running_mean,
    bool has_running_var, const at::Tensor& running_var, double momentum);
std::tuple<at::Tensor, at::Tensor> lstm_cell_exact(
    const at::Tensor& input, c10::ArrayRef<at::Tensor> hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh);
at::Tensor rnn_tanh_cell_exact(
    const at::Tensor& input, const at::Tensor& hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh);
at::Tensor rnn_relu_cell_exact(
    const at::Tensor& input, const at::Tensor& hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh);
at::Tensor gru_cell_exact(
    const at::Tensor& input, const at::Tensor& hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh);
std::tuple<at::Tensor, at::Tensor> rnn_tanh_exact(
    const at::Tensor& input, const at::Tensor& hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);
std::tuple<at::Tensor, at::Tensor> rnn_relu_exact(
    const at::Tensor& input, const at::Tensor& hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);
std::tuple<at::Tensor, at::Tensor> gru_exact(
    const at::Tensor& input, const at::Tensor& hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);
std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_exact(
    const at::Tensor& input, c10::ArrayRef<at::Tensor> hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);
std::tuple<at::Tensor, at::Tensor> rnn_tanh_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    const at::Tensor& hx, c10::ArrayRef<at::Tensor> params,
    bool has_biases, std::int64_t num_layers, double dropout, bool train,
    bool bidirectional);
std::tuple<at::Tensor, at::Tensor> rnn_relu_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    const at::Tensor& hx, c10::ArrayRef<at::Tensor> params,
    bool has_biases, std::int64_t num_layers, double dropout, bool train,
    bool bidirectional);
std::tuple<at::Tensor, at::Tensor> gru_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    const at::Tensor& hx, c10::ArrayRef<at::Tensor> params,
    bool has_biases, std::int64_t num_layers, double dropout, bool train,
    bool bidirectional);
std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    c10::ArrayRef<at::Tensor> hx, c10::ArrayRef<at::Tensor> params,
    bool has_biases, std::int64_t num_layers, double dropout, bool train,
    bool bidirectional);
at::Tensor quantized_batch_norm_exact(
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, const at::Tensor& mean,
    const at::Tensor& var, double eps, double output_scale,
    std::int64_t output_zero_point);
void quantized_batch_norm_exact_out(
    at::Tensor& out, const at::Tensor& input, bool has_weight,
    const at::Tensor& weight, bool has_bias, const at::Tensor& bias,
    const at::Tensor& mean, const at::Tensor& var, double eps,
    double output_scale, std::int64_t output_zero_point);
at::Tensor thnn_conv2d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding);
void thnn_conv2d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding);
at::Tensor slow_conv3d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding);
void slow_conv3d_forward_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding);
void slow_conv3d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding);
void slow_conv_dilated2d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation);
void slow_conv_dilated3d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation);
void slow_conv_transpose2d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation);
void slow_conv_transpose3d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation);
at::Tensor slow_conv_dilated2d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation);
at::Tensor slow_conv_dilated3d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation);
at::Tensor slow_conv_transpose2d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation);
at::Tensor slow_conv_transpose3d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation);

at::Tensor multilabel_margin_loss_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, std::int64_t reduction,
    const at::Tensor& is_target);
void multilabel_margin_loss_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target,
    std::int64_t reduction, const at::Tensor& is_target);
std::tuple<at::Tensor, at::Tensor> multilabel_margin_loss_forward_exact(
    const at::Tensor& input, const at::Tensor& target, std::int64_t reduction);
void multilabel_margin_loss_forward_exact_out(
    at::Tensor& output, at::Tensor& is_target, const at::Tensor& input,
    const at::Tensor& target, std::int64_t reduction);
at::Tensor tensor_nested_to_padded_tensor_exact(
    const at::Tensor& tensor, double padding, bool has_output_size,
    c10::ArrayRef<std::int64_t> output_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, bool has_bias_sizes,
    c10::ArrayRef<std::int64_t> bias_sizes,
    c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias);
void convolution_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, bool has_bias_sizes,
    c10::ArrayRef<std::int64_t> bias_sizes,
    c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias);
at::Tensor convolution_exact(
    const at::Tensor& input, const at::Tensor& weight, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups);
void convolution_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias,
    c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups);

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid);
void grid_sampler_2d_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_grid,
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid);

std::tuple<at::Tensor, at::Tensor> grid_sampler_3d_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid);
void grid_sampler_3d_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_grid,
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid);

at::Tensor tensor_require_grad(const at::Tensor& tensor);
void tensor_backward(const at::Tensor& tensor);
void tensor_backward_with_gradient(const at::Tensor& tensor,
                                   const at::Tensor& gradient,
                                   bool retain_graph, bool create_graph);
at::Tensor tensor_autograd_grad(const at::Tensor& output,
                                const at::Tensor& input,
                                const at::Tensor& grad_output,
                                bool retain_graph, bool create_graph);
std::vector<at::Tensor> tensor_autograd_grad_many(
    c10::ArrayRef<at::Tensor> outputs, c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<at::Tensor> grad_outputs, bool retain_graph,
    bool create_graph);
at::Tensor tensor_backward_many_token(c10::ArrayRef<at::Tensor> outputs,
                                      c10::ArrayRef<at::Tensor> grad_outputs,
                                      bool retain_graph, bool create_graph);
at::Tensor tensor_gradient(const at::Tensor& tensor);
at::Tensor tensor_gradient_with_graph(const at::Tensor& tensor);
at::Tensor tensor_sgd_step(const at::Tensor& tensor, double learning_rate);
at::Tensor tensor_categorical_sample(const at::Tensor& probabilities,
                                     std::int64_t seed);
at::Tensor tensor_mse_loss_gradient(const at::Tensor& grad_output,
                                    const at::Tensor& predicted,
                                    const at::Tensor& target,
                                    std::int64_t reduction);
at::Tensor tensor_cross_entropy_loss(const at::Tensor& logits,
                                     const at::Tensor& target,
                                     std::int64_t reduction,
                                     std::int64_t ignore_index,
                                     double label_smoothing);

}  // namespace at::native
