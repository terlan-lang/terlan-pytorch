#include "tensor_ops.hpp"

#include <cmath>
#include <stdexcept>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/MapAllocator.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/mse_loss_backward.h>
#include <ATen/ops/multinomial.h>
#include <libshm.h>
#include <torch/torch.h>

namespace at::native {

bool is_vulkan_available_exact() {
  return at::is_vulkan_available();
}

bool cudnn_is_acceptable_exact(const at::Tensor& input) {
  return at::cudnn_is_acceptable(input);
}

bool sym_is_contiguous_exact(const at::Tensor& input, std::int64_t memory_format) {
  return at::__dispatch_sym_is_contiguous(
      input, static_cast<at::MemoryFormat>(memory_format))
      .expect_bool();
}

bool deterministic_algorithms_enabled_exact() {
  return at::globalContext().deterministicAlgorithms();
}

bool deterministic_algorithms_warn_only_exact() {
  return at::globalContext().deterministicAlgorithmsWarnOnly();
}

std::int64_t deterministic_debug_mode_exact() {
  if (!at::globalContext().deterministicAlgorithms()) return 0;
  return at::globalContext().deterministicAlgorithmsWarnOnly() ? 1 : 2;
}

void set_deterministic_algorithms_exact(bool enabled, bool warn_only) {
  at::globalContext().setDeterministicAlgorithms(enabled, warn_only);
}

void set_deterministic_debug_mode_exact(std::int64_t mode) {
  if (mode < 0 || mode > 2) {
    throw std::invalid_argument("deterministic debug mode must be 0, 1, or 2");
  }
  at::globalContext().setDeterministicAlgorithms(mode != 0, mode == 1);
}

std::unique_ptr<Generator> generator_create(std::int64_t seed) {
  if (seed < 0) throw std::invalid_argument("generator seed must be nonnegative");
  auto generator = at::make_generator<at::CPUGeneratorImpl>();
  generator.set_current_seed(static_cast<std::uint64_t>(seed));
  return std::make_unique<Generator>(std::move(generator));
}

std::unique_ptr<Generator> manual_seed_exact(std::int64_t seed) {
  at::manual_seed(static_cast<std::uint64_t>(seed));
  return std::make_unique<Generator>(
      at::globalContext().defaultGenerator(c10::DeviceType::CPU));
}

std::int64_t initial_seed_exact() {
  return static_cast<std::int64_t>(
      at::globalContext().defaultGenerator(c10::DeviceType::CPU).current_seed());
}

std::int64_t seed_exact() {
  return static_cast<std::int64_t>(const_cast<at::Generator&>(
      at::globalContext().defaultGenerator(c10::DeviceType::CPU)).seed());
}

at::Tensor get_rng_state_exact() {
  return at::globalContext().defaultGenerator(c10::DeviceType::CPU).get_state();
}

void set_rng_state_exact(const at::Tensor& state) {
  const_cast<at::Generator&>(
      at::globalContext().defaultGenerator(c10::DeviceType::CPU))
      .set_state(state);
}

std::unique_ptr<StorageHandle> storage_from_tensor(const at::Tensor& tensor) {
  return std::make_unique<StorageHandle>(tensor.storage());
}

bool storage_is_shared(const at::Tensor& tensor) {
  if (tensor.is_cuda()) return true;
  const auto& data = tensor.storage().data_ptr();
  return at::MapAllocator::fromDataPtr(data) != nullptr ||
      THManagedMapAllocator::fromDataPtr(data) != nullptr;
}

void storage_share_memory(at::Tensor& tensor) {
  at::share_memory_(tensor);
}

at::Tensor nonzero_static_exact(const at::Tensor& tensor,
                                std::int64_t size,
                                std::int64_t fill_value) {
  return at::nonzero_static(tensor, size, fill_value);
}

void nonzero_static_exact_out(at::Tensor& output,
                              const at::Tensor& tensor,
                              std::int64_t size,
                              std::int64_t fill_value) {
  at::nonzero_static_symint_out(output, tensor, c10::SymInt(size), fill_value);
}

at::Tensor value_selecting_reduction_backward_exact(
    const at::Tensor& gradient, std::int64_t dimension,
    const at::Tensor& indices, c10::ArrayRef<std::int64_t> sizes,
    bool keep_dimension) {
  return at::value_selecting_reduction_backward(
      gradient, dimension, indices, sizes, keep_dimension);
}

at::Tensor storage_set(const at::Tensor& tensor, const StorageHandle& source) {
  return at::set(tensor, source.value_);
}

at::Tensor& storage_set_out(at::Tensor& output, const at::Tensor& tensor,
                            const StorageHandle& source) {
  return at::set_outf(tensor, source.value_, output);
}

at::Tensor storage_set_offset(const at::Tensor& tensor,
                              const StorageHandle& source,
                              std::int64_t storage_offset,
                              c10::ArrayRef<std::int64_t> size,
                              c10::ArrayRef<std::int64_t> stride) {
  return at::set(tensor, source.value_, storage_offset, size, stride);
}

at::Tensor& storage_set_offset_out(at::Tensor& output, const at::Tensor& tensor,
                                   const StorageHandle& source,
                                   std::int64_t storage_offset,
                                   c10::ArrayRef<std::int64_t> size,
                                   c10::ArrayRef<std::int64_t> stride) {
  return at::set_outf(tensor, source.value_, storage_offset, size, stride,
                      output);
}

at::Tensor& storage_set_in_place(at::Tensor& tensor,
                                 const StorageHandle& source) {
  return tensor.set_(source.value_);
}

at::Tensor& storage_set_in_place_offset(
    at::Tensor& tensor, const StorageHandle& source,
    std::int64_t storage_offset, c10::ArrayRef<std::int64_t> size,
    c10::ArrayRef<std::int64_t> stride) {
  return tensor.set_(source.value_, storage_offset, size, stride);
}

std::unique_ptr<HistogramddResult> histogramdd_int_result(
    const at::Tensor& input, std::int64_t bins, bool has_range,
    rust::Slice<const double> range, bool has_weight, const at::Tensor& weight,
    bool density) {
  c10::ArrayRef<double> range_ref(range.data(), range.size());
  std::optional<c10::ArrayRef<double>> selected_range =
      has_range ? std::optional<c10::ArrayRef<double>>(range_ref) : std::nullopt;
  std::optional<at::Tensor> selected_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return std::make_unique<HistogramddResult>(
      at::histogramdd(input, bins, selected_range, selected_weight, density));
}

std::unique_ptr<HistogramddResult> histogramdd_bins_result(
    const at::Tensor& input, rust::Slice<const std::int64_t> bins, bool has_range,
    rust::Slice<const double> range, bool has_weight, const at::Tensor& weight,
    bool density) {
  c10::ArrayRef<std::int64_t> bins_ref(bins.data(), bins.size());
  c10::ArrayRef<double> range_ref(range.data(), range.size());
  std::optional<c10::ArrayRef<double>> selected_range =
      has_range ? std::optional<c10::ArrayRef<double>>(range_ref) : std::nullopt;
  std::optional<at::Tensor> selected_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return std::make_unique<HistogramddResult>(
      at::histogramdd(input, bins_ref, selected_range, selected_weight, density));
}

std::unique_ptr<HistogramddResult> histogramdd_tensor_bins_result(
    const at::Tensor& input, c10::ArrayRef<at::Tensor> bins, bool has_range,
    rust::Slice<const double> range, bool has_weight, const at::Tensor& weight,
    bool density) {
  c10::ArrayRef<double> range_ref(range.data(), range.size());
  std::optional<c10::ArrayRef<double>> selected_range =
      has_range ? std::optional<c10::ArrayRef<double>>(range_ref) : std::nullopt;
  std::optional<at::Tensor> selected_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return std::make_unique<HistogramddResult>(at::histogramdd(
      input, bins, selected_range, selected_weight, density));
}

at::Tensor histogramdd_result_hist(const HistogramddResult& result) {
  return result.hist();
}

std::vector<at::Tensor> histogramdd_result_bin_edges(
    const HistogramddResult& result) {
  return result.bin_edges();
}

at::Tensor generator_rand(const Generator& generator,
                          c10::ArrayRef<std::int64_t> size,
                          bool has_dtype, std::int64_t dtype,
                          bool has_layout, std::int64_t layout,
                          bool has_device, std::int64_t device_type,
                          std::int64_t device_index,
                          bool has_pin_memory, bool pin_memory) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  return at::rand(size, std::optional<at::Generator>(generator.value_),
                  selected_dtype, selected_layout, selected_device,
                  selected_pin_memory);
}

at::Tensor& generator_rand_out(at::Tensor& output,
                               c10::ArrayRef<std::int64_t> size,
                               const Generator& generator) {
  return at::rand_out(output, size,
                      std::optional<at::Generator>(generator.value_));
}

at::Tensor generator_randn(const Generator& generator,
                           c10::ArrayRef<std::int64_t> size,
                           bool has_dtype, std::int64_t dtype,
                           bool has_layout, std::int64_t layout,
                           bool has_device, std::int64_t device_type,
                           std::int64_t device_index,
                           bool has_pin_memory, bool pin_memory) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  return at::randn(size, std::optional<at::Generator>(generator.value_),
                   selected_dtype, selected_layout, selected_device,
                   selected_pin_memory);
}

at::Tensor& generator_randn_out(at::Tensor& output,
                                c10::ArrayRef<std::int64_t> size,
                                const Generator& generator) {
  return at::randn_out(output, size,
                       std::optional<at::Generator>(generator.value_));
}

at::Tensor generator_randperm(const Generator& generator,
                              std::int64_t n,
                              bool has_dtype, std::int64_t dtype,
                              bool has_layout, std::int64_t layout,
                              bool has_device, std::int64_t device_type,
                              std::int64_t device_index,
                              bool has_pin_memory, bool pin_memory) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  return at::randperm(n, std::optional<at::Generator>(generator.value_),
                      selected_dtype, selected_layout, selected_device,
                      selected_pin_memory);
}

at::Tensor& generator_randperm_out(at::Tensor& output,
                                   std::int64_t n,
                                   const Generator& generator) {
  return at::randperm_out(output, n,
                          std::optional<at::Generator>(generator.value_));
}

at::Tensor generator_rand_like(const Generator& generator,
                               const at::Tensor& input,
                               bool has_dtype, std::int64_t dtype,
                               bool has_layout, std::int64_t layout,
                               bool has_device, std::int64_t device_type,
                               std::int64_t device_index,
                               bool has_pin_memory, bool pin_memory,
                               bool has_memory_format, std::int64_t memory_format) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::rand_like(input, std::optional<at::Generator>(generator.value_),
                       selected_dtype, selected_layout, selected_device,
                       selected_pin_memory, selected_memory_format);
}

at::Tensor& generator_rand_like_out(at::Tensor& output,
                                    const at::Tensor& input,
                                    const Generator& generator,
                                    bool has_memory_format,
                                    std::int64_t memory_format) {
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::rand_like_out(output, input,
                           std::optional<at::Generator>(generator.value_),
                           selected_memory_format);
}

at::Tensor generator_randn_like(const Generator& generator,
                                const at::Tensor& input,
                                bool has_dtype, std::int64_t dtype,
                                bool has_layout, std::int64_t layout,
                                bool has_device, std::int64_t device_type,
                                std::int64_t device_index,
                                bool has_pin_memory, bool pin_memory,
                                bool has_memory_format, std::int64_t memory_format) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::randn_like(input, std::optional<at::Generator>(generator.value_),
                        selected_dtype, selected_layout, selected_device,
                        selected_pin_memory, selected_memory_format);
}

at::Tensor& generator_randn_like_out(at::Tensor& output,
                                     const at::Tensor& input,
                                     const Generator& generator,
                                     bool has_memory_format,
                                     std::int64_t memory_format) {
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::randn_like_out(output, input,
                            std::optional<at::Generator>(generator.value_),
                            selected_memory_format);
}

at::Tensor generator_randint(const Generator& generator,
                             std::int64_t high,
                             c10::ArrayRef<std::int64_t> size,
                             bool has_dtype, std::int64_t dtype,
                             bool has_layout, std::int64_t layout,
                             bool has_device, std::int64_t device_type,
                             std::int64_t device_index,
                             bool has_pin_memory, bool pin_memory) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  return at::randint(high, size, std::optional<at::Generator>(generator.value_),
                     selected_dtype, selected_layout, selected_device,
                     selected_pin_memory);
}

at::Tensor& generator_randint_out(at::Tensor& output,
                                  std::int64_t high,
                                  c10::ArrayRef<std::int64_t> size,
                                  const Generator& generator) {
  return at::randint_out(output, high, size,
                         std::optional<at::Generator>(generator.value_));
}

at::Tensor generator_randint_low(const Generator& generator,
                                 std::int64_t low, std::int64_t high,
                                 c10::ArrayRef<std::int64_t> size,
                                 bool has_dtype, std::int64_t dtype,
                                 bool has_layout, std::int64_t layout,
                                 bool has_device, std::int64_t device_type,
                                 std::int64_t device_index,
                                 bool has_pin_memory, bool pin_memory) {
  auto selected_dtype = has_dtype
      ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<at::Device>(at::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  return at::randint(low, high, size,
                     std::optional<at::Generator>(generator.value_),
                     selected_dtype, selected_layout, selected_device,
                     selected_pin_memory);
}

at::Tensor& generator_randint_low_out(at::Tensor& output,
                                      std::int64_t low, std::int64_t high,
                                      c10::ArrayRef<std::int64_t> size,
                                      const Generator& generator) {
  return at::randint_out(output, low, high, size,
                         std::optional<at::Generator>(generator.value_));
}

namespace {
struct GeneratorLikeOptions {
  std::optional<at::ScalarType> dtype;
  std::optional<at::Layout> layout;
  std::optional<at::Device> device;
  std::optional<bool> pin_memory;
  std::optional<at::MemoryFormat> memory_format;
};

GeneratorLikeOptions generator_like_options(
    bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout,
    bool has_device, std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory,
    bool has_memory_format, std::int64_t memory_format) {
  return {
      has_dtype ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype))
                : std::nullopt,
      has_layout ? std::optional<at::Layout>(static_cast<at::Layout>(layout))
                 : std::nullopt,
      has_device ? std::optional<at::Device>(at::Device(
                       static_cast<c10::DeviceType>(device_type), device_index))
                 : std::nullopt,
      has_pin_memory ? std::optional<bool>(pin_memory) : std::nullopt,
      has_memory_format
          ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
          : std::nullopt};
}
}  // namespace

at::Tensor generator_randint_like(const Generator& generator,
                                  const at::Tensor& input,
                                  std::int64_t high,
                                  bool has_dtype, std::int64_t dtype,
                                  bool has_layout, std::int64_t layout,
                                  bool has_device, std::int64_t device_type,
                                  std::int64_t device_index,
                                  bool has_pin_memory, bool pin_memory,
                                  bool has_memory_format, std::int64_t memory_format) {
  auto options = generator_like_options(
      has_dtype, dtype, has_layout, layout, has_device, device_type,
      device_index, has_pin_memory, pin_memory, has_memory_format,
      memory_format);
  return at::randint_like(input, high,
                          std::optional<at::Generator>(generator.value_),
                          options.dtype, options.layout, options.device,
                          options.pin_memory, options.memory_format);
}

at::Tensor& generator_randint_like_out(at::Tensor& output,
                                       const at::Tensor& input,
                                       std::int64_t high,
                                       const Generator& generator,
                                       bool has_memory_format,
                                       std::int64_t memory_format) {
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::randint_like_out(output, input, high,
                              std::optional<at::Generator>(generator.value_),
                              selected_memory_format);
}

at::Tensor generator_randint_like_tensor(
    const Generator& generator, const at::Tensor& input,
    const at::Tensor& high, bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout, bool has_device,
    std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory, bool has_memory_format,
    std::int64_t memory_format) {
  auto options = generator_like_options(
      has_dtype, dtype, has_layout, layout, has_device, device_type,
      device_index, has_pin_memory, pin_memory, has_memory_format,
      memory_format);
  return at::randint_like(input, high,
                          std::optional<at::Generator>(generator.value_),
                          options.dtype, options.layout, options.device,
                          options.pin_memory, options.memory_format);
}

at::Tensor& generator_randint_like_tensor_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& high,
    const Generator& generator, bool has_memory_format,
    std::int64_t memory_format) {
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::randint_like_out(output, input, high,
                              std::optional<at::Generator>(generator.value_),
                              selected_memory_format);
}

at::Tensor generator_randint_like_low(
    const Generator& generator, const at::Tensor& input, std::int64_t low,
    std::int64_t high, bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout, bool has_device,
    std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory, bool has_memory_format,
    std::int64_t memory_format) {
  auto options = generator_like_options(
      has_dtype, dtype, has_layout, layout, has_device, device_type,
      device_index, has_pin_memory, pin_memory, has_memory_format,
      memory_format);
  return at::randint_like(input, low, high,
                          std::optional<at::Generator>(generator.value_),
                          options.dtype, options.layout, options.device,
                          options.pin_memory, options.memory_format);
}

at::Tensor& generator_randint_like_low_out(
    at::Tensor& output, const at::Tensor& input, std::int64_t low,
    std::int64_t high, const Generator& generator, bool has_memory_format,
    std::int64_t memory_format) {
  auto selected_memory_format = has_memory_format
      ? std::optional<at::MemoryFormat>(static_cast<at::MemoryFormat>(memory_format))
      : std::nullopt;
  return at::randint_like_out(output, input, low, high,
                              std::optional<at::Generator>(generator.value_),
                              selected_memory_format);
}

void dequantize_tensors_out(c10::ArrayRef<at::Tensor> tensors,
                            c10::ArrayRef<at::Tensor> out) {
  at::dequantize_out(out, tensors);
}

void quantize_per_tensor_tensors_out(c10::ArrayRef<at::Tensor> tensors,
                                     const at::Tensor& scales,
                                     const at::Tensor& zero_points,
                                     std::int64_t dtype,
                                     c10::ArrayRef<at::Tensor> out) {
  at::quantize_per_tensor_out(out, tensors, scales, zero_points,
                              static_cast<c10::ScalarType>(dtype));
}

void split_copy_tensor_out(const at::Tensor& input,
                           std::int64_t split_size,
                           std::int64_t dimension,
                           c10::ArrayRef<at::Tensor> out) {
  at::split_copy_outf(input, split_size, dimension, out);
}

void split_with_sizes_copy_out(const at::Tensor& input,
                               c10::ArrayRef<std::int64_t> split_sizes,
                               std::int64_t dimension,
                               c10::ArrayRef<at::Tensor> out) {
  at::split_with_sizes_copy_outf(input, split_sizes, dimension, out);
}

void unbind_copy_int_out(const at::Tensor& input,
                         std::int64_t dimension,
                         c10::ArrayRef<at::Tensor> out) {
  at::unbind_copy_outf(input, dimension, out);
}

void unsafe_split_tensor_out(const at::Tensor& input,
                             std::int64_t split_size,
                             std::int64_t dimension,
                             c10::ArrayRef<at::Tensor> out) {
  at::unsafe_split_outf(input, split_size, dimension, out);
}

void unsafe_split_with_sizes_out(const at::Tensor& input,
                                 c10::ArrayRef<std::int64_t> split_sizes,
                                 std::int64_t dimension,
                                 c10::ArrayRef<at::Tensor> out) {
  at::unsafe_split_with_sizes_outf(input, split_sizes, dimension, out);
}

void tensor_copy_in_place(at::Tensor& output, const at::Tensor& input,
                          bool non_blocking) {
  output.copy_(input, non_blocking);
}

std::tuple<at::Tensor, at::Tensor> native_dropout_exact(
    const at::Tensor& input, double p, bool train) {
  return at::native_dropout(input, p, std::optional<bool>(train));
}

void native_dropout_exact_out(at::Tensor& output, at::Tensor& mask,
                              const at::Tensor& input, double p,
                              bool train) {
  at::native_dropout_outf(input, p, std::optional<bool>(train), output, mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_exact(
    const at::Tensor& input, c10::ArrayRef<std::int64_t> normalized_shape,
    bool has_weight, const at::Tensor& weight, bool has_bias,
    const at::Tensor& bias, double eps) {
  return at::native_layer_norm(
      input, normalized_shape,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, eps);
}

void native_layer_norm_exact_out(
    at::Tensor& output, at::Tensor& mean, at::Tensor& rstd,
    const at::Tensor& input, c10::ArrayRef<std::int64_t> normalized_shape,
    bool has_weight, const at::Tensor& weight, bool has_bias,
    const at::Tensor& bias, double eps) {
  at::native_layer_norm_out(
      output, mean, rstd, input, normalized_shape,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_exact(
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, double eps) {
  return at::native_group_norm(
      input, has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, n, c, hxw,
      group, eps);
}

void native_group_norm_exact_out(
    at::Tensor& output, at::Tensor& mean, at::Tensor& rstd,
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, double eps) {
  at::native_group_norm_out(
      output, mean, rstd, input,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, n, c, hxw,
      group, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_exact(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& mean, const at::Tensor& rstd, bool has_weight,
    const at::Tensor& weight, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias) {
  return at::native_group_norm_backward(
      grad_out, input, mean, rstd,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt, n, c,
      hxw, group, {output_mask_input, output_mask_weight, output_mask_bias});
}

void native_group_norm_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& mean, const at::Tensor& rstd, bool has_weight,
    const at::Tensor& weight, std::int64_t n, std::int64_t c,
    std::int64_t hxw, std::int64_t group, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias) {
  at::native_group_norm_backward_out(
      grad_input, grad_weight, grad_bias, grad_out, input, mean, rstd,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt, n, c,
      hxw, group, {output_mask_input, output_mask_weight, output_mask_bias});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_layer_norm_backward_exact(
    const at::Tensor& grad_out, const at::Tensor& input,
    c10::ArrayRef<std::int64_t> normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias) {
  return at::native_layer_norm_backward(
      grad_out, input, normalized_shape, mean, rstd,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt,
      {output_mask_input, output_mask_weight, output_mask_bias});
}

void native_layer_norm_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_out, const at::Tensor& input,
    c10::ArrayRef<std::int64_t> normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool output_mask_input,
    bool output_mask_weight, bool output_mask_bias) {
  at::native_layer_norm_backward_out(
      grad_input, grad_weight, grad_bias, grad_out, input, normalized_shape,
      mean, rstd,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt,
      {output_mask_input, output_mask_weight, output_mask_bias});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_exact(
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool training, double momentum,
    double eps) {
  return at::native_batch_norm(
      input,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt,
      has_running_mean ? std::optional<at::Tensor>(running_mean) : std::nullopt,
      has_running_var ? std::optional<at::Tensor>(running_var) : std::nullopt,
      training, momentum, eps);
}

void native_batch_norm_exact_out(
    at::Tensor& output, at::Tensor& save_mean, at::Tensor& save_invstd,
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool training, double momentum,
    double eps) {
  at::native_batch_norm_out(
      output, save_mean, save_invstd, input,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt,
      has_running_mean ? std::optional<at::Tensor>(running_mean) : std::nullopt,
      has_running_var ? std::optional<at::Tensor>(running_var) : std::nullopt,
      training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm_backward_exact(
    const at::Tensor& grad_out, const at::Tensor& input, bool has_weight,
    const at::Tensor& weight, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool has_save_mean,
    const at::Tensor& save_mean, bool has_save_invstd,
    const at::Tensor& save_invstd, bool train, double eps,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias) {
  return at::native_batch_norm_backward(
      grad_out, input,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_running_mean ? std::optional<at::Tensor>(running_mean) : std::nullopt,
      has_running_var ? std::optional<at::Tensor>(running_var) : std::nullopt,
      has_save_mean ? std::optional<at::Tensor>(save_mean) : std::nullopt,
      has_save_invstd ? std::optional<at::Tensor>(save_invstd) : std::nullopt,
      train, eps, {output_mask_input, output_mask_weight, output_mask_bias});
}

void native_batch_norm_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_out, const at::Tensor& input, bool has_weight,
    const at::Tensor& weight, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, bool has_save_mean,
    const at::Tensor& save_mean, bool has_save_invstd,
    const at::Tensor& save_invstd, bool train, double eps,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias) {
  at::native_batch_norm_backward_out(
      grad_input, grad_weight, grad_bias, grad_out, input,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_running_mean ? std::optional<at::Tensor>(running_mean) : std::nullopt,
      has_running_var ? std::optional<at::Tensor>(running_var) : std::nullopt,
      has_save_mean ? std::optional<at::Tensor>(save_mean) : std::nullopt,
      has_save_invstd ? std::optional<at::Tensor>(save_invstd) : std::nullopt,
      train, eps, {output_mask_input, output_mask_weight, output_mask_bias});
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward_exact(
    const at::Tensor& input) {
  return at::log_sigmoid_forward(input);
}

void log_sigmoid_forward_exact_out(at::Tensor& output, at::Tensor& buffer,
                                   const at::Tensor& input) {
  at::log_sigmoid_forward_out(output, buffer, input);
}

at::Tensor log_sigmoid_backward_exact(const at::Tensor& grad_output,
                                      const at::Tensor& input,
                                      const at::Tensor& buffer) {
  return at::log_sigmoid_backward(grad_output, input, buffer);
}

void log_sigmoid_backward_exact_out(at::Tensor& grad_input,
                                    const at::Tensor& grad_output,
                                    const at::Tensor& input,
                                    const at::Tensor& buffer) {
  at::log_sigmoid_backward_out(grad_input, grad_output, input, buffer);
}

std::tuple<at::Tensor, at::Tensor> nll_loss_forward_exact(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return at::nll_loss_forward(input, target, optional_weight, reduction,
                              ignore_index);
}

void nll_loss_forward_exact_out(
    at::Tensor& output, at::Tensor& total_weight, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  at::nll_loss_forward_out(output, total_weight, input, target,
                           optional_weight, reduction, ignore_index);
}

at::Tensor nll_loss_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index,
    const at::Tensor& total_weight) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return at::nll_loss_backward(grad_output, input, target, optional_weight,
                               reduction, ignore_index, total_weight);
}

void nll_loss_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index, const at::Tensor& total_weight) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  at::nll_loss_backward_out(grad_input, grad_output, input, target,
                            optional_weight, reduction, ignore_index,
                            total_weight);
}

std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward_exact(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return at::nll_loss2d_forward(input, target, optional_weight, reduction,
                                ignore_index);
}

void nll_loss2d_forward_exact_out(
    at::Tensor& output, at::Tensor& total_weight, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  at::nll_loss2d_forward_out(output, total_weight, input, target,
                             optional_weight, reduction, ignore_index);
}

at::Tensor nll_loss2d_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, bool has_weight, const at::Tensor& weight,
    std::int64_t reduction, std::int64_t ignore_index,
    const at::Tensor& total_weight) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return at::nll_loss2d_backward(grad_output, input, target, optional_weight,
                                 reduction, ignore_index, total_weight);
}

void nll_loss2d_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index, const at::Tensor& total_weight) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  at::nll_loss2d_backward_out(grad_input, grad_output, input, target,
                              optional_weight, reduction, ignore_index,
                              total_weight);
}

at::Tensor nll_loss_nd_exact(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction,
    std::int64_t ignore_index) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return at::nll_loss_nd(input, target, optional_weight, reduction,
                         ignore_index);
}

at::Tensor rrelu_with_noise_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, double lower, double upper, bool training,
    bool self_is_result) {
  return at::rrelu_with_noise_backward(
      grad_output, self, noise, lower, upper, training, self_is_result);
}

std::tuple<at::Tensor, at::Tensor> rrelu_with_noise_functional_exact(
    const at::Tensor& self, const at::Tensor& noise, bool has_generator,
    const Generator& generator, double lower, double upper, bool training) {
  return at::rrelu_with_noise_functional(
      self, noise, at::Scalar(lower), at::Scalar(upper), training,
      has_generator ? std::optional<at::Generator>(generator.value_)
                    : std::nullopt);
}

at::Tensor rrelu_with_noise_exact(
    const at::Tensor& self, const at::Tensor& noise, bool has_generator,
    const Generator& generator, double lower, double upper, bool training) {
  auto mutable_noise = noise;
  return at::rrelu_with_noise(
      self, mutable_noise, at::Scalar(lower), at::Scalar(upper), training,
      has_generator ? std::optional<at::Generator>(generator.value_)
                    : std::nullopt);
}

void rrelu_with_noise_exact_out(
    at::Tensor& output, const at::Tensor& self, const at::Tensor& noise,
    bool has_generator, const Generator& generator, double lower, double upper,
    bool training) {
  auto mutable_noise = noise;
  at::rrelu_with_noise_outf(
      self, mutable_noise, at::Scalar(lower), at::Scalar(upper), training,
      has_generator ? std::optional<at::Generator>(generator.value_)
                    : std::nullopt, output);
}

void rrelu_with_noise_backward_exact_out(
    at::Tensor& output, const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, double lower, double upper, bool training,
    bool self_is_result) {
  at::rrelu_with_noise_backward_out(
      output, grad_output, self, noise, lower, upper, training,
      self_is_result);
}

at::Tensor multi_margin_loss_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, double p, double margin, bool has_weight,
    const at::Tensor& weight, std::int64_t reduction) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  return at::multi_margin_loss_backward(
      grad_output, input, target, p, margin, optional_weight, reduction);
}

void multi_margin_loss_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target, double p, double margin,
    bool has_weight, const at::Tensor& weight, std::int64_t reduction) {
  const std::optional<at::Tensor> optional_weight =
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt;
  at::multi_margin_loss_backward_out(
      grad_input, grad_output, input, target, p, margin, optional_weight,
      reduction);
}

std::tuple<at::Tensor, at::Tensor> batch_norm_update_stats_exact(
    const at::Tensor& input, bool has_running_mean,
    const at::Tensor& running_mean, bool has_running_var,
    const at::Tensor& running_var, double momentum) {
  const std::optional<at::Tensor> optional_running_mean =
      has_running_mean ? std::optional<at::Tensor>(running_mean) : std::nullopt;
  const std::optional<at::Tensor> optional_running_var =
      has_running_var ? std::optional<at::Tensor>(running_var) : std::nullopt;
  return at::batch_norm_update_stats(
      input, optional_running_mean, optional_running_var, momentum);
}

void batch_norm_update_stats_exact_out(
    at::Tensor& out_mean, at::Tensor& out_var, const at::Tensor& input,
    bool has_running_mean, const at::Tensor& running_mean,
    bool has_running_var, const at::Tensor& running_var, double momentum) {
  const std::optional<at::Tensor> optional_running_mean =
      has_running_mean ? std::optional<at::Tensor>(running_mean) : std::nullopt;
  const std::optional<at::Tensor> optional_running_var =
      has_running_var ? std::optional<at::Tensor>(running_var) : std::nullopt;
  at::batch_norm_update_stats_out(
      out_mean, out_var, input, optional_running_mean, optional_running_var,
      momentum);
}

std::tuple<at::Tensor, at::Tensor> lstm_cell_exact(
    const at::Tensor& input, c10::ArrayRef<at::Tensor> hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh) {
  return at::lstm_cell(
      input, hx, w_ih, w_hh,
      has_b_ih ? std::optional<at::Tensor>(b_ih) : std::nullopt,
      has_b_hh ? std::optional<at::Tensor>(b_hh) : std::nullopt);
}

at::Tensor rnn_tanh_cell_exact(
    const at::Tensor& input, const at::Tensor& hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh) {
  return at::rnn_tanh_cell(
      input, hx, w_ih, w_hh,
      has_b_ih ? std::optional<at::Tensor>(b_ih) : std::nullopt,
      has_b_hh ? std::optional<at::Tensor>(b_hh) : std::nullopt);
}

at::Tensor rnn_relu_cell_exact(
    const at::Tensor& input, const at::Tensor& hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh) {
  return at::rnn_relu_cell(
      input, hx, w_ih, w_hh,
      has_b_ih ? std::optional<at::Tensor>(b_ih) : std::nullopt,
      has_b_hh ? std::optional<at::Tensor>(b_hh) : std::nullopt);
}

at::Tensor gru_cell_exact(
    const at::Tensor& input, const at::Tensor& hx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, bool has_b_ih,
    const at::Tensor& b_ih, bool has_b_hh, const at::Tensor& b_hh) {
  return at::gru_cell(
      input, hx, w_ih, w_hh,
      has_b_ih ? std::optional<at::Tensor>(b_ih) : std::nullopt,
      has_b_hh ? std::optional<at::Tensor>(b_hh) : std::nullopt);
}

std::tuple<at::Tensor, at::Tensor> rnn_tanh_exact(
    const at::Tensor& input, const at::Tensor& hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first) {
  return at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout,
                      train, bidirectional, batch_first);
}

std::tuple<at::Tensor, at::Tensor> rnn_relu_exact(
    const at::Tensor& input, const at::Tensor& hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first) {
  return at::rnn_relu(input, hx, params, has_biases, num_layers, dropout,
                      train, bidirectional, batch_first);
}

std::tuple<at::Tensor, at::Tensor> gru_exact(
    const at::Tensor& input, const at::Tensor& hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first) {
  return at::gru(input, hx, params, has_biases, num_layers, dropout, train,
                 bidirectional, batch_first);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_exact(
    const at::Tensor& input, c10::ArrayRef<at::Tensor> hx,
    c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first) {
  return at::lstm(input, hx, params, has_biases, num_layers, dropout, train,
                  bidirectional, batch_first);
}

std::tuple<at::Tensor, at::Tensor> rnn_tanh_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    const at::Tensor& hx, c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train, bool bidirectional) {
  return at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers,
                      dropout, train, bidirectional);
}

std::tuple<at::Tensor, at::Tensor> rnn_relu_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    const at::Tensor& hx, c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train, bool bidirectional) {
  return at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers,
                      dropout, train, bidirectional);
}

std::tuple<at::Tensor, at::Tensor> gru_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    const at::Tensor& hx, c10::ArrayRef<at::Tensor> params, bool has_biases,
    std::int64_t num_layers, double dropout, bool train, bool bidirectional) {
  return at::gru(data, batch_sizes, hx, params, has_biases, num_layers,
                 dropout, train, bidirectional);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_data_exact(
    const at::Tensor& data, const at::Tensor& batch_sizes,
    c10::ArrayRef<at::Tensor> hx, c10::ArrayRef<at::Tensor> params,
    bool has_biases, std::int64_t num_layers, double dropout, bool train,
    bool bidirectional) {
  return at::lstm(data, batch_sizes, hx, params, has_biases, num_layers,
                  dropout, train, bidirectional);
}

at::Tensor quantized_batch_norm_exact(
    const at::Tensor& input, bool has_weight, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias, const at::Tensor& mean,
    const at::Tensor& var, double eps, double output_scale,
    std::int64_t output_zero_point) {
  return at::quantized_batch_norm(
      input, has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, mean, var,
      eps, output_scale, output_zero_point);
}

void quantized_batch_norm_exact_out(
    at::Tensor& out, const at::Tensor& input, bool has_weight,
    const at::Tensor& weight, bool has_bias, const at::Tensor& bias,
    const at::Tensor& mean, const at::Tensor& var, double eps,
    double output_scale, std::int64_t output_zero_point) {
  at::quantized_batch_norm_out(
      out, input,
      has_weight ? std::optional<at::Tensor>(weight) : std::nullopt,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, mean, var,
      eps, output_scale, output_zero_point);
}

at::Tensor thnn_conv2d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding) {
  return at::thnn_conv2d(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding);
}

void thnn_conv2d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding) {
  at::thnn_conv2d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output);
}

at::Tensor slow_conv3d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding) {
  return at::slow_conv3d(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding);
}

void slow_conv3d_forward_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding) {
  at::slow_conv3d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output);
}

void slow_conv3d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding) {
  at::slow_conv3d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output);
}

void slow_conv_dilated2d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation) {
  at::slow_conv_dilated2d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, dilation, output);
}

void slow_conv_dilated3d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation) {
  at::slow_conv_dilated3d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, dilation, output);
}

void slow_conv_transpose2d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation) {
  at::slow_conv_transpose2d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output_padding, dilation, output);
}

void slow_conv_transpose3d_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation) {
  at::slow_conv_transpose3d_outf(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output_padding, dilation, output);
}

at::Tensor slow_conv_dilated2d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation) {
  return at::slow_conv_dilated2d(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, dilation);
}

at::Tensor slow_conv_dilated3d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding, c10::ArrayRef<std::int64_t> dilation) {
  return at::slow_conv_dilated3d(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, dilation);
}

at::Tensor slow_conv_transpose2d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation) {
  return at::slow_conv_transpose2d(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output_padding, dilation);
}

at::Tensor slow_conv_transpose3d_exact(
    const at::Tensor& input, const at::Tensor& weight,
    c10::ArrayRef<std::int64_t> kernel_size, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> output_padding,
    c10::ArrayRef<std::int64_t> dilation) {
  return at::slow_conv_transpose3d(
      input, weight, kernel_size,
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt, stride,
      padding, output_padding, dilation);
}


at::Tensor multilabel_margin_loss_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& target, std::int64_t reduction,
    const at::Tensor& is_target) {
  return at::multilabel_margin_loss_backward(
      grad_output, input, target, reduction, is_target);
}

void multilabel_margin_loss_backward_exact_out(
    at::Tensor& grad_input, const at::Tensor& grad_output,
    const at::Tensor& input, const at::Tensor& target,
    std::int64_t reduction, const at::Tensor& is_target) {
  at::multilabel_margin_loss_backward_out(
      grad_input, grad_output, input, target, reduction, is_target);
}

std::tuple<at::Tensor, at::Tensor> multilabel_margin_loss_forward_exact(
    const at::Tensor& input, const at::Tensor& target, std::int64_t reduction) {
  return at::multilabel_margin_loss_forward(input, target, reduction);
}

void multilabel_margin_loss_forward_exact_out(
    at::Tensor& output, at::Tensor& is_target, const at::Tensor& input,
    const at::Tensor& target, std::int64_t reduction) {
  at::multilabel_margin_loss_forward_out(
      output, is_target, input, target, reduction);
}

at::Tensor tensor_nested_to_padded_tensor_exact(
    const at::Tensor& tensor, double padding, bool has_output_size,
    c10::ArrayRef<std::int64_t> output_size) {
  const std::optional<c10::ArrayRef<std::int64_t>> selected_size =
      has_output_size ? std::optional<c10::ArrayRef<std::int64_t>>(output_size)
                      : std::nullopt;
  return at::nested_to_padded_tensor(tensor, padding, selected_size);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, bool has_bias_sizes,
    c10::ArrayRef<std::int64_t> bias_sizes,
    c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias) {
  const at::OptionalIntArrayRef optional_bias_sizes =
      has_bias_sizes ? at::OptionalIntArrayRef(bias_sizes) : std::nullopt;
  return at::convolution_backward(
      grad_output, input, weight, optional_bias_sizes, stride, padding,
      dilation, transposed, output_padding, groups,
      {output_mask_input, output_mask_weight, output_mask_bias});
}

void convolution_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_weight, at::Tensor& grad_bias,
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, bool has_bias_sizes,
    c10::ArrayRef<std::int64_t> bias_sizes,
    c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups,
    bool output_mask_input, bool output_mask_weight, bool output_mask_bias) {
  const at::OptionalIntArrayRef optional_bias_sizes =
      has_bias_sizes ? at::OptionalIntArrayRef(bias_sizes) : std::nullopt;
  at::convolution_backward_out(
      grad_input, grad_weight, grad_bias, grad_output, input, weight,
      optional_bias_sizes, stride, padding, dilation, transposed,
      output_padding, groups,
      {output_mask_input, output_mask_weight, output_mask_bias});
}

at::Tensor convolution_exact(
    const at::Tensor& input, const at::Tensor& weight, bool has_bias,
    const at::Tensor& bias, c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups) {
  const std::optional<at::Tensor> optional_bias =
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt;
  return at::convolution(input, weight, optional_bias, stride, padding,
                         dilation, transposed, output_padding, groups);
}

void convolution_exact_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    bool has_bias, const at::Tensor& bias,
    c10::ArrayRef<std::int64_t> stride,
    c10::ArrayRef<std::int64_t> padding,
    c10::ArrayRef<std::int64_t> dilation, bool transposed,
    c10::ArrayRef<std::int64_t> output_padding, std::int64_t groups) {
  const std::optional<at::Tensor> optional_bias =
      has_bias ? std::optional<at::Tensor>(bias) : std::nullopt;
  at::convolution_out(output, input, weight, optional_bias, stride, padding,
                      dilation, transposed, output_padding, groups);
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid) {
  return at::grid_sampler_2d_backward(
      grad_output, input, grid, interpolation_mode, padding_mode,
      align_corners, {output_mask_input, output_mask_grid});
}

void grid_sampler_2d_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_grid,
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid) {
  at::grid_sampler_2d_backward_out(
      grad_input, grad_grid, grad_output, input, grid, interpolation_mode,
      padding_mode, align_corners, {output_mask_input, output_mask_grid});
}

std::tuple<at::Tensor, at::Tensor> grid_sampler_3d_backward_exact(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid) {
  return at::grid_sampler_3d_backward(
      grad_output, input, grid, interpolation_mode, padding_mode,
      align_corners, {output_mask_input, output_mask_grid});
}

void grid_sampler_3d_backward_exact_out(
    at::Tensor& grad_input, at::Tensor& grad_grid,
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, std::int64_t interpolation_mode,
    std::int64_t padding_mode, bool align_corners, bool output_mask_input,
    bool output_mask_grid) {
  at::grid_sampler_3d_backward_out(
      grad_input, grad_grid, grad_output, input, grid, interpolation_mode,
      padding_mode, align_corners, {output_mask_input, output_mask_grid});
}

at::Tensor tensor_require_grad(const at::Tensor& tensor) {
  if (!tensor.defined() || (!tensor.is_floating_point() && !tensor.is_complex()))
    throw std::invalid_argument("requires_grad needs floating or complex tensor");
  auto result = tensor.detach().clone();
  result.set_requires_grad(true);
  return result;
}

void tensor_backward(const at::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() != 1 || !tensor.requires_grad())
    throw std::invalid_argument("backward requires a scalar requiring-grad tensor");
  tensor.backward();
}

void tensor_backward_with_gradient(const at::Tensor& tensor,
                                   const at::Tensor& gradient,
                                   bool retain_graph, bool create_graph) {
  if (!tensor.defined() || !tensor.requires_grad())
    throw std::invalid_argument(
        "backward requires a defined requiring-grad tensor");
  if (!gradient.defined())
    throw std::invalid_argument("backward gradient must be defined");
  // PyTorch's Python default is retain_graph=None, which resolves to
  // create_graph when higher-order gradients are requested.  Terlan's
  // generated surface carries a concrete Bool default, so preserve that
  // observable default here while still accepting an explicit true value.
  tensor.backward(gradient, retain_graph || create_graph, create_graph);
}

at::Tensor tensor_autograd_grad(const at::Tensor& output,
                                const at::Tensor& input,
                                const at::Tensor& grad_output,
                                bool retain_graph, bool create_graph) {
  if (!output.defined() || !output.requires_grad())
    throw std::invalid_argument(
        "autograd_grad requires a defined requiring-grad output");
  if (!input.defined() || !input.requires_grad())
    throw std::invalid_argument(
        "autograd_grad requires a defined requiring-grad input");
  if (!grad_output.defined())
    throw std::invalid_argument("autograd_grad grad_output must be defined");
  const auto gradients = torch::autograd::grad(
      {output}, {input}, {grad_output}, retain_graph || create_graph,
      create_graph, false);
  if (gradients.empty() || !gradients.front().defined())
    throw std::runtime_error("autograd_grad returned no gradient");
  return gradients.front();
}

std::vector<at::Tensor> tensor_autograd_grad_many(
    c10::ArrayRef<at::Tensor> outputs, c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<at::Tensor> grad_outputs, bool retain_graph,
    bool create_graph) {
  if (outputs.empty() || inputs.empty() || outputs.size() != grad_outputs.size())
    throw std::invalid_argument(
        "autograd_grad_many requires matching non-empty output and seed lists");
  for (const auto& output : outputs) {
    if (!output.defined() || !output.requires_grad())
      throw std::invalid_argument(
          "autograd_grad_many requires requiring-grad outputs");
  }
  for (const auto& input : inputs) {
    if (!input.defined() || !input.requires_grad())
      throw std::invalid_argument(
          "autograd_grad_many requires requiring-grad inputs");
  }
  for (const auto& grad_output : grad_outputs) {
    if (!grad_output.defined())
      throw std::invalid_argument(
          "autograd_grad_many seeds must be defined");
  }
  return torch::autograd::grad(
      std::vector<at::Tensor>(outputs.begin(), outputs.end()),
      std::vector<at::Tensor>(inputs.begin(), inputs.end()),
      std::vector<at::Tensor>(grad_outputs.begin(), grad_outputs.end()),
      retain_graph || create_graph, create_graph, false);
}

at::Tensor tensor_backward_many_token(c10::ArrayRef<at::Tensor> outputs,
                                      c10::ArrayRef<at::Tensor> grad_outputs,
                                      bool retain_graph, bool create_graph) {
  if (outputs.empty() || outputs.size() != grad_outputs.size())
    throw std::invalid_argument(
        "backward_many requires matching non-empty output and seed lists");
  for (const auto& output : outputs) {
    if (!output.defined() || !output.requires_grad())
      throw std::invalid_argument(
          "backward_many requires requiring-grad outputs");
  }
  for (const auto& grad_output : grad_outputs) {
    if (!grad_output.defined())
      throw std::invalid_argument("backward_many seeds must be defined");
  }
  torch::autograd::backward(
      std::vector<at::Tensor>(outputs.begin(), outputs.end()),
      std::vector<at::Tensor>(grad_outputs.begin(), grad_outputs.end()),
      retain_graph || create_graph, create_graph);
  // Return an owned tensor token so the generated CXX list-input adapter can
  // preserve normal handle ownership; the Terlan compatibility alias disposes
  // this token and exposes the public Unit-shaped backward_many contract.
  return outputs.front();
}

at::Tensor tensor_gradient(const at::Tensor& tensor) {
  auto gradient = tensor.grad();
  if (!gradient.defined()) throw std::runtime_error("gradient is unavailable");
  return gradient.detach().clone();
}

at::Tensor tensor_gradient_with_graph(const at::Tensor& tensor) {
  auto gradient = tensor.grad();
  if (!gradient.defined()) throw std::runtime_error("gradient is unavailable");
  return gradient;
}

at::Tensor tensor_sgd_step(const at::Tensor& tensor, double learning_rate) {
  if (!tensor.defined() || !std::isfinite(learning_rate) || learning_rate < 0.0)
    throw std::invalid_argument("invalid SGD step");
  auto gradient = tensor.grad();
  if (!gradient.defined()) throw std::runtime_error("gradient is unavailable");
  torch::NoGradGuard no_grad;
  auto result = (tensor - learning_rate * gradient).detach();
  result.set_requires_grad(true);
  return result;
}

at::Tensor tensor_categorical_sample(const at::Tensor& probabilities,
                                     std::int64_t seed) {
  if (seed < 0 || !probabilities.defined() || !probabilities.is_floating_point() ||
      probabilities.dim() == 0 || probabilities.size(-1) <= 0 || probabilities.numel() == 0)
    throw std::invalid_argument("invalid categorical-sample input");
  auto cpu = probabilities.to(at::kCPU);
  auto generator = at::make_generator<at::CPUGeneratorImpl>();
  generator.set_current_seed(static_cast<std::uint64_t>(seed));
  auto result = at::multinomial(cpu, 1, false, generator);
  return result.to(probabilities.device());
}

at::Tensor tensor_mse_loss_gradient(const at::Tensor& grad_output,
                                    const at::Tensor& predicted,
                                    const at::Tensor& target,
                                    std::int64_t reduction) {
  return at::mse_loss_backward(grad_output, predicted, target, reduction);
}

at::Tensor tensor_cross_entropy_loss(const at::Tensor& logits,
                                     const at::Tensor& target,
                                     std::int64_t reduction,
                                     std::int64_t ignore_index,
                                     double label_smoothing) {
  return at::cross_entropy_loss(logits, target, std::nullopt, reduction,
                                ignore_index, label_smoothing);
}

}  // namespace at::native
