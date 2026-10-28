#include "tensor_compat.hpp"

#include <stdexcept>

#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/full.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/_nested_tensor_from_tensor_list.h>
#include <ATen/ops/from_file.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/to_padded_tensor.h>
#include <ATen/ops/sym_constrain_range.h>
#include <ATen/ops/sym_constrain_range_for_size.h>

namespace at::native {
namespace {

c10::TensorOptions options(std::int64_t dtype, std::int64_t device_type,
                            std::int64_t device_index) {
  if (device_type == static_cast<std::int64_t>(c10::DeviceType::CPU)) {
    if (device_index != -1) throw std::invalid_argument("CPU device index must be -1");
    return c10::TensorOptions().dtype(static_cast<c10::ScalarType>(dtype));
  }
  if (device_type != static_cast<std::int64_t>(c10::DeviceType::CUDA) || device_index < 0)
    throw std::invalid_argument("invalid device");
  return c10::TensorOptions()
      .dtype(static_cast<c10::ScalarType>(dtype))
      .device(c10::Device(static_cast<c10::DeviceType>(device_type), device_index));
}

void validate_shape(c10::ArrayRef<std::int64_t> shape) {
  for (auto value : shape) if (value < 0) throw std::invalid_argument("negative tensor dimension");
}

}  // namespace

at::Tensor tensor_from_int(std::int64_t value) {
  return at::scalar_tensor(value, c10::TensorOptions().dtype(at::kLong));
}
at::Tensor tensor_from_float(double value) {
  return at::scalar_tensor(value, c10::TensorOptions().dtype(at::kDouble));
}
at::Tensor tensor_from_bool(bool value) {
  return at::scalar_tensor(value, c10::TensorOptions().dtype(at::kBool));
}

at::Tensor tensor_from_floats(rust::Slice<const double> values,
                              c10::ArrayRef<std::int64_t> shape) {
  validate_shape(shape);
  auto result = at::tensor(c10::ArrayRef<double>(values.data(), values.size()),
                           c10::TensorOptions().dtype(at::kDouble));
  return result.reshape(shape);
}
at::Tensor tensor_from_ints(c10::ArrayRef<std::int64_t> values,
                            c10::ArrayRef<std::int64_t> shape) {
  validate_shape(shape);
  auto result = at::tensor(values, c10::TensorOptions().dtype(at::kLong));
  return result.reshape(shape);
}
at::Tensor tensor_full_float(c10::ArrayRef<std::int64_t> shape, double value) {
  validate_shape(shape);
  return at::full(shape, value, c10::TensorOptions().dtype(at::kDouble));
}
at::Tensor tensor_full_typed(c10::ArrayRef<std::int64_t> shape, double value,
                             std::int64_t dtype, std::int64_t device_type,
                             std::int64_t device_index) {
  validate_shape(shape); return at::full(shape, value, options(dtype, device_type, device_index));
}
at::Tensor tensor_empty_typed(c10::ArrayRef<std::int64_t> shape,
                              std::int64_t dtype, std::int64_t device_type,
                              std::int64_t device_index) {
  validate_shape(shape); return at::empty(shape, options(dtype, device_type, device_index));
}
at::Tensor tensor_arange_typed(double start, double end, double step,
                               std::int64_t dtype, std::int64_t device_type,
                               std::int64_t device_index, bool pin_memory) {
  if (step == 0.0) throw std::invalid_argument("arange step cannot be zero");
  auto opts = options(dtype, device_type, device_index).pinned_memory(pin_memory);
  return at::arange(start, end, step, opts);
}
at::Tensor tensor_linspace_typed(double start, double end, std::int64_t steps,
                                 std::int64_t dtype, std::int64_t device_type,
                                 std::int64_t device_index, bool pin_memory) {
  if (steps < 0) throw std::invalid_argument("linspace steps must be nonnegative");
  auto opts = options(dtype, device_type, device_index).pinned_memory(pin_memory);
  return at::linspace(start, end, steps, opts);
}
at::Tensor tensor_logspace_typed(double start, double end, std::int64_t steps,
                                 double base, std::int64_t dtype,
                                 std::int64_t device_type,
                                 std::int64_t device_index, bool pin_memory) {
  if (steps < 0) throw std::invalid_argument("logspace steps must be nonnegative");
  auto opts = options(dtype, device_type, device_index).pinned_memory(pin_memory);
  return at::logspace(start, end, steps, base, opts);
}
at::Tensor tensor_eye_typed(std::int64_t rows, std::int64_t columns,
                            std::int64_t dtype, std::int64_t device_type,
                            std::int64_t device_index, bool pin_memory) {
  if (rows < 0 || columns < 0) throw std::invalid_argument("eye dimensions must be nonnegative");
  auto opts = options(dtype, device_type, device_index).pinned_memory(pin_memory);
  return at::eye(rows, columns, opts);
}

std::int64_t tensor_item_int(const at::Tensor& tensor) { return tensor.item<std::int64_t>(); }
double tensor_item_float(const at::Tensor& tensor) { return tensor.item<double>(); }
bool tensor_item_bool(const at::Tensor& tensor) { return tensor.item<bool>(); }
std::int64_t tensor_storage_offset(const at::Tensor& tensor) { return tensor.storage_offset(); }
std::int64_t tensor_dtype(const at::Tensor& tensor) { return static_cast<std::int64_t>(tensor.scalar_type()); }
std::int64_t tensor_device_type(const at::Tensor& tensor) { return static_cast<std::int64_t>(tensor.device().type()); }
std::int64_t tensor_device_index(const at::Tensor& tensor) { return tensor.device().index(); }
at::Tensor tensor_to_device(const at::Tensor& tensor, std::int64_t device_type,
                            std::int64_t device_index) {
  return tensor.to(c10::Device(static_cast<c10::DeviceType>(device_type), device_index));
}
at::Tensor tensor_to_dtype(const at::Tensor& tensor, std::int64_t dtype) {
  if (dtype < 0 || dtype >= static_cast<std::int64_t>(c10::ScalarType::NumOptions)) {
    throw std::invalid_argument("unsupported dtype code");
  }
  return tensor.to(static_cast<c10::ScalarType>(dtype));
}
at::Tensor tensor_to_other(const at::Tensor& tensor, const at::Tensor& other,
                           bool non_blocking) {
  return tensor.to(other, non_blocking, false, std::nullopt);
}
at::Tensor tensor_to_device_dtype(const at::Tensor& tensor, std::int64_t device_type,
                                  std::int64_t device_index, std::int64_t dtype,
                                  bool non_blocking, bool copy) {
  auto device = c10::Device(static_cast<c10::DeviceType>(device_type), device_index);
  return tensor.to(device, static_cast<c10::ScalarType>(dtype), non_blocking, copy,
                   std::nullopt);
}
at::Tensor tensor_to_dtype_layout(
    const at::Tensor& tensor, bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout, bool has_device,
    std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory, bool non_blocking, bool copy,
    bool has_memory_format, std::int64_t memory_format) {
  auto selected_dtype = has_dtype
      ? std::optional<c10::ScalarType>(static_cast<c10::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<c10::Layout>(static_cast<c10::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device != 0
      ? std::optional<c10::Device>(c10::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  auto selected_memory_format = has_memory_format
      ? std::optional<c10::MemoryFormat>(
            static_cast<c10::MemoryFormat>(memory_format))
      : std::nullopt;
  return tensor.to(selected_dtype, selected_layout, selected_device,
                   selected_pin_memory, non_blocking, copy,
                   selected_memory_format);
}
at::Tensor tensor_nested_from_tensor_list(c10::ArrayRef<at::Tensor> tensors) {
  return at::_nested_tensor_from_tensor_list(tensors);
}
at::Tensor tensor_to_padded_tensor(const at::Tensor& tensor, double padding,
                                   bool has_output_size,
                                   c10::ArrayRef<std::int64_t> output_size) {
  auto selected_size = has_output_size
      ? std::optional<c10::ArrayRef<std::int64_t>>(output_size)
      : std::nullopt;
  return at::symint::to_padded_tensor<std::int64_t>(tensor, padding,
                                                    selected_size);
}
at::Tensor& tensor_to_padded_tensor_out(
    at::Tensor& output, const at::Tensor& tensor, double padding,
    bool has_output_size, c10::ArrayRef<std::int64_t> output_size) {
  auto selected_size = has_output_size
      ? std::optional<c10::ArrayRef<std::int64_t>>(output_size)
      : std::nullopt;
  // LibTorch 2.13's generated nested-tensor `.out` kernel is registered but
  // fails on the CPU distribution. Preserve the public retained-output
  // contract by using the executable functional kernel, then materializing
  // its result into the caller-owned storage.
  auto padded = at::symint::to_padded_tensor<std::int64_t>(
      tensor, padding, selected_size);
  output.resize_(padded.sizes());
  output.copy_(padded);
  return output;
}
at::Tensor tensor_from_file(
    std::string_view filename, bool has_shared, bool shared,
    bool has_size, std::int64_t size, bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout, std::int64_t has_device,
    std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory) {
  auto selected_shared = has_shared ? std::optional<bool>(shared) : std::nullopt;
  auto selected_size = has_size ? std::optional<std::int64_t>(size) : std::nullopt;
  auto selected_dtype = has_dtype
      ? std::optional<c10::ScalarType>(static_cast<c10::ScalarType>(dtype))
      : std::nullopt;
  auto selected_layout = has_layout
      ? std::optional<c10::Layout>(static_cast<c10::Layout>(layout))
      : std::nullopt;
  auto selected_device = has_device
      ? std::optional<c10::Device>(c10::Device(
            static_cast<c10::DeviceType>(device_type), device_index))
      : std::nullopt;
  auto selected_pin_memory = has_pin_memory
      ? std::optional<bool>(pin_memory)
      : std::nullopt;
  return at::from_file(filename, selected_shared, selected_size, selected_dtype,
                       selected_layout, selected_device, selected_pin_memory);
}
at::Tensor& tensor_from_file_out(
    at::Tensor& output, std::string_view filename, bool has_shared,
    bool shared, bool has_size, std::int64_t size) {
  auto selected_shared = has_shared ? std::optional<bool>(shared) : std::nullopt;
  auto selected_size = has_size ? std::optional<std::int64_t>(size) : std::nullopt;
  return at::from_file_outf(filename, selected_shared, selected_size, output);
}

void sym_constrain_range_exact(bool scalar_is_integer, std::int64_t scalar_integer,
                               double scalar_floating, bool has_min,
                               std::int64_t min, bool has_max,
                               std::int64_t max) {
  at::sym_constrain_range(
      scalar_is_integer ? at::Scalar(scalar_integer) : at::Scalar(scalar_floating),
      has_min ? std::optional<std::int64_t>(min) : std::nullopt,
      has_max ? std::optional<std::int64_t>(max) : std::nullopt);
}

void sym_constrain_range_for_size_exact(
    bool scalar_is_integer, std::int64_t scalar_integer, double scalar_floating,
    bool has_min, std::int64_t min, bool has_max, std::int64_t max) {
  at::sym_constrain_range_for_size(
      scalar_is_integer ? at::Scalar(scalar_integer) : at::Scalar(scalar_floating),
      has_min ? std::optional<std::int64_t>(min) : std::nullopt,
      has_max ? std::optional<std::int64_t>(max) : std::nullopt);
}

at::Tensor fused_moving_avg_obs_fake_quant_exact(
    const at::Tensor& self, const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on, const at::Tensor& running_min,
    const at::Tensor& running_max, const at::Tensor& scale,
    const at::Tensor& zero_point, double averaging_const,
    std::int64_t quant_min, std::int64_t quant_max, std::int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
  // Tensor handles are cheap shared references. Local non-const handles let
  // the upstream kernel retain its in-out reference contract while mutations
  // remain visible through the caller-owned handles.
  auto running_min_mut = running_min;
  auto running_max_mut = running_max;
  auto scale_mut = scale;
  auto zero_point_mut = zero_point;
  return at::fused_moving_avg_obs_fake_quant(
      self, observer_on, fake_quant_on, running_min_mut, running_max_mut,
      scale_mut, zero_point_mut, averaging_const, quant_min, quant_max, ch_axis,
      per_row_fake_quant, symmetric_quant);
}

namespace {

c10::List<std::optional<at::Tensor>> optional_indices(
    c10::ArrayRef<at::Tensor> indices, c10::ArrayRef<std::int64_t> present) {
  if (indices.size() != present.size())
    throw std::invalid_argument("index presence mask length mismatch");
  c10::List<std::optional<at::Tensor>> result;
  for (std::size_t i = 0; i < indices.size(); ++i)
    result.push_back(present[i] != 0 ? std::optional<at::Tensor>(indices[i])
                                : std::nullopt);
  return result;
}

}  // namespace

at::Tensor tensor_index(const at::Tensor& tensor,
                        c10::ArrayRef<at::Tensor> indices,
                        c10::ArrayRef<std::int64_t> present) {
  return at::index(tensor, optional_indices(indices, present));
}

at::Tensor& tensor_index_out(at::Tensor& output, const at::Tensor& tensor,
                             c10::ArrayRef<at::Tensor> indices,
                             c10::ArrayRef<std::int64_t> present) {
  return at::index_out(output, tensor, optional_indices(indices, present));
}

at::Tensor tensor_index_put(const at::Tensor& tensor,
                            c10::ArrayRef<at::Tensor> indices,
                            c10::ArrayRef<std::int64_t> present,
                            const at::Tensor& values, bool accumulate) {
  return at::index_put(tensor, optional_indices(indices, present), values,
                       accumulate);
}

at::Tensor& tensor_index_put_out(at::Tensor& output, const at::Tensor& tensor,
                                 c10::ArrayRef<at::Tensor> indices,
                                 c10::ArrayRef<std::int64_t> present,
                                 const at::Tensor& values, bool accumulate) {
  return at::index_put_out(output, tensor, optional_indices(indices, present),
                           values, accumulate);
}

at::Tensor& tensor_index_put_in_place(
    at::Tensor& tensor, c10::ArrayRef<at::Tensor> indices,
    c10::ArrayRef<std::int64_t> present, const at::Tensor& values, bool accumulate) {
  return at::index_put_(tensor, optional_indices(indices, present), values,
                        accumulate);
}

namespace {
std::vector<std::int64_t> present_as_ints(c10::ArrayRef<bool> present) {
  std::vector<std::int64_t> result;
  result.reserve(present.size());
  for (bool value : present) result.push_back(value ? 1 : 0);
  return result;
}
}  // namespace

at::Tensor tensor_index(const at::Tensor& tensor,
                        c10::ArrayRef<at::Tensor> indices,
                        c10::ArrayRef<bool> present) {
  auto values = present_as_ints(present);
  return tensor_index(tensor, indices, values);
}
at::Tensor& tensor_index_out(at::Tensor& output, const at::Tensor& tensor,
                             c10::ArrayRef<at::Tensor> indices,
                             c10::ArrayRef<bool> present) {
  auto values = present_as_ints(present);
  return tensor_index_out(output, tensor, indices, values);
}
at::Tensor tensor_index_put(const at::Tensor& tensor,
                            c10::ArrayRef<at::Tensor> indices,
                            c10::ArrayRef<bool> present,
                            const at::Tensor& values, bool accumulate) {
  auto mask = present_as_ints(present);
  return tensor_index_put(tensor, indices, mask, values, accumulate);
}
at::Tensor& tensor_index_put_out(at::Tensor& output, const at::Tensor& tensor,
                                 c10::ArrayRef<at::Tensor> indices,
                                 c10::ArrayRef<bool> present,
                                 const at::Tensor& values, bool accumulate) {
  auto mask = present_as_ints(present);
  return tensor_index_put_out(output, tensor, indices, mask, values, accumulate);
}
at::Tensor& tensor_index_put_in_place(
    at::Tensor& tensor, c10::ArrayRef<at::Tensor> indices,
    c10::ArrayRef<bool> present, const at::Tensor& values, bool accumulate) {
  auto mask = present_as_ints(present);
  return tensor_index_put_in_place(tensor, indices, mask, values, accumulate);
}

std::int64_t tensor_qscheme(const at::Tensor& tensor) {
  return static_cast<std::int64_t>(tensor.qscheme());
}
std::vector<at::Tensor> tensor_gradient_array(const at::Tensor& tensor,
                                               c10::ArrayRef<std::int64_t> dim,
                                               std::int64_t edge_order) {
  return at::gradient(tensor, dim, edge_order);
}
std::vector<at::Tensor> tensor_gradient_scalar_array(
    const at::Tensor& tensor, double spacing,
    c10::ArrayRef<std::int64_t> dim, std::int64_t edge_order) {
  return at::gradient(tensor, c10::Scalar(spacing), dim, edge_order);
}
std::vector<at::Tensor> tensor_gradient_scalar_int(
    const at::Tensor& tensor, bool has_spacing, double spacing,
    bool has_dim, std::int64_t dim, std::int64_t edge_order) {
  auto selected_spacing = has_spacing ? std::optional<at::Scalar>(c10::Scalar(spacing))
                                      : std::nullopt;
  auto selected_dim = has_dim ? std::optional<std::int64_t>(dim) : std::nullopt;
  return at::gradient(tensor, selected_spacing, selected_dim, edge_order);
}
std::vector<at::Tensor> tensor_gradient_scalar_array_array(
    const at::Tensor& tensor, rust::Slice<const double> spacing,
    c10::ArrayRef<std::int64_t> dim, std::int64_t edge_order) {
  std::vector<at::Scalar> scalars;
  scalars.reserve(spacing.size());
  for (double value : spacing) scalars.emplace_back(value);
  return at::gradient(tensor, c10::ArrayRef<at::Scalar>(scalars), dim, edge_order);
}
std::vector<at::Tensor> tensor_gradient_scalar_array_int(
    const at::Tensor& tensor, rust::Slice<const double> spacing,
    bool has_dim, std::int64_t dim, std::int64_t edge_order) {
  std::vector<at::Scalar> scalars;
  scalars.reserve(spacing.size());
  for (double value : spacing) scalars.emplace_back(value);
  auto selected_dim = has_dim ? std::optional<std::int64_t>(dim) : std::nullopt;
  return at::gradient(tensor, c10::ArrayRef<at::Scalar>(scalars), selected_dim, edge_order);
}
std::vector<at::Tensor> tensor_gradient_tensor_array(
    const at::Tensor& tensor, c10::ArrayRef<at::Tensor> spacing,
    c10::ArrayRef<std::int64_t> dim, std::int64_t edge_order) {
  return at::gradient(tensor, spacing, dim, edge_order);
}
std::vector<at::Tensor> tensor_gradient_tensor_array_int(
    const at::Tensor& tensor, c10::ArrayRef<at::Tensor> spacing,
    bool has_dim, std::int64_t dim, std::int64_t edge_order) {
  auto selected_dim = has_dim ? std::optional<std::int64_t>(dim) : std::nullopt;
  return at::gradient(tensor, spacing, selected_dim, edge_order);
}

}  // namespace at::native
