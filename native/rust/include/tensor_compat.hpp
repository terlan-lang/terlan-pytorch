#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>

#if __has_include("rust/cxx.h")
#include "rust/cxx.h"
#else
namespace rust { inline namespace cxxbridge1 { template <typename T> class Slice; } }
#endif

namespace at::native {

at::Tensor tensor_from_int(std::int64_t value);
at::Tensor tensor_from_float(double value);
at::Tensor tensor_from_bool(bool value);
at::Tensor tensor_from_floats(rust::Slice<const double> values,
                              c10::ArrayRef<std::int64_t> shape);
at::Tensor tensor_from_ints(c10::ArrayRef<std::int64_t> values,
                            c10::ArrayRef<std::int64_t> shape);
at::Tensor tensor_full_float(c10::ArrayRef<std::int64_t> shape, double value);
at::Tensor tensor_full_typed(c10::ArrayRef<std::int64_t> shape, double value,
                             std::int64_t dtype, std::int64_t device_type,
                             std::int64_t device_index);
at::Tensor tensor_empty_typed(c10::ArrayRef<std::int64_t> shape,
                              std::int64_t dtype, std::int64_t device_type,
                              std::int64_t device_index);
at::Tensor tensor_arange_typed(double start, double end, double step,
                               std::int64_t dtype, std::int64_t device_type,
                               std::int64_t device_index, bool pin_memory);
at::Tensor tensor_linspace_typed(double start, double end, std::int64_t steps,
                                 std::int64_t dtype, std::int64_t device_type,
                                 std::int64_t device_index, bool pin_memory);
at::Tensor tensor_logspace_typed(double start, double end, std::int64_t steps,
                                 double base, std::int64_t dtype,
                                 std::int64_t device_type,
                                 std::int64_t device_index, bool pin_memory);
at::Tensor tensor_eye_typed(std::int64_t rows, std::int64_t columns,
                            std::int64_t dtype, std::int64_t device_type,
                            std::int64_t device_index, bool pin_memory);
std::int64_t tensor_item_int(const at::Tensor& tensor);
double tensor_item_float(const at::Tensor& tensor);
bool tensor_item_bool(const at::Tensor& tensor);
std::int64_t tensor_storage_offset(const at::Tensor& tensor);
std::int64_t tensor_dtype(const at::Tensor& tensor);
std::int64_t tensor_device_type(const at::Tensor& tensor);
std::int64_t tensor_device_index(const at::Tensor& tensor);
at::Tensor tensor_to_device(const at::Tensor& tensor, std::int64_t device_type,
                            std::int64_t device_index);
at::Tensor tensor_to_dtype(const at::Tensor& tensor, std::int64_t dtype);
at::Tensor tensor_to_other(const at::Tensor& tensor, const at::Tensor& other,
                           bool non_blocking);
at::Tensor tensor_to_device_dtype(const at::Tensor& tensor, std::int64_t device_type,
                                  std::int64_t device_index, std::int64_t dtype,
                                  bool non_blocking, bool copy);
at::Tensor tensor_to_dtype_layout(
    const at::Tensor& tensor, bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout, bool has_device,
    std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory, bool non_blocking, bool copy,
    bool has_memory_format, std::int64_t memory_format);
at::Tensor tensor_nested_from_tensor_list(c10::ArrayRef<at::Tensor> tensors);
at::Tensor tensor_to_padded_tensor(const at::Tensor& tensor, double padding,
                                   bool has_output_size,
                                   c10::ArrayRef<std::int64_t> output_size);
at::Tensor& tensor_to_padded_tensor_out(
    at::Tensor& output, const at::Tensor& tensor, double padding,
    bool has_output_size, c10::ArrayRef<std::int64_t> output_size);
at::Tensor tensor_from_file(
    std::string_view filename, bool has_shared, bool shared,
    bool has_size, std::int64_t size, bool has_dtype, std::int64_t dtype,
    bool has_layout, std::int64_t layout, std::int64_t has_device,
    std::int64_t device_type, std::int64_t device_index,
    bool has_pin_memory, bool pin_memory);
at::Tensor& tensor_from_file_out(
    at::Tensor& output, std::string_view filename, bool has_shared,
    bool shared, bool has_size, std::int64_t size);
void sym_constrain_range_exact(bool scalar_is_integer, std::int64_t scalar_integer,
                               double scalar_floating, bool has_min,
                               std::int64_t min, bool has_max,
                               std::int64_t max);
void sym_constrain_range_for_size_exact(
    bool scalar_is_integer, std::int64_t scalar_integer, double scalar_floating,
    bool has_min, std::int64_t min, bool has_max, std::int64_t max);
at::Tensor fused_moving_avg_obs_fake_quant_exact(
    const at::Tensor& self, const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on, const at::Tensor& running_min,
    const at::Tensor& running_max, const at::Tensor& scale,
    const at::Tensor& zero_point, double averaging_const,
    std::int64_t quant_min, std::int64_t quant_max, std::int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant);
at::Tensor tensor_index(const at::Tensor& tensor,
                        c10::ArrayRef<at::Tensor> indices,
                        c10::ArrayRef<std::int64_t> present);
at::Tensor& tensor_index_out(at::Tensor& output, const at::Tensor& tensor,
                             c10::ArrayRef<at::Tensor> indices,
                             c10::ArrayRef<std::int64_t> present);
at::Tensor tensor_index_put(const at::Tensor& tensor,
                            c10::ArrayRef<at::Tensor> indices,
                            c10::ArrayRef<std::int64_t> present,
                            const at::Tensor& values, bool accumulate);
at::Tensor& tensor_index_put_out(at::Tensor& output, const at::Tensor& tensor,
                                 c10::ArrayRef<at::Tensor> indices,
                                 c10::ArrayRef<std::int64_t> present,
                                 const at::Tensor& values, bool accumulate);
at::Tensor& tensor_index_put_in_place(
    at::Tensor& tensor, c10::ArrayRef<at::Tensor> indices,
    c10::ArrayRef<std::int64_t> present, const at::Tensor& values, bool accumulate);
at::Tensor tensor_index(const at::Tensor& tensor,
                        c10::ArrayRef<at::Tensor> indices,
                        c10::ArrayRef<bool> present);
at::Tensor& tensor_index_out(at::Tensor& output, const at::Tensor& tensor,
                             c10::ArrayRef<at::Tensor> indices,
                             c10::ArrayRef<bool> present);
at::Tensor tensor_index_put(const at::Tensor& tensor,
                            c10::ArrayRef<at::Tensor> indices,
                            c10::ArrayRef<bool> present,
                            const at::Tensor& values, bool accumulate);
at::Tensor& tensor_index_put_out(at::Tensor& output, const at::Tensor& tensor,
                                 c10::ArrayRef<at::Tensor> indices,
                                 c10::ArrayRef<bool> present,
                                 const at::Tensor& values, bool accumulate);
at::Tensor& tensor_index_put_in_place(
    at::Tensor& tensor, c10::ArrayRef<at::Tensor> indices,
    c10::ArrayRef<bool> present, const at::Tensor& values, bool accumulate);
std::int64_t tensor_qscheme(const at::Tensor& tensor);
std::vector<at::Tensor> tensor_gradient_array(const at::Tensor& tensor,
                                               c10::ArrayRef<std::int64_t> dim,
                                               std::int64_t edge_order);
std::vector<at::Tensor> tensor_gradient_scalar_array(
    const at::Tensor& tensor, double spacing,
    c10::ArrayRef<std::int64_t> dim, std::int64_t edge_order);
std::vector<at::Tensor> tensor_gradient_scalar_int(
    const at::Tensor& tensor, bool has_spacing, double spacing,
    bool has_dim, std::int64_t dim, std::int64_t edge_order);
std::vector<at::Tensor> tensor_gradient_scalar_array_array(
    const at::Tensor& tensor, rust::Slice<const double> spacing,
    c10::ArrayRef<std::int64_t> dim, std::int64_t edge_order);
std::vector<at::Tensor> tensor_gradient_scalar_array_int(
    const at::Tensor& tensor, rust::Slice<const double> spacing,
    bool has_dim, std::int64_t dim, std::int64_t edge_order);
std::vector<at::Tensor> tensor_gradient_tensor_array(
    const at::Tensor& tensor, c10::ArrayRef<at::Tensor> spacing,
    c10::ArrayRef<std::int64_t> dim, std::int64_t edge_order);
std::vector<at::Tensor> tensor_gradient_tensor_array_int(
    const at::Tensor& tensor, c10::ArrayRef<at::Tensor> spacing,
    bool has_dim, std::int64_t dim, std::int64_t edge_order);

}  // namespace at::native
