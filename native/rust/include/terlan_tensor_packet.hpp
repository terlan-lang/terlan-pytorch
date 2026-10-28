#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <ATen/ATen.h>

#if __has_include("rust/cxx.h")
#include "rust/cxx.h"
#else
namespace rust {
inline namespace cxxbridge1 {
template <typename T>
class Slice;
}  // namespace cxxbridge1
}  // namespace rust
#endif

namespace terlan::tensor_packet {

/**
 * Validates and copies one pointer-free TNXP value into independently owned
 * LibTorch storage.
 */
std::unique_ptr<at::Tensor> import_tensor_packet(
    rust::Slice<const std::uint8_t> packet) noexcept;

/**
 * Copies one tensor through a contiguous CPU snapshot into a pointer-free TNXP
 * value. CUDA synchronization is completed before this function returns.
 */
std::unique_ptr<std::vector<std::uint8_t>> export_tensor_packet(
    const at::Tensor& tensor) noexcept;

}  // namespace terlan::tensor_packet
