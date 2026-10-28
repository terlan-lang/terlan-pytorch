#pragma once

#include "libtorch_bindings.hpp"

#include <memory>
#include <string>

namespace at::native {

std::unique_ptr<std::string> terlan_enum_pytorch_tensor_dtype(const at::Tensor& value) noexcept;
std::unique_ptr<std::string> terlan_enum_pytorch_tensor_layout(const at::Tensor& value) noexcept;

}  // namespace at::native
