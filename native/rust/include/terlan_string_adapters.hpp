#pragma once

#include "libtorch_bindings.hpp"

#include <memory>
#include <string>

namespace at::native {

std::unique_ptr<std::string> terlan_string_pytorch_tensor_device(const at::Tensor& value) noexcept;

}  // namespace at::native
