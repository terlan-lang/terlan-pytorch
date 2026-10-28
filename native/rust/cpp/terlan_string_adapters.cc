#include "include/terlan_string_adapters.hpp"

namespace at::native {

std::unique_ptr<std::string> terlan_string_pytorch_tensor_device(const at::Tensor& value) noexcept {
  try {
    const auto projected = value.device();
    return std::make_unique<std::string>(projected.str());
  } catch (...) {
    return nullptr;
  }
}

}  // namespace at::native
