#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>
#include "rust/cxx.h"
#include "libtorch_bindings.hpp"

namespace at::native {

struct TorchScriptModel {
  explicit TorchScriptModel(torch::jit::Module&& value) : module(std::move(value)) {}
  torch::jit::Module module;
};

std::unique_ptr<TorchScriptModel> tensor_load_torchscript(rust::Str path);
std::unique_ptr<at::Tensor> tensor_torchscript_forward(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<at::Tensor> tensor_torchscript_forward2(const TorchScriptModel& model, const at::Tensor& first, const at::Tensor& second);
TensorPair tensor_torchscript_forward_pair(const TorchScriptModel& model, const at::Tensor& input);
double tensor_torchscript_forward_float(const TorchScriptModel& model, const at::Tensor& input);
std::int64_t tensor_torchscript_forward_int(const TorchScriptModel& model, const at::Tensor& input);
bool tensor_torchscript_forward_bool(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<std::string> tensor_torchscript_forward_string(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<std::vector<std::int64_t>> tensor_torchscript_forward_int_list(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<std::vector<double>> tensor_torchscript_forward_float_list(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<std::vector<std::uint8_t>> tensor_torchscript_forward_bool_list(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<std::vector<std::string>> tensor_torchscript_forward_string_list(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<at::Tensor> tensor_torchscript_forward_optional_tensor(const TorchScriptModel& model, const at::Tensor& input);
std::vector<at::Tensor> tensor_torchscript_forward_tensor_list(const TorchScriptModel& model, const at::Tensor& input);
std::unique_ptr<at::Tensor> tensor_run_torchscript(const at::Tensor& input, rust::Str path);
std::unique_ptr<at::Tensor> tensor_pair_first(const TensorPair& pair);
std::unique_ptr<at::Tensor> tensor_pair_second(const TensorPair& pair);

}  // namespace at::native
