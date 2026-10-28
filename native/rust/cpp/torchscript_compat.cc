#include "torchscript_compat.hpp"

#include <initializer_list>
#include <stdexcept>

namespace at::native {

namespace {
torch::jit::IValue forward(const TorchScriptModel& model,
                           std::initializer_list<torch::jit::IValue> inputs) {
  // LibTorch's JIT forward mutates execution state despite the logical
  // immutable model contract exposed to Terlan; keep the resource handle
  // immutable while applying the library's required non-const call.
  return const_cast<torch::jit::Module&>(model.module).forward(inputs);
}
}

std::unique_ptr<TorchScriptModel> tensor_load_torchscript(rust::Str path) {
  auto module = torch::jit::load(std::string(path.data(), path.size()));
  module.eval();
  return std::make_unique<TorchScriptModel>(std::move(module));
}

std::unique_ptr<at::Tensor> tensor_torchscript_forward(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isTensor()) throw std::runtime_error("TorchScript output is not a tensor");
  return std::make_unique<at::Tensor>(result.toTensor());
}

std::unique_ptr<at::Tensor> tensor_torchscript_forward2(const TorchScriptModel& model, const at::Tensor& first, const at::Tensor& second) {
  auto result = forward(model, {torch::jit::IValue(first), torch::jit::IValue(second)});
  if (!result.isTensor()) throw std::runtime_error("TorchScript output is not a tensor");
  return std::make_unique<at::Tensor>(result.toTensor());
}

TensorPair tensor_torchscript_forward_pair(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isTuple() || result.toTuple()->elements().size() != 2) throw std::runtime_error("TorchScript output is not a tensor pair");
  const auto& values = result.toTuple()->elements();
  if (!values[0].isTensor() || !values[1].isTensor()) throw std::runtime_error("TorchScript pair contains a non-tensor");
  return {values[0].toTensor(), values[1].toTensor()};
}

double tensor_torchscript_forward_float(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isDouble()) throw std::runtime_error("TorchScript output is not a float");
  return result.toDouble();
}

std::int64_t tensor_torchscript_forward_int(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isInt()) throw std::runtime_error("TorchScript output is not an integer");
  return result.toInt();
}

bool tensor_torchscript_forward_bool(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isBool()) throw std::runtime_error("TorchScript output is not a bool");
  return result.toBool();
}

std::unique_ptr<std::string> tensor_torchscript_forward_string(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isString()) throw std::runtime_error("TorchScript output is not a string");
  return std::make_unique<std::string>(result.toStringRef());
}

std::unique_ptr<std::vector<std::int64_t>> tensor_torchscript_forward_int_list(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isIntList()) throw std::runtime_error("TorchScript output is not an integer list");
  return std::make_unique<std::vector<std::int64_t>>(result.toIntList().vec());
}

std::unique_ptr<std::vector<double>> tensor_torchscript_forward_float_list(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isDoubleList()) throw std::runtime_error("TorchScript output is not a float list");
  return std::make_unique<std::vector<double>>(result.toDoubleList().vec());
}

std::unique_ptr<std::vector<std::uint8_t>> tensor_torchscript_forward_bool_list(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isBoolList()) throw std::runtime_error("TorchScript output is not a boolean list");
  const auto values = result.toBoolList();
  auto output = std::make_unique<std::vector<std::uint8_t>>();
  output->reserve(values.size());
  for (const bool value : values) output->push_back(value ? 1 : 0);
  return output;
}

std::unique_ptr<std::vector<std::string>> tensor_torchscript_forward_string_list(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isList()) throw std::runtime_error("TorchScript output is not a string list");
  const auto values = result.toList();
  auto output = std::make_unique<std::vector<std::string>>();
  output->reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    const auto value = values.get(index);
    if (!value.isString()) throw std::runtime_error("TorchScript string list contains a non-string");
    output->push_back(value.toStringRef());
  }
  return output;
}

std::unique_ptr<at::Tensor> tensor_torchscript_forward_optional_tensor(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (result.isNone()) return nullptr;
  if (!result.isTensor()) throw std::runtime_error("TorchScript optional output is not a tensor or None");
  return std::make_unique<at::Tensor>(result.toTensor());
}

std::vector<at::Tensor> tensor_torchscript_forward_tensor_list(const TorchScriptModel& model, const at::Tensor& input) {
  auto result = forward(model, {torch::jit::IValue(input)});
  if (!result.isTensorList()) throw std::runtime_error("TorchScript output is not a tensor list");
  return result.toTensorList().vec();
}

std::unique_ptr<at::Tensor> tensor_run_torchscript(const at::Tensor& input, rust::Str path) {
  auto model = tensor_load_torchscript(path);
  return tensor_torchscript_forward(*model, input);
}

std::unique_ptr<at::Tensor> tensor_pair_first(const TensorPair& pair) {
  return std::make_unique<at::Tensor>(std::get<0>(pair));
}

std::unique_ptr<at::Tensor> tensor_pair_second(const TensorPair& pair) {
  return std::make_unique<at::Tensor>(std::get<1>(pair));
}

}  // namespace at::native
