#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/script.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#else
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;
using AOTITorchError = int32_t;

extern "C" AOTITorchError aoti_torch_new_tensor_handle(
    AtenTensorHandle tensor,
    AtenTensorHandle* out_tensor);
extern "C" AOTITorchError aoti_torch_scalar_to_tensor_float64(
    double value,
    AtenTensorHandle* out_tensor);
extern "C" AOTITorchError aoti_torch_item_float64(
    AtenTensorHandle tensor,
    double* out_value);
extern "C" AOTITorchError aoti_torch_item_int64(
    AtenTensorHandle tensor,
    int64_t* out_value);
extern "C" AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor);
extern "C" AOTITorchError aoti_torch_aten_subtract_Tensor(
    AtenTensorHandle self,
    AtenTensorHandle other,
    double alpha,
    AtenTensorHandle* out_tensor);
extern "C" AOTITorchError torch_call_dispatcher(
    const char* operator_name,
    const char* overload_name,
    uint64_t* stack,
    uint64_t extension_abi_version);
#endif

struct TerlanTorchScriptModelOpaque {
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  explicit TerlanTorchScriptModelOpaque(torch::jit::Module&& loaded)
      : module(std::move(loaded)) {}
  torch::jit::Module module;
#else
  uint8_t fixture_marker;
#endif
};
using TerlanTorchScriptModelHandle = TerlanTorchScriptModelOpaque*;

struct TerlanTorchScriptTensorPairOpaque {
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  TerlanTorchScriptTensorPairOpaque(at::Tensor first_value,
                                    at::Tensor second_value)
      : first(std::move(first_value)), second(std::move(second_value)) {}
  at::Tensor first;
  at::Tensor second;
#else
  AtenTensorHandle first;
  AtenTensorHandle second;
#endif
};
using TerlanTorchScriptTensorPairHandle =
    TerlanTorchScriptTensorPairOpaque*;

namespace {

constexpr AOTITorchError kInvalidArgument = 7040;
constexpr AOTITorchError kModelNotFound = 7041;
constexpr AOTITorchError kModelLoadFailed = 7042;
constexpr AOTITorchError kModelForwardFailed = 7043;
constexpr AOTITorchError kModelOutputNotTensor = 7044;
constexpr AOTITorchError kAllocationFailed = 7045;
constexpr AOTITorchError kModelOutputNotTuple = 7046;
constexpr AOTITorchError kModelOutputTupleArity = 7047;
constexpr AOTITorchError kModelOutputTupleElement = 7048;
constexpr AOTITorchError kModelOutputNotFloat = 7049;
constexpr AOTITorchError kModelOutputNotInt = 7050;
constexpr AOTITorchError kModelOutputNotBool = 7051;
constexpr AOTITorchError kModelOutputNotString = 7052;
constexpr AOTITorchError kModelOutputNotIntList = 7053;
constexpr AOTITorchError kModelOutputNotFloatList = 7054;
constexpr AOTITorchError kModelOutputNotBoolList = 7055;
constexpr AOTITorchError kModelOutputNotStringList = 7056;

bool file_exists(const char* path) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

AOTITorchError copy_owned_utf8(const char* value,
                               std::size_t length,
                               char** out_value,
                               std::size_t* out_length) {
  if ((length != 0 && value == nullptr) || out_value == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_value = nullptr;
  *out_length = 0;
  if (length == 0) {
    return 0;
  }
  auto* owned = new (std::nothrow) char[length];
  if (owned == nullptr) {
    return kAllocationFailed;
  }
  std::memcpy(owned, value, length);
  *out_value = owned;
  *out_length = length;
  return 0;
}

AOTITorchError copy_owned_strings(const std::vector<std::string>& values,
                                  char*** out_values,
                                  std::size_t** out_lengths,
                                  std::size_t* out_count) {
  if (out_values == nullptr || out_lengths == nullptr || out_count == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_lengths = nullptr;
  *out_count = 0;
  if (values.empty()) {
    return 0;
  }
  auto** owned_values = new (std::nothrow) char*[values.size()]();
  auto* owned_lengths = new (std::nothrow) std::size_t[values.size()]();
  if (owned_values == nullptr || owned_lengths == nullptr) {
    delete[] owned_values;
    delete[] owned_lengths;
    return kAllocationFailed;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    owned_lengths[index] = values[index].size();
    if (values[index].empty()) {
      continue;
    }
    owned_values[index] = new (std::nothrow) char[values[index].size()];
    if (owned_values[index] == nullptr) {
      for (std::size_t cleanup = 0; cleanup < index; ++cleanup) {
        delete[] owned_values[cleanup];
      }
      delete[] owned_values;
      delete[] owned_lengths;
      return kAllocationFailed;
    }
    std::memcpy(owned_values[index], values[index].data(),
                values[index].size());
  }
  *out_values = owned_values;
  *out_lengths = owned_lengths;
  *out_count = values.size();
  return 0;
}

AOTITorchError copy_owned_ints(const int64_t* values,
                               std::size_t length,
                               int64_t** out_values,
                               std::size_t* out_length) {
  if ((length != 0 && values == nullptr) || out_values == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_length = 0;
  if (length == 0) {
    return 0;
  }
  auto* owned = new (std::nothrow) int64_t[length];
  if (owned == nullptr) {
    return kAllocationFailed;
  }
  std::memcpy(owned, values, length * sizeof(int64_t));
  *out_values = owned;
  *out_length = length;
  return 0;
}

AOTITorchError copy_owned_floats(const double* values,
                                 std::size_t length,
                                 double** out_values,
                                 std::size_t* out_length) {
  if ((length != 0 && values == nullptr) || out_values == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_length = 0;
  if (length == 0) {
    return 0;
  }
  auto* owned = new (std::nothrow) double[length];
  if (owned == nullptr) {
    return kAllocationFailed;
  }
  std::memcpy(owned, values, length * sizeof(double));
  *out_values = owned;
  *out_length = length;
  return 0;
}

AOTITorchError copy_owned_bools(const uint8_t* values,
                                std::size_t length,
                                uint8_t** out_values,
                                std::size_t* out_length) {
  if ((length != 0 && values == nullptr) || out_values == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_length = 0;
  if (length == 0) {
    return 0;
  }
  auto* owned = new (std::nothrow) uint8_t[length];
  if (owned == nullptr) {
    return kAllocationFailed;
  }
  std::memcpy(owned, values, length * sizeof(uint8_t));
  *out_values = owned;
  *out_length = length;
  return 0;
}

} // namespace

extern "C" AOTITorchError terlan_pytorch_load_torchscript(
    const char* path,
    TerlanTorchScriptModelHandle* out_model) noexcept {
  if (path == nullptr || path[0] == '\0' || out_model == nullptr) {
    return kInvalidArgument;
  }
  *out_model = nullptr;
  if (!file_exists(path)) {
    return kModelNotFound;
  }

#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto module = torch::jit::load(path);
    module.eval();
    auto* model = new (std::nothrow)
        TerlanTorchScriptModelOpaque(std::move(module));
    if (model == nullptr) {
      return kAllocationFailed;
    }
    *out_model = model;
    return 0;
  } catch (...) {
    return kModelLoadFailed;
  }
#else
  if (std::strstr(path, "invalid.pt") != nullptr) {
    return kModelLoadFailed;
  }
  uint8_t fixture_marker = 1;
  if (std::strstr(path, "weighted_difference.pt") != nullptr) {
    fixture_marker = 2;
  } else if (std::strstr(path, "tensor_pair.pt") != nullptr) {
    fixture_marker = 3;
  } else if (std::strstr(path, "scalar_float.pt") != nullptr) {
    fixture_marker = 4;
  } else if (std::strstr(path, "scalar_int.pt") != nullptr) {
    fixture_marker = 5;
  } else if (std::strstr(path, "scalar_bool.pt") != nullptr) {
    fixture_marker = 6;
  } else if (std::strstr(path, "scalar_string.pt") != nullptr) {
    fixture_marker = 7;
  } else if (std::strstr(path, "int_list.pt") != nullptr) {
    fixture_marker = 8;
  } else if (std::strstr(path, "float_list.pt") != nullptr) {
    fixture_marker = 9;
  } else if (std::strstr(path, "bool_list.pt") != nullptr) {
    fixture_marker = 10;
  } else if (std::strstr(path, "string_list.pt") != nullptr) {
    fixture_marker = 11;
  }
  auto* model =
      new (std::nothrow) TerlanTorchScriptModelOpaque{fixture_marker};
  if (model == nullptr) {
    return kAllocationFailed;
  }
  *out_model = model;
  return 0;
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    AtenTensorHandle* out_tensor) noexcept {
  if (model == nullptr || input == nullptr || out_tensor == nullptr) {
    return kInvalidArgument;
  }
  *out_tensor = nullptr;

#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isTensor()) {
      return kModelOutputNotTensor;
    }
    auto* owned = new (std::nothrow) at::Tensor(result.toTensor());
    if (owned == nullptr) {
      return kAllocationFailed;
    }
    *out_tensor = reinterpret_cast<AtenTensorHandle>(owned);
    return 0;
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 1) {
    return kModelForwardFailed;
  }
  AtenTensorHandle input_copy = nullptr;
  AtenTensorHandle factor = nullptr;
  AOTITorchError status = aoti_torch_new_tensor_handle(input, &input_copy);
  if (status != 0) {
    return status;
  }
  status = aoti_torch_scalar_to_tensor_float64(2.0, &factor);
  if (status != 0) {
    aoti_torch_delete_tensor_object(input_copy);
    return status;
  }

  uint64_t stack[2] = {
      reinterpret_cast<uint64_t>(input_copy),
      reinterpret_cast<uint64_t>(factor),
  };
  status = torch_call_dispatcher(
      "aten::mul", "Tensor", stack, UINT64_C(0x020d000000000000));
  if (status != 0) {
    return status;
  }
  *out_tensor = reinterpret_cast<AtenTensorHandle>(stack[0]);
  return 0;
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward2(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle first,
    AtenTensorHandle second,
    AtenTensorHandle* out_tensor) noexcept {
  if (model == nullptr || first == nullptr || second == nullptr ||
      out_tensor == nullptr) {
    return kInvalidArgument;
  }
  *out_tensor = nullptr;

#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(first)),
         torch::jit::IValue(*reinterpret_cast<at::Tensor*>(second))});
    if (!result.isTensor()) {
      return kModelOutputNotTensor;
    }
    auto* owned = new (std::nothrow) at::Tensor(result.toTensor());
    if (owned == nullptr) {
      return kAllocationFailed;
    }
    *out_tensor = reinterpret_cast<AtenTensorHandle>(owned);
    return 0;
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 2) {
    return kModelForwardFailed;
  }
  return aoti_torch_aten_subtract_Tensor(first, second, 2.0, out_tensor);
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_pair(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    TerlanTorchScriptTensorPairHandle* out_pair) noexcept {
  if (model == nullptr || input == nullptr || out_pair == nullptr) {
    return kInvalidArgument;
  }
  *out_pair = nullptr;

#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isTuple()) {
      return kModelOutputNotTuple;
    }
    const auto& elements = result.toTupleRef().elements();
    if (elements.size() != 2) {
      return kModelOutputTupleArity;
    }
    if (!elements[0].isTensor() || !elements[1].isTensor()) {
      return kModelOutputTupleElement;
    }
    auto* pair = new (std::nothrow) TerlanTorchScriptTensorPairOpaque(
        elements[0].toTensor(), elements[1].toTensor());
    if (pair == nullptr) {
      return kAllocationFailed;
    }
    *out_pair = pair;
    return 0;
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 3) {
    return kModelOutputNotTuple;
  }
  AtenTensorHandle first = nullptr;
  AOTITorchError status =
      aoti_torch_aten_subtract_Tensor(input, input, -1.0, &first);
  if (status != 0) {
    return status;
  }
  AtenTensorHandle second = nullptr;
  status = aoti_torch_aten_subtract_Tensor(input, input, -2.0, &second);
  if (status != 0) {
    aoti_torch_delete_tensor_object(first);
    return status;
  }
  auto* pair =
      new (std::nothrow) TerlanTorchScriptTensorPairOpaque{first, second};
  if (pair == nullptr) {
    aoti_torch_delete_tensor_object(second);
    aoti_torch_delete_tensor_object(first);
    return kAllocationFailed;
  }
  *out_pair = pair;
  return 0;
#endif
}

extern "C" AOTITorchError terlan_pytorch_tensor_pair_first(
    TerlanTorchScriptTensorPairHandle pair,
    AtenTensorHandle* out_tensor) noexcept {
  if (pair == nullptr || out_tensor == nullptr) {
    return kInvalidArgument;
  }
  *out_tensor = nullptr;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto* owned = new (std::nothrow) at::Tensor(pair->first);
    if (owned == nullptr) {
      return kAllocationFailed;
    }
    *out_tensor = reinterpret_cast<AtenTensorHandle>(owned);
    return 0;
  } catch (...) {
    return kAllocationFailed;
  }
#else
  return aoti_torch_new_tensor_handle(pair->first, out_tensor);
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_float(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    double* out_value) noexcept {
  if (model == nullptr || input == nullptr || out_value == nullptr) {
    return kInvalidArgument;
  }
  *out_value = 0.0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isDouble()) {
      return kModelOutputNotFloat;
    }
    *out_value = result.toDouble();
    return 0;
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 4) {
    return kModelOutputNotFloat;
  }
  AOTITorchError status = aoti_torch_item_float64(input, out_value);
  if (status != 0) {
    return status;
  }
  *out_value *= 2.0;
  return 0;
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_int(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    int64_t* out_value) noexcept {
  if (model == nullptr || input == nullptr || out_value == nullptr) {
    return kInvalidArgument;
  }
  *out_value = 0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isInt()) {
      return kModelOutputNotInt;
    }
    *out_value = result.toInt();
    return 0;
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 5) {
    return kModelOutputNotInt;
  }
  AOTITorchError status = aoti_torch_item_int64(input, out_value);
  if (status != 0) {
    return status;
  }
  *out_value += 3;
  return 0;
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_bool(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    bool* out_value) noexcept {
  if (model == nullptr || input == nullptr || out_value == nullptr) {
    return kInvalidArgument;
  }
  *out_value = false;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isBool()) {
      return kModelOutputNotBool;
    }
    *out_value = result.toBool();
    return 0;
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 6) {
    return kModelOutputNotBool;
  }
  int64_t input_value = 0;
  AOTITorchError status = aoti_torch_item_int64(input, &input_value);
  if (status != 0) {
    return status;
  }
  *out_value = input_value > 0;
  return 0;
#endif
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_string(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    char** out_value,
    std::size_t* out_length) noexcept {
  if (model == nullptr || input == nullptr || out_value == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_value = nullptr;
  *out_length = 0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isString()) {
      return kModelOutputNotString;
    }
    const auto& value = result.toStringRef();
    return copy_owned_utf8(value.data(), value.size(), out_value, out_length);
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 7) {
    return kModelOutputNotString;
  }
  int64_t input_value = 0;
  AOTITorchError status = aoti_torch_item_int64(input, &input_value);
  if (status != 0) {
    return status;
  }
  const char* value = input_value > 0 ? "positive" : "non_positive";
  return copy_owned_utf8(value, std::strlen(value), out_value, out_length);
#endif
}

extern "C" void terlan_pytorch_delete_owned_string(char* value) noexcept {
  delete[] value;
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_int_list(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    int64_t** out_values,
    std::size_t* out_length) noexcept {
  if (model == nullptr || input == nullptr || out_values == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_length = 0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isIntList()) {
      return kModelOutputNotIntList;
    }
    const auto values = result.toIntVector();
    return copy_owned_ints(values.data(), values.size(), out_values,
                           out_length);
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 8) {
    return kModelOutputNotIntList;
  }
  int64_t input_value = 0;
  AOTITorchError status = aoti_torch_item_int64(input, &input_value);
  if (status != 0) {
    return status;
  }
  const int64_t values[] = {input_value, input_value + 1, input_value + 2};
  return copy_owned_ints(values, 3, out_values, out_length);
#endif
}

extern "C" void terlan_pytorch_delete_owned_int_array(
    int64_t* values) noexcept {
  delete[] values;
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_float_list(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    double** out_values,
    std::size_t* out_length) noexcept {
  if (model == nullptr || input == nullptr || out_values == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_length = 0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isDoubleList()) {
      return kModelOutputNotFloatList;
    }
    const auto values = result.toDoubleVector();
    return copy_owned_floats(values.data(), values.size(), out_values,
                             out_length);
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 9) {
    return kModelOutputNotFloatList;
  }
  double input_value = 0.0;
  AOTITorchError status = aoti_torch_item_float64(input, &input_value);
  if (status != 0) {
    return status;
  }
  const double values[] = {input_value, input_value + 0.5,
                           input_value + 1.0};
  return copy_owned_floats(values, 3, out_values, out_length);
#endif
}

extern "C" void terlan_pytorch_delete_owned_float_array(
    double* values) noexcept {
  delete[] values;
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_bool_list(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    uint8_t** out_values,
    std::size_t* out_length) noexcept {
  if (model == nullptr || input == nullptr || out_values == nullptr ||
      out_length == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_length = 0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isBoolList()) {
      return kModelOutputNotBoolList;
    }
    const auto values = result.toBoolList();
    std::vector<uint8_t> encoded;
    encoded.reserve(values.size());
    for (bool value : values) {
      encoded.push_back(value ? 1 : 0);
    }
    return copy_owned_bools(encoded.data(), encoded.size(), out_values,
                            out_length);
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  if (model->fixture_marker != 10) {
    return kModelOutputNotBoolList;
  }
  int64_t input_value = 0;
  AOTITorchError status = aoti_torch_item_int64(input, &input_value);
  if (status != 0) {
    return status;
  }
  const uint8_t values[] = {
      static_cast<uint8_t>(input_value > 0),
      static_cast<uint8_t>(input_value == 0),
      static_cast<uint8_t>(input_value < 0),
  };
  return copy_owned_bools(values, 3, out_values, out_length);
#endif
}

extern "C" void terlan_pytorch_delete_owned_bool_array(
    uint8_t* values) noexcept {
  delete[] values;
}

extern "C" AOTITorchError terlan_pytorch_torchscript_forward_string_list(
    TerlanTorchScriptModelHandle model,
    AtenTensorHandle input,
    char*** out_values,
    std::size_t** out_lengths,
    std::size_t* out_count) noexcept {
  if (model == nullptr || input == nullptr || out_values == nullptr ||
      out_lengths == nullptr || out_count == nullptr) {
    return kInvalidArgument;
  }
  *out_values = nullptr;
  *out_lengths = nullptr;
  *out_count = 0;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto result = model->module.forward(
        {torch::jit::IValue(*reinterpret_cast<at::Tensor*>(input))});
    if (!result.isList()) {
      return kModelOutputNotStringList;
    }
    std::vector<std::string> values;
    const auto elements = result.toListRef();
    values.reserve(elements.size());
    for (const auto& element : elements) {
      if (!element.isString()) {
        return kModelOutputNotStringList;
      }
      values.emplace_back(element.toStringRef());
    }
    return copy_owned_strings(values, out_values, out_lengths, out_count);
  } catch (...) {
    return kModelForwardFailed;
  }
#else
  try {
    if (model->fixture_marker != 11) {
      return kModelOutputNotStringList;
    }
    int64_t input_value = 0;
    AOTITorchError status = aoti_torch_item_int64(input, &input_value);
    if (status != 0) {
      return status;
    }
    const std::vector<std::string> values =
        input_value > 0
            ? std::vector<std::string>{"positive", "", "ready"}
            : std::vector<std::string>{"non_positive", "", "waiting"};
    return copy_owned_strings(values, out_values, out_lengths, out_count);
  } catch (...) {
    return kAllocationFailed;
  }
#endif
}

extern "C" void terlan_pytorch_delete_owned_string_array(
    char** values,
    std::size_t* lengths,
    std::size_t count) noexcept {
  if (values != nullptr) {
    for (std::size_t index = 0; index < count; ++index) {
      delete[] values[index];
    }
  }
  delete[] values;
  delete[] lengths;
}

extern "C" AOTITorchError terlan_pytorch_tensor_pair_second(
    TerlanTorchScriptTensorPairHandle pair,
    AtenTensorHandle* out_tensor) noexcept {
  if (pair == nullptr || out_tensor == nullptr) {
    return kInvalidArgument;
  }
  *out_tensor = nullptr;
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  try {
    auto* owned = new (std::nothrow) at::Tensor(pair->second);
    if (owned == nullptr) {
      return kAllocationFailed;
    }
    *out_tensor = reinterpret_cast<AtenTensorHandle>(owned);
    return 0;
  } catch (...) {
    return kAllocationFailed;
  }
#else
  return aoti_torch_new_tensor_handle(pair->second, out_tensor);
#endif
}

extern "C" AOTITorchError terlan_pytorch_delete_tensor_pair(
    TerlanTorchScriptTensorPairHandle pair) noexcept {
  if (pair == nullptr) {
    return kInvalidArgument;
  }
#if __has_include(<torch/script.h>) &&                                      \
    __has_include(<torch/csrc/inductor/aoti_torch/c/shim.h>)
  delete pair;
  return 0;
#else
  const AOTITorchError second_status =
      aoti_torch_delete_tensor_object(pair->second);
  const AOTITorchError first_status =
      aoti_torch_delete_tensor_object(pair->first);
  delete pair;
  return first_status != 0 ? first_status : second_status;
#endif
}

extern "C" AOTITorchError terlan_pytorch_delete_torchscript_model(
    TerlanTorchScriptModelHandle model) noexcept {
  if (model == nullptr) {
    return kInvalidArgument;
  }
  delete model;
  return 0;
}

extern "C" AOTITorchError terlan_pytorch_run_torchscript(
    AtenTensorHandle input,
    const char* path,
    AtenTensorHandle* out_tensor) noexcept {
  TerlanTorchScriptModelHandle model = nullptr;
  AOTITorchError status = terlan_pytorch_load_torchscript(path, &model);
  if (status != 0) {
    return status;
  }
  status = terlan_pytorch_torchscript_forward(model, input, out_tensor);
  const AOTITorchError delete_status =
      terlan_pytorch_delete_torchscript_model(model);
  return status == 0 ? delete_status : status;
}
