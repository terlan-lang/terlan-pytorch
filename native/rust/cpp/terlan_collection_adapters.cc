#include "include/terlan_collection_adapters.hpp"

#include <utility>

namespace at::native {

std::unique_ptr<std::vector<std::int64_t>> terlan_collection_pytorch_tensor_shape(const at::Tensor& value) noexcept {
  try {
    const auto projected = value.sizes();
    return std::make_unique<std::vector<std::int64_t>>(projected.begin(), projected.end());
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<std::vector<std::int64_t>> terlan_collection_pytorch_tensor_strides(const at::Tensor& value) noexcept {
  try {
    const auto projected = value.strides();
    return std::make_unique<std::vector<std::int64_t>>(projected.begin(), projected.end());
  } catch (...) {
    return nullptr;
  }
}

TerlanPytorchTensorBroadcastTensorsResourceListResult::TerlanPytorchTensorBroadcastTensorsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorBroadcastTensorsResourceListResult::TerlanPytorchTensorBroadcastTensorsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorBroadcastTensorsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorBroadcastTensorsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorBroadcastTensorsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorBroadcastTensorsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorBroadcastTensorsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorBroadcastTensorsResourceListResult> terlan_collection_pytorch_tensor_broadcast_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchTensorBroadcastTensorsResourceListResult>(at::broadcast_tensors(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorBroadcastTensorsResourceListResult>("pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensor_broadcast_tensors_resources_is_ok(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensor_broadcast_tensors_resources_len(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_broadcast_tensors_resources_element(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensor_broadcast_tensors_resources_code(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensor_broadcast_tensors_resources_message(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorMeshgridResourceListResult::TerlanPytorchTensorMeshgridResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorMeshgridResourceListResult::TerlanPytorchTensorMeshgridResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorMeshgridResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorMeshgridResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorMeshgridResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorMeshgridResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorMeshgridResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorMeshgridResourceListResult> terlan_collection_pytorch_tensor_meshgrid_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<at::Tensor> (*)(at::TensorList);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::meshgrid);
      return std::make_unique<TerlanPytorchTensorMeshgridResourceListResult>(terlan_selected_overload(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorMeshgridResourceListResult>("pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensor_meshgrid_resources_is_ok(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensor_meshgrid_resources_len(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_meshgrid_resources_element(const TerlanPytorchTensorMeshgridResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensor_meshgrid_resources_code(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensor_meshgrid_resources_message(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorMeshgridIndexingResourceListResult::TerlanPytorchTensorMeshgridIndexingResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorMeshgridIndexingResourceListResult::TerlanPytorchTensorMeshgridIndexingResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorMeshgridIndexingResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorMeshgridIndexingResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorMeshgridIndexingResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorMeshgridIndexingResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorMeshgridIndexingResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorMeshgridIndexingResourceListResult> terlan_collection_pytorch_tensor_meshgrid_indexing_resources(const TerlanPytorchTensorTensorResourceListInput& tensors, rust::Str indexing) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<at::Tensor> (*)(at::TensorList, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::meshgrid);
      return std::make_unique<TerlanPytorchTensorMeshgridIndexingResourceListResult>(terlan_selected_overload(tensors.values(), c10::string_view(indexing.data(), indexing.size())));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorMeshgridIndexingResourceListResult>("pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensor_meshgrid_indexing_resources_is_ok(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensor_meshgrid_indexing_resources_len(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_meshgrid_indexing_resources_element(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensor_meshgrid_indexing_resources_code(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensor_meshgrid_indexing_resources_message(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorAutogradGradManyResourceListResult::TerlanPytorchTensorAutogradGradManyResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorAutogradGradManyResourceListResult::TerlanPytorchTensorAutogradGradManyResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorAutogradGradManyResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorAutogradGradManyResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorAutogradGradManyResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorAutogradGradManyResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorAutogradGradManyResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorAutogradGradManyResourceListResult> terlan_collection_pytorch_tensor_autograd_grad_many_resources(const TerlanPytorchTensorTensorResourceListInput& outputs, const TerlanPytorchTensorTensorResourceListInput& inputs, const TerlanPytorchTensorTensorResourceListInput& grad_outputs, bool retain_graph, bool create_graph) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchTensorAutogradGradManyResourceListResult>(at::native::tensor_autograd_grad_many(outputs.values(), inputs.values(), grad_outputs.values(), retain_graph, create_graph));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorAutogradGradManyResourceListResult>("pytorch_operation_failed", "LibTorch numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensor_autograd_grad_many_resources_is_ok(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensor_autograd_grad_many_resources_len(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_autograd_grad_many_resources_element(const TerlanPytorchTensorAutogradGradManyResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensor_autograd_grad_many_resources_code(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensor_autograd_grad_many_resources_message(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult> terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources(const TorchScriptModel & arg0, const at::Tensor & arg1) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult>(at::native::tensor_torchscript_forward_tensor_list(arg0, arg1));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult>("pytorch_torchscript_failed", "TorchScript operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_is_ok(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_len(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_element(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_code(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_message(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionChunkResourceListResult::TerlanPytorchShapePartitionChunkResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionChunkResourceListResult::TerlanPytorchShapePartitionChunkResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionChunkResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionChunkResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionChunkResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionChunkResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionChunkResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionChunkResourceListResult> terlan_collection_pytorch_shapepartition_chunk_resources(const at::Tensor& value, int64_t chunks, int64_t dim) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchShapePartitionChunkResourceListResult>(value.chunk(chunks, dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionChunkResourceListResult>("pytorch_operation_failed", "LibTorch tensor partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_chunk_resources_is_ok(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_chunk_resources_len(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_chunk_resources_element(const TerlanPytorchShapePartitionChunkResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_chunk_resources_code(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_chunk_resources_message(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionSplitSizeResourceListResult::TerlanPytorchShapePartitionSplitSizeResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionSplitSizeResourceListResult::TerlanPytorchShapePartitionSplitSizeResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionSplitSizeResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionSplitSizeResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionSplitSizeResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionSplitSizeResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionSplitSizeResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionSplitSizeResourceListResult> terlan_collection_pytorch_shapepartition_split_size_resources(const at::Tensor& value, int64_t split_size, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (at::Tensor::*)(int64_t, int64_t) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::split);
      return std::make_unique<TerlanPytorchShapePartitionSplitSizeResourceListResult>((value.*terlan_selected_overload)(split_size, dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionSplitSizeResourceListResult>("pytorch_operation_failed", "LibTorch tensor partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_split_size_resources_is_ok(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_split_size_resources_len(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_split_size_resources_element(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_split_size_resources_code(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_split_size_resources_message(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionSplitSizesResourceListResult::TerlanPytorchShapePartitionSplitSizesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionSplitSizesResourceListResult::TerlanPytorchShapePartitionSplitSizesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionSplitSizesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionSplitSizesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionSplitSizesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionSplitSizesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionSplitSizesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionSplitSizesResourceListResult> terlan_collection_pytorch_shapepartition_split_sizes_resources(const at::Tensor& value, rust::Slice<const std::int64_t> split_size, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (at::Tensor::*)(IntArrayRef, int64_t) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::split);
      return std::make_unique<TerlanPytorchShapePartitionSplitSizesResourceListResult>((value.*terlan_selected_overload)(IntArrayRef(split_size.data(), split_size.size()), dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionSplitSizesResourceListResult>("pytorch_operation_failed", "LibTorch tensor partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_split_sizes_resources_is_ok(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_split_sizes_resources_len(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_split_sizes_resources_element(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_split_sizes_resources_code(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_split_sizes_resources_message(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_tensor_split_sections_resources(const at::Tensor& value, int64_t sections, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (at::Tensor::*)(int64_t, int64_t) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::tensor_split);
      return std::make_unique<TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult>((value.*terlan_selected_overload)(sections, dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult>("pytorch_operation_failed", "LibTorch tensor partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_is_ok(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_len(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_element(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_code(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_message(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_tensor_split_indices_resources(const at::Tensor& value, rust::Slice<const std::int64_t> indices, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (at::Tensor::*)(IntArrayRef, int64_t) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::tensor_split);
      return std::make_unique<TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult>((value.*terlan_selected_overload)(IntArrayRef(indices.data(), indices.size()), dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult>("pytorch_operation_failed", "LibTorch tensor partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_is_ok(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_len(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_element(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_code(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_message(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::TerlanPytorchShapePartitionTensorSplitTensorResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::TerlanPytorchShapePartitionTensorSplitTensorResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionTensorSplitTensorResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionTensorSplitTensorResourceListResult> terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources(const at::Tensor& value, const Tensor & tensor_indices_or_sections, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (at::Tensor::*)(const Tensor &, int64_t) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::tensor_split);
      return std::make_unique<TerlanPytorchShapePartitionTensorSplitTensorResourceListResult>((value.*terlan_selected_overload)(tensor_indices_or_sections, dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionTensorSplitTensorResourceListResult>("pytorch_operation_failed", "LibTorch tensor partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_is_ok(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_len(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_element(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_code(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_message(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionUnbindResourceListResult::TerlanPytorchShapePartitionUnbindResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionUnbindResourceListResult::TerlanPytorchShapePartitionUnbindResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionUnbindResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionUnbindResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionUnbindResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionUnbindResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionUnbindResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionUnbindResourceListResult> terlan_collection_pytorch_shapepartition_unbind_resources(const at::Tensor& value, int64_t dim) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchShapePartitionUnbindResourceListResult>(value.unbind(dim));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionUnbindResourceListResult>("pytorch_operation_failed", "LibTorch tensor unbind failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_unbind_resources_is_ok(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_unbind_resources_len(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_unbind_resources_element(const TerlanPytorchShapePartitionUnbindResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_unbind_resources_code(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_unbind_resources_message(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionDsplitSectionsResourceListResult::TerlanPytorchShapePartitionDsplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionDsplitSectionsResourceListResult::TerlanPytorchShapePartitionDsplitSectionsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionDsplitSectionsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionDsplitSectionsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionDsplitSectionsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionDsplitSectionsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionDsplitSectionsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionDsplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_dsplit_sections_resources(const Tensor & self, int64_t sections) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::dsplit);
      return std::make_unique<TerlanPytorchShapePartitionDsplitSectionsResourceListResult>(terlan_selected_overload(self, sections));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionDsplitSectionsResourceListResult>("pytorch_operation_failed", "LibTorch depth split failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_dsplit_sections_resources_is_ok(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_dsplit_sections_resources_len(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_dsplit_sections_resources_element(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_dsplit_sections_resources_code(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_dsplit_sections_resources_message(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionHsplitSectionsResourceListResult::TerlanPytorchShapePartitionHsplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionHsplitSectionsResourceListResult::TerlanPytorchShapePartitionHsplitSectionsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionHsplitSectionsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionHsplitSectionsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionHsplitSectionsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionHsplitSectionsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionHsplitSectionsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionHsplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_hsplit_sections_resources(const Tensor & self, int64_t sections) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::hsplit);
      return std::make_unique<TerlanPytorchShapePartitionHsplitSectionsResourceListResult>(terlan_selected_overload(self, sections));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionHsplitSectionsResourceListResult>("pytorch_operation_failed", "LibTorch horizontal split failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_hsplit_sections_resources_is_ok(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_hsplit_sections_resources_len(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_hsplit_sections_resources_element(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_hsplit_sections_resources_code(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_hsplit_sections_resources_message(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionVsplitSectionsResourceListResult::TerlanPytorchShapePartitionVsplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionVsplitSectionsResourceListResult::TerlanPytorchShapePartitionVsplitSectionsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionVsplitSectionsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionVsplitSectionsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionVsplitSectionsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionVsplitSectionsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionVsplitSectionsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionVsplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_vsplit_sections_resources(const Tensor & self, int64_t sections) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::vsplit);
      return std::make_unique<TerlanPytorchShapePartitionVsplitSectionsResourceListResult>(terlan_selected_overload(self, sections));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionVsplitSectionsResourceListResult>("pytorch_operation_failed", "LibTorch vertical split failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_vsplit_sections_resources_is_ok(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_vsplit_sections_resources_len(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_vsplit_sections_resources_element(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_vsplit_sections_resources_code(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_vsplit_sections_resources_message(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionDsplitIndicesResourceListResult::TerlanPytorchShapePartitionDsplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionDsplitIndicesResourceListResult::TerlanPytorchShapePartitionDsplitIndicesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionDsplitIndicesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionDsplitIndicesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionDsplitIndicesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionDsplitIndicesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionDsplitIndicesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionDsplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_dsplit_indices_resources(const Tensor & self, rust::Slice<const std::int64_t> indices) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &, IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::dsplit);
      return std::make_unique<TerlanPytorchShapePartitionDsplitIndicesResourceListResult>(terlan_selected_overload(self, IntArrayRef(indices.data(), indices.size())));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionDsplitIndicesResourceListResult>("pytorch_operation_failed", "LibTorch depth split failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_dsplit_indices_resources_is_ok(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_dsplit_indices_resources_len(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_dsplit_indices_resources_element(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_dsplit_indices_resources_code(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_dsplit_indices_resources_message(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionHsplitIndicesResourceListResult::TerlanPytorchShapePartitionHsplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionHsplitIndicesResourceListResult::TerlanPytorchShapePartitionHsplitIndicesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionHsplitIndicesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionHsplitIndicesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionHsplitIndicesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionHsplitIndicesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionHsplitIndicesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionHsplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_hsplit_indices_resources(const Tensor & self, rust::Slice<const std::int64_t> indices) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &, IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::hsplit);
      return std::make_unique<TerlanPytorchShapePartitionHsplitIndicesResourceListResult>(terlan_selected_overload(self, IntArrayRef(indices.data(), indices.size())));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionHsplitIndicesResourceListResult>("pytorch_operation_failed", "LibTorch horizontal split failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_hsplit_indices_resources_is_ok(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_hsplit_indices_resources_len(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_hsplit_indices_resources_element(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_hsplit_indices_resources_code(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_hsplit_indices_resources_message(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionVsplitIndicesResourceListResult::TerlanPytorchShapePartitionVsplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionVsplitIndicesResourceListResult::TerlanPytorchShapePartitionVsplitIndicesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionVsplitIndicesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionVsplitIndicesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionVsplitIndicesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionVsplitIndicesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionVsplitIndicesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionVsplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_vsplit_indices_resources(const Tensor & self, rust::Slice<const std::int64_t> indices) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &, IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::vsplit);
      return std::make_unique<TerlanPytorchShapePartitionVsplitIndicesResourceListResult>(terlan_selected_overload(self, IntArrayRef(indices.data(), indices.size())));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionVsplitIndicesResourceListResult>("pytorch_operation_failed", "LibTorch vertical split failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartition_vsplit_indices_resources_is_ok(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartition_vsplit_indices_resources_len(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_vsplit_indices_resources_element(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartition_vsplit_indices_resources_code(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartition_vsplit_indices_resources_message(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::TerlanPytorchWhereNativeNonzeroNumpyResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::TerlanPytorchWhereNativeNonzeroNumpyResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchWhereNativeNonzeroNumpyResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchWhereNativeNonzeroNumpyResourceListResult> terlan_collection_pytorch_wherenative_nonzero_numpy_resources(const Tensor & self) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchWhereNativeNonzeroNumpyResourceListResult>(at::nonzero_numpy(self));
    } catch (...) {
      return std::make_unique<TerlanPytorchWhereNativeNonzeroNumpyResourceListResult>("pytorch_operation_failed", "LibTorch nonzero_numpy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_wherenative_nonzero_numpy_resources_is_ok(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_wherenative_nonzero_numpy_resources_len(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_wherenative_nonzero_numpy_resources_element(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_wherenative_nonzero_numpy_resources_code(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_wherenative_nonzero_numpy_resources_message(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchWhereNativeWhereIndicesResourceListResult::TerlanPytorchWhereNativeWhereIndicesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchWhereNativeWhereIndicesResourceListResult::TerlanPytorchWhereNativeWhereIndicesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchWhereNativeWhereIndicesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchWhereNativeWhereIndicesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchWhereNativeWhereIndicesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchWhereNativeWhereIndicesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchWhereNativeWhereIndicesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchWhereNativeWhereIndicesResourceListResult> terlan_collection_pytorch_wherenative_where_indices_resources(const Tensor & condition) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::where);
      return std::make_unique<TerlanPytorchWhereNativeWhereIndicesResourceListResult>(terlan_selected_overload(condition));
    } catch (...) {
      return std::make_unique<TerlanPytorchWhereNativeWhereIndicesResourceListResult>("pytorch_operation_failed", "LibTorch condition-selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_wherenative_where_indices_resources_is_ok(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_wherenative_where_indices_resources_len(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_wherenative_where_indices_resources_element(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_wherenative_where_indices_resources_code(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_wherenative_where_indices_resources_message(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientArrayResourceListResult::TerlanPytorchGradientsGradientArrayResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientArrayResourceListResult::TerlanPytorchGradientsGradientArrayResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientArrayResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientArrayResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientArrayResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientArrayResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientArrayResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_array_resources(const at::Tensor & tensor, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientArrayResourceListResult>(at::native::tensor_gradient_array(tensor, at::IntArrayRef(dim.data(), dim.size()), edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientArrayResourceListResult>("pytorch_operation_failed", "LibTorch numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_array_resources_is_ok(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_array_resources_len(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_array_resources_element(const TerlanPytorchGradientsGradientArrayResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_array_resources_code(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_array_resources_message(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientScalarArrayResourceListResult::TerlanPytorchGradientsGradientScalarArrayResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientScalarArrayResourceListResult::TerlanPytorchGradientsGradientScalarArrayResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientScalarArrayResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientScalarArrayResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientScalarArrayResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientScalarArrayResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientScalarArrayResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientScalarArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_array_resources(const at::Tensor & tensor, double spacing, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientScalarArrayResourceListResult>(at::native::tensor_gradient_scalar_array(tensor, spacing, at::IntArrayRef(dim.data(), dim.size()), edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientScalarArrayResourceListResult>("pytorch_operation_failed", "LibTorch scalar-spaced numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_scalar_array_resources_is_ok(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_array_resources_len(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_array_resources_element(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_resources_code(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_resources_message(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientScalarIntResourceListResult::TerlanPytorchGradientsGradientScalarIntResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientScalarIntResourceListResult::TerlanPytorchGradientsGradientScalarIntResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientScalarIntResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientScalarIntResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientScalarIntResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientScalarIntResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientScalarIntResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientScalarIntResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_int_resources(const at::Tensor & tensor, bool has_spacing, double spacing, bool has_dim, int64_t dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientScalarIntResourceListResult>(at::native::tensor_gradient_scalar_int(tensor, has_spacing, spacing, has_dim, dim, edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientScalarIntResourceListResult>("pytorch_operation_failed", "LibTorch optional numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_scalar_int_resources_is_ok(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_int_resources_len(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_int_resources_element(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_int_resources_code(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_int_resources_message(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources(const at::Tensor & tensor, rust::Slice<const double> spacing, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult>(at::native::tensor_gradient_scalar_array_array(tensor, spacing, at::IntArrayRef(dim.data(), dim.size()), edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult>("pytorch_operation_failed", "LibTorch scalar-array numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_is_ok(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_len(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_element(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_code(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_message(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::TerlanPytorchGradientsGradientScalarArrayIntResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::TerlanPytorchGradientsGradientScalarArrayIntResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientScalarArrayIntResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientScalarArrayIntResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources(const at::Tensor & tensor, rust::Slice<const double> spacing, bool has_dim, int64_t dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientScalarArrayIntResourceListResult>(at::native::tensor_gradient_scalar_array_int(tensor, spacing, has_dim, dim, edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientScalarArrayIntResourceListResult>("pytorch_operation_failed", "LibTorch optional scalar-array numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_is_ok(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_len(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_element(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_code(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_message(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientTensorArrayResourceListResult::TerlanPytorchGradientsGradientTensorArrayResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientTensorArrayResourceListResult::TerlanPytorchGradientsGradientTensorArrayResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientTensorArrayResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientTensorArrayResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientTensorArrayResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientTensorArrayResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientTensorArrayResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientTensorArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_tensor_array_resources(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& spacing, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientTensorArrayResourceListResult>(at::native::tensor_gradient_tensor_array(tensor, spacing.values(), at::IntArrayRef(dim.data(), dim.size()), edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientTensorArrayResourceListResult>("pytorch_operation_failed", "LibTorch tensor-array numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_tensor_array_resources_is_ok(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_tensor_array_resources_len(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_tensor_array_resources_element(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_resources_code(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_resources_message(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::TerlanPytorchGradientsGradientTensorArrayIntResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::TerlanPytorchGradientsGradientTensorArrayIntResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchGradientsGradientTensorArrayIntResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchGradientsGradientTensorArrayIntResourceListResult> terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& spacing, bool has_dim, int64_t dim, int64_t edge_order) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchGradientsGradientTensorArrayIntResourceListResult>(at::native::tensor_gradient_tensor_array_int(tensor, spacing.values(), has_dim, dim, edge_order));
    } catch (...) {
      return std::make_unique<TerlanPytorchGradientsGradientTensorArrayIntResourceListResult>("pytorch_operation_failed", "LibTorch optional tensor-array numerical gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_is_ok(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_len(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_element(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_code(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_message(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult> terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult>(at::broadcast_tensors(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult>("pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_is_ok(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_len(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_element(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_code(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_message(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorListTransformsMeshgridResourceListResult::TerlanPytorchTensorListTransformsMeshgridResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorListTransformsMeshgridResourceListResult::TerlanPytorchTensorListTransformsMeshgridResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorListTransformsMeshgridResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorListTransformsMeshgridResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorListTransformsMeshgridResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorListTransformsMeshgridResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorListTransformsMeshgridResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorListTransformsMeshgridResourceListResult> terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<at::Tensor> (*)(at::TensorList);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::meshgrid);
      return std::make_unique<TerlanPytorchTensorListTransformsMeshgridResourceListResult>(terlan_selected_overload(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorListTransformsMeshgridResourceListResult>("pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_is_ok(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_len(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_element(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_code(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_message(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult> terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources(const TerlanPytorchTensorTensorResourceListInput& tensors, rust::Str indexing) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<at::Tensor> (*)(at::TensorList, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::meshgrid);
      return std::make_unique<TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult>(terlan_selected_overload(tensors.values(), c10::string_view(indexing.data(), indexing.size())));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult>("pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_is_ok(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_len(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_element(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_code(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_message(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult> terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources(const Tensor & flat, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult>(at::unflatten_dense_tensors(flat, tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult>("pytorch_operation_failed", "LibTorch tensor-list transform failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_is_ok(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_len(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_element(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_code(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_message(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchViewAtleast1dSequenceResourceListResult::TerlanPytorchViewAtleast1dSequenceResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchViewAtleast1dSequenceResourceListResult::TerlanPytorchViewAtleast1dSequenceResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchViewAtleast1dSequenceResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchViewAtleast1dSequenceResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchViewAtleast1dSequenceResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchViewAtleast1dSequenceResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchViewAtleast1dSequenceResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchViewAtleast1dSequenceResourceListResult> terlan_collection_pytorch_view_atleast_1d_sequence_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(TensorList);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::atleast_1d);
      return std::make_unique<TerlanPytorchViewAtleast1dSequenceResourceListResult>(terlan_selected_overload(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchViewAtleast1dSequenceResourceListResult>("pytorch_operation_failed", "LibTorch view operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_view_atleast_1d_sequence_resources_is_ok(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_view_atleast_1d_sequence_resources_len(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_view_atleast_1d_sequence_resources_element(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_view_atleast_1d_sequence_resources_code(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_view_atleast_1d_sequence_resources_message(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchViewAtleast2dSequenceResourceListResult::TerlanPytorchViewAtleast2dSequenceResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchViewAtleast2dSequenceResourceListResult::TerlanPytorchViewAtleast2dSequenceResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchViewAtleast2dSequenceResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchViewAtleast2dSequenceResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchViewAtleast2dSequenceResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchViewAtleast2dSequenceResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchViewAtleast2dSequenceResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchViewAtleast2dSequenceResourceListResult> terlan_collection_pytorch_view_atleast_2d_sequence_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(TensorList);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::atleast_2d);
      return std::make_unique<TerlanPytorchViewAtleast2dSequenceResourceListResult>(terlan_selected_overload(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchViewAtleast2dSequenceResourceListResult>("pytorch_operation_failed", "LibTorch view operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_view_atleast_2d_sequence_resources_is_ok(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_view_atleast_2d_sequence_resources_len(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_view_atleast_2d_sequence_resources_element(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_view_atleast_2d_sequence_resources_code(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_view_atleast_2d_sequence_resources_message(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchViewAtleast3dSequenceResourceListResult::TerlanPytorchViewAtleast3dSequenceResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchViewAtleast3dSequenceResourceListResult::TerlanPytorchViewAtleast3dSequenceResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchViewAtleast3dSequenceResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchViewAtleast3dSequenceResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchViewAtleast3dSequenceResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchViewAtleast3dSequenceResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchViewAtleast3dSequenceResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchViewAtleast3dSequenceResourceListResult> terlan_collection_pytorch_view_atleast_3d_sequence_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::vector<Tensor> (*)(TensorList);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::atleast_3d);
      return std::make_unique<TerlanPytorchViewAtleast3dSequenceResourceListResult>(terlan_selected_overload(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchViewAtleast3dSequenceResourceListResult>("pytorch_operation_failed", "LibTorch view operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_view_atleast_3d_sequence_resources_is_ok(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_view_atleast_3d_sequence_resources_len(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_view_atleast_3d_sequence_resources_element(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_view_atleast_3d_sequence_resources_code(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_view_atleast_3d_sequence_resources_message(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult> terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors, const Tensor & scales, const Tensor & zero_points, std::int64_t dtype) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult>(at::quantize_per_tensor(tensors.values(), scales, zero_points, static_cast<c10::ScalarType>(dtype)));
    } catch (...) {
      return std::make_unique<TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult>("pytorch_operation_failed", "LibTorch tensor-list quantization operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_is_ok(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_len(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_element(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_code(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_message(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult> terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult>(at::dequantize(tensors.values()));
    } catch (...) {
      return std::make_unique<TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult>("pytorch_operation_failed", "LibTorch tensor-list dequantization operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_is_ok(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_len(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_element(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_code(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_message(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult> terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources(const HistogramddResult & result) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult>(at::native::histogramdd_result_bin_edges(result));
    } catch (...) {
      return std::make_unique<TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult>("pytorch_operation_failed", "LibTorch histogramdd edge projection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_is_ok(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_len(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_element(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_code(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_message(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept { return result.message(); }

TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult(std::vector<at::Tensor> values) noexcept
    : ok_(true), values_(std::move(values)) {}
TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult(std::string code, std::string message) noexcept
    : ok_(false), code_(std::move(code)), message_(std::move(message)) {}
bool TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::is_ok() const noexcept { return ok_; }
std::size_t TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::len() const noexcept { return values_.size(); }
std::unique_ptr<at::Tensor> TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::element(std::size_t index) const noexcept {
  try {
    if (index >= values_.size()) { return nullptr; }
    return std::make_unique<at::Tensor>(values_[index]);
  } catch (...) {
    return nullptr;
  }
}
const std::string& TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::code() const noexcept { return code_; }
const std::string& TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult::message() const noexcept { return message_; }

std::unique_ptr<TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult> terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources(const at::Tensor & input, int64_t chunks, int64_t dimension) noexcept {
  try {
    try {
      return std::make_unique<TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult>(at::native::unsafe_chunk_exact(input, chunks, dimension));
    } catch (...) {
      return std::make_unique<TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult>("pytorch_operation_failed", "LibTorch unsafe chunk partition failed.");
    }
  } catch (...) {
    return nullptr;
  }
}
bool terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_is_ok(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept { return result.is_ok(); }
std::size_t terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_len(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept { return result.len(); }
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_element(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result, std::size_t index) noexcept { return result.element(index); }
const std::string& terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_code(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept { return result.code(); }
const std::string& terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_message(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept { return result.message(); }

}  // namespace at::native
