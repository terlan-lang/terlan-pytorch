#pragma once

#include "libtorch_bindings.hpp"
#include "terlan_owned_value_adapters.hpp"
#include "rust/cxx.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace at::native {

std::unique_ptr<std::vector<std::int64_t>> terlan_collection_pytorch_tensor_shape(const at::Tensor& value) noexcept;
std::unique_ptr<std::vector<std::int64_t>> terlan_collection_pytorch_tensor_strides(const at::Tensor& value) noexcept;
class TerlanPytorchTensorBroadcastTensorsResourceListResult final {
 public:
  explicit TerlanPytorchTensorBroadcastTensorsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorBroadcastTensorsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorBroadcastTensorsResourceListResult> terlan_collection_pytorch_tensor_broadcast_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_tensor_broadcast_tensors_resources_is_ok(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensor_broadcast_tensors_resources_len(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_broadcast_tensors_resources_element(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensor_broadcast_tensors_resources_code(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensor_broadcast_tensors_resources_message(const TerlanPytorchTensorBroadcastTensorsResourceListResult& result) noexcept;

class TerlanPytorchTensorMeshgridResourceListResult final {
 public:
  explicit TerlanPytorchTensorMeshgridResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorMeshgridResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorMeshgridResourceListResult> terlan_collection_pytorch_tensor_meshgrid_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_tensor_meshgrid_resources_is_ok(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensor_meshgrid_resources_len(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_meshgrid_resources_element(const TerlanPytorchTensorMeshgridResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensor_meshgrid_resources_code(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensor_meshgrid_resources_message(const TerlanPytorchTensorMeshgridResourceListResult& result) noexcept;

class TerlanPytorchTensorMeshgridIndexingResourceListResult final {
 public:
  explicit TerlanPytorchTensorMeshgridIndexingResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorMeshgridIndexingResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorMeshgridIndexingResourceListResult> terlan_collection_pytorch_tensor_meshgrid_indexing_resources(const TerlanPytorchTensorTensorResourceListInput& tensors, rust::Str indexing) noexcept;
bool terlan_collection_pytorch_tensor_meshgrid_indexing_resources_is_ok(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensor_meshgrid_indexing_resources_len(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_meshgrid_indexing_resources_element(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensor_meshgrid_indexing_resources_code(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensor_meshgrid_indexing_resources_message(const TerlanPytorchTensorMeshgridIndexingResourceListResult& result) noexcept;

class TerlanPytorchTensorAutogradGradManyResourceListResult final {
 public:
  explicit TerlanPytorchTensorAutogradGradManyResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorAutogradGradManyResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorAutogradGradManyResourceListResult> terlan_collection_pytorch_tensor_autograd_grad_many_resources(const TerlanPytorchTensorTensorResourceListInput& outputs, const TerlanPytorchTensorTensorResourceListInput& inputs, const TerlanPytorchTensorTensorResourceListInput& grad_outputs, bool retain_graph, bool create_graph) noexcept;
bool terlan_collection_pytorch_tensor_autograd_grad_many_resources_is_ok(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensor_autograd_grad_many_resources_len(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_autograd_grad_many_resources_element(const TerlanPytorchTensorAutogradGradManyResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensor_autograd_grad_many_resources_code(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensor_autograd_grad_many_resources_message(const TerlanPytorchTensorAutogradGradManyResourceListResult& result) noexcept;

class TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult final {
 public:
  explicit TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult> terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources(const TorchScriptModel & arg0, const at::Tensor & arg1) noexcept;
bool terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_is_ok(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_len(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_element(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_code(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensor_torchscript_forward_tensor_list_resources_message(const TerlanPytorchTensorTorchscriptForwardTensorListResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionChunkResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionChunkResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionChunkResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionChunkResourceListResult> terlan_collection_pytorch_shapepartition_chunk_resources(const at::Tensor& value, int64_t chunks, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_chunk_resources_is_ok(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_chunk_resources_len(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_chunk_resources_element(const TerlanPytorchShapePartitionChunkResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_chunk_resources_code(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_chunk_resources_message(const TerlanPytorchShapePartitionChunkResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionSplitSizeResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionSplitSizeResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionSplitSizeResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionSplitSizeResourceListResult> terlan_collection_pytorch_shapepartition_split_size_resources(const at::Tensor& value, int64_t split_size, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_split_size_resources_is_ok(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_split_size_resources_len(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_split_size_resources_element(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_split_size_resources_code(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_split_size_resources_message(const TerlanPytorchShapePartitionSplitSizeResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionSplitSizesResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionSplitSizesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionSplitSizesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionSplitSizesResourceListResult> terlan_collection_pytorch_shapepartition_split_sizes_resources(const at::Tensor& value, rust::Slice<const std::int64_t> split_size, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_split_sizes_resources_is_ok(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_split_sizes_resources_len(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_split_sizes_resources_element(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_split_sizes_resources_code(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_split_sizes_resources_message(const TerlanPytorchShapePartitionSplitSizesResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_tensor_split_sections_resources(const at::Tensor& value, int64_t sections, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_is_ok(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_len(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_element(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_code(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_sections_resources_message(const TerlanPytorchShapePartitionTensorSplitSectionsResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_tensor_split_indices_resources(const at::Tensor& value, rust::Slice<const std::int64_t> indices, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_is_ok(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_len(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_element(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_code(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_indices_resources_message(const TerlanPytorchShapePartitionTensorSplitIndicesResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionTensorSplitTensorResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionTensorSplitTensorResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionTensorSplitTensorResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionTensorSplitTensorResourceListResult> terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources(const at::Tensor& value, const Tensor & tensor_indices_or_sections, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_is_ok(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_len(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_element(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_code(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_tensor_split_tensor_resources_message(const TerlanPytorchShapePartitionTensorSplitTensorResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionUnbindResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionUnbindResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionUnbindResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionUnbindResourceListResult> terlan_collection_pytorch_shapepartition_unbind_resources(const at::Tensor& value, int64_t dim) noexcept;
bool terlan_collection_pytorch_shapepartition_unbind_resources_is_ok(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_unbind_resources_len(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_unbind_resources_element(const TerlanPytorchShapePartitionUnbindResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_unbind_resources_code(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_unbind_resources_message(const TerlanPytorchShapePartitionUnbindResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionDsplitSectionsResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionDsplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionDsplitSectionsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionDsplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_dsplit_sections_resources(const Tensor & self, int64_t sections) noexcept;
bool terlan_collection_pytorch_shapepartition_dsplit_sections_resources_is_ok(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_dsplit_sections_resources_len(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_dsplit_sections_resources_element(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_dsplit_sections_resources_code(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_dsplit_sections_resources_message(const TerlanPytorchShapePartitionDsplitSectionsResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionHsplitSectionsResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionHsplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionHsplitSectionsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionHsplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_hsplit_sections_resources(const Tensor & self, int64_t sections) noexcept;
bool terlan_collection_pytorch_shapepartition_hsplit_sections_resources_is_ok(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_hsplit_sections_resources_len(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_hsplit_sections_resources_element(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_hsplit_sections_resources_code(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_hsplit_sections_resources_message(const TerlanPytorchShapePartitionHsplitSectionsResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionVsplitSectionsResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionVsplitSectionsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionVsplitSectionsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionVsplitSectionsResourceListResult> terlan_collection_pytorch_shapepartition_vsplit_sections_resources(const Tensor & self, int64_t sections) noexcept;
bool terlan_collection_pytorch_shapepartition_vsplit_sections_resources_is_ok(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_vsplit_sections_resources_len(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_vsplit_sections_resources_element(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_vsplit_sections_resources_code(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_vsplit_sections_resources_message(const TerlanPytorchShapePartitionVsplitSectionsResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionDsplitIndicesResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionDsplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionDsplitIndicesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionDsplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_dsplit_indices_resources(const Tensor & self, rust::Slice<const std::int64_t> indices) noexcept;
bool terlan_collection_pytorch_shapepartition_dsplit_indices_resources_is_ok(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_dsplit_indices_resources_len(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_dsplit_indices_resources_element(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_dsplit_indices_resources_code(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_dsplit_indices_resources_message(const TerlanPytorchShapePartitionDsplitIndicesResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionHsplitIndicesResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionHsplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionHsplitIndicesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionHsplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_hsplit_indices_resources(const Tensor & self, rust::Slice<const std::int64_t> indices) noexcept;
bool terlan_collection_pytorch_shapepartition_hsplit_indices_resources_is_ok(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_hsplit_indices_resources_len(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_hsplit_indices_resources_element(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_hsplit_indices_resources_code(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_hsplit_indices_resources_message(const TerlanPytorchShapePartitionHsplitIndicesResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionVsplitIndicesResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionVsplitIndicesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionVsplitIndicesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionVsplitIndicesResourceListResult> terlan_collection_pytorch_shapepartition_vsplit_indices_resources(const Tensor & self, rust::Slice<const std::int64_t> indices) noexcept;
bool terlan_collection_pytorch_shapepartition_vsplit_indices_resources_is_ok(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartition_vsplit_indices_resources_len(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartition_vsplit_indices_resources_element(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_vsplit_indices_resources_code(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartition_vsplit_indices_resources_message(const TerlanPytorchShapePartitionVsplitIndicesResourceListResult& result) noexcept;

class TerlanPytorchWhereNativeNonzeroNumpyResourceListResult final {
 public:
  explicit TerlanPytorchWhereNativeNonzeroNumpyResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchWhereNativeNonzeroNumpyResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchWhereNativeNonzeroNumpyResourceListResult> terlan_collection_pytorch_wherenative_nonzero_numpy_resources(const Tensor & self) noexcept;
bool terlan_collection_pytorch_wherenative_nonzero_numpy_resources_is_ok(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_wherenative_nonzero_numpy_resources_len(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_wherenative_nonzero_numpy_resources_element(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_wherenative_nonzero_numpy_resources_code(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_wherenative_nonzero_numpy_resources_message(const TerlanPytorchWhereNativeNonzeroNumpyResourceListResult& result) noexcept;

class TerlanPytorchWhereNativeWhereIndicesResourceListResult final {
 public:
  explicit TerlanPytorchWhereNativeWhereIndicesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchWhereNativeWhereIndicesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchWhereNativeWhereIndicesResourceListResult> terlan_collection_pytorch_wherenative_where_indices_resources(const Tensor & condition) noexcept;
bool terlan_collection_pytorch_wherenative_where_indices_resources_is_ok(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_wherenative_where_indices_resources_len(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_wherenative_where_indices_resources_element(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_wherenative_where_indices_resources_code(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_wherenative_where_indices_resources_message(const TerlanPytorchWhereNativeWhereIndicesResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientArrayResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientArrayResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientArrayResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_array_resources(const at::Tensor & tensor, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_array_resources_is_ok(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_array_resources_len(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_array_resources_element(const TerlanPytorchGradientsGradientArrayResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_array_resources_code(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_array_resources_message(const TerlanPytorchGradientsGradientArrayResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientScalarArrayResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientScalarArrayResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientScalarArrayResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientScalarArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_array_resources(const at::Tensor & tensor, double spacing, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_scalar_array_resources_is_ok(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_array_resources_len(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_array_resources_element(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_resources_code(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_resources_message(const TerlanPytorchGradientsGradientScalarArrayResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientScalarIntResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientScalarIntResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientScalarIntResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientScalarIntResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_int_resources(const at::Tensor & tensor, bool has_spacing, double spacing, bool has_dim, int64_t dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_scalar_int_resources_is_ok(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_int_resources_len(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_int_resources_element(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_int_resources_code(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_int_resources_message(const TerlanPytorchGradientsGradientScalarIntResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources(const at::Tensor & tensor, rust::Slice<const double> spacing, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_is_ok(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_len(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_element(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_code(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_array_resources_message(const TerlanPytorchGradientsGradientScalarArrayArrayResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientScalarArrayIntResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientScalarArrayIntResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientScalarArrayIntResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientScalarArrayIntResourceListResult> terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources(const at::Tensor & tensor, rust::Slice<const double> spacing, bool has_dim, int64_t dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_is_ok(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_len(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_element(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_code(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_scalar_array_int_resources_message(const TerlanPytorchGradientsGradientScalarArrayIntResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientTensorArrayResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientTensorArrayResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientTensorArrayResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientTensorArrayResourceListResult> terlan_collection_pytorch_gradients_gradient_tensor_array_resources(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& spacing, rust::Slice<const std::int64_t> dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_tensor_array_resources_is_ok(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_tensor_array_resources_len(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_tensor_array_resources_element(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_resources_code(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_resources_message(const TerlanPytorchGradientsGradientTensorArrayResourceListResult& result) noexcept;

class TerlanPytorchGradientsGradientTensorArrayIntResourceListResult final {
 public:
  explicit TerlanPytorchGradientsGradientTensorArrayIntResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchGradientsGradientTensorArrayIntResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchGradientsGradientTensorArrayIntResourceListResult> terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources(const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& spacing, bool has_dim, int64_t dim, int64_t edge_order) noexcept;
bool terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_is_ok(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_len(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_element(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_code(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_gradients_gradient_tensor_array_int_resources_message(const TerlanPytorchGradientsGradientTensorArrayIntResourceListResult& result) noexcept;

class TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult final {
 public:
  explicit TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult> terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_is_ok(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_len(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_element(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_code(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_broadcast_tensors_resources_message(const TerlanPytorchTensorListTransformsBroadcastTensorsResourceListResult& result) noexcept;

class TerlanPytorchTensorListTransformsMeshgridResourceListResult final {
 public:
  explicit TerlanPytorchTensorListTransformsMeshgridResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorListTransformsMeshgridResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorListTransformsMeshgridResourceListResult> terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_is_ok(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_len(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_element(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_code(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_resources_message(const TerlanPytorchTensorListTransformsMeshgridResourceListResult& result) noexcept;

class TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult final {
 public:
  explicit TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult> terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources(const TerlanPytorchTensorTensorResourceListInput& tensors, rust::Str indexing) noexcept;
bool terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_is_ok(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_len(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_element(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_code(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_meshgrid_indexing_resources_message(const TerlanPytorchTensorListTransformsMeshgridIndexingResourceListResult& result) noexcept;

class TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult final {
 public:
  explicit TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult> terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources(const Tensor & flat, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_is_ok(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_len(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_element(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_code(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_tensorlisttransforms_unflatten_dense_tensors_resources_message(const TerlanPytorchTensorListTransformsUnflattenDenseTensorsResourceListResult& result) noexcept;

class TerlanPytorchViewAtleast1dSequenceResourceListResult final {
 public:
  explicit TerlanPytorchViewAtleast1dSequenceResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchViewAtleast1dSequenceResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchViewAtleast1dSequenceResourceListResult> terlan_collection_pytorch_view_atleast_1d_sequence_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_view_atleast_1d_sequence_resources_is_ok(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_view_atleast_1d_sequence_resources_len(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_view_atleast_1d_sequence_resources_element(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_view_atleast_1d_sequence_resources_code(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_view_atleast_1d_sequence_resources_message(const TerlanPytorchViewAtleast1dSequenceResourceListResult& result) noexcept;

class TerlanPytorchViewAtleast2dSequenceResourceListResult final {
 public:
  explicit TerlanPytorchViewAtleast2dSequenceResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchViewAtleast2dSequenceResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchViewAtleast2dSequenceResourceListResult> terlan_collection_pytorch_view_atleast_2d_sequence_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_view_atleast_2d_sequence_resources_is_ok(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_view_atleast_2d_sequence_resources_len(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_view_atleast_2d_sequence_resources_element(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_view_atleast_2d_sequence_resources_code(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_view_atleast_2d_sequence_resources_message(const TerlanPytorchViewAtleast2dSequenceResourceListResult& result) noexcept;

class TerlanPytorchViewAtleast3dSequenceResourceListResult final {
 public:
  explicit TerlanPytorchViewAtleast3dSequenceResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchViewAtleast3dSequenceResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchViewAtleast3dSequenceResourceListResult> terlan_collection_pytorch_view_atleast_3d_sequence_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_view_atleast_3d_sequence_resources_is_ok(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_view_atleast_3d_sequence_resources_len(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_view_atleast_3d_sequence_resources_element(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_view_atleast_3d_sequence_resources_code(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_view_atleast_3d_sequence_resources_message(const TerlanPytorchViewAtleast3dSequenceResourceListResult& result) noexcept;

class TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult final {
 public:
  explicit TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult> terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors, const Tensor & scales, const Tensor & zero_points, std::int64_t dtype) noexcept;
bool terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_is_ok(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_len(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_element(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_code(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_quantizationcompletion_quantize_per_tensor_tensors_resources_message(const TerlanPytorchQuantizationCompletionQuantizePerTensorTensorsResourceListResult& result) noexcept;

class TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult final {
 public:
  explicit TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult> terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources(const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept;
bool terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_is_ok(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_len(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_element(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_code(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_quantizationcompletion_dequantize_tensors_resources_message(const TerlanPytorchQuantizationCompletionDequantizeTensorsResourceListResult& result) noexcept;

class TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult final {
 public:
  explicit TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult> terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources(const HistogramddResult & result) noexcept;
bool terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_is_ok(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_len(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_element(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_code(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_histogramddnative_histogramdd_result_bin_edges_resources_message(const TerlanPytorchHistogramddNativeHistogramddResultBinEdgesResourceListResult& result) noexcept;

class TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult final {
 public:
  explicit TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult(std::vector<at::Tensor> values) noexcept;
  TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult(std::string code, std::string message) noexcept;
  bool is_ok() const noexcept;
  std::size_t len() const noexcept;
  std::unique_ptr<at::Tensor> element(std::size_t index) const noexcept;
  const std::string& code() const noexcept;
  const std::string& message() const noexcept;

 private:
  bool ok_;
  std::vector<at::Tensor> values_;
  std::string code_;
  std::string message_;
};

std::unique_ptr<TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult> terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources(const at::Tensor & input, int64_t chunks, int64_t dimension) noexcept;
bool terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_is_ok(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept;
std::size_t terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_len(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept;
std::unique_ptr<at::Tensor> terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_element(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result, std::size_t index) noexcept;
const std::string& terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_code(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept;
const std::string& terlan_collection_pytorch_shapepartitioncompletion_unsafe_chunk_resources_message(const TerlanPytorchShapePartitionCompletionUnsafeChunkResourceListResult& result) noexcept;


}  // namespace at::native
