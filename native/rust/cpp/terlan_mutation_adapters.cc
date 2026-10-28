#include "include/terlan_mutation_adapters.hpp"

#include <utility>

namespace at::native {

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_typerelationsnative_retain_grad_native(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.retain_grad();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_requires_grad_in_place(at::Tensor& self_value, bool _requires_grad) noexcept {
  try {
    try {
      (void)self_value.requires_grad_(_requires_grad);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch autograd participation mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_gather_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) noexcept {
  try {
    try {
      (void)at::gather_out(out, self, dim, index, sparse_grad);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_embedding_out(Tensor & out, const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) noexcept {
  try {
    try {
      (void)at::embedding_out(out, weight, indices, padding_idx, scale_grad_by_freq, sparse);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_embedding_renorm_in_place(Tensor & self, const Tensor & indices, double max_norm, double norm_type) noexcept {
  try {
    try {
      (void)at::embedding_renorm_(self, indices, max_norm, norm_type);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_embedding_renorm_out(Tensor & out, const Tensor & self, const Tensor & indices, double max_norm, double norm_type) noexcept {
  try {
    try {
      (void)at::embedding_renorm_out(out, self, indices, max_norm, norm_type);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_copy_out(Tensor & out, const Tensor & self, const Tensor & src, bool non_blocking) noexcept {
  try {
    try {
      (void)at::copy_out(out, self, src, non_blocking);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "LibTorch Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_to_padded_tensor_out(at::Tensor & output, const at::Tensor & tensor, double padding, bool has_output_size, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::native::tensor_to_padded_tensor_out(output, tensor, padding, has_output_size, c10::ArrayRef<int64_t>(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nested to-padded output conversion failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_from_file_out(at::Tensor & output, rust::Str filename, bool has_shared, bool shared, bool has_size, int64_t size) noexcept {
  try {
    try {
      (void)at::native::tensor_from_file_out(output, std::string_view(filename.data(), filename.size()), has_shared, shared, has_size, size);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch file-backed output construction failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_gelu_out(Tensor & out, const Tensor & self, rust::Str approximate) noexcept {
  try {
    try {
      (void)at::gelu_out(out, self, c10::string_view(approximate.data(), approximate.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_gelu_in_place(Tensor & self, rust::Str approximate) noexcept {
  try {
    try {
      (void)at::gelu_(self, c10::string_view(approximate.data(), approximate.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_copy_in_place(at::Tensor & arg0, const at::Tensor & arg1, bool arg2) noexcept {
  try {
    try {
      (void)at::native::tensor_copy_in_place(arg0, arg1, arg2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_resize_out_cxx(at::Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> size, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::resize_out(out, self, IntArrayRef(size.data(), size.size()), has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_resize_in_place_cxx(at::Tensor & self, rust::Slice<const std::int64_t> size, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::native::resize_(self, IntArrayRef(size.data(), size.size()), has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_empty_strided_out_cxx(Tensor & out, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::empty_strided_out(out, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensor_empty_permuted_out_cxx(Tensor & out, rust::Slice<const std::int64_t> physical_layout, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::empty_permuted_out(out, IntArrayRef(physical_layout.data(), physical_layout.size()), IntArrayRef(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_block_diag_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::block_diag_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_column_stack_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::column_stack_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_dstack_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::dstack_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_hstack_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::hstack_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_row_stack_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::row_stack_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_vstack_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::vstack_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_cat_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    try {
      (void)at::cat_out(out, tensors.values(), dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_stack_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    try {
      (void)at::stack_out(out, tensors.values(), dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_concat_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    try {
      (void)at::concat_out(out, tensors.values(), dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_tensorjoining_concatenate_out(at::Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors, int64_t dim) noexcept {
  try {
    try {
      (void)at::concatenate_out(out, tensors.values(), dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor joining output mutation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_dropout_out(at::Tensor & output, at::Tensor & mask, const at::Tensor & input, double p, bool train) noexcept {
  try {
    try {
      (void)at::native::native_dropout_exact_out(output, mask, input, p, train);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_dropout output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_layer_norm_out(at::Tensor & output, at::Tensor & mean, at::Tensor & rstd, const at::Tensor & input, rust::Slice<const std::int64_t> normalized_shape, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, double eps) noexcept {
  try {
    try {
      (void)at::native::native_layer_norm_exact_out(output, mean, rstd, input, c10::ArrayRef<std::int64_t>(normalized_shape.data(), normalized_shape.size()), has_weight, weight, has_bias, bias, eps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_layer_norm output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_group_norm_out(at::Tensor & output, at::Tensor & mean, at::Tensor & rstd, const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, std::int64_t n, std::int64_t c, std::int64_t hxw, std::int64_t group, double eps) noexcept {
  try {
    try {
      (void)at::native::native_group_norm_exact_out(output, mean, rstd, input, has_weight, weight, has_bias, bias, n, c, hxw, group, eps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_group_norm output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_group_norm_backward_out(at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, bool has_weight, const at::Tensor & weight, std::int64_t n, std::int64_t c, std::int64_t hxw, std::int64_t group, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    try {
      (void)at::native::native_group_norm_backward_exact_out(grad_input, grad_weight, grad_bias, grad_out, input, mean, rstd, has_weight, weight, n, c, hxw, group, output_mask_input, output_mask_weight, output_mask_bias);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_group_norm_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_layer_norm_backward_out(at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias, const at::Tensor & grad_out, const at::Tensor & input, rust::Slice<const std::int64_t> normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    try {
      (void)at::native::native_layer_norm_backward_exact_out(grad_input, grad_weight, grad_bias, grad_out, input, c10::ArrayRef<std::int64_t>(normalized_shape.data(), normalized_shape.size()), mean, rstd, has_weight, weight, has_bias, bias, output_mask_input, output_mask_weight, output_mask_bias);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_layer_norm_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_batch_norm_out(at::Tensor & output, at::Tensor & save_mean, at::Tensor & save_invstd, const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, bool training, double momentum, double eps) noexcept {
  try {
    try {
      (void)at::native::native_batch_norm_exact_out(output, save_mean, save_invstd, input, has_weight, weight, has_bias, bias, has_running_mean, running_mean, has_running_var, running_var, training, momentum, eps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_batch_norm output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_native_batch_norm_backward_out(at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias, const at::Tensor & grad_out, const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, bool has_save_mean, const at::Tensor & save_mean, bool has_save_invstd, const at::Tensor & save_invstd, bool train, double eps, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    try {
      (void)at::native::native_batch_norm_backward_exact_out(grad_input, grad_weight, grad_bias, grad_out, input, has_weight, weight, has_running_mean, running_mean, has_running_var, running_var, has_save_mean, save_mean, has_save_invstd, save_invstd, train, eps, output_mask_input, output_mask_weight, output_mask_bias);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor native_batch_norm_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_log_sigmoid_forward_out(at::Tensor & output, at::Tensor & buffer, const at::Tensor & input) noexcept {
  try {
    try {
      (void)at::native::log_sigmoid_forward_exact_out(output, buffer, input);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor log_sigmoid_forward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_log_sigmoid_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & buffer) noexcept {
  try {
    try {
      (void)at::native::log_sigmoid_backward_exact_out(grad_input, grad_output, input, buffer);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor log_sigmoid_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_regularization_batch_norm_update_stats_out(at::Tensor & out_mean, at::Tensor & out_var, const at::Tensor & input, bool has_running_mean, const at::Tensor & running_mean, bool has_running_var, const at::Tensor & running_var, double momentum) noexcept {
  try {
    try {
      (void)at::native::batch_norm_update_stats_exact_out(out_mean, out_var, input, has_running_mean, running_mean, has_running_var, running_var, momentum);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor batch_norm_update_stats output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dropoutcompletion_alpha_dropout_in_place(Tensor & self, double p, bool train) noexcept {
  try {
    try {
      (void)at::alpha_dropout_(self, p, train);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place alpha dropout failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dropoutcompletion_dropout_in_place(Tensor & self, double p, bool train) noexcept {
  try {
    try {
      (void)at::dropout_(self, p, train);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place dropout failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dropoutcompletion_feature_alpha_dropout_in_place(Tensor & self, double p, bool train) noexcept {
  try {
    try {
      (void)at::feature_alpha_dropout_(self, p, train);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place feature alpha dropout failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dropoutcompletion_feature_dropout_in_place(Tensor & self, double p, bool train) noexcept {
  try {
    try {
      (void)at::feature_dropout_(self, p, train);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place feature dropout failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_glucompletion_glu_out(Tensor & out, const Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::glu_out(out, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch GLU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_glucompletion_glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::glu_backward_out(grad_input, grad_output, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch GLU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_glucompletion_glu_jvp_out(Tensor & out, const Tensor & glu, const Tensor & x, const Tensor & dx, int64_t dim) noexcept {
  try {
    try {
      (void)at::glu_jvp_out(out, glu, x, dx, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch GLU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_glucompletion_glu_backward_jvp_out(Tensor & out, const Tensor & grad_x, const Tensor & grad_glu, const Tensor & x, const Tensor & dgrad_glu, const Tensor & dx, int64_t dim) noexcept {
  try {
    try {
      (void)at::glu_backward_jvp_out(out, grad_x, grad_glu, x, dgrad_glu, dx, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch GLU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_avg_pool1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::adaptive_avg_pool1d_out(out, self, IntArrayRef(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::adaptive_avg_pool2d_out(out, self, IntArrayRef(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_avg_pool3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::adaptive_avg_pool3d_out(out, self, IntArrayRef(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    try {
      (void)at::adaptive_avg_pool3d_backward_out(grad_input, grad_output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_max_pool2d_out(Tensor & out, Tensor & indices, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::adaptive_max_pool2d_out(out, indices, self, IntArrayRef(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_max_pool3d_out(Tensor & out, Tensor & indices, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::adaptive_max_pool3d_out(out, indices, self, IntArrayRef(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) noexcept {
  try {
    try {
      (void)at::adaptive_max_pool2d_backward_out(grad_input, grad_output, self, indices);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) noexcept {
  try {
    try {
      (void)at::adaptive_max_pool3d_backward_out(grad_input, grad_output, self, indices);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    try {
      (void)at::avg_pool1d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    try {
      (void)at::avg_pool2d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    try {
      (void)at::avg_pool3d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool2d_out_divisor(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    try {
      (void)at::avg_pool2d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool3d_out_divisor(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    try {
      (void)at::avg_pool3d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    try {
      (void)at::avg_pool2d_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad) noexcept {
  try {
    try {
      (void)at::avg_pool3d_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool2d_backward_out_divisor(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    try {
      (void)at::avg_pool2d_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_avg_pool3d_backward_out_divisor(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, bool ceil_mode, bool count_include_pad, std::int64_t divisor_override) noexcept {
  try {
    try {
      (void)at::avg_pool3d_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), ceil_mode, count_include_pad, ::std::optional<int64_t>(divisor_override));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & random_samples) noexcept {
  try {
    try {
      (void)at::fractional_max_pool2d_out(output, indices, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), random_samples);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_fractional_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & random_samples) noexcept {
  try {
    try {
      (void)at::fractional_max_pool3d_out(output, indices, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), random_samples);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & indices) noexcept {
  try {
    try {
      (void)at::fractional_max_pool2d_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), indices);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_fractional_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> output_size, const Tensor & indices) noexcept {
  try {
    try {
      (void)at::fractional_max_pool3d_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(output_size.data(), output_size.size()), indices);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    try {
      (void)at::max_pool2d_with_indices_out(out, indices, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_max_pool3d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    try {
      (void)at::max_pool3d_with_indices_out(out, indices, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode, const Tensor & indices) noexcept {
  try {
    try {
      (void)at::max_pool2d_with_indices_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode, indices);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode, const Tensor & indices) noexcept {
  try {
    try {
      (void)at::max_pool3d_with_indices_backward_out(grad_input, grad_output, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode, indices);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_max_unpool2d_out(Tensor & out, const Tensor & self, const Tensor & indices, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::max_unpool2d_out(out, self, indices, IntArrayRef(output_size.data(), output_size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_pooling_max_unpool3d_out(Tensor & out, const Tensor & self, const Tensor & indices, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::max_unpool3d_out(out, self, indices, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch pooling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_spatial_affine_grid_generator_out(Tensor & out, const Tensor & theta, rust::Slice<const std::int64_t> size, bool align_corners) noexcept {
  try {
    try {
      (void)at::affine_grid_generator_out(out, theta, IntArrayRef(size.data(), size.size()), align_corners);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch spatial operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_spatial_grid_sampler_2d_out(Tensor & out, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept {
  try {
    try {
      (void)at::grid_sampler_2d_out(out, input, grid, interpolation_mode, padding_mode, align_corners);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch spatial operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_spatial_grid_sampler_3d_out(Tensor & out, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) noexcept {
  try {
    try {
      (void)at::grid_sampler_3d_out(out, input, grid, interpolation_mode, padding_mode, align_corners);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch spatial operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_spatial_grid_sampler_2d_backward_out(at::Tensor & grad_input, at::Tensor & grad_grid, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, std::int64_t interpolation_mode, std::int64_t padding_mode, bool align_corners, bool output_mask_input, bool output_mask_grid) noexcept {
  try {
    try {
      (void)at::native::grid_sampler_2d_backward_exact_out(grad_input, grad_grid, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask_input, output_mask_grid);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor grid_sampler_2d_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_spatial_grid_sampler_3d_backward_out(at::Tensor & grad_input, at::Tensor & grad_grid, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, std::int64_t interpolation_mode, std::int64_t padding_mode, bool align_corners, bool output_mask_input, bool output_mask_grid) noexcept {
  try {
    try {
      (void)at::native::grid_sampler_3d_backward_exact_out(grad_input, grad_grid, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask_input, output_mask_grid);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor grid_sampler_3d_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapegradient_diagonal_scatter_out(Tensor & out, const Tensor & self, const Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    try {
      (void)at::diagonal_scatter_out(out, self, src, offset, dim1, dim2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapegradient_diagonal_backward_out(Tensor & out, const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    try {
      (void)at::diagonal_backward_out(out, grad_output, IntArrayRef(input_sizes.data(), input_sizes.size()), offset, dim1, dim2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapegradient_select_backward_out(Tensor & out, const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t index) noexcept {
  try {
    try {
      (void)at::select_backward_out(out, grad_output, IntArrayRef(input_sizes.data(), input_sizes.size()), dim, index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapegradient_slice_backward_out(Tensor & out, const Tensor & grad_output, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_backward_out(out, grad_output, IntArrayRef(input_sizes.data(), input_sizes.size()), dim, start, end, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapegradient_unfold_backward_out(Tensor & out, const Tensor & grad_in, rust::Slice<const std::int64_t> input_sizes, int64_t dim, int64_t size, int64_t step) noexcept {
  try {
    try {
      (void)at::unfold_backward_out(out, grad_in, IntArrayRef(input_sizes.data(), input_sizes.size()), dim, size, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingbackwardcompletion_reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::reflection_pad1d_backward_out(grad_input, grad_output, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingbackwardcompletion_reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::reflection_pad2d_backward_out(grad_input, grad_output, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingbackwardcompletion_reflection_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::reflection_pad3d_backward_out(grad_input, grad_output, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingbackwardcompletion_replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::replication_pad1d_backward_out(grad_input, grad_output, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingbackwardcompletion_replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::replication_pad2d_backward_out(grad_input, grad_output, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingbackwardcompletion_replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::replication_pad3d_backward_out(grad_input, grad_output, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nantonumnative_nan_to_num_configured_out(Tensor & out, const Tensor & self, bool has_nan, double nan, bool has_posinf, double posinf, bool has_neginf, double neginf) noexcept {
  try {
    try {
      (void)at::nan_to_num_out(out, self, has_nan ? ::std::optional<double>(nan) : std::nullopt, has_posinf ? ::std::optional<double>(posinf) : std::nullopt, has_neginf ? ::std::optional<double>(neginf) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nan-to-number operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nantonumnative_nan_to_num_configured_in_place(Tensor & self, bool has_nan, double nan, bool has_posinf, double posinf, bool has_neginf, double neginf) noexcept {
  try {
    try {
      (void)at::nan_to_num_(self, has_nan ? ::std::optional<double>(nan) : std::nullopt, has_posinf ? ::std::optional<double>(posinf) : std::nullopt, has_neginf ? ::std::optional<double>(neginf) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nan-to-number operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_strided_as_strided_in_place(at::Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::as_strided_(self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_strided_as_strided_copy_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::as_strided_copy_out(out, self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_strided_as_strided_scatter_out(Tensor & out, const Tensor & self, const Tensor & src, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::as_strided_scatter_out(out, self, src, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_strided_as_strided_in_place_offset(at::Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept {
  try {
    try {
      (void)at::as_strided_(self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), ::std::optional<int64_t>(storage_offset));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_strided_as_strided_copy_out_offset(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept {
  try {
    try {
      (void)at::as_strided_copy_out(out, self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), ::std::optional<int64_t>(storage_offset));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_strided_as_strided_scatter_out_offset(Tensor & out, const Tensor & self, const Tensor & src, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride, std::int64_t storage_offset) noexcept {
  try {
    try {
      (void)at::as_strided_scatter_out(out, self, src, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()), ::std::optional<int64_t>(storage_offset));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexfillnative_index_fill_scalar_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::index_fill_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexfillnative_index_fill_scalar_in_place(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::index_fill_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexfillnative_index_fill_tensor_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::index_fill_out);
      (void)terlan_selected_overload(out, self, dim, index, value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexfillnative_index_fill_tensor_in_place(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::index_fill_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lerpnative_lerp_scalar_out(Tensor & out, const Tensor & self, const Tensor & end, bool weight_is_int, std::int64_t weight_int, double weight_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lerp_out);
      (void)terlan_selected_overload(out, self, end, weight_is_int ? c10::Scalar(weight_int) : c10::Scalar(weight_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear-interpolation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lerpnative_lerp_tensor_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lerp_out);
      (void)terlan_selected_overload(out, self, end, weight);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear-interpolation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lerpnative_lerp_tensor_in_place(at::Tensor& self_value, const Tensor & end, const Tensor & weight) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::lerp_);
      (void)(self_value.*terlan_selected_overload)(end, weight);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear-interpolation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_wherenative_where_tensor_out(Tensor & out, const Tensor & condition, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::where_out(out, condition, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch condition-selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexmutationnative_index_add_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)at::index_add_out(out, self, dim, index, source, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-mutation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexmutationnative_index_add_in_place(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & source, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.index_add_(dim, index, source, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-mutation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexmutationnative_index_copy_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) noexcept {
  try {
    try {
      (void)at::index_copy_out(out, self, dim, index, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-mutation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexmutationnative_index_copy_in_place(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & source) noexcept {
  try {
    try {
      (void)self_value.index_copy_(dim, index, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-mutation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexmutationnative_index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) noexcept {
  try {
    try {
      (void)at::index_select_out(out, self, dim, index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch index-mutation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_fill_scalar_out(Tensor & out, const Tensor & self, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fill_out);
      (void)terlan_selected_overload(out, self, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_fill_scalar_in_place(at::Tensor& self_value, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::fill_);
      (void)(self_value.*terlan_selected_overload)(value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_fill_tensor_out(Tensor & out, const Tensor & self, const Tensor & value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fill_out);
      (void)terlan_selected_overload(out, self, value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_fill_tensor_in_place(at::Tensor& self_value, const Tensor & value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::fill_);
      (void)(self_value.*terlan_selected_overload)(value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_fill_diagonal_in_place(at::Tensor& self_value, bool value_is_int, std::int64_t value_int, double value_float, bool wrap) noexcept {
  try {
    try {
      (void)self_value.fill_diagonal_(value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), wrap);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_masked_fill_scalar_out(Tensor & out, const Tensor & self, const Tensor & mask, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::masked_fill_out);
      (void)terlan_selected_overload(out, self, mask, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_masked_fill_scalar_in_place(at::Tensor& self_value, const Tensor & mask, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::masked_fill_);
      (void)(self_value.*terlan_selected_overload)(mask, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_masked_fill_tensor_out(Tensor & out, const Tensor & self, const Tensor & mask, const Tensor & value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::masked_fill_out);
      (void)terlan_selected_overload(out, self, mask, value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fillnative_masked_fill_tensor_in_place(at::Tensor& self_value, const Tensor & mask, const Tensor & value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::masked_fill_);
      (void)(self_value.*terlan_selected_overload)(mask, value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fill operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_hardshrink_out_int(Tensor & out, const Tensor & self, std::int64_t lambd) noexcept {
  try {
    try {
      (void)at::hardshrink_out(out, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_hardshrink_out_float(Tensor & out, const Tensor & self, double lambd) noexcept {
  try {
    try {
      (void)at::hardshrink_out(out, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_hardshrink_backward_out_int(Tensor & grad_input, const Tensor & grad_out, const Tensor & self, std::int64_t lambd) noexcept {
  try {
    try {
      (void)at::hardshrink_backward_out(grad_input, grad_out, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_hardshrink_backward_out_float(Tensor & grad_input, const Tensor & grad_out, const Tensor & self, double lambd) noexcept {
  try {
    try {
      (void)at::hardshrink_backward_out(grad_input, grad_out, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softshrink_out_int(Tensor & out, const Tensor & self, std::int64_t lambd) noexcept {
  try {
    try {
      (void)at::softshrink_out(out, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softshrink_out_float(Tensor & out, const Tensor & self, double lambd) noexcept {
  try {
    try {
      (void)at::softshrink_out(out, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softshrink_backward_out_int(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, std::int64_t lambd) noexcept {
  try {
    try {
      (void)at::softshrink_backward_out(grad_input, grad_output, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softshrink_backward_out_float(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, double lambd) noexcept {
  try {
    try {
      (void)at::softshrink_backward_out(grad_input, grad_output, self, c10::Scalar(lambd));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softplus_out_int(Tensor & out, const Tensor & self, std::int64_t beta, std::int64_t threshold) noexcept {
  try {
    try {
      (void)at::softplus_out(out, self, c10::Scalar(beta), c10::Scalar(threshold));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softplus_out_float(Tensor & out, const Tensor & self, double beta, double threshold) noexcept {
  try {
    try {
      (void)at::softplus_out(out, self, c10::Scalar(beta), c10::Scalar(threshold));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softplus_backward_out_int(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, std::int64_t beta, std::int64_t threshold) noexcept {
  try {
    try {
      (void)at::softplus_backward_out(grad_input, grad_output, self, c10::Scalar(beta), c10::Scalar(threshold));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_softplus_backward_out_float(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, double beta, double threshold) noexcept {
  try {
    try {
      (void)at::softplus_backward_out(grad_input, grad_output, self, c10::Scalar(beta), c10::Scalar(threshold));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_threshold_out_int(Tensor & out, const Tensor & self, std::int64_t threshold, std::int64_t value) noexcept {
  try {
    try {
      (void)at::threshold_out(out, self, c10::Scalar(threshold), c10::Scalar(value));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_threshold_out_float(Tensor & out, const Tensor & self, double threshold, double value) noexcept {
  try {
    try {
      (void)at::threshold_out(out, self, c10::Scalar(threshold), c10::Scalar(value));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_threshold_backward_out_int(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, std::int64_t threshold) noexcept {
  try {
    try {
      (void)at::threshold_backward_out(grad_input, grad_output, self, c10::Scalar(threshold));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_activationcompletion_threshold_backward_out_float(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, double threshold) noexcept {
  try {
    try {
      (void)at::threshold_backward_out(grad_input, grad_output, self, c10::Scalar(threshold));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch activation-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_celu_out_int(Tensor & out, const Tensor & self, std::int64_t alpha) noexcept {
  try {
    try {
      (void)at::celu_out(out, self, c10::Scalar(alpha));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_celu_out_float(Tensor & out, const Tensor & self, double alpha) noexcept {
  try {
    try {
      (void)at::celu_out(out, self, c10::Scalar(alpha));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_celu_in_place_int(Tensor & self, std::int64_t alpha) noexcept {
  try {
    try {
      (void)at::celu_(self, c10::Scalar(alpha));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_celu_in_place_float(Tensor & self, double alpha) noexcept {
  try {
    try {
      (void)at::celu_(self, c10::Scalar(alpha));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_elu_out_int(Tensor & out, const Tensor & self, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale) noexcept {
  try {
    try {
      (void)at::elu_out(out, self, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_elu_out_float(Tensor & out, const Tensor & self, double alpha, double scale, double input_scale) noexcept {
  try {
    try {
      (void)at::elu_out(out, self, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_elu_in_place_int(Tensor & self, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale) noexcept {
  try {
    try {
      (void)at::elu_(self, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_elu_in_place_float(Tensor & self, double alpha, double scale, double input_scale) noexcept {
  try {
    try {
      (void)at::elu_(self, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_elu_backward_out_int(Tensor & grad_input, const Tensor & grad_output, std::int64_t alpha, std::int64_t scale, std::int64_t input_scale, bool is_result, const Tensor & self_or_result) noexcept {
  try {
    try {
      (void)at::elu_backward_out(grad_input, grad_output, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale), is_result, self_or_result);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_elu_backward_out_float(Tensor & grad_input, const Tensor & grad_output, double alpha, double scale, double input_scale, bool is_result, const Tensor & self_or_result) noexcept {
  try {
    try {
      (void)at::elu_backward_out(grad_input, grad_output, c10::Scalar(alpha), c10::Scalar(scale), c10::Scalar(input_scale), is_result, self_or_result);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_hardtanh_out_int(Tensor & out, const Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept {
  try {
    try {
      (void)at::hardtanh_out(out, self, c10::Scalar(min_val), c10::Scalar(max_val));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_hardtanh_out_float(Tensor & out, const Tensor & self, double min_val, double max_val) noexcept {
  try {
    try {
      (void)at::hardtanh_out(out, self, c10::Scalar(min_val), c10::Scalar(max_val));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_hardtanh_in_place_int(Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept {
  try {
    try {
      (void)at::hardtanh_(self, c10::Scalar(min_val), c10::Scalar(max_val));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_hardtanh_in_place_float(Tensor & self, double min_val, double max_val) noexcept {
  try {
    try {
      (void)at::hardtanh_(self, c10::Scalar(min_val), c10::Scalar(max_val));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_hardtanh_backward_out_int(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, std::int64_t min_val, std::int64_t max_val) noexcept {
  try {
    try {
      (void)at::hardtanh_backward_out(grad_input, grad_output, self, c10::Scalar(min_val), c10::Scalar(max_val));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_hardtanh_backward_out_float(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, double min_val, double max_val) noexcept {
  try {
    try {
      (void)at::hardtanh_backward_out(grad_input, grad_output, self, c10::Scalar(min_val), c10::Scalar(max_val));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_leaky_relu_out_int(Tensor & out, const Tensor & self, std::int64_t negative_slope) noexcept {
  try {
    try {
      (void)at::leaky_relu_out(out, self, c10::Scalar(negative_slope));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_leaky_relu_out_float(Tensor & out, const Tensor & self, double negative_slope) noexcept {
  try {
    try {
      (void)at::leaky_relu_out(out, self, c10::Scalar(negative_slope));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_leaky_relu_in_place_int(Tensor & self, std::int64_t negative_slope) noexcept {
  try {
    try {
      (void)at::leaky_relu_(self, c10::Scalar(negative_slope));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_leaky_relu_in_place_float(Tensor & self, double negative_slope) noexcept {
  try {
    try {
      (void)at::leaky_relu_(self, c10::Scalar(negative_slope));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_leaky_relu_backward_out_int(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, std::int64_t negative_slope, bool self_is_result) noexcept {
  try {
    try {
      (void)at::leaky_relu_backward_out(grad_input, grad_output, self, c10::Scalar(negative_slope), self_is_result);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_extendedactivationcompletion_leaky_relu_backward_out_float(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, double negative_slope, bool self_is_result) noexcept {
  try {
    try {
      (void)at::leaky_relu_backward_out(grad_input, grad_output, self, c10::Scalar(negative_slope), self_is_result);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch extended-activation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_nll_loss_forward_out(at::Tensor & output, at::Tensor & total_weight, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept {
  try {
    try {
      (void)at::native::nll_loss_forward_exact_out(output, total_weight, input, target, has_weight, weight, reduction, ignore_index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor nll_loss_forward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_nll_loss_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index, const at::Tensor & total_weight) noexcept {
  try {
    try {
      (void)at::native::nll_loss_backward_exact_out(grad_input, grad_output, input, target, has_weight, weight, reduction, ignore_index, total_weight);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor nll_loss_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_nll_loss2d_forward_out(at::Tensor & output, at::Tensor & total_weight, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index) noexcept {
  try {
    try {
      (void)at::native::nll_loss2d_forward_exact_out(output, total_weight, input, target, has_weight, weight, reduction, ignore_index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor nll_loss2d_forward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_nll_loss2d_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, bool has_weight, const at::Tensor & weight, std::int64_t reduction, std::int64_t ignore_index, const at::Tensor & total_weight) noexcept {
  try {
    try {
      (void)at::native::nll_loss2d_backward_exact_out(grad_input, grad_output, input, target, has_weight, weight, reduction, ignore_index, total_weight);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor nll_loss2d_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_huber_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction, double delta) noexcept {
  try {
    try {
      (void)at::huber_loss_out(out, self, target, reduction, delta);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_huber_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double delta) noexcept {
  try {
    try {
      (void)at::huber_loss_backward_out(grad_input, grad_output, self, target, reduction, delta);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_mse_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::mse_loss_out(out, self, target, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::mse_loss_backward_out(grad_input, grad_output, self, target, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_smooth_l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction, double beta) noexcept {
  try {
    try {
      (void)at::smooth_l1_loss_out(out, self, target, reduction, beta);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, double beta) noexcept {
  try {
    try {
      (void)at::smooth_l1_loss_backward_out(grad_input, grad_output, self, target, reduction, beta);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_soft_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::soft_margin_loss_out(out, self, target, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_loss_soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::soft_margin_loss_backward_out(grad_input, grad_output, self, target, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_channel_shuffle_out(Tensor & out, const Tensor & self, int64_t groups) noexcept {
  try {
    try {
      (void)at::channel_shuffle_out(out, self, groups);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_transpose_copy_out(Tensor & out, const Tensor & self, int64_t dim0, int64_t dim1) noexcept {
  try {
    try {
      (void)at::transpose_copy_out(out, self, dim0, dim1);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_select_copy_out(Tensor & out, const Tensor & self, int64_t dim, int64_t index) noexcept {
  try {
    try {
      (void)at::select_copy_out(out, self, dim, index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_select_scatter_out(Tensor & out, const Tensor & self, const Tensor & src, int64_t dim, int64_t index) noexcept {
  try {
    try {
      (void)at::select_scatter_out(out, self, src, dim, index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_narrow_copy_out(Tensor & out, const Tensor & self, int64_t dim, int64_t start, int64_t length) noexcept {
  try {
    try {
      (void)at::narrow_copy_out(out, self, dim, start, length);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_unfold_copy_out(Tensor & out, const Tensor & self, int64_t dimension, int64_t size, int64_t step) noexcept {
  try {
    try {
      (void)at::unfold_copy_out(out, self, dimension, size, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_zero_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::zero_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_im2col_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> dilation, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::im2col_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(dilation.data(), dilation.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_col2im_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> dilation, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::col2im_out(out, self, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(dilation.data(), dilation.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shapecompletion_tensordot_out(Tensor & out, const Tensor & self, const Tensor & other, rust::Slice<const std::int64_t> dims_self, rust::Slice<const std::int64_t> dims_other) noexcept {
  try {
    try {
      (void)at::tensordot_out(out, self, other, IntArrayRef(dims_self.data(), dims_self.size()), IntArrayRef(dims_other.data(), dims_other.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_masked_scatter_out(Tensor & out, const Tensor & self, const Tensor & mask, const Tensor & source) noexcept {
  try {
    try {
      (void)at::masked_scatter_out(out, self, mask, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexing operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_masked_scatter_in_place(at::Tensor& self_value, const Tensor & mask, const Tensor & source) noexcept {
  try {
    try {
      (void)self_value.masked_scatter_(mask, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexing operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_put_out(Tensor & out, const Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) noexcept {
  try {
    try {
      (void)at::put_out(out, self, index, source, accumulate);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexing operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_put_in_place(at::Tensor& self_value, const Tensor & index, const Tensor & source, bool accumulate) noexcept {
  try {
    try {
      (void)self_value.put_(index, source, accumulate);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexing operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_index_tensor_out(at::Tensor & output, const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present) noexcept {
  try {
    try {
      (void)at::native::tensor_index_out(output, tensor, indices.values(), c10::ArrayRef<std::int64_t>(present.data(), present.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexed output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_index_put_out(at::Tensor & output, const at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present, const at::Tensor & values, bool accumulate) noexcept {
  try {
    try {
      (void)at::native::tensor_index_put_out(output, tensor, indices.values(), c10::ArrayRef<std::int64_t>(present.data(), present.size()), values, accumulate);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexed output update failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_indexing_index_put_in_place(at::Tensor & tensor, const TerlanPytorchTensorTensorResourceListInput& indices, rust::Slice<const std::int64_t> present, const at::Tensor & values, bool accumulate) noexcept {
  try {
    try {
      (void)at::native::tensor_index_put_in_place(tensor, indices.values(), c10::ArrayRef<std::int64_t>(present.data(), present.size()), values, accumulate);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch indexed in-place update failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_gradients_embedding_dense_backward_out(Tensor & out, const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) noexcept {
  try {
    try {
      (void)at::embedding_dense_backward_out(out, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_gradients_gelu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, rust::Str approximate) noexcept {
  try {
    try {
      (void)at::gelu_backward_out(grad_input, grad_output, self, c10::string_view(approximate.data(), approximate.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_gradients_logit_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) noexcept {
  try {
    try {
      (void)at::logit_backward_out(grad_input, grad_output, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_gradients_logit_backward_epsilon_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, double eps) noexcept {
  try {
    try {
      (void)at::logit_backward_out(grad_input, grad_output, self, ::std::optional<double>(eps));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch gradient operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bmm_out);
      (void)terlan_selected_overload(out, self, mat2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_ger_out(Tensor & out, const Tensor & self, const Tensor & vec2) noexcept {
  try {
    try {
      (void)at::ger_out(out, self, vec2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_matmul_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::matmul_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_mm_out(Tensor & out, const Tensor & self, const Tensor & mat2) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mm_out);
      (void)terlan_selected_overload(out, self, mat2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_mv_out(Tensor & out, const Tensor & self, const Tensor & vec) noexcept {
  try {
    try {
      (void)at::mv_out(out, self, vec);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_orgqr_out(Tensor & out, const Tensor & self, const Tensor & input2) noexcept {
  try {
    try {
      (void)at::orgqr_out(out, self, input2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_dense_outer_out(Tensor & out, const Tensor & self, const Tensor & vec2) noexcept {
  try {
    try {
      (void)at::outer_out(out, self, vec2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dense operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_linear_out_no_bias(Tensor & out, const Tensor & input, const Tensor & weight) noexcept {
  try {
    try {
      (void)at::linear_out(out, input, weight, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_linear_out_with_bias(Tensor & out, const Tensor & input, const Tensor & weight, const at::Tensor& bias) noexcept {
  try {
    try {
      (void)at::linear_out(out, input, weight, std::optional<at::Tensor>(bias));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_log_softmax_out_default_dtype(Tensor & out, const Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::log_softmax_out(out, self, dim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_log_softmax_out_dtype(Tensor & out, const Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    try {
      (void)at::log_softmax_out(out, self, dim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_diff_out_basic(Tensor & out, const Tensor & self, int64_t n, int64_t dim) noexcept {
  try {
    try {
      (void)at::diff_out(out, self, n, dim, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_diff_out_prepend(Tensor & out, const Tensor & self, int64_t n, int64_t dim, const at::Tensor& prepend) noexcept {
  try {
    try {
      (void)at::diff_out(out, self, n, dim, std::optional<at::Tensor>(prepend), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_diff_out_append(Tensor & out, const Tensor & self, int64_t n, int64_t dim, const at::Tensor& append) noexcept {
  try {
    try {
      (void)at::diff_out(out, self, n, dim, std::nullopt, std::optional<at::Tensor>(append));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_functionalcompletion_diff_out_both(Tensor & out, const Tensor & self, int64_t n, int64_t dim, const at::Tensor& prepend, const at::Tensor& append) noexcept {
  try {
    try {
      (void)at::diff_out(out, self, n, dim, std::optional<at::Tensor>(prepend), std::optional<at::Tensor>(append));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch functional-completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_paddingnative_constant_pad_configured_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> pad, bool value_is_int, std::int64_t integer_value, double floating_value) noexcept {
  try {
    try {
      (void)at::constant_pad_nd_out(out, self, IntArrayRef(pad.data(), pad.size()), value_is_int ? c10::Scalar(integer_value) : c10::Scalar(floating_value));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch padding operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_out(out, self, target, std::nullopt, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_weight_out(Tensor & out, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_out(out, self, target, std::optional<at::Tensor>(weight), reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_backward_out(grad_input, grad_output, self, target, std::nullopt, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_backward_weight_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_backward_out(grad_input, grad_output, self, target, std::optional<at::Tensor>(weight), reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_with_logits_out(out, self, target, std::nullopt, std::nullopt, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_weight_out(Tensor & out, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_with_logits_out(out, self, target, std::optional<at::Tensor>(weight), std::nullopt, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_positive_weight_out(Tensor & out, const Tensor & self, const Tensor & target, const at::Tensor& pos_weight, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_with_logits_out(out, self, target, std::nullopt, std::optional<at::Tensor>(pos_weight), reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binarycrossentropycompletion_binary_cross_entropy_with_logits_both_weights_out(Tensor & out, const Tensor & self, const Tensor & target, const at::Tensor& weight, const at::Tensor& pos_weight, int64_t reduction) noexcept {
  try {
    try {
      (void)at::binary_cross_entropy_with_logits_out(out, self, target, std::optional<at::Tensor>(weight), std::optional<at::Tensor>(pos_weight), reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-cross-entropy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::div_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_int_mode_in_place(at::Tensor& self_value, std::int64_t other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &, ::std::optional<c10::string_view>) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::div_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_int_mode_out(Tensor & out, const Tensor & self, std::int64_t other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::div_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_float_mode_in_place(at::Tensor& self_value, double other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &, ::std::optional<c10::string_view>) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::div_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_right_float_mode_out(Tensor & out, const Tensor & self, double other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_tensor_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_tensor_mode_out(Tensor & out, const Tensor & self, const Tensor & other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::div_out);
      (void)terlan_selected_overload(out, self, other, ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_tensor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::div_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_div_tensor_mode_in_place(at::Tensor& self_value, const Tensor & other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, ::std::optional<c10::string_view>) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::div_);
      (void)(self_value.*terlan_selected_overload)(other, ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_right_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_right_int_mode_in_place(at::Tensor& self_value, std::int64_t other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &, ::std::optional<c10::string_view>) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_right_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_right_float_mode_in_place(at::Tensor& self_value, double other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &, ::std::optional<c10::string_view>) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other), ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_tensor_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_tensor_mode_out(Tensor & out, const Tensor & self, const Tensor & other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, ::std::optional<c10::string_view>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::divide_out);
      (void)terlan_selected_overload(out, self, other, ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_tensor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::divide_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_divide_tensor_mode_in_place(at::Tensor& self_value, const Tensor & other, rust::Str rounding_mode) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, ::std::optional<c10::string_view>) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::divide_);
      (void)(self_value.*terlan_selected_overload)(other, ::std::optional<c10::string_view>(std::basic_string_view<char>(rounding_mode.data(), rounding_mode.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_true_divide_right_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::true_divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_true_divide_right_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::true_divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_true_divide_tensor_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::true_divide_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_division_true_divide_tensor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::true_divide_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch division operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_copy_out_all(Tensor & out, const Tensor & self, int64_t dim, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_copy_out(out, self, dim, std::nullopt, std::nullopt, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_copy_out_from(Tensor & out, const Tensor & self, int64_t dim, std::int64_t start, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_copy_out(out, self, dim, ::std::optional<int64_t>(start), std::nullopt, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_copy_out_until(Tensor & out, const Tensor & self, int64_t dim, std::int64_t end, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_copy_out(out, self, dim, std::nullopt, ::std::optional<int64_t>(end), step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_copy_out_bounds(Tensor & out, const Tensor & self, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_copy_out(out, self, dim, ::std::optional<int64_t>(start), ::std::optional<int64_t>(end), step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_scatter_out_all(Tensor & out, const Tensor & self, const Tensor & src, int64_t dim, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_scatter_out(out, self, src, dim, std::nullopt, std::nullopt, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_scatter_out_from(Tensor & out, const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_scatter_out(out, self, src, dim, ::std::optional<int64_t>(start), std::nullopt, step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_scatter_out_until(Tensor & out, const Tensor & self, const Tensor & src, int64_t dim, std::int64_t end, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_scatter_out(out, self, src, dim, std::nullopt, ::std::optional<int64_t>(end), step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_slicenative_slice_scatter_out_bounds(Tensor & out, const Tensor & self, const Tensor & src, int64_t dim, std::int64_t start, std::int64_t end, int64_t step) noexcept {
  try {
    try {
      (void)at::slice_scatter_out(out, self, src, dim, ::std::optional<int64_t>(start), ::std::optional<int64_t>(end), step);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch slice operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_fft_out(Tensor & out, const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_fft_out(out, self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_hfft_out(Tensor & out, const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_hfft_out(out, self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_ifft_out(Tensor & out, const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_ifft_out(out, self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_ihfft_out(Tensor & out, const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_ihfft_out(out, self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_irfft_out(Tensor & out, const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_irfft_out(out, self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_rfft_out(Tensor & out, const Tensor & self, int64_t dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_rfft_out(out, self, std::nullopt, dim, ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_fft2_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_fft2_out(out, self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_hfft2_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_hfft2_out(out, self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_ifft2_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_ifft2_out(out, self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_ihfft2_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_ihfft2_out(out, self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_irfft2_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_irfft2_out(out, self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_rfft2_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_rfft2_out(out, self, std::nullopt, IntArrayRef(dim.data(), dim.size()), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_fftn_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_fftn_out(out, self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_hfftn_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_hfftn_out(out, self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_ifftn_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_ifftn_out(out, self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_ihfftn_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_ihfftn_out(out, self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_irfftn_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_irfftn_out(out, self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_rfftn_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, rust::Str norm) noexcept {
  try {
    try {
      (void)at::fft_rfftn_out(out, self, std::nullopt, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), ::std::optional<c10::string_view>(std::basic_string_view<char>(norm.data(), norm.size())));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_fftfreq_out(Tensor & out, int64_t n, double d) noexcept {
  try {
    try {
      (void)at::fft_fftfreq_out(out, n, d);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fft_rfftfreq_out(Tensor & out, int64_t n, double d) noexcept {
  try {
    try {
      (void)at::fft_rfftfreq_out(out, n, d);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch FFT operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_cummax_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::cummax_out(values, indices, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_cummin_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::cummin_out(values, indices, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_median_dim_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::tuple<Tensor &, Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::median_out);
      (void)terlan_selected_overload(values, indices, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::mode_out(values, indices, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_nanmedian_dim_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::tuple<Tensor &, Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanmedian_out);
      (void)terlan_selected_overload(values, indices, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::kthvalue_out(values, indices, self, k, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::tuple<Tensor &, Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sort_out);
      (void)terlan_selected_overload(values, indices, self, dim, descending);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) noexcept {
  try {
    try {
      (void)at::topk_out(values, indices, self, k, dim, largest, sorted);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch selection operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_selection_nonzero_static_out(at::Tensor & output, const at::Tensor & tensor, std::int64_t size, std::int64_t fill_value) noexcept {
  try {
    try {
      (void)at::native::nonzero_static_exact_out(output, tensor, size, fill_value);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fixed-size nonzero output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_numericalcompletionnative_argsort_stable_out(Tensor & out, const Tensor & self, bool stable, int64_t dim, bool descending) noexcept {
  try {
    try {
      (void)at::argsort_out(out, self, stable, dim, descending);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch numerical completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_numericalcompletionnative_chain_matmul_out_native(Tensor & out, const TerlanPytorchTensorTensorResourceListInput& matrices) noexcept {
  try {
    try {
      (void)at::chain_matmul_out(out, matrices.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch numerical completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_numericalcompletionnative_clone_out_default_memory(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::clone_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch numerical completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_numericalcompletionnative_clone_out_memory(Tensor & out, const Tensor & self, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::clone_out(out, self, ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch numerical completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_numericalcompletionnative_cross_out_default_dimension(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::cross_out(out, self, other, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch numerical completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_numericalcompletionnative_cross_out_dimension(Tensor & out, const Tensor & self, const Tensor & other, std::int64_t dim) noexcept {
  try {
    try {
      (void)at::cross_out(out, self, other, ::std::optional<int64_t>(dim));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch numerical completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_complex_out(Tensor & out, const Tensor & real, const Tensor & imag) noexcept {
  try {
    try {
      (void)at::complex_out(out, real, imag);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_dist_out(Tensor & out, const Tensor & self, const Tensor & other, double p) noexcept {
  try {
    try {
      (void)at::dist_out(out, self, other, c10::Scalar(p));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_expand_copy_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> size, bool implicit) noexcept {
  try {
    try {
      (void)at::expand_copy_out(out, self, IntArrayRef(size.data(), size.size()), implicit);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_lu_solve_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) noexcept {
  try {
    try {
      (void)at::lu_solve_out(out, self, LU_data, LU_pivots);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_native_dropout_backward_out(Tensor & out, const Tensor & grad_output, const Tensor & mask, double scale) noexcept {
  try {
    try {
      (void)at::native_dropout_backward_out(out, grad_output, mask, scale);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_ormqr_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) noexcept {
  try {
    try {
      (void)at::ormqr_out(out, self, input2, input3, left, transpose);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_polar_out(Tensor & out, const Tensor & abs, const Tensor & angle) noexcept {
  try {
    try {
      (void)at::polar_out(out, abs, angle);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_reflection_pad1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::reflection_pad1d_out(out, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_reflection_pad2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::reflection_pad2d_out(out, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_reflection_pad3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::reflection_pad3d_out(out, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_replication_pad1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::replication_pad1d_out(out, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_replication_pad2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::replication_pad2d_out(out, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_replication_pad3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::replication_pad3d_out(out, self, IntArrayRef(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_renorm_out(Tensor & out, const Tensor & self, double p, int64_t dim, double maxnorm) noexcept {
  try {
    try {
      (void)at::renorm_out(out, self, c10::Scalar(p), dim, c10::Scalar(maxnorm));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_slogdet_out(Tensor & sign, Tensor & logabsdet, const Tensor & self) noexcept {
  try {
    try {
      (void)at::slogdet_out(sign, logabsdet, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_outputcompletion_softmax_out(Tensor & out, const Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::softmax_out(out, self, dim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch output completion operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linearstoragecompletion_conv_tbc_out(Tensor & out, const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) noexcept {
  try {
    try {
      (void)at::conv_tbc_out(out, self, weight, bias, pad);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linearstoragecompletion_set_empty_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::set_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linearstoragecompletion_set_empty_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::set_);
      (void)(self_value.*terlan_selected_overload)();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linearstoragecompletion_set_source_tensor_out(Tensor & out, const Tensor & self, const Tensor & source) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::set_out);
      (void)terlan_selected_overload(out, self, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linearstoragesourcecompletion_set_source_tensor_in_place(at::Tensor & self, const at::Tensor & source) noexcept {
  try {
    try {
      (void)at::native::set_source_tensor_in_place(self, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch set_ source-tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linearstoragesourcecompletion_set_source_tensor_storage_offset_in_place(at::Tensor & self, const at::Tensor & source, int64_t storage_offset, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::native::set_source_tensor_storage_offset_in_place(self, source, storage_offset, at::IntArrayRef(size.data(), size.size()), at::IntArrayRef(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch set_ source-tensor storage-offset operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_convolutionnative_convolution_backward_out(at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool has_bias_sizes, rust::Slice<const std::int64_t> bias_sizes, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool transposed, rust::Slice<const std::int64_t> output_padding, std::int64_t groups, bool output_mask_input, bool output_mask_weight, bool output_mask_bias) noexcept {
  try {
    try {
      (void)at::native::convolution_backward_exact_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, has_bias_sizes, c10::ArrayRef<std::int64_t>(bias_sizes.data(), bias_sizes.size()), c10::ArrayRef<std::int64_t>(stride.data(), stride.size()), c10::ArrayRef<std::int64_t>(padding.data(), padding.size()), c10::ArrayRef<std::int64_t>(dilation.data(), dilation.size()), transposed, c10::ArrayRef<std::int64_t>(output_padding.data(), output_padding.size()), groups, output_mask_input, output_mask_weight, output_mask_bias);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor convolution_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_convolutionnative_convolution_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool transposed, rust::Slice<const std::int64_t> output_padding, std::int64_t groups) noexcept {
  try {
    try {
      (void)at::native::convolution_exact_out(output, input, weight, has_bias, bias, c10::ArrayRef<std::int64_t>(stride.data(), stride.size()), c10::ArrayRef<std::int64_t>(padding.data(), padding.size()), c10::ArrayRef<std::int64_t>(dilation.data(), dilation.size()), transposed, c10::ArrayRef<std::int64_t>(output_padding.data(), output_padding.size()), groups);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor convolution output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_correction_out_default(Tensor & out, const Tensor & self, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, std::nullopt, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_correction_out_default(Tensor & out, const Tensor & self, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, std::nullopt, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_correction_out_dimensions(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_correction_out_dimensions(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_correction_out_int(Tensor & out, const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_correction_out_int(Tensor & out, const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_correction_out_float(Tensor & out, const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_correction_out_float(Tensor & out, const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_correction_out_dimensions_int(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_correction_out_dimensions_int(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_correction_out_dimensions_float(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_correction_out_dimensions_float(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, OptionalIntArrayRef, const ::std::optional<Scalar> &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch correction-statistics output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_mean_correction_out_default(Tensor & out0, Tensor & out1, const Tensor & self, bool keepdim) noexcept {
  try {
    try {
      (void)at::std_mean_out(out0, out1, self, std::nullopt, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_mean_correction_out_default(Tensor & out0, Tensor & out1, const Tensor & self, bool keepdim) noexcept {
  try {
    try {
      (void)at::var_mean_out(out0, out1, self, std::nullopt, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_std_mean_correction_out_dimensions(Tensor & out0, Tensor & out1, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::std_mean_out(out0, out1, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticscompletion_var_mean_correction_out_dimensions(Tensor & out0, Tensor & out1, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::var_mean_out(out0, out1, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_repeatinterleavenative_repeat_indices_out_native(Tensor & out, const Tensor & repeats) noexcept {
  try {
    try {
      (void)at::repeat_interleave_out(out, repeats, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch repeat-interleave output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_repeatinterleavenative_repeat_indices_out_output_size_native(Tensor & out, const Tensor & repeats, std::int64_t output_size) noexcept {
  try {
    try {
      (void)at::repeat_interleave_out(out, repeats, ::std::optional<int64_t>(output_size));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch repeat-interleave output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)at::addbmm_out(out, self, batch1, batch2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)at::addcdiv_out(out, self, tensor1, tensor2, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)at::addcmul_out(out, self, tensor1, tensor2, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::addmm_out);
      (void)terlan_selected_overload(out, self, mat1, mat2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addmm_out_dtype(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, std::int64_t out_dtype, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, ScalarType, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::addmm_out);
      (void)terlan_selected_overload(out, self, mat1, mat2, static_cast<c10::ScalarType>(out_dtype), beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addmv_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)at::addmv_out(out, self, mat, vec, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)at::addr_out(out, self, vec1, vec2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::baddbmm_out);
      (void)terlan_selected_overload(out, self, batch1, batch2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_baddbmm_out_dtype(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, std::int64_t out_dtype, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, ScalarType, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::baddbmm_out);
      (void)terlan_selected_overload(out, self, batch1, batch2, static_cast<c10::ScalarType>(out_dtype), beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addbmm_in_place(at::Tensor& self_value, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.addbmm_(batch1, batch2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addcdiv_in_place(at::Tensor& self_value, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.addcdiv_(tensor1, tensor2, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addcmul_in_place(at::Tensor& self_value, const Tensor & tensor1, const Tensor & tensor2, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.addcmul_(tensor1, tensor2, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addmm_in_place(at::Tensor& self_value, const Tensor & mat1, const Tensor & mat2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.addmm_(mat1, mat2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addmv_in_place(at::Tensor& self_value, const Tensor & mat, const Tensor & vec, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.addmv_(mat, vec, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_addr_in_place(at::Tensor& self_value, const Tensor & vec1, const Tensor & vec2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.addr_(vec1, vec2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_fusedarithmeticnative_fused_baddbmm_in_place(at::Tensor& self_value, const Tensor & batch1, const Tensor & batch2, bool beta_is_int, std::int64_t beta_int, double beta_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
      (void)self_value.baddbmm_(batch1, batch2, beta_is_int ? c10::Scalar(beta_int) : c10::Scalar(beta_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fused-arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_add_tensor_out_cxx(Tensor & out, const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::add_out);
      (void)terlan_selected_overload(out, self, other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_sub_tensor_out_cxx(Tensor & out, const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sub_out);
      (void)terlan_selected_overload(out, self, other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_rsub_tensor_out_cxx(Tensor & out, const Tensor & self, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::rsub_out);
      (void)terlan_selected_overload(out, self, other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_add_tensor_in_place_cxx(at::Tensor& self_value, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::add_);
      (void)(self_value.*terlan_selected_overload)(other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_sub_tensor_in_place_cxx(at::Tensor& self_value, const Tensor & other, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::sub_);
      (void)(self_value.*terlan_selected_overload)(other, alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_add_scalar_out_cxx(Tensor & out, const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::add_out);
      (void)terlan_selected_overload(out, self, right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_sub_scalar_out_cxx(Tensor & out, const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sub_out);
      (void)terlan_selected_overload(out, self, right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_rsub_scalar_out_cxx(Tensor & out, const Tensor & self, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::rsub_out);
      (void)terlan_selected_overload(out, self, right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_add_scalar_in_place_cxx(at::Tensor& self_value, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::add_);
      (void)(self_value.*terlan_selected_overload)(right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scaledarithmeticnative_scaled_sub_scalar_in_place_cxx(at::Tensor& self_value, bool right_is_int, std::int64_t right_int, double right_float, bool alpha_is_int, std::int64_t alpha_int, double alpha_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::sub_);
      (void)(self_value.*terlan_selected_overload)(right_is_int ? c10::Scalar(right_int) : c10::Scalar(right_float), alpha_is_int ? c10::Scalar(alpha_int) : c10::Scalar(alpha_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scaled arithmetic operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_scalar_min_out(Tensor & out, const Tensor & self, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_min_out);
      (void)terlan_selected_overload(out, self, bound_is_int ? c10::Scalar(bound_int) : c10::Scalar(bound_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_scalar_max_out(Tensor & out, const Tensor & self, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_max_out);
      (void)terlan_selected_overload(out, self, bound_is_int ? c10::Scalar(bound_int) : c10::Scalar(bound_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_scalar_min_max_out(Tensor & out, const Tensor & self, bool min_is_int, std::int64_t min_int, double min_float, bool max_is_int, std::int64_t max_int, double max_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const ::std::optional<Scalar> &, const ::std::optional<Scalar> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_out);
      (void)terlan_selected_overload(out, self, std::optional<c10::Scalar>(min_is_int ? c10::Scalar(min_int) : c10::Scalar(min_float)), std::optional<c10::Scalar>(max_is_int ? c10::Scalar(max_int) : c10::Scalar(max_float)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_scalar_min_in_place(at::Tensor& self_value, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::clamp_min_);
      (void)(self_value.*terlan_selected_overload)(bound_is_int ? c10::Scalar(bound_int) : c10::Scalar(bound_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_scalar_max_in_place(at::Tensor& self_value, bool bound_is_int, std::int64_t bound_int, double bound_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::clamp_max_);
      (void)(self_value.*terlan_selected_overload)(bound_is_int ? c10::Scalar(bound_int) : c10::Scalar(bound_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_scalar_min_max_in_place(at::Tensor& self_value, bool min_is_int, std::int64_t min_int, double min_float, bool max_is_int, std::int64_t max_int, double max_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const ::std::optional<Scalar> &, const ::std::optional<Scalar> &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::clamp_);
      (void)(self_value.*terlan_selected_overload)(std::optional<c10::Scalar>(min_is_int ? c10::Scalar(min_int) : c10::Scalar(min_float)), std::optional<c10::Scalar>(max_is_int ? c10::Scalar(max_int) : c10::Scalar(max_float)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_tensor_min_out(Tensor & out, const Tensor & self, const Tensor & min) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_min_out);
      (void)terlan_selected_overload(out, self, min);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_tensor_max_out(Tensor & out, const Tensor & self, const Tensor & max) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_max_out);
      (void)terlan_selected_overload(out, self, max);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_tensor_min_max_out(Tensor & out, const Tensor & self, const at::Tensor& min, const at::Tensor& max) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const ::std::optional<Tensor> &, const ::std::optional<Tensor> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::clamp_out);
      (void)terlan_selected_overload(out, self, std::optional<at::Tensor>(min), std::optional<at::Tensor>(max));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_tensor_min_in_place(at::Tensor& self_value, const Tensor & min) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::clamp_min_);
      (void)(self_value.*terlan_selected_overload)(min);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_tensor_max_in_place(at::Tensor& self_value, const Tensor & max) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::clamp_max_);
      (void)(self_value.*terlan_selected_overload)(max);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_clampnative_clamp_tensor_min_max_in_place(at::Tensor& self_value, const at::Tensor& min, const at::Tensor& max) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const ::std::optional<Tensor> &, const ::std::optional<Tensor> &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::clamp_);
      (void)(self_value.*terlan_selected_overload)(std::optional<at::Tensor>(min), std::optional<at::Tensor>(max));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch clamp operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_aggregationnative_aggregation_bincount_weighted_out(Tensor & out, const Tensor & self, const at::Tensor& weights, int64_t minlength) noexcept {
  try {
    try {
      (void)at::bincount_out(out, self, std::optional<at::Tensor>(weights), minlength);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch aggregation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_aggregationnative_aggregation_bincount_unweighted_out(Tensor & out, const Tensor & self, int64_t minlength) noexcept {
  try {
    try {
      (void)at::bincount_out(out, self, std::nullopt, minlength);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch aggregation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_aggregationnative_aggregation_histc_int_out(Tensor & out, const Tensor & self, int64_t bins, std::int64_t min, std::int64_t max) noexcept {
  try {
    try {
      (void)at::histc_out(out, self, bins, c10::Scalar(min), c10::Scalar(max));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch aggregation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_aggregationnative_aggregation_histc_float_out(Tensor & out, const Tensor & self, int64_t bins, double min, double max) noexcept {
  try {
    try {
      (void)at::histc_out(out, self, bins, c10::Scalar(min), c10::Scalar(max));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch aggregation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_aggregationnative_aggregation_index_reduce_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)at::index_reduce_out(out, self, dim, index, source, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch aggregation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_aggregationnative_aggregation_index_reduce_in_place(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & source, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)self_value.index_reduce_(dim, index, source, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch aggregation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_windownative_window_bartlett_out(Tensor & out, int64_t window_length, bool periodic) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bartlett_window_out);
      (void)terlan_selected_overload(out, window_length, periodic);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch window operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_windownative_window_blackman_out(Tensor & out, int64_t window_length, bool periodic) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::blackman_window_out);
      (void)terlan_selected_overload(out, window_length, periodic);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch window operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_windownative_window_hamming_out(Tensor & out, int64_t window_length, bool periodic, double alpha, double beta) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t, bool, double, double);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::hamming_window_out);
      (void)terlan_selected_overload(out, window_length, periodic, alpha, beta);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch window operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_windownative_window_hann_out(Tensor & out, int64_t window_length, bool periodic) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::hann_window_out);
      (void)terlan_selected_overload(out, window_length, periodic);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch window operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_windownative_window_kaiser_out(Tensor & out, int64_t window_length, bool periodic, double beta) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t, bool, double);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::kaiser_window_out);
      (void)terlan_selected_overload(out, window_length, periodic, beta);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch window operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_fmax_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::fmax_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_fmin_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::fmin_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_maximum_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::maximum_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_minimum_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::minimum_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_fmod_tensor_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fmod_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_fmod_tensor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::fmod_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logical_and_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::logical_and_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logical_and_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.logical_and_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logical_or_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::logical_or_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logical_or_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.logical_or_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logical_xor_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::logical_xor_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logical_xor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.logical_xor_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_arctan2_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::arctan2_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_atan2_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::atan2_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_copysign_tensor_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::copysign_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_dot_out(Tensor & out, const Tensor & self, const Tensor & tensor) noexcept {
  try {
    try {
      (void)at::dot_out(out, self, tensor);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_gcd_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::gcd_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_heaviside_out(Tensor & out, const Tensor & self, const Tensor & values) noexcept {
  try {
    try {
      (void)at::heaviside_out(out, self, values);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_hypot_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::hypot_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_igamma_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::igamma_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_igammac_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::igammac_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_inner_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::inner_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_kron_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::kron_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_lcm_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::lcm_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_ldexp_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::ldexp_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logaddexp_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::logaddexp_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_logaddexp2_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::logaddexp2_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_masked_select_out(Tensor & out, const Tensor & self, const Tensor & mask) noexcept {
  try {
    try {
      (void)at::masked_select_out(out, self, mask);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_max_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::max_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_min_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::min_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_nextafter_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::nextafter_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_take_out(Tensor & out, const Tensor & self, const Tensor & index) noexcept {
  try {
    try {
      (void)at::take_out(out, self, index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_vdot_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::vdot_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_arctan2_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.arctan2_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_atan2_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.atan2_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_copysign_tensor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::copysign_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_gcd_in_place(Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::gcd_(self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_heaviside_in_place(at::Tensor& self_value, const Tensor & values) noexcept {
  try {
    try {
      (void)self_value.heaviside_(values);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_hypot_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.hypot_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_igamma_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.igamma_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_igammac_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.igammac_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_lcm_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.lcm_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_ldexp_tensor_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.ldexp_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_nextafter_in_place(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
      (void)self_value.nextafter_(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_max_dim_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::tuple<Tensor &, Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::max_out);
      (void)terlan_selected_overload(max, max_values, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binary_min_dim_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ::std::tuple<Tensor &, Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::min_out);
      (void)terlan_selected_overload(min, min_indices, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_src(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, src);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_legacy_add(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Tensor &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_legacy_multiply(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Tensor &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_add(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    try {
      (void)at::scatter_add_out(out, self, dim, index, src);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_modern_sum(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)at::scatter_reduce_out(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_modern_prod(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)at::scatter_reduce_out(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_modern_mean(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)at::scatter_reduce_out(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_modern_amax(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)at::scatter_reduce_out(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_out_modern_amin(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)at::scatter_reduce_out(out, self, dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_src(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, src);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_legacy_add(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Tensor &, c10::string_view) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, src, c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_legacy_multiply(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Tensor &, c10::string_view) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, src, c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_add(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src) noexcept {
  try {
    try {
      (void)self_value.scatter_add_(dim, index, src);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_modern_sum(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)self_value.scatter_reduce_(dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_modern_prod(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)self_value.scatter_reduce_(dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_modern_mean(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)self_value.scatter_reduce_(dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_modern_amax(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)self_value.scatter_reduce_(dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_tensor_in_modern_amin(at::Tensor& self_value, int64_t dim, const Tensor & index, const Tensor & src, rust::Str reduce, bool include_self) noexcept {
  try {
    try {
      (void)self_value.scatter_reduce_(dim, index, src, c10::string_view(reduce.data(), reduce.size()), include_self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_out_int(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_out_float(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_out_reduce_int_add(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_out_reduce_int_multiply(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_out_reduce_float_add(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_out_reduce_float_multiply(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, const Scalar &, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::scatter_out);
      (void)terlan_selected_overload(out, self, dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_in_int(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_in_float(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_in_reduce_int_add(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &, c10::string_view) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_in_reduce_int_multiply(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &, c10::string_view) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_in_reduce_float_add(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &, c10::string_view) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_scatternative_scatter_scalar_in_reduce_float_multiply(at::Tensor& self_value, int64_t dim, const Tensor & index, bool value_is_int, std::int64_t value_int, double value_float, rust::Str reduce) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t, const Tensor &, const Scalar &, c10::string_view) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::scatter_);
      (void)(self_value.*terlan_selected_overload)(dim, index, value_is_int ? c10::Scalar(value_int) : c10::Scalar(value_float), c10::string_view(reduce.data(), reduce.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scatter operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_arange_out_end(Tensor & out, bool end_is_integer, std::int64_t end_integer, double end_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::arange_out);
      (void)terlan_selected_overload(out, end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_arange_out_start_end_step(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool step_is_integer, std::int64_t step_integer, double step_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::arange_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), step_is_integer ? c10::Scalar(step_integer) : c10::Scalar(step_floating));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_eye_out_square(Tensor & out, int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eye_out);
      (void)terlan_selected_overload(out, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_eye_out_rect(Tensor & out, int64_t n, int64_t m) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, int64_t, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eye_out);
      (void)terlan_selected_overload(out, n, m);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_linspace_out_scalar_scalar(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Scalar &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_linspace_out_tensor_tensor(Tensor & out, const Tensor & start, const Tensor & end, int64_t steps) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace_out);
      (void)terlan_selected_overload(out, start, end, steps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_linspace_out_tensor_scalar(Tensor & out, const Tensor & start, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace_out);
      (void)terlan_selected_overload(out, start, end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_linspace_out_scalar_tensor(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, const Tensor & end, int64_t steps) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linspace_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end, steps);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_logspace_out_scalar_scalar(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, double base) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Scalar &, int64_t, double);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps, base);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_logspace_out_tensor_tensor(Tensor & out, const Tensor & start, const Tensor & end, int64_t steps, double base) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t, double);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace_out);
      (void)terlan_selected_overload(out, start, end, steps, base);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_logspace_out_tensor_scalar(Tensor & out, const Tensor & start, bool end_is_integer, std::int64_t end_integer, double end_floating, int64_t steps, double base) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, int64_t, double);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace_out);
      (void)terlan_selected_overload(out, start, end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), steps, base);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_logspace_out_scalar_tensor(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, const Tensor & end, int64_t steps, double base) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &, int64_t, double);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::logspace_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end, steps, base);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_range_out_start_end(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::range_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_range_range_out_start_end_step(Tensor & out, bool start_is_integer, std::int64_t start_integer, double start_floating, bool end_is_integer, std::int64_t end_integer, double end_floating, bool step_is_integer, std::int64_t step_integer, double step_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::range_out);
      (void)terlan_selected_overload(out, start_is_integer ? c10::Scalar(start_integer) : c10::Scalar(start_floating), end_is_integer ? c10::Scalar(end_integer) : c10::Scalar(end_floating), step_is_integer ? c10::Scalar(step_integer) : c10::Scalar(step_floating));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_tril_indices_out_exact(Tensor & out, int64_t row, int64_t col, int64_t offset) noexcept {
  try {
    try {
      (void)at::tril_indices_out(out, row, col, offset);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_rangenative_triu_indices_out_exact(Tensor & out, int64_t row, int64_t col, int64_t offset) noexcept {
  try {
    try {
      (void)at::triu_indices_out(out, row, col, offset);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch range operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_u_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_v_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_chebyshev_polynomial_w_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_h_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_h_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_h_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_h_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_he_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_he_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_he_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_hermite_polynomial_he_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_laguerre_polynomial_l_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_laguerre_polynomial_l_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_laguerre_polynomial_l_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_laguerre_polynomial_l_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_legendre_polynomial_p_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_legendre_polynomial_p_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_legendre_polynomial_p_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_legendre_polynomial_p_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_t_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_u_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_v_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_right_int_out(Tensor & out, const Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_left_int_out(Tensor & out, std::int64_t x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_right_float_out(Tensor & out, const Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_shifted_chebyshev_polynomial_w_left_float_out(Tensor & out, double x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlog1py_right_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlog1py_left_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlog1py_right_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlog1py_left_float_out(Tensor & out, double self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlogy_right_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlogy_left_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlogy_right_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_xlogy_left_float_out(Tensor & out, double self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_zeta_right_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_zeta_left_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_zeta_right_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_specialscalarnative_ss_zeta_left_float_out(Tensor & out, double self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_empty_out(Tensor & out, rust::Slice<const std::int64_t> size, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::empty_out(out, IntArrayRef(size.data(), size.size()), has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_full_out(Tensor & out, rust::Slice<const std::int64_t> size, bool fill_value_is_integer, std::int64_t fill_value_integer, double fill_value_floating) noexcept {
  try {
    try {
      (void)at::full_out(out, IntArrayRef(size.data(), size.size()), fill_value_is_integer ? c10::Scalar(fill_value_integer) : c10::Scalar(fill_value_floating));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_ones_out(Tensor & out, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::ones_out(out, IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_zeros_out(Tensor & out, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::zeros_out(out, IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_empty_like_out(Tensor & out, const Tensor & self, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::empty_like_out(out, self, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_full_like_out(Tensor & out, const Tensor & self, bool fill_value_is_integer, std::int64_t fill_value_integer, double fill_value_floating, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::full_like_out(out, self, fill_value_is_integer ? c10::Scalar(fill_value_integer) : c10::Scalar(fill_value_floating), has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_ones_like_out(Tensor & out, const Tensor & self, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::ones_like_out(out, self, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_creationcompletionnative_creation_zeros_like_out(Tensor & out, const Tensor & self, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::zeros_like_out(out, self, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch creation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_bitwise_and(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_and_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_bitwise_and(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_and_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_bitwise_or(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_or_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_bitwise_or(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_or_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_bitwise_xor(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_xor_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_bitwise_xor(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_xor_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_bitwise_left_shift(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_left_shift_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_bitwise_left_shift(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_left_shift_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_bitwise_right_shift(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_right_shift_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_bitwise_right_shift(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_right_shift_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_remainder(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_remainder(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::remainder_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_pow(Tensor & out, const Tensor & self, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow_out);
      (void)terlan_selected_overload(out, self, exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_pow(at::Tensor& self_value, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::pow_);
      (void)(self_value.*terlan_selected_overload)(exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_float_power(Tensor & out, const Tensor & self, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power_out);
      (void)terlan_selected_overload(out, self, exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_float_power(at::Tensor& self_value, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::float_power_);
      (void)(self_value.*terlan_selected_overload)(exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_mul(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mul_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_mul(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::mul_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_floor_divide(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::floor_divide_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_floor_divide(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::floor_divide_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_xlogy(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_xlogy(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::xlogy_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_and_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_and_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_and_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_and_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_or_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_or_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_or_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_or_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_xor_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_xor_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_xor_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_xor_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_left_shift_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_left_shift_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_left_shift_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_left_shift_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_right_shift_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_right_shift_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_bitwise_right_shift_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::bitwise_right_shift_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_remainder_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_remainder_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::remainder_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_pow_int_out(Tensor & out, const Tensor & self, std::int64_t exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_pow_int_in_place(at::Tensor& self_value, std::int64_t exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::pow_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_float_power_int_out(Tensor & out, const Tensor & self, std::int64_t exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_float_power_int_in_place(at::Tensor& self_value, std::int64_t exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::float_power_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_mul_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mul_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_mul_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::mul_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::floor_divide_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::floor_divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_xlogy_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_xlogy_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::xlogy_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_copysign_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::copysign_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_copysign_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::copysign_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_remainder_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_remainder_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::remainder_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_pow_float_out(Tensor & out, const Tensor & self, double exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_pow_float_in_place(at::Tensor& self_value, double exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::pow_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_float_power_float_out(Tensor & out, const Tensor & self, double exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_float_power_float_in_place(at::Tensor& self_value, double exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::float_power_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(exponent));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_mul_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mul_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_mul_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::mul_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::floor_divide_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_floor_divide_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::floor_divide_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_xlogy_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_xlogy_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::xlogy_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_fmod_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::fmod_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_fmod_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::fmod_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_copysign_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::copysign_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_copysign_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::copysign_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_bitwise_and_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_and_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_bitwise_or_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_or_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_bitwise_xor_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_xor_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_bitwise_left_shift_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_left_shift_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_bitwise_right_shift_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bitwise_right_shift_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_remainder_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_pow_int_out(Tensor & out, std::int64_t self, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_float_power_int_out(Tensor & out, std::int64_t self, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_xlogy_int_out(Tensor & out, std::int64_t self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_remainder_float_out(Tensor & out, double self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::remainder_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_pow_float_out(Tensor & out, double self, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::pow_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_float_power_float_out(Tensor & out, double self, const Tensor & exponent) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::float_power_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), exponent);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_left_xlogy_float_out(Tensor & out, double self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::xlogy_out);
      (void)terlan_selected_overload(out, c10::Scalar(self), other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binary-scalar operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_eq_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eq_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_eq_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::eq_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ne_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ne_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ne_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ne_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_lt_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lt_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_lt_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::lt_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_le_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::le_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_le_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::le_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_gt_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::gt_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_gt_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::gt_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ge_float_out(Tensor & out, const Tensor & self, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ge_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ge_float_in_place(at::Tensor& self_value, double other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ge_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_eq_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eq_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_eq_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::eq_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ne_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ne_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ne_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ne_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_lt_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lt_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_lt_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::lt_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_le_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::le_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_le_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::le_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_gt_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::gt_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_gt_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::gt_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ge_int_out(Tensor & out, const Tensor & self, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ge_out);
      (void)terlan_selected_overload(out, self, c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_scalar_right_ge_int_in_place(at::Tensor& self_value, std::int64_t other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Scalar &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ge_);
      (void)(self_value.*terlan_selected_overload)(c10::Scalar(other));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_eq(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::eq_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_eq(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::eq_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_ne(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ne_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_ne(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ne_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_lt(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::lt_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_lt(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::lt_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_le(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::le_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_le(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::le_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_gt(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::gt_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_gt(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::gt_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_out_ge(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::ge_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_binaryscalarnative_binary_tensor_in_ge(at::Tensor& self_value, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(const Tensor &) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::ge_);
      (void)(self_value.*terlan_selected_overload)(other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch comparison operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_isin_tensor_tensor_out(Tensor & out, const Tensor & elements, const Tensor & test_elements, bool assume_unique, bool invert) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::isin_out);
      (void)terlan_selected_overload(out, elements, test_elements, assume_unique, invert);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_isin_tensor_scalar_out(Tensor & out, const Tensor & elements, bool test_element_is_integer, std::int64_t test_element_integer, double test_element_floating, bool assume_unique, bool invert) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::isin_out);
      (void)terlan_selected_overload(out, elements, test_element_is_integer ? c10::Scalar(test_element_integer) : c10::Scalar(test_element_floating), assume_unique, invert);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_isin_scalar_tensor_out(Tensor & out, bool element_is_integer, std::int64_t element_integer, double element_floating, const Tensor & test_elements, bool assume_unique, bool invert) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::isin_out);
      (void)terlan_selected_overload(out, element_is_integer ? c10::Scalar(element_integer) : c10::Scalar(element_floating), test_elements, assume_unique, invert);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_searchsorted_tensor_out(Tensor & out, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right, bool has_side, rust::Str side, bool has_sorter, const at::Tensor& sorter) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, bool, bool, ::std::optional<c10::string_view>, const ::std::optional<Tensor> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::searchsorted_out);
      (void)terlan_selected_overload(out, sorted_sequence, self, out_int32, right, has_side ? ::std::optional<c10::string_view>(std::basic_string_view<char>(side.data(), side.size())) : std::nullopt, has_sorter ? std::optional<at::Tensor>(sorter) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_searchsorted_scalar_out(Tensor & out, const Tensor & sorted_sequence, bool self_is_integer, std::int64_t self_integer, double self_floating, bool out_int32, bool right, bool has_side, rust::Str side, bool has_sorter, const at::Tensor& sorter) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Scalar &, bool, bool, ::std::optional<c10::string_view>, const ::std::optional<Tensor> &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::searchsorted_out);
      (void)terlan_selected_overload(out, sorted_sequence, self_is_integer ? c10::Scalar(self_integer) : c10::Scalar(self_floating), out_int32, right, has_side ? ::std::optional<c10::string_view>(std::basic_string_view<char>(side.data(), side.size())) : std::nullopt, has_sorter ? std::optional<at::Tensor>(sorter) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_bucketize_tensor_out(Tensor & out, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bucketize_out);
      (void)terlan_selected_overload(out, self, boundaries, out_int32, right);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_lookupnative_lookup_bucketize_scalar_out(Tensor & out, bool self_is_integer, std::int64_t self_integer, double self_floating, const Tensor & boundaries, bool out_int32, bool right) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Scalar &, const Tensor &, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::bucketize_out);
      (void)terlan_selected_overload(out, self_is_integer ? c10::Scalar(self_integer) : c10::Scalar(self_floating), boundaries, out_int32, right);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lookup operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_acos_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::acos_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_acos_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::acos_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_acosh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::acosh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_acosh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::acosh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_asin_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::asin_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_asin_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::asin_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_asinh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::asinh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_asinh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::asinh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_atan_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::atan_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_atan_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::atan_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_atanh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::atanh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_atanh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::atanh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_cos_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::cos_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_cos_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::cos_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_cosh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::cosh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_cosh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::cosh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sin_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sin_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sin_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sin_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sinh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sinh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sinh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sinh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_tan_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::tan_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_tan_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::tan_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_tanh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::tanh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_tanh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::tanh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_abs_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::abs_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_abs_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::abs_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_bitwise_not_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::bitwise_not_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_bitwise_not_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.bitwise_not_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_ceil_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::ceil_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_ceil_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::ceil_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_deg2rad_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::deg2rad_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_deg2rad_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::deg2rad_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_erf_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::erf_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_erf_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::erf_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_erfc_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::erfc_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_erfc_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::erfc_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_erfinv_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::erfinv_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_erfinv_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.erfinv_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_exp2_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::exp2_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_exp2_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::exp2_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_expm1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::expm1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_expm1_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::expm1_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_floor_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::floor_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_floor_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::floor_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_frac_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::frac_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_frac_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::frac_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_hardsigmoid_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::hardsigmoid_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_hardsigmoid_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::hardsigmoid_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_hardswish_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::hardswish_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_hardswish_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::hardswish_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log10_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log10_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log10_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log10_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log1p_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log1p_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log1p_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log1p_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log2_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log2_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log2_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log2_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_logical_not_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::logical_not_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_logical_not_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.logical_not_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_mish_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::mish_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_mish_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::mish_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_rad2deg_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::rad2deg_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_rad2deg_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::rad2deg_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_reciprocal_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::reciprocal_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_reciprocal_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::reciprocal_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_round_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::round_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_round_in_place(at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::round_);
      (void)terlan_selected_overload(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_rsqrt_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::rsqrt_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_rsqrt_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::rsqrt_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sigmoid_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sigmoid_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sigmoid_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sigmoid_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sign_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sign_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sign_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.sign_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_silu_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::silu_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_silu_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::silu_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sinc_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sinc_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sinc_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sinc_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_square_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::square_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_square_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::square_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_trunc_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::trunc_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_trunc_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::trunc_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_relu6_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::relu6_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_selu_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::selu_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_absolute_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::absolute_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_absolute_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.absolute_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arccos_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arccos_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arccos_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arccos_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arccosh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arccosh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arccosh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arccosh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arcsin_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arcsin_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arcsin_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arcsin_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arcsinh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arcsinh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arcsinh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arcsinh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arctan_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arctan_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arctan_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arctan_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arctanh_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arctanh_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_arctanh_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::arctanh_(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_all_all_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::all_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_any_all_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::any_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_isinf_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::isinf_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_isnan_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::isnan_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_isneginf_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::isneginf_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_isposinf_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::isposinf_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_signbit_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::signbit_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_max_unary_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::max_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_median_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::median_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_min_unary_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::min_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_nonzero_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::nonzero_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_trace_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::trace_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_hardsigmoid_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::hardsigmoid_backward_out(grad_input, grad_output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_hardswish_backward_out(at::Tensor & out, const at::Tensor & grad_output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::hardswish_backward_out(out, grad_output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sigmoid_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & output) noexcept {
  try {
    try {
      (void)at::sigmoid_backward_out(grad_input, grad_output, output);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_silu_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::silu_backward_out(grad_input, grad_output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_tanh_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & output) noexcept {
  try {
    try {
      (void)at::tanh_backward_out(grad_input, grad_output, output);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_round_decimals_out(at::Tensor & out, const at::Tensor & self, int64_t decimals) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::round_out);
      (void)terlan_selected_overload(out, self, decimals);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_round_decimals_in_place(at::Tensor & self, int64_t decimals) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::round_);
      (void)terlan_selected_overload(self, decimals);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_frexp_out(at::Tensor & mantissa, at::Tensor & exponent, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::frexp_out(mantissa, exponent, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_alias_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::alias_copy_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_angle_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::angle_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_conj_physical_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::conj_physical_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_detach_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::detach_copy_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_digamma_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::digamma_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_exp_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::exp_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_fix_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::fix_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_i0_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::i0_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_inverse_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::inverse_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_lgamma_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::lgamma_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_lift_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::lift_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_lift_fresh_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::lift_fresh_copy_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log_sigmoid_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::log_sigmoid_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_msort_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::msort_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_neg_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::neg_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_negative_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::negative_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_relu_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::relu_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sgn_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sgn_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sqrt_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::sqrt_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_t_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::t_copy_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_view_as_complex_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::view_as_complex_copy_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_view_as_real_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::view_as_real_copy_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_conj_physical_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.conj_physical_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_detach_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.detach_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_digamma_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.digamma_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_exp_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.exp_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_fix_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.fix_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_i0_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.i0_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_lgamma_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.lgamma_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_log_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.log_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_neg_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.neg_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_negative_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.negative_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_relu_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.relu_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sgn_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.sgn_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_squeeze_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::squeeze_);
      (void)(self_value.*terlan_selected_overload)();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_sqrt_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.sqrt_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_t_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.t_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_unarystoragenative_zero_in_place(at::Tensor& self_value) noexcept {
  try {
    try {
      (void)self_value.zero_();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unary-storage operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_diag_out(at::Tensor & out, const at::Tensor & self, int64_t diagonal) noexcept {
  try {
    try {
      (void)at::diag_out(out, self, diagonal);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_matrix_power_out(at::Tensor & out, const at::Tensor & self, int64_t n) noexcept {
  try {
    try {
      (void)at::matrix_power_out(out, self, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_mvlgamma_out(at::Tensor & out, const at::Tensor & self, int64_t p) noexcept {
  try {
    try {
      (void)at::mvlgamma_out(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_pixel_shuffle_out(at::Tensor & out, const at::Tensor & self, int64_t upscale_factor) noexcept {
  try {
    try {
      (void)at::pixel_shuffle_out(out, self, upscale_factor);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_pixel_unshuffle_out(at::Tensor & out, const at::Tensor & self, int64_t downscale_factor) noexcept {
  try {
    try {
      (void)at::pixel_unshuffle_out(out, self, downscale_factor);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_squeeze_copy_dimension_out(at::Tensor & out, const at::Tensor & self, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, int64_t);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze_copy_out);
      (void)terlan_selected_overload(out, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_tril_out(at::Tensor & out, const at::Tensor & self, int64_t diagonal) noexcept {
  try {
    try {
      (void)at::tril_out(out, self, diagonal);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_triu_out(at::Tensor & out, const at::Tensor & self, int64_t diagonal) noexcept {
  try {
    try {
      (void)at::triu_out(out, self, diagonal);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_unsqueeze_copy_out(at::Tensor & out, const at::Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::unsqueeze_copy_out(out, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_flip_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    try {
      (void)at::flip_out(out, self, at::IntArrayRef(dims.data(), dims.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_permute_copy_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    try {
      (void)at::permute_copy_out(out, self, at::IntArrayRef(dims.data(), dims.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_repeat_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> repeats) noexcept {
  try {
    try {
      (void)at::repeat_out(out, self, at::IntArrayRef(repeats.data(), repeats.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_squeeze_copy_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze_copy_out);
      (void)terlan_selected_overload(out, self, at::IntArrayRef(dim.data(), dim.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_view_copy_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::view_copy_out);
      (void)terlan_selected_overload(out, self, at::IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_squeeze_copy_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::squeeze_copy_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_diag_embed_out(at::Tensor & out, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    try {
      (void)at::diag_embed_out(out, self, offset, dim1, dim2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_diagonal_copy_out(at::Tensor & out, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) noexcept {
  try {
    try {
      (void)at::diagonal_copy_out(out, self, offset, dim1, dim2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_roll_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> shifts, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    try {
      (void)at::roll_out(out, self, at::IntArrayRef(shifts.data(), shifts.size()), at::IntArrayRef(dims.data(), dims.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_rot90_out(at::Tensor & out, const at::Tensor & self, int64_t k, rust::Slice<const std::int64_t> dims) noexcept {
  try {
    try {
      (void)at::rot90_out(out, self, k, at::IntArrayRef(dims.data(), dims.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_squeeze_dimension_in_place(at::Tensor& self_value, int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(int64_t) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::squeeze_);
      (void)(self_value.*terlan_selected_overload)(dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_mvlgamma_in_place(at::Tensor& self_value, int64_t p) noexcept {
  try {
    try {
      (void)self_value.mvlgamma_(p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_tril_in_place(at::Tensor& self_value, int64_t diagonal) noexcept {
  try {
    try {
      (void)self_value.tril_(diagonal);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_triu_in_place(at::Tensor& self_value, int64_t diagonal) noexcept {
  try {
    try {
      (void)self_value.triu_(diagonal);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_unsqueeze_in_place(at::Tensor& self_value, int64_t dim) noexcept {
  try {
    try {
      (void)self_value.unsqueeze_(dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_squeeze_dimensions_in_place(at::Tensor& self_value, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (at::Tensor::*)(IntArrayRef) const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::Tensor::squeeze_);
      (void)(self_value.*terlan_selected_overload)(IntArrayRef(dim.data(), dim.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_swapaxes_in_place(at::Tensor& self_value, int64_t axis0, int64_t axis1) noexcept {
  try {
    try {
      (void)self_value.swapaxes_(axis0, axis1);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_swapdims_in_place(at::Tensor& self_value, int64_t dim0, int64_t dim1) noexcept {
  try {
    try {
      (void)self_value.swapdims_(dim0, dim1);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_shape_transpose_in_place(at::Tensor& self_value, int64_t dim0, int64_t dim1) noexcept {
  try {
    try {
      (void)self_value.transpose_(dim0, dim1);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shape operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_polygamma_out(at::Tensor & out, const at::Tensor & self, int64_t n) noexcept {
  try {
    try {
      (void)at::polygamma_out(out, n, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_polygamma_in_place(at::Tensor& self_value, int64_t n) noexcept {
  try {
    try {
      (void)self_value.polygamma_(n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_logit_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::logit_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_logit_epsilon_out(at::Tensor & out, const at::Tensor & self, double eps) noexcept {
  try {
    try {
      (void)at::logit_out(out, self, ::std::optional<double>(eps));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_logit_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::logit_(self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_logit_epsilon_in_place(at::Tensor & self, double eps) noexcept {
  try {
    try {
      (void)at::logit_(self, ::std::optional<double>(eps));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_airy_ai_out(at::Tensor & out, const at::Tensor & x) noexcept {
  try {
    try {
      (void)at::special_airy_ai_out(out, x);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_bessel_j0_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_bessel_j0_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_bessel_j1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_bessel_j1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_bessel_y0_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_bessel_y0_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_bessel_y1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_bessel_y1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_digamma_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_digamma_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_entr_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_entr_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_erf_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_erf_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_erfc_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_erfc_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_erfcx_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_erfcx_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_erfinv_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_erfinv_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_exp2_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_exp2_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_expit_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_expit_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_expm1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_expm1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_gammaln_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_gammaln_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_i0_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_i0_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_i0e_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_i0e_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_i1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_i1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_i1e_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_i1e_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_log1p_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_log1p_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_log_ndtr_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_log_ndtr_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_modified_bessel_i0_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_modified_bessel_i0_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_modified_bessel_i1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_modified_bessel_i1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_modified_bessel_k0_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_modified_bessel_k0_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_modified_bessel_k1_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_modified_bessel_k1_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_ndtr_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_ndtr_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_ndtri_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_ndtri_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_psi_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_psi_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_scaled_modified_bessel_k0_out(at::Tensor & out, const at::Tensor & x) noexcept {
  try {
    try {
      (void)at::special_scaled_modified_bessel_k0_out(out, x);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_scaled_modified_bessel_k1_out(at::Tensor & out, const at::Tensor & x) noexcept {
  try {
    try {
      (void)at::special_scaled_modified_bessel_k1_out(out, x);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_sinc_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_sinc_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_spherical_bessel_j0_out(at::Tensor & out, const at::Tensor & x) noexcept {
  try {
    try {
      (void)at::special_spherical_bessel_j0_out(out, x);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_gammainc_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) noexcept {
  try {
    try {
      (void)at::special_gammainc_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_gammaincc_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) noexcept {
  try {
    try {
      (void)at::special_gammaincc_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_xlog1py_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlog1py_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_xlogy_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_xlogy_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_zeta_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_zeta_out);
      (void)terlan_selected_overload(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_logit_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::special_logit_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_logit_epsilon_out(at::Tensor & out, const at::Tensor & self, double eps) noexcept {
  try {
    try {
      (void)at::special_logit_out(out, self, ::std::optional<double>(eps));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_logsumexp_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::special_logsumexp_out(out, self, at::IntArrayRef(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_multigammaln_out(at::Tensor & out, const at::Tensor & self, int64_t p) noexcept {
  try {
    try {
      (void)at::special_multigammaln_out(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_polygamma_out(at::Tensor & out, const at::Tensor & self, int64_t n) noexcept {
  try {
    try {
      (void)at::special_polygamma_out(out, n, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_round_out(at::Tensor & out, const at::Tensor & self, int64_t decimals) noexcept {
  try {
    try {
      (void)at::special_round_out(out, self, decimals);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_t_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_u_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_v_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_w_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_hermite_polynomial_h_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_h_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_hermite_polynomial_he_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_hermite_polynomial_he_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_laguerre_polynomial_l_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_laguerre_polynomial_l_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_legendre_polynomial_p_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_legendre_polynomial_p_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_shifted_chebyshev_polynomial_t_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_shifted_chebyshev_polynomial_u_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_u_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_shifted_chebyshev_polynomial_v_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_v_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_shifted_chebyshev_polynomial_w_out(Tensor & out, const Tensor & x, const Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_shifted_chebyshev_polynomial_w_out);
      (void)terlan_selected_overload(out, x, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_t_order_int_out(at::Tensor & out, const at::Tensor & x, std::int64_t n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, const at::Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_t_order_float_out(at::Tensor & out, const at::Tensor & x, double n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, const at::Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, x, c10::Scalar(n));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_t_input_int_out(at::Tensor & out, std::int64_t x, const at::Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Scalar &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_special_special_chebyshev_polynomial_t_input_float_out(at::Tensor & out, double x, const at::Tensor & n) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Scalar &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::special_chebyshev_polynomial_t_out);
      (void)terlan_selected_overload(out, c10::Scalar(x), n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch special operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rand_generator_out(at::Tensor & output, rust::Slice<const std::int64_t> size, const Generator & generator) noexcept {
  try {
    try {
      (void)at::native::generator_rand_out(output, c10::ArrayRef<std::int64_t>(size.data(), size.size()), generator);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed rand output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randn_generator_out(at::Tensor & output, rust::Slice<const std::int64_t> size, const Generator & generator) noexcept {
  try {
    try {
      (void)at::native::generator_randn_out(output, c10::ArrayRef<std::int64_t>(size.data(), size.size()), generator);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed randn output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randperm_generator_out(at::Tensor & output, int64_t n, const Generator & generator) noexcept {
  try {
    try {
      (void)at::native::generator_randperm_out(output, n, generator);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed randperm output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_rand_like_out(at::Tensor & output, const Tensor & input, const Generator & generator, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    try {
      (void)at::native::generator_rand_like_out(output, input, generator, has_memory_format, memory_format);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed rand_like output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_randn_like_out(at::Tensor & output, const Tensor & input, const Generator & generator, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    try {
      (void)at::native::generator_randn_like_out(output, input, generator, has_memory_format, memory_format);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed randn_like output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_randint_out(at::Tensor & output, int64_t high, rust::Slice<const std::int64_t> size, const Generator & generator) noexcept {
  try {
    try {
      (void)at::native::generator_randint_out(output, high, c10::ArrayRef<std::int64_t>(size.data(), size.size()), generator);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed randint output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_randint_low_out(at::Tensor & output, int64_t low, int64_t high, rust::Slice<const std::int64_t> size, const Generator & generator) noexcept {
  try {
    try {
      (void)at::native::generator_randint_low_out(output, low, high, c10::ArrayRef<std::int64_t>(size.data(), size.size()), generator);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed low randint output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_randint_like_out(at::Tensor & output, const Tensor & input, int64_t high, const Generator & generator, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    try {
      (void)at::native::generator_randint_like_out(output, input, high, generator, has_memory_format, memory_format);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed randint_like output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_randint_like_tensor_out(at::Tensor & output, const Tensor & input, const Tensor & high, const Generator & generator, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    try {
      (void)at::native::generator_randint_like_tensor_out(output, input, high, generator, has_memory_format, memory_format);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed tensor-bound randint_like output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_generator_randint_like_low_out(at::Tensor & output, const Tensor & input, int64_t low, int64_t high, const Generator & generator, bool has_memory_format, int64_t memory_format) noexcept {
  try {
    try {
      (void)at::native::generator_randint_like_low_out(output, input, low, high, generator, has_memory_format, memory_format);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch generator-backed low randint_like output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rand_out(at::Tensor & out, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::native::rand_out_ordered(out, at::IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch rand output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randn_out(at::Tensor & out, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::native::randn_out_ordered(out, at::IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randn output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randint_out(at::Tensor & out, int64_t high, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::native::randint_out_ordered(out, high, at::IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randint high output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randint_low_out(at::Tensor & out, int64_t low, int64_t high, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::native::randint_low_out_ordered(out, low, high, at::IntArrayRef(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randint low/high output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rand_like_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::rand_like_out_ordered(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch rand_like output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randn_like_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::randn_like_out_ordered(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randn_like output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randint_like_out_default(at::Tensor & out, const at::Tensor & self, int64_t high) noexcept {
  try {
    try {
      (void)at::native::randint_like_out_default(out, self, high);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randint_like output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randint_like_tensor_out_default(at::Tensor & out, const at::Tensor & self, const at::Tensor & high) noexcept {
  try {
    try {
      (void)at::native::randint_like_tensor_out_default(out, self, high);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor-bound randint_like output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_randint_like_low_dtype_out_default(at::Tensor & out, const at::Tensor & self, int64_t low, int64_t high) noexcept {
  try {
    try {
      (void)at::native::randint_like_low_dtype_out_default(out, self, low, high);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randint_like low/high output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_random_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::random_out_ordered(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch random output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_random_from_out(at::Tensor & out, const at::Tensor & self, int64_t from, int64_t to) noexcept {
  try {
    try {
      (void)at::native::random_from_out_ordered(out, self, from, to);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch random from output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_random_to_out(at::Tensor & out, const at::Tensor & self, int64_t to) noexcept {
  try {
    try {
      (void)at::native::random_to_out_ordered(out, self, to);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch random to output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_random_in_place(at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::random_in_place(self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch random in-place operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_random_from_in_place(at::Tensor & self, int64_t from, int64_t to) noexcept {
  try {
    try {
      (void)at::native::random_from_in_place(self, from, to);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch random from in-place operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_random_to_in_place(at::Tensor & self, int64_t to) noexcept {
  try {
    try {
      (void)at::native::random_to_in_place(self, to);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch random to in-place operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rrelu_in_place(at::Tensor & self, double lower, double upper, bool training) noexcept {
  try {
    try {
      (void)at::native::rrelu_in_place(self, lower, upper, training);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch rrelu in-place operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rrelu_with_noise_in_place(at::Tensor & self, at::Tensor & noise, double lower, double upper, bool training) noexcept {
  try {
    try {
      (void)at::native::rrelu_with_noise_in_place(self, noise, lower, upper, training);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch rrelu-with-noise in-place operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rrelu_with_noise_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & noise, bool has_generator, const Generator & generator, double lower, double upper, bool training) noexcept {
  try {
    try {
      (void)at::native::rrelu_with_noise_exact_out(output, self, noise, has_generator, generator, lower, upper, training);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch rrelu-with-noise value output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randomcompletion_rrelu_with_noise_backward_out(at::Tensor & output, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, double lower, double upper, bool training, bool self_is_result) noexcept {
  try {
    try {
      (void)at::native::rrelu_with_noise_backward_exact_out(output, grad_output, self, noise, lower, upper, training, self_is_result);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor rrelu_with_noise_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_segmentreducecompletion_segment_reduce_lengths_out(at::Tensor & out, const at::Tensor & data, rust::Str reduce, const at::Tensor & lengths) noexcept {
  try {
    try {
      (void)at::native::segment_reduce_lengths_out(out, data, std::string_view(reduce.data(), reduce.size()), lengths);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch segment_reduce output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_histogramcompletion_histogram_bin_ct_out(at::Tensor & hist, at::Tensor & bin_edges, const at::Tensor & self, int64_t bins, bool has_range, rust::Slice<const double> range, bool has_weight, const at::Tensor & weight, bool density) noexcept {
  try {
    try {
      (void)at::native::histogram_bin_ct_out_configured(hist, bin_edges, self, bins, has_range, range, has_weight, weight, density);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch configured histogram bin-count output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_histogramcompletion_histogram_bins_tensor_out(at::Tensor & hist, at::Tensor & bin_edges, const at::Tensor & self, const at::Tensor & bins, bool has_weight, const at::Tensor & weight, bool density) noexcept {
  try {
    try {
      (void)at::native::histogram_bins_tensor_out_configured(hist, bin_edges, self, bins, has_weight, weight, density);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch configured histogram Tensor-bin output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_uniquecompletion_unique_consecutive_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & self, bool return_inverse, bool return_counts) noexcept {
  try {
    try {
      (void)at::native::unique_consecutive_out_default(out0, out1, out2, self, return_inverse, return_counts, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unique_consecutive output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_uniquecompletion_unique_dim_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) noexcept {
  try {
    try {
      (void)at::native::unique_dim_out_default(out0, out1, out2, self, dim, sorted, return_inverse, return_counts);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unique_dim output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_uniquecompletion_unique_dim_consecutive_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) noexcept {
  try {
    try {
      (void)at::native::unique_dim_consecutive_out_default(out0, out1, out2, self, dim, return_inverse, return_counts);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch unique_dim_consecutive output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_svdcompletion_svd_out(at::Tensor & U, at::Tensor & S, at::Tensor & V, const at::Tensor & self, bool some, bool compute_uv) noexcept {
  try {
    try {
      (void)at::native::svd_out_default(U, S, V, self, some, compute_uv);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch SVD output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizedallocationcompletion_empty_quantized_out(at::Tensor & out, rust::Slice<const std::int64_t> size, const at::Tensor & qtensor) noexcept {
  try {
    try {
      (void)at::native::empty_quantized_out_default(out, at::IntArrayRef(size.data(), size.size()), qtensor);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch empty quantized output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumsum_out_dtype(at::Tensor & out, const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    try {
      (void)at::cumsum_out(out, self, dim, ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumprod_out_dtype(at::Tensor & out, const at::Tensor & self, int64_t dim, std::int64_t dtype) noexcept {
  try {
    try {
      (void)at::cumprod_out(out, self, dim, ::std::optional<at::ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumsum_in_place(at::Tensor& self_value, int64_t dim) noexcept {
  try {
    try {
      (void)self_value.cumsum_(dim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumsum_in_place_dtype(at::Tensor& self_value, int64_t dim, std::int64_t dtype) noexcept {
  try {
    try {
      (void)self_value.cumsum_(dim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumprod_in_place(at::Tensor& self_value, int64_t dim) noexcept {
  try {
    try {
      (void)self_value.cumprod_(dim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumprod_in_place_dtype(at::Tensor& self_value, int64_t dim, std::int64_t dtype) noexcept {
  try {
    try {
      (void)self_value.cumprod_(dim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_norm_dimensions_copy_in_place(at::Tensor & arg0, const at::Tensor & arg1, bool arg2) noexcept {
  try {
    try {
      (void)at::native::tensor_copy_in_place(arg0, arg1, arg2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nanmean_out_native(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::nanmean_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nanmean_dimensions_out_native(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::nanmean_out(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nanmean_dimensions_dtype_out_native(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t dtype, bool keepdim) noexcept {
  try {
    try {
      (void)at::nanmean_out(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nansum_out_native(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::nansum_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nansum_dimensions_out_native(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::nansum_out(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nansum_dimensions_dtype_out_native(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t dtype, bool keepdim) noexcept {
  try {
    try {
      (void)at::nansum_out(out, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, ::std::optional<ScalarType>(static_cast<c10::ScalarType>(dtype)));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_amax_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::amax_out(out, self, at::IntArrayRef(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_amin_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::amin_out(out, self, at::IntArrayRef(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_all_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::all_out);
      (void)terlan_selected_overload(out, self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_any_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::any_out);
      (void)terlan_selected_overload(out, self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_all_dimension_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::all_out);
      (void)terlan_selected_overload(out, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_any_dimension_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, int64_t, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::any_out);
      (void)terlan_selected_overload(out, self, dim, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_argmax_out(at::Tensor & out, const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::argmax_out(out, self, ::std::optional<int64_t>(dim), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_argmin_out(at::Tensor & out, const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::argmin_out(out, self, ::std::optional<int64_t>(dim), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_argmax_all_out(at::Tensor & out, const at::Tensor & self, bool keepdim) noexcept {
  try {
    try {
      (void)at::argmax_out(out, self, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_argmin_all_out(at::Tensor & out, const at::Tensor & self, bool keepdim) noexcept {
  try {
    try {
      (void)at::argmin_out(out, self, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_count_nonzero_out(at::Tensor & out, const at::Tensor & self, std::int64_t dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::count_nonzero_out);
      (void)terlan_selected_overload(out, self, ::std::optional<int64_t>(dim));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_count_nonzero_all_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, ::std::optional<int64_t>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::count_nonzero_out);
      (void)terlan_selected_overload(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_count_nonzero_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::IntArrayRef);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::count_nonzero_out);
      (void)terlan_selected_overload(out, self, at::IntArrayRef(dim.data(), dim.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_aminmax_dim_out(at::Tensor & min, at::Tensor & max, const at::Tensor & self, std::int64_t dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::aminmax_out(min, max, self, ::std::optional<int64_t>(dim), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_aminmax_all_out(at::Tensor & min, at::Tensor & max, const at::Tensor & self, bool keepdim) noexcept {
  try {
    try {
      (void)at::aminmax_out(min, max, self, std::nullopt, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nanmedian_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanmedian_out);
      (void)terlan_selected_overload(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_logsumexp_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::logsumexp_out(out, self, at::IntArrayRef(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_mean_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mean_out);
      (void)terlan_selected_overload(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_prod_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::prod_out);
      (void)terlan_selected_overload(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_sum_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sum_out);
      (void)terlan_selected_overload(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_mean_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mean_out);
      (void)terlan_selected_overload(out, self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_sum_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sum_out);
      (void)terlan_selected_overload(out, self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_mean_all_dimensions_out(at::Tensor & out, const at::Tensor & self, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::mean_out);
      (void)terlan_selected_overload(out, self, std::nullopt, keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_sum_all_dimensions_out(at::Tensor & out, const at::Tensor & self, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::sum_out);
      (void)terlan_selected_overload(out, self, std::nullopt, keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_prod_dimension_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, int64_t, bool, ::std::optional<at::ScalarType>);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::prod_out);
      (void)terlan_selected_overload(out, self, dim, keepdim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_std_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), unbiased, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_var_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool unbiased, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, at::OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), unbiased, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_std_all_dimensions_out(at::Tensor & out, const at::Tensor & self, bool unbiased, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::std_out);
      (void)terlan_selected_overload(out, self, std::nullopt, unbiased, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_var_all_dimensions_out(at::Tensor & out, const at::Tensor & self, bool unbiased, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::OptionalIntArrayRef, bool, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::var_out);
      (void)terlan_selected_overload(out, self, std::nullopt, unbiased, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumprod_out(at::Tensor & out, const at::Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::cumprod_out(out, self, dim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cumsum_out(at::Tensor & out, const at::Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::cumsum_out(out, self, dim, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_logcumsumexp_out(at::Tensor & out, const at::Tensor & self, int64_t dim) noexcept {
  try {
    try {
      (void)at::logcumsumexp_out(out, self, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cholesky_out(at::Tensor & out, const at::Tensor & self, bool upper) noexcept {
  try {
    try {
      (void)at::cholesky_out(out, self, upper);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cholesky_inverse_out(at::Tensor & out, const at::Tensor & self, bool upper) noexcept {
  try {
    try {
      (void)at::cholesky_inverse_out(out, self, upper);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nuclear_norm_out(at::Tensor & out, const at::Tensor & self, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nuclear_norm_out);
      (void)terlan_selected_overload(out, self, keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_cholesky_solve_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & input2, bool upper) noexcept {
  try {
    try {
      (void)at::cholesky_solve_out(out, self, input2, upper);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_frobenius_norm_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::frobenius_norm_out(out, self, at::IntArrayRef(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_statistics_nuclear_norm_dimensions_out(at::Tensor & out, const at::Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, at::IntArrayRef, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nuclear_norm_out);
      (void)terlan_selected_overload(out, self, at::IntArrayRef(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch statistics operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_std_mean_correction_out_int(Tensor & out0, Tensor & out1, const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::std_mean_out(out0, out1, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_std_mean_correction_out_float(Tensor & out0, Tensor & out1, const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::std_mean_out(out0, out1, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_std_mean_correction_out_dimensions_int(Tensor & out0, Tensor & out1, const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::std_mean_out(out0, out1, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_std_mean_correction_out_dimensions_float(Tensor & out0, Tensor & out1, const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::std_mean_out(out0, out1, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_var_mean_correction_out_int(Tensor & out0, Tensor & out1, const Tensor & self, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::var_mean_out(out0, out1, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_var_mean_correction_out_float(Tensor & out0, Tensor & out1, const Tensor & self, double correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::var_mean_out(out0, out1, self, std::nullopt, std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_var_mean_correction_out_dimensions_int(Tensor & out0, Tensor & out1, const Tensor & self, rust::Slice<const std::int64_t> dim, std::int64_t correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::var_mean_out(out0, out1, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_correctionstatisticspairnative_var_mean_correction_out_dimensions_float(Tensor & out0, Tensor & out1, const Tensor & self, rust::Slice<const std::int64_t> dim, double correction, bool keepdim) noexcept {
  try {
    try {
      (void)at::var_mean_out(out0, out1, self, OptionalIntArrayRef(c10::ArrayRef<std::int64_t>(dim.data(), dim.size())), std::optional<c10::Scalar>(c10::Scalar(correction)), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_pair_operation_failed", "Correction pair operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_sparse_mask_out(Tensor & out, const Tensor & self, const Tensor & mask) noexcept {
  try {
    try {
      (void)at::compositeexplicitautograd::sparse_mask_out(out, self, mask);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_copy_sparse_to_sparse_out(Tensor & out, const Tensor & self, const Tensor & src, bool non_blocking) noexcept {
  try {
    try {
      (void)at::compositeexplicitautograd::copy_sparse_to_sparse_out(out, self, src, non_blocking);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_resize_as_sparse_in_place(at::Tensor & self, const at::Tensor & shape) noexcept {
  try {
    try {
      (void)at::native::sparse_resize_as_in_place(self, shape);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_sparse_resize_in_place(at::Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept {
  try {
    try {
      (void)at::native::sparse_resize_in_place(self, c10::ArrayRef<int64_t>(size.data(), size.size()), sparse_dim, dense_dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_sparse_resize_and_clear_in_place(at::Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept {
  try {
    try {
      (void)at::native::sparse_resize_and_clear_in_place(self, c10::ArrayRef<int64_t>(size.data(), size.size()), sparse_dim, dense_dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_crow_indices_copy_out(at::Tensor & output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::sparse_crow_indices_copy_out(output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_col_indices_copy_out(at::Tensor & output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::sparse_col_indices_copy_out(output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_ccol_indices_copy_out(at::Tensor & output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::sparse_ccol_indices_copy_out(output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_row_indices_copy_out(at::Tensor & output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::sparse_row_indices_copy_out(output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_indices_copy_out(at::Tensor & output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::sparse_indices_copy_out(output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_values_copy_out(at::Tensor & output, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::sparse_values_copy_out(output, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_copy_sparse_to_sparse_in_place(at::Tensor & self, const at::Tensor & source, bool non_blocking) noexcept {
  try {
    try {
      (void)at::native::sparse_copy_in_place(self, source, non_blocking);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_resize_as_sparse_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & shape) noexcept {
  try {
    try {
      (void)at::native::sparse_resize_as_out(output, self, shape);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_sparse_resize_out(at::Tensor & output, const at::Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept {
  try {
    try {
      (void)at::native::sparse_resize_out(output, self, c10::ArrayRef<int64_t>(size.data(), size.size()), sparse_dim, dense_dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_sparse_resize_and_clear_out(at::Tensor & output, const at::Tensor & self, rust::Slice<const std::int64_t> size, int64_t sparse_dim, int64_t dense_dim) noexcept {
  try {
    try {
      (void)at::native::sparse_resize_and_clear_out(output, self, c10::ArrayRef<int64_t>(size.data(), size.size()), sparse_dim, dense_dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_coo_empty_out(at::Tensor & output, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::native::sparse_coo_empty_out(output, c10::ArrayRef<int64_t>(size.data(), size.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparse_sampled_addmm_numeric_out(at::Tensor & output, const at::Tensor & mask, const at::Tensor & left, const at::Tensor & right, double beta, bool beta_is_integer, double alpha, bool alpha_is_integer) noexcept {
  try {
    try {
      (void)at::native::sparse_sampled_addmm_numeric_out(output, mask, left, right, beta, beta_is_integer, alpha, alpha_is_integer);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cholesky_ex_out(Tensor & L, Tensor & info, const Tensor & self, bool upper, bool check_errors) noexcept {
  try {
    try {
      (void)at::linalg_cholesky_ex_out(L, info, self, upper, check_errors);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_eig_out(Tensor & eigenvalues, Tensor & eigenvectors, const Tensor & self) noexcept {
  try {
    try {
      (void)at::linalg_eig_out(eigenvalues, eigenvectors, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_eigh_out(Tensor & eigvals, Tensor & eigvecs, const Tensor & self, rust::Str UPLO) noexcept {
  try {
    try {
      (void)at::linalg_eigh_out(eigvals, eigvecs, self, c10::string_view(UPLO.data(), UPLO.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_inv_ex_out(Tensor & inverse, Tensor & info, const Tensor & A, bool check_errors) noexcept {
  try {
    try {
      (void)at::linalg_inv_ex_out(inverse, info, A, check_errors);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_ldl_factor_out(Tensor & LD, Tensor & pivots, const Tensor & self, bool hermitian) noexcept {
  try {
    try {
      (void)at::linalg_ldl_factor_out(LD, pivots, self, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_ldl_factor_ex_out(Tensor & LD, Tensor & pivots, Tensor & info, const Tensor & self, bool hermitian, bool check_errors) noexcept {
  try {
    try {
      (void)at::linalg_ldl_factor_ex_out(LD, pivots, info, self, hermitian, check_errors);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_lu_out(Tensor & P, Tensor & L, Tensor & U, const Tensor & A, bool pivot) noexcept {
  try {
    try {
      (void)at::linalg_lu_out(P, L, U, A, pivot);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_lu_factor_out(Tensor & LU, Tensor & pivots, const Tensor & A, bool pivot) noexcept {
  try {
    try {
      (void)at::linalg_lu_factor_out(LU, pivots, A, pivot);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_lu_factor_ex_out(Tensor & LU, Tensor & pivots, Tensor & info, const Tensor & A, bool pivot, bool check_errors) noexcept {
  try {
    try {
      (void)at::linalg_lu_factor_ex_out(LU, pivots, info, A, pivot, check_errors);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_qr_out(Tensor & Q, Tensor & R, const Tensor & A, rust::Str mode) noexcept {
  try {
    try {
      (void)at::linalg_qr_out(Q, R, A, c10::string_view(mode.data(), mode.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_slogdet_out(Tensor & sign, Tensor & logabsdet, const Tensor & A) noexcept {
  try {
    try {
      (void)at::linalg_slogdet_out(sign, logabsdet, A);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_solve_ex_out(Tensor & result, Tensor & info, const Tensor & A, const Tensor & B, bool left, bool check_errors) noexcept {
  try {
    try {
      (void)at::linalg_solve_ex_out(result, info, A, B, left, check_errors);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_det_out(Tensor & out, const Tensor & A) noexcept {
  try {
    try {
      (void)at::linalg_det_out(out, A);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cholesky_out(Tensor & out, const Tensor & self, bool upper) noexcept {
  try {
    try {
      (void)at::linalg_cholesky_out(out, self, upper);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cross_out(Tensor & out, const Tensor & self, const Tensor & other, int64_t dim) noexcept {
  try {
    try {
      (void)at::linalg_cross_out(out, self, other, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_inv_out(Tensor & out, const Tensor & A) noexcept {
  try {
    try {
      (void)at::linalg_inv_out(out, A);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_ldl_solve_out(Tensor & out, const Tensor & LD, const Tensor & pivots, const Tensor & B, bool hermitian) noexcept {
  try {
    try {
      (void)at::linalg_ldl_solve_out(out, LD, pivots, B, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_lu_solve_out(Tensor & out, const Tensor & LU, const Tensor & pivots, const Tensor & B, bool left, bool adjoint) noexcept {
  try {
    try {
      (void)at::linalg_lu_solve_out(out, LU, pivots, B, left, adjoint);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matmul_out(Tensor & out, const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      (void)at::linalg_matmul_out(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_solve_out(Tensor & out, const Tensor & A, const Tensor & B, bool left) noexcept {
  try {
    try {
      (void)at::linalg_solve_out(out, A, B, left);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_solve_triangular_out(Tensor & out, const Tensor & self, const Tensor & B, bool upper, bool left, bool unitriangular) noexcept {
  try {
    try {
      (void)at::linalg_solve_triangular_out(out, self, B, upper, left, unitriangular);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_vecdot_out(Tensor & out, const Tensor & x, const Tensor & y, int64_t dim) noexcept {
  try {
    try {
      (void)at::linalg_vecdot_out(out, x, y, dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_eigvals_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::linalg_eigvals_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_eigvalsh_out(Tensor & out, const Tensor & self, rust::Str UPLO) noexcept {
  try {
    try {
      (void)at::linalg_eigvalsh_out(out, self, c10::string_view(UPLO.data(), UPLO.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_householder_product_out(Tensor & out, const Tensor & input, const Tensor & tau) noexcept {
  try {
    try {
      (void)at::linalg_householder_product_out(out, input, tau);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_exp_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::linalg_matrix_exp_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_power_out(Tensor & out, const Tensor & self, int64_t n) noexcept {
  try {
    try {
      (void)at::linalg_matrix_power_out(out, self, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_svdvals_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::linalg_svdvals_out_default(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_tensorinv_out(Tensor & out, const Tensor & self, int64_t ind) noexcept {
  try {
    try {
      (void)at::linalg_tensorinv_out(out, self, ind);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_tensorsolve_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) noexcept {
  try {
    try {
      (void)at::native::linalg_tensorsolve_out_default(out, self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_multi_dot_out(Tensor & out, const TerlanPytorchTensorTensorResourceListInput& tensors) noexcept {
  try {
    try {
      (void)at::linalg_multi_dot_out(out, tensors.values());
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_default(at::Tensor & out, const at::Tensor & self, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_default(out, self, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_absolute(Tensor & out, const Tensor & self, double tol, bool hermitian) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, double, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linalg_matrix_rank_out);
      (void)terlan_selected_overload(out, self, tol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_relative(Tensor & out, const Tensor & self, double tol, bool hermitian) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, double, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linalg_matrix_rank_out);
      (void)terlan_selected_overload(out, self, tol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tolerances(at::Tensor & out, const at::Tensor & self, double atol, double rtol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tolerances(out, self, atol, rtol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tensor_default(at::Tensor & out, const at::Tensor & self, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tensor_default(out, self, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tensor_absolute(at::Tensor & out, const at::Tensor & self, const at::Tensor & atol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tensor_absolute(out, self, atol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tensor_relative(at::Tensor & out, const at::Tensor & self, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tensor_relative(out, self, rtol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tensor_tolerances(at::Tensor & out, const at::Tensor & self, const at::Tensor & atol, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tensor_tolerances(out, self, atol, rtol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tolerance(at::Tensor & out, const at::Tensor & self, double tol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tolerance(out, self, tol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_rank_out_tensor_tolerance(at::Tensor & out, const at::Tensor & self, const at::Tensor & tol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_rank_out_tensor_tolerance(out, self, tol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_default(at::Tensor & out, const at::Tensor & self, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_default(out, self, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_absolute(Tensor & out, const Tensor & self, double rcond, bool hermitian) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, double, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linalg_pinv_out);
      (void)terlan_selected_overload(out, self, rcond, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_relative(Tensor & out, const Tensor & self, double rcond, bool hermitian) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(Tensor &, const Tensor &, double, bool);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::linalg_pinv_out);
      (void)terlan_selected_overload(out, self, rcond, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_tolerances(at::Tensor & out, const at::Tensor & self, double atol, double rtol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_tolerances(out, self, atol, rtol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_tensor_default(at::Tensor & out, const at::Tensor & self, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_tensor_default(out, self, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_tensor_absolute(at::Tensor & out, const at::Tensor & self, const at::Tensor & atol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_tensor_absolute(out, self, atol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_tensor_relative(at::Tensor & out, const at::Tensor & self, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_tensor_relative(out, self, rtol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_tensor_tolerances(at::Tensor & out, const at::Tensor & self, const at::Tensor & atol, const at::Tensor & rtol, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_tensor_tolerances(out, self, atol, rtol, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_rcond(at::Tensor & out, const at::Tensor & self, double rcond, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_rcond(out, self, rcond, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_pinv_out_tensor_rcond(at::Tensor & out, const at::Tensor & self, const at::Tensor & rcond, bool hermitian) noexcept {
  try {
    try {
      (void)at::native::linalg_pinv_out_tensor_rcond(out, self, rcond, hermitian);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cond_string_out(at::Tensor & out, const at::Tensor & self, rust::Str p) noexcept {
  try {
    try {
      (void)at::native::linalg_cond_out_string(out, self, std::string_view(p.data(), p.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_norm_string_out(at::Tensor & out, const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_norm_out_string(out, self, std::string_view(p.data(), p.size()), c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_norm_string_out_dtype(at::Tensor & out, const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_norm_out_string_dtype(out, self, std::string_view(p.data(), p.size()), c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_string_out(at::Tensor & out, const at::Tensor & self, rust::Str p, bool keepdim) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_string(out, self, std::string_view(p.data(), p.size()), keepdim);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_string_out_dimensions(at::Tensor & out, const at::Tensor & self, rust::Str p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_string_dimensions(out, self, std::string_view(p.data(), p.size()), c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cond_default_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::linalg_cond_out_default(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cond_int_out(at::Tensor & out, const at::Tensor & self, int64_t p) noexcept {
  try {
    try {
      (void)at::native::linalg_cond_out_int(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_cond_float_out(at::Tensor & out, const at::Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::native::linalg_cond_out_float(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_norm_int_out(at::Tensor & out, const at::Tensor & self, int64_t p) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_norm_out_int(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_norm_float_out(at::Tensor & out, const at::Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_norm_out_float(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_norm_int_configured_out(at::Tensor & out, const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_norm_out_int_configured(out, self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_matrix_norm_float_configured_out(at::Tensor & out, const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_matrix_norm_out_float_configured(out, self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_default_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_default(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_int_out(at::Tensor & out, const at::Tensor & self, int64_t p) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_int(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_float_out(at::Tensor & out, const at::Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_float(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_int_configured_out(at::Tensor & out, const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_int_configured(out, self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_norm_float_configured_out(at::Tensor & out, const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_norm_out_float_configured(out, self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_vector_norm_int_out(at::Tensor & out, const at::Tensor & self, int64_t p) noexcept {
  try {
    try {
      (void)at::native::linalg_vector_norm_out_int(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_vector_norm_float_out(at::Tensor & out, const at::Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::native::linalg_vector_norm_out_float(out, self, p);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_vector_norm_int_configured_out(at::Tensor & out, const at::Tensor & self, int64_t p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_vector_norm_out_int_configured(out, self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_vector_norm_float_configured_out(at::Tensor & out, const at::Tensor & self, double p, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t dtype) noexcept {
  try {
    try {
      (void)at::native::linalg_vector_norm_out_float_configured(out, self, p, c10::ArrayRef<int64_t>(dim.data(), dim.size()), keepdim, dtype);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_lstsq_out(at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values, const at::Tensor & a, const at::Tensor & b) noexcept {
  try {
    try {
      (void)at::native::linalg_lstsq_out_default(solution, residuals, rank, singular_values, a, b);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_lstsq_out_configured(at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values, const at::Tensor & a, const at::Tensor & b, double rcond, rust::Str driver) noexcept {
  try {
    try {
      (void)at::native::linalg_lstsq_out_configured(solution, residuals, rank, singular_values, a, b, rcond, std::string_view(driver.data(), driver.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_linalg_linalg_svd_out(at::Tensor & u, at::Tensor & s, at::Tensor & vh, const at::Tensor & self, bool full) noexcept {
  try {
    try {
      (void)at::native::linalg_svd_out_default(u, s, vh, self, full);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_linalg_operation_failed", "LibTorch linalg operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_cuda_record_stream(at::Tensor & tensor, const CudaStream & stream) noexcept {
  try {
    try {
      (void)at::native::cuda_stream_record(tensor, stream);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_cuda_operation_failed", "LibTorch CUDA stream operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_optimizer_add_parameter(at::native::Adam & arg0, const at::Tensor & arg1) noexcept {
  try {
    try {
      (void)at::native::adam_add_parameter(arg0, arg1);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_optimizer_operation_failed", "LibTorch optimizer operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_optimizer_zero_grad(at::native::Adam & arg0) noexcept {
  try {
    try {
      (void)at::native::adam_zero_grad(arg0);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_optimizer_operation_failed", "LibTorch optimizer operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_optimizer_step(at::native::Adam & arg0) noexcept {
  try {
    try {
      (void)at::native::adam_step(arg0);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_optimizer_operation_failed", "LibTorch optimizer operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_optimizer_restore(at::native::Adam & arg0, rust::Slice<const std::uint8_t> arg1) noexcept {
  try {
    try {
      (void)at::native::adam_restore(arg0, arg1);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_optimizer_operation_failed", "LibTorch optimizer operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_optimizer_cancel(at::native::Adam & arg0) noexcept {
  try {
    try {
      (void)at::native::adam_cancel(arg0);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_optimizer_operation_failed", "LibTorch optimizer operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_losscompletion_multilabel_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) noexcept {
  try {
    try {
      (void)at::multilabel_margin_loss_out(out, self, target, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_losscompletion_multilabel_margin_loss_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, std::int64_t reduction, const at::Tensor & is_target) noexcept {
  try {
    try {
      (void)at::native::multilabel_margin_loss_backward_exact_out(grad_input, grad_output, input, target, reduction, is_target);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor multilabel_margin_loss_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_losscompletion_multilabel_margin_loss_forward_out(at::Tensor & output, at::Tensor & is_target, const at::Tensor & input, const at::Tensor & target, std::int64_t reduction) noexcept {
  try {
    try {
      (void)at::native::multilabel_margin_loss_forward_exact_out(output, is_target, input, target, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor multilabel_margin_loss_forward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_fake_quantize_per_tensor_affine_cachemask_out(at::Tensor & out0, at::Tensor & out1, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    try {
      (void)at::native::fake_quantize_per_tensor_affine_cachemask_out_default(out0, out1, self, scale, zero_point, quant_min, quant_max);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fake quantize per-tensor cachemask output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_fake_quantize_per_channel_affine_cachemask_out(at::Tensor & out0, at::Tensor & out1, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) noexcept {
  try {
    try {
      (void)at::native::fake_quantize_per_channel_affine_cachemask_out_default(out0, out1, self, scale, zero_point, axis, quant_min, quant_max);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch fake quantize per-channel cachemask output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_quantize_per_channel_out(Tensor & out, const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, std::int64_t dtype) noexcept {
  try {
    try {
      (void)at::quantize_per_channel_out(out, self, scales, zero_points, axis, static_cast<c10::ScalarType>(dtype));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch per-channel quantization output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_quantize_per_tensor_tensor_qparams_out(Tensor & out, const Tensor & self, const Tensor & scale, const Tensor & zero_point, std::int64_t dtype) noexcept {
  try {
    try {
      (void)at::quantize_per_tensor_out(out, self, scale, zero_point, static_cast<c10::ScalarType>(dtype));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor-qparams quantization output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_quantize_per_tensor_out(Tensor & out, const Tensor & self, double scale, int64_t zero_point, std::int64_t dtype) noexcept {
  try {
    try {
      (void)at::quantize_per_tensor_out(out, self, scale, zero_point, static_cast<c10::ScalarType>(dtype));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch per-tensor quantization output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_quantize_per_tensor_dynamic_out(Tensor & out, const Tensor & self, std::int64_t dtype, bool reduce_range) noexcept {
  try {
    try {
      (void)at::quantize_per_tensor_dynamic_out(out, self, static_cast<c10::ScalarType>(dtype), reduce_range);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dynamic per-tensor quantization output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_int_repr_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::int_repr_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch quantized integer representation output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_dequantize_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::dequantize_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dequantization output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_q_per_channel_scales_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::q_per_channel_scales_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch per-channel quantization scales output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizationcompletion_q_per_channel_zero_points_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::q_per_channel_zero_points_out(out, self);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch per-channel quantization zero-points output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_exponential_out(at::Tensor & out, const at::Tensor & self, double lambd) noexcept {
  try {
    try {
      (void)at::exponential_out(out, self, lambd, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch exponential sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_poisson_out(at::Tensor & out, const at::Tensor & self) noexcept {
  try {
    try {
      (void)at::poisson_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch Poisson sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_binomial_out(at::Tensor & out, const at::Tensor & count, const at::Tensor & prob) noexcept {
  try {
    try {
      (void)at::binomial_out(out, count, prob, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch binomial sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_cauchy_out(at::Tensor & out, const at::Tensor & self, double median, double sigma) noexcept {
  try {
    try {
      (void)at::cauchy_out(out, self, median, sigma, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch Cauchy sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_geometric_out(at::Tensor & out, const at::Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::geometric_out(out, self, p, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch geometric sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_log_normal_out(at::Tensor & out, const at::Tensor & self, double mean, double std) noexcept {
  try {
    try {
      (void)at::log_normal_out(out, self, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch log-normal sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_uniform_out(at::Tensor & out, const at::Tensor & self, double from, double to) noexcept {
  try {
    try {
      (void)at::uniform_out(out, self, from, to, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch uniform sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_uniform_inplace(at::Tensor & self, double from, double to) noexcept {
  try {
    try {
      (void)at::native::uniform_(self, from, to, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place uniform sampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_normal_out(at::Tensor & out, const at::Tensor & self, double mean, double std) noexcept {
  try {
    try {
      (void)at::normal_out(out, self, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch normal sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_normal_inplace(at::Tensor & self, double mean, double std) noexcept {
  try {
    try {
      (void)at::native::normal_(self, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place normal sampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_exponential_inplace(at::Tensor & self, double lambd) noexcept {
  try {
    try {
      (void)at::native::exponential_(self, lambd, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place exponential sampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_cauchy_inplace(at::Tensor & self, double median, double sigma) noexcept {
  try {
    try {
      (void)at::native::cauchy_(self, median, sigma, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place Cauchy sampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_geometric_inplace(at::Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::native::geometric_(self, p, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place geometric sampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_log_normal_inplace(at::Tensor & self, double mean, double std) noexcept {
  try {
    try {
      (void)at::native::log_normal_(self, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place log-normal sampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_normal_tensor_tensor_out(at::Tensor & out, const at::Tensor & mean, const at::Tensor & std) noexcept {
  try {
    try {
      (void)at::normal_out(out, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor normal sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_normal_tensor_float_out(at::Tensor & out, const at::Tensor & mean, double std) noexcept {
  try {
    try {
      (void)at::normal_out(out, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor/scalar normal sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_normal_float_tensor_out(at::Tensor & out, double mean, const at::Tensor & std) noexcept {
  try {
    try {
      (void)at::normal_out(out, mean, std, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scalar/tensor normal sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_normal_float_float_out(at::Tensor & out, double mean, double std, rust::Slice<const std::int64_t> size) noexcept {
  try {
    try {
      (void)at::normal_out(out, mean, std, at::IntArrayRef(size.data(), size.size()), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch sized normal sampling output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_bernoulli_out(Tensor & out, const Tensor & self) noexcept {
  try {
    try {
      (void)at::bernoulli_out(out, self, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch Bernoulli output sampling failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_bernoulli_probability_out(Tensor & out, const Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::bernoulli_out(out, self, p, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scalar-probability Bernoulli output sampling failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_bernoulli_tensor_out(Tensor & out, const Tensor & self, const Tensor & p) noexcept {
  try {
    try {
      (void)at::bernoulli_out(out, self, p, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor-probability Bernoulli output sampling failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_bernoulli_inplace(Tensor & self, double p) noexcept {
  try {
    try {
      (void)at::native::bernoulli_(self, p, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place Bernoulli probability sampling failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_bernoulli_tensor_inplace(Tensor & self, const Tensor & p) noexcept {
  try {
    try {
      (void)at::native::bernoulli_(self, p, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch in-place tensor-probability Bernoulli sampling failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_distributioncompletion_multinomial_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement) noexcept {
  try {
    try {
      (void)at::multinomial_out(out, self, num_samples, replacement, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch multinomial output sampling failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_hashcompletion_hash_tensor_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> dim, bool keepdim, int64_t mode) noexcept {
  try {
    try {
      (void)at::hash_tensor_out(out, self, IntArrayRef(dim.data(), dim.size()), keepdim, mode);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch hash operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_resizeascompletion_resize_as_out(at::Tensor & out, const Tensor & self, const Tensor & the_template, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::resize_as_out(out, self, the_template, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch resize-as operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_resizeascompletion_resize_as_in_place(at::Tensor & self, const Tensor & the_template, bool has_memory_format, std::int64_t memory_format) noexcept {
  try {
    try {
      (void)at::resize_as_(self, the_template, has_memory_format ? ::std::optional<MemoryFormat>(static_cast<c10::MemoryFormat>(memory_format)) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch resize-as operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_newemptystridedcompletion_new_empty_strided_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::new_empty_strided_out(out, self, IntArrayRef(size.data(), size.size()), IntArrayRef(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch new-empty-strided operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nlllosscompletion_nll_loss_out_default(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    try {
      (void)at::nll_loss_out(out, self, target, std::nullopt, reduction, ignore_index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch NLL loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nlllosscompletion_nll_loss_out_weight(Tensor & out, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    try {
      (void)at::nll_loss_out(out, self, target, std::optional<at::Tensor>(weight), reduction, ignore_index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch NLL loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_multimarginlosscompletion_multi_margin_loss_out_default(Tensor & out, const Tensor & self, const Tensor & target, double p, double margin, int64_t reduction) noexcept {
  try {
    try {
      (void)at::multi_margin_loss_out(out, self, target, c10::Scalar(p), c10::Scalar(margin), std::nullopt, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch multi-margin loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_multimarginlosscompletion_multi_margin_loss_out_weight(Tensor & out, const Tensor & self, const Tensor & target, double p, double margin, const at::Tensor& weight, int64_t reduction) noexcept {
  try {
    try {
      (void)at::multi_margin_loss_out(out, self, target, c10::Scalar(p), c10::Scalar(margin), std::optional<at::Tensor>(weight), reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch multi-margin loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_multimarginlosscompletion_multi_margin_loss_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, double p, double margin, bool has_weight, const at::Tensor & weight, std::int64_t reduction) noexcept {
  try {
    try {
      (void)at::native::multi_margin_loss_backward_exact_out(grad_input, grad_output, input, target, p, margin, has_weight, weight, reduction);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_tensor_operation_failed", "Tensor multi_margin_loss_backward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebilinear2dcompletion_upsample_bilinear2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_bilinear2d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch bilinear upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebicubic2dcompletion_upsample_bicubic2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_bicubic2d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch bicubic upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nllloss2dcompletion_nll_loss2d_out_default(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    try {
      (void)at::nll_loss2d_out(out, self, target, std::nullopt, reduction, ignore_index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch 2-D NLL loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nllloss2dcompletion_nll_loss2d_out_weight(Tensor & out, const Tensor & self, const Tensor & target, const at::Tensor& weight, int64_t reduction, int64_t ignore_index) noexcept {
  try {
    try {
      (void)at::nll_loss2d_out(out, self, target, std::optional<at::Tensor>(weight), reduction, ignore_index);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch 2-D NLL loss operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_viewcopydtypecompletion_view_copy_dtype_out(at::Tensor & out, const at::Tensor & self, std::int64_t dtype) noexcept {
  try {
    try {
    using TerlanSelectedOverload = at::Tensor & (*)(at::Tensor &, const at::Tensor &, ScalarType);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::view_copy_out);
      (void)terlan_selected_overload(out, self, static_cast<c10::ScalarType>(dtype));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch dtype view-copy operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantilecompletion_quantile_tensor_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::quantile_out);
      (void)terlan_selected_overload(out, self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor quantile operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantilecompletion_quantile_scalar_out(at::Tensor & out, const at::Tensor & self, double q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(at::Tensor &, const at::Tensor &, double, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::quantile_out);
      (void)terlan_selected_overload(out, self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch scalar quantile operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nanquantilecompletion_nanquantile_tensor_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanquantile_out);
      (void)terlan_selected_overload(out, self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch NaN-quantile operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_nanquantilecompletion_nanquantile_scalar_out(at::Tensor & out, const at::Tensor & self, double q, std::int64_t dim, bool keepdim, rust::Str interpolation) noexcept {
  try {
    try {
    using TerlanSelectedOverload = Tensor & (*)(at::Tensor &, const at::Tensor &, double, ::std::optional<int64_t>, bool, c10::string_view);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::nanquantile_out);
      (void)terlan_selected_overload(out, self, q, ::std::optional<int64_t>(dim), keepdim, c10::string_view(interpolation.data(), interpolation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch NaN-quantile operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_randpermcompletion_randperm_out(Tensor & out, int64_t n) noexcept {
  try {
    try {
      (void)at::randperm_out(out, n);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch randperm operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplelinear1dcompletion_upsample_linear1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_linear1d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch linear upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplenearest1dcompletion_upsample_nearest1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest1d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nearest upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplenearest2dcompletion_upsample_nearest2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest2d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nearest upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplenearest3dcompletion_upsample_nearest3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest3d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nearest upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_setdatacompletion_set_data(Tensor & self, const Tensor & new_data) noexcept {
  try {
    try {
      (void)at::native::set_data(self, new_data);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch set_data operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest1d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest2d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest3d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), std::nullopt, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_linear1d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_trilinear3d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_bilinear2d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplebackwardcompletion_upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, rust::Slice<const std::int64_t> output_size, rust::Slice<const std::int64_t> input_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_bicubic2d_backward_out(grad_input, grad_output, IntArrayRef(output_size.data(), output_size.size()), IntArrayRef(input_size.data(), input_size.size()), align_corners, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch upsampling backward operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsampletrilinear3dcompletion_upsample_trilinear3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_trilinear3d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch trilinear upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_luunpackcompletion_lu_unpack_out_default(at::Tensor & p, at::Tensor & l, at::Tensor & u, const at::Tensor & lu_data, const at::Tensor & lu_pivots, bool unpack_data, bool unpack_pivots) noexcept {
  try {
    try {
      (void)at::native::lu_unpack_out_default(p, l, u, lu_data, lu_pivots, unpack_data, unpack_pivots);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch lu_unpack output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparsematmulcompletion_hspmm_out_default(at::Tensor & out, const at::Tensor & mat1, const at::Tensor & mat2) noexcept {
  try {
    try {
      (void)at::native::hspmm_out_default(out, mat1, mat2);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch hspmm output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_sparsematmulcompletion_sspaddmm_out_default(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, double beta, double alpha) noexcept {
  try {
    try {
      (void)at::native::sspaddmm_out_default(out, self, mat1, mat2, beta, alpha);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch sspaddmm output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplevectorcompletion_upsample_bilinear2d_vec_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size, bool align_corners) noexcept {
  try {
    try {
      (void)at::upsample_bilinear2d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), align_corners, std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch bilinear upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_upsamplevectorcompletion_upsample_nearest2d_vec_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> output_size) noexcept {
  try {
    try {
      (void)at::upsample_nearest2d_out(out, self, IntArrayRef(output_size.data(), output_size.size()), std::nullopt, std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch nearest upsampling operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_storagenative_storage_share_memory(at::Tensor & tensor) noexcept {
  try {
    try {
      (void)at::native::storage_share_memory(tensor);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch shared-memory conversion failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_storagenative_storage_set_out(at::Tensor & output, const at::Tensor & tensor, const StorageHandle & source) noexcept {
  try {
    try {
      (void)at::native::storage_set_out(output, tensor, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch storage set output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_storagenative_storage_set_offset_out(at::Tensor & output, const at::Tensor & tensor, const StorageHandle & source, std::int64_t storage_offset, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::native::storage_set_offset_out(output, tensor, source, storage_offset, c10::ArrayRef<std::int64_t>(size.data(), size.size()), c10::ArrayRef<std::int64_t>(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch storage offset output failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_storagenative_storage_set_in_place(at::Tensor & tensor, const StorageHandle & source) noexcept {
  try {
    try {
      (void)at::native::storage_set_in_place(tensor, source);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch storage in-place set failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_storagenative_storage_set_in_place_offset(at::Tensor & tensor, const StorageHandle & source, std::int64_t storage_offset, rust::Slice<const std::int64_t> size, rust::Slice<const std::int64_t> stride) noexcept {
  try {
    try {
      (void)at::native::storage_set_in_place_offset(tensor, source, storage_offset, c10::ArrayRef<std::int64_t>(size.data(), size.size()), c10::ArrayRef<std::int64_t>(stride.data(), stride.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch storage offset in-place set failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_thnn_conv2d_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::native::thnn_conv2d_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor THNN 2D convolution output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_slow_conv3d_forward_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::native::slow_conv3d_forward_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor slow_conv3d_forward output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_slow_conv3d_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding) noexcept {
  try {
    try {
      (void)at::native::slow_conv3d_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor slow_conv3d output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_slow_conv_dilated2d_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    try {
      (void)at::native::slow_conv_dilated2d_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor slow_conv_dilated2d output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_slow_conv_dilated3d_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    try {
      (void)at::native::slow_conv_dilated3d_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor slow_conv_dilated3d output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_slow_conv_transpose2d_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    try {
      (void)at::native::slow_conv_transpose2d_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(output_padding.data(), output_padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor slow_conv_transpose2d output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_legacyconvolution_slow_conv_transpose3d_out(at::Tensor & output, const at::Tensor & input, const at::Tensor & weight, rust::Slice<const std::int64_t> kernel_size, bool has_bias, const at::Tensor & bias, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> output_padding, rust::Slice<const std::int64_t> dilation) noexcept {
  try {
    try {
      (void)at::native::slow_conv_transpose3d_exact_out(output, input, weight, c10::ArrayRef<long>(kernel_size.data(), kernel_size.size()), has_bias, bias, c10::ArrayRef<long>(stride.data(), stride.size()), c10::ArrayRef<long>(padding.data(), padding.size()), c10::ArrayRef<long>(output_padding.data(), output_padding.size()), c10::ArrayRef<long>(dilation.data(), dilation.size()));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "Tensor slow_conv_transpose3d output operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizedpooling_quantized_max_pool1d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    try {
      (void)at::quantized_max_pool1d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_quantized_operation_failed", "PyTorch quantized CPU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizedpooling_quantized_max_pool2d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    try {
      (void)at::quantized_max_pool2d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_quantized_operation_failed", "PyTorch quantized CPU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizedpooling_quantized_max_pool3d_out(Tensor & out, const Tensor & self, rust::Slice<const std::int64_t> kernel_size, rust::Slice<const std::int64_t> stride, rust::Slice<const std::int64_t> padding, rust::Slice<const std::int64_t> dilation, bool ceil_mode) noexcept {
  try {
    try {
      (void)at::quantized_max_pool3d_out(out, self, IntArrayRef(kernel_size.data(), kernel_size.size()), IntArrayRef(stride.data(), stride.size()), IntArrayRef(padding.data(), padding.size()), IntArrayRef(dilation.data(), dilation.size()), ceil_mode);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_quantized_operation_failed", "PyTorch quantized CPU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_mutation_pytorch_quantizednormalization_quantized_batch_norm_out(at::Tensor & out, const at::Tensor & input, bool has_weight, const at::Tensor & weight, bool has_bias, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, std::int64_t output_zero_point) noexcept {
  try {
    try {
      (void)at::native::quantized_batch_norm_exact_out(out, input, has_weight, weight, has_bias, bias, mean, var, eps, output_scale, output_zero_point);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_quantized_operation_failed", "PyTorch quantized CPU operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

}  // namespace at::native
