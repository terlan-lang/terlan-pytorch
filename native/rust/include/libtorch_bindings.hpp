#pragma once

#include "terlan_tensor_packet.hpp"

#include <tuple>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/ops/amax.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/ctc_loss.h>
#include <ATen/ops/frobenius_norm.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/nanmean.h>
#include <ATen/ops/nuclear_norm.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/nansum.h>
#include <ATen/ops/pad_sequence.h>
#include <ATen/ops/sparse_dim.h>
#include <ATen/ops/dense_dim.h>
#include <ATen/ops/einsum.h>
#include <ATen/ops/is_coalesced.h>
#include <ATen/ops/indices.h>
#include <ATen/ops/values.h>
#include <ATen/ops/crow_indices.h>
#include <ATen/ops/col_indices.h>
#include <ATen/ops/ccol_indices.h>
#include <ATen/ops/row_indices.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#include <ATen/ops/to_sparse.h>
#include <ATen/ops/to_dense.h>
#include <ATen/ops/coalesce.h>
#include <ATen/ops/sparse_mask.h>
#include <ATen/ops/to_sparse_csr.h>
#include <ATen/ops/to_sparse_csc.h>
#include <ATen/ops/to_sparse_bsr.h>
#include <ATen/ops/to_sparse_bsc.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/resize_as_sparse.h>
#include <ATen/ops/sparse_resize.h>
#include <ATen/ops/sparse_resize_and_clear.h>
#include <ATen/ops/sparse_sampled_addmm.h>
#include <ATen/ops/to_dense_backward.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/lu_solve.h>
#include <ATen/ops/ormqr.h>
#include <ATen/ops/linalg__powsum.h>
#include <ATen/ops/linalg_cholesky.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cond.h>
#include <ATen/ops/linalg_cross.h>
#include <ATen/ops/linalg_det.h>
#include <ATen/ops/linalg_diagonal.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_ldl_factor.h>
#include <ATen/ops/linalg_ldl_factor_ex.h>
#include <ATen/ops/linalg_ldl_solve.h>
#include <ATen/ops/linalg_lstsq.h>
#include <ATen/ops/linalg_lu.h>
#include <ATen/ops/linalg_lu_factor.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_matmul.h>
#include <ATen/ops/linalg_matrix_exp.h>
#include <ATen/ops/linalg_matrix_norm.h>
#include <ATen/ops/linalg_matrix_power.h>
#include <ATen/ops/linalg_matrix_rank.h>
#include <ATen/ops/linalg_multi_dot.h>
#include <ATen/ops/linalg_norm.h>
#include <ATen/ops/linalg_pinv.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_slogdet.h>
#include <ATen/ops/linalg_solve.h>
#include <ATen/ops/linalg_solve_ex.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_tensorinv.h>
#include <ATen/ops/linalg_tensorsolve.h>
#include <ATen/ops/linalg_vander.h>
#include <ATen/ops/linalg_vecdot.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/qr.h>
#include <ATen/ops/slogdet.h>
#include <ATen/ops/triangular_solve.h>
#include <ATen/ops/sparse_dim_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/dense_dim_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/is_coalesced_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/indices_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/values_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/crow_indices_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/col_indices_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/ccol_indices_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/row_indices_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/to_sparse_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/to_dense_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/coalesce_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/sparse_mask_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/to_sparse_csr_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/to_sparse_csc_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/to_sparse_bsr_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/to_sparse_bsc_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/copy_sparse_to_sparse_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/resize_as_sparse_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/sparse_resize_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/sparse_resize_and_clear_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/sym_constrain_range.h>
#include <ATen/ops/sym_constrain_range_for_size.h>
#include <ATen/ops/choose_qparams_optimized.h>
#include <ATen/ops/fused_moving_avg_obs_fake_quant.h>

// Small package-owned C++ conveniences for sparse contracts whose public
// Terlan shape carries option-presence bits or scalar-kind tags. These remain
// ordinary C++ functions consumed by the generated CXX bridge.
namespace at::native {

inline at::Tensor sparse_indices_copy(const at::Tensor& self) { return self.indices().clone(); }
inline at::Tensor& sparse_indices_copy_out(at::Tensor& output, const at::Tensor& self) {
  output = self.indices().clone(); return output;
}
inline at::Tensor sparse_values_copy(const at::Tensor& self) { return self.values().clone(); }
inline at::Tensor& sparse_values_copy_out(at::Tensor& output, const at::Tensor& self) {
  output = self.values().clone(); return output;
}
inline at::Tensor sparse_crow_indices_copy(const at::Tensor& self) { return self.crow_indices().clone(); }
inline at::Tensor& sparse_crow_indices_copy_out(at::Tensor& output, const at::Tensor& self) { output = self.crow_indices().clone(); return output; }
inline at::Tensor sparse_col_indices_copy(const at::Tensor& self) { return self.col_indices().clone(); }
inline at::Tensor& sparse_col_indices_copy_out(at::Tensor& output, const at::Tensor& self) { output = self.col_indices().clone(); return output; }
inline at::Tensor sparse_ccol_indices_copy(const at::Tensor& self) { return self.ccol_indices().clone(); }
inline at::Tensor& sparse_ccol_indices_copy_out(at::Tensor& output, const at::Tensor& self) { output = self.ccol_indices().clone(); return output; }
inline at::Tensor sparse_row_indices_copy(const at::Tensor& self) { return self.row_indices().clone(); }
inline at::Tensor& sparse_row_indices_copy_out(at::Tensor& output, const at::Tensor& self) { output = self.row_indices().clone(); return output; }
inline at::Tensor sparse_mask(const at::Tensor& self, const at::Tensor& mask) { return self.sparse_mask(mask); }
inline at::Tensor& sparse_resize_as_out(at::Tensor& output, const at::Tensor& self, const at::Tensor& shape) {
  output = at::resize_as_sparse(self.clone(), shape); return output;
}
inline at::Tensor& sparse_resize_out(at::Tensor& output, const at::Tensor& self, c10::ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  output = at::sparse_resize(self.clone(), size, sparse_dim, dense_dim); return output;
}
inline at::Tensor& sparse_resize_and_clear_out(at::Tensor& output, const at::Tensor& self, c10::ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  output = at::sparse_resize_and_clear(self.clone(), size, sparse_dim, dense_dim); return output;
}
inline at::Tensor& sparse_copy_in_place(at::Tensor& self, const at::Tensor& source, bool non_blocking) {
  at::native::copy_sparse_(self, source, non_blocking); return self;
}
inline at::Tensor& sparse_resize_as_in_place(at::Tensor& self, const at::Tensor& shape) {
  at::native::resize_as_sparse_(self, shape); return self;
}
inline at::Tensor& sparse_resize_in_place(at::Tensor& self, c10::ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  at::native::sparse_resize_(self, size, sparse_dim, dense_dim); return self;
}
inline at::Tensor& sparse_resize_and_clear_in_place(at::Tensor& self, c10::ArrayRef<int64_t> size, int64_t sparse_dim, int64_t dense_dim) {
  at::native::sparse_resize_and_clear_(self, size, sparse_dim, dense_dim); return self;
}
inline at::Tensor sparse_to_dense_configured(const at::Tensor& self, int64_t dtype, bool has_dtype, bool masked_grad, bool has_masked_grad) {
  return self.to_dense(has_dtype ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype)) : std::nullopt,
                       has_masked_grad ? std::optional<bool>(masked_grad) : std::nullopt);
}
inline at::Tensor sparse_to_sparse_configured(const at::Tensor& self, int64_t layout, bool has_layout,
                                              c10::ArrayRef<int64_t> block_size, bool has_block_size,
                                              int64_t dense_dim, bool has_dense_dim) {
  auto selected_layout = has_layout ? std::optional<at::Layout>(static_cast<at::Layout>(layout)) : std::nullopt;
  auto blocks = has_block_size ? at::OptionalIntArrayRef(block_size) : at::OptionalIntArrayRef(std::nullopt);
  auto dense = has_dense_dim ? std::optional<int64_t>(dense_dim) : std::nullopt;
  return self.to_sparse(selected_layout, blocks, dense);
}
inline at::Tensor sparse_coo_empty_configured(c10::ArrayRef<int64_t> size, int64_t dtype, bool has_dtype,
                                              int64_t layout, bool has_layout, int64_t device_type,
                                              int64_t device_index, bool has_device, bool pin_memory,
                                              bool has_pin_memory) {
  auto dt = has_dtype ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype)) : std::nullopt;
  auto lo = has_layout ? std::optional<at::Layout>(static_cast<at::Layout>(layout)) : std::nullopt;
  auto dev = has_device ? std::optional<at::Device>(device_type == 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index))) : std::nullopt;
  auto pin = has_pin_memory ? std::optional<bool>(pin_memory) : std::nullopt;
  return at::sparse_coo_tensor(size, dt, lo, dev, pin);
}
inline at::Tensor& sparse_coo_empty_out(at::Tensor& output, c10::ArrayRef<int64_t> size) {
  output = at::sparse_coo_tensor(size, output.options().layout(at::kSparse)); return output;
}
inline at::Tensor sparse_coo_from_components_configured(const at::Tensor& indices, const at::Tensor& values,
                                                        c10::ArrayRef<int64_t> size, bool has_size,
                                                        int64_t dtype, bool has_dtype, int64_t layout,
                                                        bool has_layout, int64_t device_type, int64_t device_index,
                                                        bool has_device, bool pin_memory, bool has_pin_memory,
                                                        bool is_coalesced, bool has_is_coalesced) {
  auto dt = has_dtype ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype)) : std::nullopt;
  auto lo = has_layout ? std::optional<at::Layout>(static_cast<at::Layout>(layout)) : std::nullopt;
  auto dev = has_device ? std::optional<at::Device>(device_type == 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index))) : std::nullopt;
  auto pin = has_pin_memory ? std::optional<bool>(pin_memory) : std::nullopt;
  auto coal = has_is_coalesced ? std::optional<bool>(is_coalesced) : std::nullopt;
  return has_size ? at::sparse_coo_tensor(indices, values, size, dt, lo, dev, pin, coal)
                  : at::sparse_coo_tensor(indices, values, dt, lo, dev, pin, coal);
}
inline at::Tensor sparse_compressed_from_components_configured(int64_t kind, const at::Tensor& compressed,
                                                               const at::Tensor& plain, const at::Tensor& values,
                                                               c10::ArrayRef<int64_t> size, bool has_size,
                                                               int64_t dtype, bool has_dtype, int64_t layout,
                                                               bool has_layout, int64_t device_type,
                                                               int64_t device_index, bool has_device,
                                                               bool pin_memory, bool has_pin_memory) {
  auto dt = has_dtype ? std::optional<at::ScalarType>(static_cast<at::ScalarType>(dtype)) : std::nullopt;
  auto lo = has_layout ? std::optional<at::Layout>(static_cast<at::Layout>(layout)) : std::nullopt;
  auto dev = has_device ? std::optional<at::Device>(device_type == 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index))) : std::nullopt;
  auto pin = has_pin_memory ? std::optional<bool>(pin_memory) : std::nullopt;
  auto call = [&](auto fn) { return has_size ? fn(compressed, plain, values, size, dt, lo, dev, pin)
                                               : fn(compressed, plain, values, dt, lo, dev, pin); };
  switch (kind) {
    case 0: return call([](auto&&... a) { return at::sparse_compressed_tensor(std::forward<decltype(a)>(a)...); });
    case 2: return call([](auto&&... a) { return at::sparse_csr_tensor(std::forward<decltype(a)>(a)...); });
    case 4: return call([](auto&&... a) { return at::sparse_csc_tensor(std::forward<decltype(a)>(a)...); });
    case 5: return call([](auto&&... a) { return at::sparse_bsr_tensor(std::forward<decltype(a)>(a)...); });
    default: return call([](auto&&... a) { return at::sparse_bsc_tensor(std::forward<decltype(a)>(a)...); });
  }
}
inline at::Tensor sparse_sampled_addmm_numeric(const at::Tensor& mask, const at::Tensor& left, const at::Tensor& right,
                                               double beta, bool beta_is_integer, double alpha, bool alpha_is_integer) {
  return at::sparse_sampled_addmm(mask, left, right,
                                  beta_is_integer ? at::Scalar(static_cast<int64_t>(beta)) : at::Scalar(beta),
                                  alpha_is_integer ? at::Scalar(static_cast<int64_t>(alpha)) : at::Scalar(alpha));
}
inline at::Tensor& sparse_sampled_addmm_numeric_out(at::Tensor& output, const at::Tensor& mask, const at::Tensor& left,
                                                    const at::Tensor& right, double beta, bool beta_is_integer,
                                                    double alpha, bool alpha_is_integer) {
  output = at::sparse_sampled_addmm(mask, left, right,
                                    beta_is_integer ? at::Scalar(static_cast<int64_t>(beta)) : at::Scalar(beta),
                                    alpha_is_integer ? at::Scalar(static_cast<int64_t>(alpha)) : at::Scalar(alpha));
  return output;
}
inline at::Tensor sparse_to_dense_backward_configured(const at::Tensor& gradient, const at::Tensor& input,
                                                      bool masked_grad, bool has_masked_grad) {
  return at::to_dense_backward(gradient, input, has_masked_grad ? std::optional<bool>(masked_grad) : std::nullopt);
}
} // namespace at::native

// Package-owned carrier alias for legacy TensorPair contracts. The generated
// CXX bridge owns the tuple; no operator wrapper or C ABI is introduced.
using TensorPair = std::tuple<at::Tensor, at::Tensor>;
namespace at::native {
using TensorPair = std::tuple<at::Tensor, at::Tensor>;
}

#include <ATen/ops/affine_grid_generator.h>
#include <ATen/ops/affine_grid_generator_backward.h>
#include <ATen/ops/adaptive_avg_pool1d.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/adaptive_avg_pool3d.h>
#include <ATen/ops/adaptive_avg_pool3d_backward.h>
#include <ATen/ops/adaptive_max_pool1d.h>
#include <ATen/ops/adaptive_max_pool2d.h>
#include <ATen/ops/adaptive_max_pool2d_backward.h>
#include <ATen/ops/adaptive_max_pool3d.h>
#include <ATen/ops/adaptive_max_pool3d_backward.h>
#include <ATen/ops/add.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arctan2.h>
#include <ATen/ops/atan2.h>
#include <ATen/ops/addbmm.h>
#include <ATen/ops/addcdiv.h>
#include <ATen/ops/addcmul.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/alpha_dropout.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/as_strided_copy.h>
#include <ATen/ops/as_strided_scatter.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/avg_pool1d.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/avg_pool3d.h>
#include <ATen/ops/avg_pool3d_backward.h>
#include <ATen/ops/baddbmm.h>
#include <ATen/ops/bilinear.h>
#include <ATen/ops/bincount.h>
#include <ATen/ops/bartlett_window.h>
#include <ATen/ops/blackman_window.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/hamming_window.h>
#include <ATen/ops/hann_window.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/kaiser_window.h>
#include <ATen/ops/broadcast_tensors.h>
#include <ATen/ops/block_diag.h>
#include <ATen/ops/bucketize.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cartesian_prod.h>
#include <ATen/ops/celu.h>
#include <ATen/ops/channel_shuffle.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/cdist.h>
#include <ATen/ops/chain_matmul.h>
#include <ATen/ops/col2im.h>
#include <ATen/ops/column_stack.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/concat.h>
#include <ATen/ops/concatenate.h>
#include <ATen/ops/cosine_embedding_loss.h>
#include <ATen/ops/cosine_similarity.h>
#include <ATen/ops/contiguous.h>
#include <ATen/ops/conv_tbc.h>
#include <ATen/ops/conv_tbc_backward.h>
#include <ATen/ops/cross.h>
#include <ATen/ops/cummaxmin_backward.h>
#include <ATen/ops/cumprod_backward.h>
#include <ATen/ops/cumulative_trapezoid.h>
#include <ATen/ops/cummax.h>
#include <ATen/ops/cummin.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diagonal_backward.h>
#include <ATen/ops/diagonal_copy.h>
#include <ATen/ops/diagonal_scatter.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/dist.h>
#include <ATen/ops/div.h>
#include <ATen/ops/dstack.h>
#include <ATen/ops/dsplit.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/feature_alpha_dropout.h>
#include <ATen/ops/feature_dropout.h>
#include <ATen/ops/expand_copy.h>
#include <ATen/ops/flatten_dense_tensors.h>
#include <ATen/ops/fractional_max_pool2d.h>
#include <ATen/ops/fractional_max_pool2d_backward.h>
#include <ATen/ops/fractional_max_pool3d.h>
#include <ATen/ops/fractional_max_pool3d_backward.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_fft2.h>
#include <ATen/ops/fft_fftfreq.h>
#include <ATen/ops/fft_fftn.h>
#include <ATen/ops/fft_fftshift.h>
#include <ATen/ops/fft_hfft.h>
#include <ATen/ops/fft_hfft2.h>
#include <ATen/ops/fft_hfftn.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/fft_ifft2.h>
#include <ATen/ops/fft_ifftn.h>
#include <ATen/ops/fft_ifftshift.h>
#include <ATen/ops/fft_ihfft.h>
#include <ATen/ops/fft_ihfft2.h>
#include <ATen/ops/fft_ihfftn.h>
#include <ATen/ops/fft_irfft.h>
#include <ATen/ops/fft_irfft2.h>
#include <ATen/ops/fft_irfftn.h>
#include <ATen/ops/fft_rfft.h>
#include <ATen/ops/fft_rfft2.h>
#include <ATen/ops/fft_rfftfreq.h>
#include <ATen/ops/fft_rfftn.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/elu_backward.h>
#include <ATen/ops/embedding_backward.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/embedding_sparse_backward.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask_backward.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_cachemask_backward.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/fill_diagonal.h>
#include <ATen/ops/glu.h>
#include <ATen/ops/glu_backward.h>
#include <ATen/ops/glu_backward_jvp.h>
#include <ATen/ops/glu_jvp.h>
#include <ATen/ops/grid_sampler.h>
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/grid_sampler_3d.h>
#include <ATen/ops/gather_backward.h>
#include <ATen/ops/gelu_backward.h>
#include <ATen/ops/group_norm.h>
#include <ATen/ops/heaviside.h>
#include <ATen/ops/hardshrink.h>
#include <ATen/ops/hardshrink_backward.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/hardtanh_backward.h>
#include <ATen/ops/hinge_embedding_loss.h>
#include <ATen/ops/hstack.h>
#include <ATen/ops/hsplit.h>
#include <ATen/ops/huber_loss.h>
#include <ATen/ops/huber_loss_backward.h>
#include <ATen/ops/histc.h>
#include <ATen/ops/igamma.h>
#include <ATen/ops/igammac.h>
#include <ATen/ops/index_add.h>
#include <ATen/ops/index_copy.h>
#include <ATen/ops/index_fill.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/index_select_backward.h>
#include <ATen/ops/index_reduce.h>
#include <ATen/ops/im2col.h>
#include <ATen/ops/isin.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/kl_div.h>
#include <ATen/ops/kthvalue.h>
#include <ATen/ops/l1_loss.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu_backward.h>
#include <ATen/ops/lerp.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/logit_backward.h>
#include <ATen/ops/lu_solve.h>
#include <ATen/ops/margin_ranking_loss.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/set.h>
#include <ATen/ops/masked_scatter.h>
#include <ATen/ops/masked_scatter_backward.h>
#include <ATen/ops/masked_select_backward.h>
#include <ATen/ops/max.h>
#include <ATen/ops/max_pool1d.h>
#include <ATen/ops/max_pool1d_with_indices.h>
#include <ATen/ops/max_pool2d.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool2d_with_indices_backward.h>
#include <ATen/ops/max_pool3d.h>
#include <ATen/ops/max_pool3d_with_indices.h>
#include <ATen/ops/max_pool3d_with_indices_backward.h>
#include <ATen/ops/max_unpool2d.h>
#include <ATen/ops/max_unpool3d.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/median.h>
#include <ATen/ops/min.h>
#include <ATen/ops/mode.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/mse_loss_backward.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nanmedian.h>
#include <ATen/ops/native_dropout_backward.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/ones_native.h>
#include <ATen/ops/ormqr.h>
#include <ATen/ops/pairwise_distance.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/poisson_nll_loss.h>
#include <ATen/ops/put.h>
#include <ATen/ops/reflection_pad1d.h>
#include <ATen/ops/reflection_pad2d.h>
#include <ATen/ops/reflection_pad3d.h>
#include <ATen/ops/reflection_pad1d_backward.h>
#include <ATen/ops/reflection_pad2d_backward.h>
#include <ATen/ops/reflection_pad3d_backward.h>
#include <ATen/ops/replication_pad1d_backward.h>
#include <ATen/ops/replication_pad2d_backward.h>
#include <ATen/ops/replication_pad3d_backward.h>
#include <ATen/ops/replication_pad1d.h>
#include <ATen/ops/replication_pad2d.h>
#include <ATen/ops/replication_pad3d.h>
#include <ATen/ops/renorm.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/norm_except_dim.h>
#include <ATen/ops/range.h>
#include <ATen/ops/row_stack.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/scatter_add.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/select_backward.h>
#include <ATen/ops/select_copy.h>
#include <ATen/ops/select_scatter.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/slice_backward.h>
#include <ATen/ops/slice_copy.h>
#include <ATen/ops/slice_scatter.h>
#include <ATen/ops/slogdet.h>
#include <ATen/ops/smooth_l1_loss.h>
#include <ATen/ops/smooth_l1_loss_backward.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/soft_margin_loss.h>
#include <ATen/ops/soft_margin_loss_backward.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/softplus_backward.h>
#include <ATen/ops/softshrink.h>
#include <ATen/ops/softshrink_backward.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/tensordot.h>
#include <ATen/ops/threshold.h>
#include <ATen/ops/threshold_backward.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/trace_backward.h>
#include <ATen/ops/transpose_copy.h>
#include <ATen/ops/trapezoid.h>
#include <ATen/ops/trapz.h>
#include <ATen/ops/triplet_margin_loss.h>
#include <ATen/ops/true_divide.h>
#include <ATen/ops/unfold.h>
#include <ATen/ops/unfold_backward.h>
#include <ATen/ops/unfold_copy.h>
#include <ATen/ops/unflatten_dense_tensors.h>
#include <ATen/ops/vstack.h>
#include <ATen/ops/vsplit.h>
#include <ATen/ops/where.h>
#include <ATen/Parallel.h>
#include <ATen/autocast_mode.h>
#include <c10/core/GradMode.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/binary_cross_entropy.h>
#include <ATen/ops/binary_cross_entropy_backward.h>
#include <ATen/ops/binary_cross_entropy_with_logits.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/divide.h>
#include <ATen/ops/expand_as.h>
#include <ATen/ops/ger.h>
#include <ATen/ops/hardsigmoid_backward.h>
#include <ATen/ops/hardswish_backward.h>
#include <ATen/ops/infinitely_differentiable_gelu_backward.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_matmul.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matrix_exp_backward.h>
#include <ATen/ops/mish_backward.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/multiply.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/orgqr.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/polar.h>
#include <ATen/ops/prelu.h>
#include <ATen/ops/reshape_as.h>
#include <ATen/ops/sigmoid_backward.h>
#include <ATen/ops/silu_backward.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/special_chebyshev_polynomial_t.h>
#include <ATen/ops/special_chebyshev_polynomial_u.h>
#include <ATen/ops/special_chebyshev_polynomial_v.h>
#include <ATen/ops/special_chebyshev_polynomial_w.h>
#include <ATen/ops/special_hermite_polynomial_h.h>
#include <ATen/ops/special_hermite_polynomial_he.h>
#include <ATen/ops/special_laguerre_polynomial_l.h>
#include <ATen/ops/special_legendre_polynomial_p.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w.h>
#include <ATen/ops/special_xlog1py.h>
#include <ATen/ops/special_xlogy.h>
#include <ATen/ops/special_zeta.h>
#include <ATen/ops/special_chebyshev_polynomial_u.h>
#include <ATen/ops/special_chebyshev_polynomial_v.h>
#include <ATen/ops/special_chebyshev_polynomial_w.h>
#include <ATen/ops/special_gammainc.h>
#include <ATen/ops/special_gammaincc.h>
#include <ATen/ops/special_hermite_polynomial_h.h>
#include <ATen/ops/special_hermite_polynomial_he.h>
#include <ATen/ops/special_laguerre_polynomial_l.h>
#include <ATen/ops/special_legendre_polynomial_p.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w.h>
#include <ATen/ops/tanh_backward.h>
#include <ATen/ops/type_as.h>
#include <ATen/ops/view_as.h>
#include <ATen/ops/adjoint.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/alias_copy.h>
#include <ATen/ops/atleast_1d.h>
#include <ATen/ops/atleast_2d.h>
#include <ATen/ops/atleast_3d.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/data.h>
#include <ATen/ops/detach.h>
#include <ATen/ops/detach_copy.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/lift.h>
#include <ATen/ops/lift_fresh.h>
#include <ATen/ops/lift_fresh_copy.h>
#include <ATen/ops/mH.h>
#include <ATen/ops/mT.h>
#include <ATen/ops/matrix_H.h>
#include <ATen/ops/numpy_T.h>
#include <ATen/ops/positive.h>
#include <ATen/ops/ravel.h>
#include <ATen/ops/real.h>
#include <ATen/ops/resolve_conj.h>
#include <ATen/ops/resolve_neg.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/squeeze_copy.h>
#include <ATen/ops/t.h>
#include <ATen/ops/t_copy.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_complex_copy.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/view_as_real_copy.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/conv3d.h>
#include <ATen/ops/conv_transpose1d.h>
#include <ATen/ops/conv_transpose2d.h>
#include <ATen/ops/conv_transpose3d.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/can_cast.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/is_same_size.h>
#include <ATen/ops/promote_types.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/repeat_interleave.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/functional/normalization.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/rand_like.h>
#include <ATen/ops/randn_like.h>
#include <ATen/ops/randint_like.h>
#include <ATen/ops/random.h>
#include <ATen/ops/random_native.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/svd.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/unique_dim.h>
#include <ATen/ops/unique_dim_consecutive.h>
#include <ATen/ops/histogram.h>
#include <ATen/ops/set.h>
#include <ATen/ops/rrelu.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/stft.h>
#include <ATen/ops/istft.h>
#include <ATen/ops/segment_reduce.h>
#include <ATen/ops/hspmm.h>
#include <ATen/ops/smm.h>
#include <ATen/ops/sspaddmm.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/rsub.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/std.h>
#include <ATen/ops/std_mean.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/var.h>
#include <ATen/ops/var_mean.h>
#include <ATen/ops/zero.h>

#include <ATen/ops/eq.h>
#include <ATen/ops/float_power.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/fmax.h>
#include <ATen/ops/fmin.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/gcd.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/greater.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/hypot.h>
#include <ATen/ops/inner.h>
#include <ATen/ops/isclose.h>
#include <ATen/ops/kron.h>
#include <ATen/ops/lcm.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/le.h>
#include <ATen/ops/less.h>
#include <ATen/ops/less_equal.h>
#include <ATen/ops/logaddexp.h>
#include <ATen/ops/logaddexp2.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/nextafter.h>
#include <ATen/ops/not_equal.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/special_xlog1py.h>
#include <ATen/ops/special_xlogy.h>
#include <ATen/ops/special_zeta.h>
#include <ATen/ops/take.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/greater_equal.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/arctan2_compositeimplicitautograd_dispatch.h>
#include <ATen/ops/atan2_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/copysign_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/heaviside_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/hypot_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/igamma_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/igammac_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/lcm_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/ldexp_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/nextafter_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/fmod_compositeexplicitautogradnonfunctional_dispatch.h>
#include <ATen/ops/logical_and_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/logical_or_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/logical_xor_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/atan2_cpu_dispatch.h>
#include <ATen/ops/copysign_cpu_dispatch.h>
#include <ATen/ops/heaviside_cpu_dispatch.h>
#include <ATen/ops/hypot_cpu_dispatch.h>
#include <ATen/ops/igamma_cpu_dispatch.h>
#include <ATen/ops/igammac_cpu_dispatch.h>
#include <ATen/ops/lcm_cpu_dispatch.h>
#include <ATen/ops/nextafter_cpu_dispatch.h>
#include <ATen/ops/fmod_cpu_dispatch.h>
#include <ATen/ops/logical_and_cpu_dispatch.h>
#include <ATen/ops/logical_or_cpu_dispatch.h>
#include <ATen/ops/logical_xor_cpu_dispatch.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/absolute.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/acosh.h>
#include <ATen/ops/all.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/argwhere.h>
#include <ATen/ops/any.h>
#include <ATen/ops/arccos.h>
#include <ATen/ops/arccosh.h>
#include <ATen/ops/arcsin.h>
#include <ATen/ops/arcsinh.h>
#include <ATen/ops/arctan.h>
#include <ATen/ops/arctanh.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/cosh.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/digamma.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/fix.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frexp.h>
#include <ATen/ops/hardsigmoid.h>
#include <ATen/ops/hardswish.h>
#include <ATen/ops/i0.h>
#include <ATen/ops/inverse.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isfinite.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/isneginf.h>
#include <ATen/ops/isposinf.h>
#include <ATen/ops/lgamma.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log10.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/log_sigmoid.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/mish.h>
#include <ATen/ops/msort.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/negative.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/relu6.h>
#include <ATen/ops/round.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/selu.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/signbit.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sinc.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/square.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/trace.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/broadcast_to.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/diagflat.h>
#include <ATen/ops/expand.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/linalg_matrix_power.h>
#include <ATen/ops/linalg_tensorinv.h>
#include <ATen/ops/logcumsumexp.h>
#include <ATen/ops/matrix_power.h>
#include <ATen/ops/moveaxis.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/mvlgamma.h>
#include <ATen/ops/native_channel_shuffle.h>
#include <ATen/ops/one_hot.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/permute_copy.h>
#include <ATen/ops/pixel_shuffle.h>
#include <ATen/ops/pixel_unshuffle.h>
#include <ATen/ops/repeat.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/rot90.h>
#include <ATen/ops/select.h>
#include <ATen/ops/special_multigammaln.h>
#include <ATen/ops/sum_to_size.h>
#include <ATen/ops/swapaxes.h>
#include <ATen/ops/swapdims.h>
#include <ATen/ops/tile.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/tril_indices.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/triu_indices.h>
#include <ATen/ops/int_repr.h>
#include <ATen/ops/is_pinned.h>
#include <ATen/ops/is_pinned_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/resize.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/resize_as.h>
#include <ATen/ops/slice_inverse.h>
#include <ATen/ops/new_empty_strided.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/nll_loss.h>
#include <ATen/ops/nll_loss2d.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bicubic2d.h>
#include <ATen/ops/upsample_linear1d.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest3d.h>
#include <ATen/ops/set_data.h>
#include <ATen/ops/set_data_native.h>
#include <ATen/ops/upsample_nearest1d_backward.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <ATen/ops/upsample_nearest3d_backward.h>
#include <ATen/ops/upsample_linear1d_backward.h>
#include <ATen/ops/upsample_trilinear3d.h>
#include <ATen/ops/upsample_trilinear3d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bicubic2d_backward.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/quantize_per_channel.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/quantize_per_tensor_dynamic.h>
#include <ATen/ops/int_repr.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_permuted.h>
#include <ATen/ops/multilabel_margin_loss.h>
#include <ATen/ops/fake_quantize_per_tensor_affine.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_cachemask.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask.h>
#include <ATen/ops/q_scale.h>
#include <ATen/ops/q_zero_point.h>
#include <ATen/ops/q_per_channel_axis.h>
#include <ATen/ops/q_per_channel_scales.h>
#include <ATen/ops/q_per_channel_zero_points.h>
#include <ATen/ops/exponential.h>
#include <ATen/ops/poisson.h>
#include <ATen/ops/binomial.h>
#include <ATen/ops/cauchy.h>
#include <ATen/ops/geometric.h>
#include <ATen/ops/log_normal.h>
#include <ATen/ops/uniform.h>
#include <ATen/ops/uniform_native.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/hash_tensor.h>
#include <ATen/ops/unflatten.h>
#include <ATen/ops/unsqueeze_copy.h>
#include <ATen/ops/view.h>
#include <ATen/ops/view_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/view_copy.h>
#include <ATen/ops/quantile.h>
#include <ATen/ops/nanquantile.h>
#include <ATen/ops/cov.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/instance_norm.h>
#include <ATen/ops/corrcoef.h>
#include <ATen/ops/det.h>
#include <ATen/ops/fliplr.h>
#include <ATen/ops/flipud.h>
#include <ATen/ops/isreal.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_matrix_exp.h>
#include <ATen/ops/logdet.h>
#include <ATen/ops/logit.h>
#include <ATen/ops/matrix_exp.h>
#include <ATen/ops/polygamma.h>
#include <ATen/ops/special_airy_ai.h>
#include <ATen/ops/special_bessel_j0.h>
#include <ATen/ops/special_bessel_j1.h>
#include <ATen/ops/special_bessel_y0.h>
#include <ATen/ops/special_bessel_y1.h>
#include <ATen/ops/special_digamma.h>
#include <ATen/ops/special_entr.h>
#include <ATen/ops/special_erf.h>
#include <ATen/ops/special_erfc.h>
#include <ATen/ops/special_erfcx.h>
#include <ATen/ops/special_erfinv.h>
#include <ATen/ops/special_exp2.h>
#include <ATen/ops/special_expit.h>
#include <ATen/ops/special_expm1.h>
#include <ATen/ops/special_gammaln.h>
#include <ATen/ops/special_i0.h>
#include <ATen/ops/special_i0e.h>
#include <ATen/ops/special_i1.h>
#include <ATen/ops/special_i1e.h>
#include <ATen/ops/special_log1p.h>
#include <ATen/ops/special_log_ndtr.h>
#include <ATen/ops/special_log_softmax.h>
#include <ATen/ops/special_logit.h>
#include <ATen/ops/special_logsumexp.h>
#include <ATen/ops/special_modified_bessel_i0.h>
#include <ATen/ops/special_modified_bessel_i1.h>
#include <ATen/ops/special_modified_bessel_k0.h>
#include <ATen/ops/special_modified_bessel_k1.h>
#include <ATen/ops/special_ndtr.h>
#include <ATen/ops/special_ndtri.h>
#include <ATen/ops/special_polygamma.h>
#include <ATen/ops/special_psi.h>
#include <ATen/ops/special_round.h>
#include <ATen/ops/special_scaled_modified_bessel_k0.h>
#include <ATen/ops/special_scaled_modified_bessel_k1.h>
#include <ATen/ops/special_sinc.h>
#include <ATen/ops/special_softmax.h>
#include <ATen/ops/special_spherical_bessel_j0.h>

// Tensor ATen declarations needed by the remaining legacy Tensor contracts.
#include <ATen/ops/clone.h>
#include <ATen/ops/unsqueeze.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/div.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/log.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/add.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/embedding_bag.h>
#include <ATen/ops/embedding_renorm.h>
#include <ATen/ops/batch_norm.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/requires_grad.h>
#include <ATen/ops/repeat_interleave.h>

// Linalg convenience adapters.  These are package-owned C++ functions (not
// a C ABI): they make ATen's optional Scalar/ScalarType/ArrayRef overloads
// explicit at the generated CXX boundary while preserving the public Terlan
// contracts.
namespace at::native {
inline std::tuple<at::Tensor, at::Tensor> terlan_geqrf(const at::Tensor& self) { return at::geqrf(self); }
inline at::Tensor cov_default(const at::Tensor& self, int64_t correction) { return at::cov(self, correction, std::nullopt, std::nullopt); }

inline at::Tensor ctc_loss_intlist(const at::Tensor& log_probs, const at::Tensor& targets,
                                   at::IntArrayRef input_lengths, at::IntArrayRef target_lengths,
                                   int64_t blank, int64_t reduction, bool zero_infinity) {
  return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

inline at::Tensor ctc_loss_tensor(const at::Tensor& log_probs, const at::Tensor& targets,
                                  const at::Tensor& input_lengths, const at::Tensor& target_lengths,
                                  int64_t blank, int64_t reduction, bool zero_infinity) {
  return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

inline at::Tensor rand_default(at::IntArrayRef size) {
  return at::rand(size);
}

inline at::Tensor& rand_out_ordered(at::Tensor& out, at::IntArrayRef size) {
  return at::rand_out(out, size);
}

inline at::Tensor randn_default(at::IntArrayRef size) {
  return at::randn(size);
}

inline at::Tensor& randn_out_ordered(at::Tensor& out, at::IntArrayRef size) {
  return at::randn_out(out, size);
}

inline at::Tensor randint_default(int64_t high, at::IntArrayRef size) {
  return at::randint(high, size);
}

inline at::Tensor& randint_out_ordered(at::Tensor& out, int64_t high, at::IntArrayRef size) {
  return at::randint_out(out, high, size);
}

inline at::Tensor randint_low_default(int64_t low, int64_t high, at::IntArrayRef size) {
  return at::randint(low, high, size);
}

inline at::Tensor& randint_low_out_ordered(at::Tensor& out, int64_t low, int64_t high, at::IntArrayRef size) {
  return at::randint_out(out, low, high, size);
}

inline at::Tensor rand_like_default(const at::Tensor& self) {
  return at::rand_like(self);
}

inline at::Tensor& rand_like_out_ordered(at::Tensor& out, const at::Tensor& self) {
  return at::rand_like_out(out, self);
}

inline at::Tensor randn_like_default(const at::Tensor& self) {
  return at::randn_like(self);
}

inline at::Tensor& randn_like_out_ordered(at::Tensor& out, const at::Tensor& self) {
  return at::randn_like_out(out, self);
}

inline at::Tensor randint_like_default(const at::Tensor& self, int64_t high) {
  return at::randint_like(self, high);
}

inline at::Tensor randint_like_low_dtype_default(const at::Tensor& self, int64_t low,
                                                 int64_t high) {
  return at::randint_like(self, low, high);
}

inline at::Tensor randint_like_tensor_default(const at::Tensor& self,
                                               const at::Tensor& high) {
  return at::randint_like(self, high);
}

inline at::Tensor& randint_like_out_default(at::Tensor& out, const at::Tensor& self,
                                            int64_t high) {
  return at::randint_like_out(out, self, high);
}

inline at::Tensor& randint_like_tensor_out_default(at::Tensor& out, const at::Tensor& self,
                                                   const at::Tensor& high) {
  return at::randint_like_out(out, self, high);
}

inline at::Tensor& randint_like_low_dtype_out_default(at::Tensor& out,
                                                      const at::Tensor& self, int64_t low,
                                                      int64_t high) {
  return at::randint_like_out(out, self, low, high);
}

inline at::Tensor random_default(const at::Tensor& self) { return at::random(self); }
inline at::Tensor& random_out_ordered(at::Tensor& out, const at::Tensor& self) { return at::random_out(out, self); }
inline at::Tensor random_from(const at::Tensor& self, int64_t from, int64_t to) { return at::random(self, from, to); }
inline at::Tensor& random_from_out_ordered(at::Tensor& out, const at::Tensor& self, int64_t from, int64_t to) { return at::random_out(out, self, from, to); }
inline at::Tensor random_to(const at::Tensor& self, int64_t to) { return at::random(self, to); }
inline at::Tensor& random_to_out_ordered(at::Tensor& out, const at::Tensor& self, int64_t to) { return at::random_out(out, self, to); }
inline at::Tensor& random_in_place(at::Tensor& self) { return at::native::random_(self); }
inline at::Tensor& random_from_in_place(at::Tensor& self, int64_t from, int64_t to) { return at::native::random_(self, from, to); }
inline at::Tensor& random_to_in_place(at::Tensor& self, int64_t to) { return at::native::random_(self, to); }

inline at::Tensor scaled_dot_product_attention_default(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal) {
  return at::scaled_dot_product_attention(
      query, key, value, std::nullopt, dropout_p, is_causal, std::nullopt, false);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_default(
    const at::Tensor& self, bool some, bool compute_uv) {
  return at::svd(self, some, compute_uv);
}

inline std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> svd_out_default(
    at::Tensor& U, at::Tensor& S, at::Tensor& V,
    const at::Tensor& self, bool some, bool compute_uv) {
  return at::svd_outf(self, some, compute_uv, U, S, V);
}

inline at::Tensor empty_quantized_default(at::IntArrayRef size, const at::Tensor& qtensor) {
  return at::empty_quantized(size, qtensor);
}

inline at::Tensor& empty_quantized_out_default(
    at::Tensor& out, at::IntArrayRef size, const at::Tensor& qtensor) {
  return at::empty_quantized_out(out, size, qtensor);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive_default(
    const at::Tensor& self, bool return_inverse, bool return_counts,
    std::optional<int64_t> dim) {
  return at::unique_consecutive(self, return_inverse, return_counts, dim);
}

inline std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> unique_consecutive_out_default(
    at::Tensor& out0, at::Tensor& out1, at::Tensor& out2, const at::Tensor& self,
    bool return_inverse, bool return_counts, std::optional<int64_t> dim) {
  return at::unique_consecutive_outf(self, return_inverse, return_counts, dim, out0, out1, out2);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_default(
    const at::Tensor& self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  return at::unique_dim(self, dim, sorted, return_inverse, return_counts);
}

inline std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> unique_dim_out_default(
    at::Tensor& out0, at::Tensor& out1, at::Tensor& out2, const at::Tensor& self,
    int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  return at::unique_dim_outf(self, dim, sorted, return_inverse, return_counts, out0, out1, out2);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_consecutive_default(
    const at::Tensor& self, int64_t dim, bool return_inverse, bool return_counts) {
  return at::unique_dim_consecutive(self, dim, return_inverse, return_counts);
}

inline std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> unique_dim_consecutive_out_default(
    at::Tensor& out0, at::Tensor& out1, at::Tensor& out2, const at::Tensor& self,
    int64_t dim, bool return_inverse, bool return_counts) {
  return at::unique_dim_consecutive_outf(self, dim, return_inverse, return_counts, out0, out1, out2);
}

inline std::tuple<at::Tensor&, at::Tensor&> fake_quantize_per_tensor_affine_cachemask_out_default(
    at::Tensor& out0, at::Tensor& out1, const at::Tensor& self, double scale,
    int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  return at::fake_quantize_per_tensor_affine_cachemask_outf(
      self, scale, zero_point, quant_min, quant_max, out0, out1);
}

inline std::tuple<at::Tensor&, at::Tensor&> fake_quantize_per_channel_affine_cachemask_out_default(
    at::Tensor& out0, at::Tensor& out1, const at::Tensor& self,
    const at::Tensor& scale, const at::Tensor& zero_point, int64_t axis,
    int64_t quant_min, int64_t quant_max) {
  return at::fake_quantize_per_channel_affine_cachemask_outf(
      self, scale, zero_point, axis, quant_min, quant_max, out0, out1);
}

inline std::tuple<at::Tensor, at::Tensor> histogram_bin_ct_default(
    const at::Tensor& self, int64_t bins, bool density) {
  return at::histogram(self, bins, std::nullopt, std::nullopt, density);
}

inline std::tuple<at::Tensor&, at::Tensor&> histogram_bin_ct_out_default(
    at::Tensor& hist, at::Tensor& bin_edges, const at::Tensor& self,
    int64_t bins, bool density) {
  return at::histogram_outf(self, bins, std::nullopt, std::nullopt, density, hist, bin_edges);
}

inline std::tuple<at::Tensor, at::Tensor> histogram_bins_tensor_default(
    const at::Tensor& self, const at::Tensor& bins, bool density) {
  return at::histogram(self, bins, std::nullopt, density);
}

inline std::tuple<at::Tensor&, at::Tensor&> histogram_bins_tensor_out_default(
    at::Tensor& hist, at::Tensor& bin_edges, const at::Tensor& self,
    const at::Tensor& bins, bool density) {
  return at::histogram_outf(self, bins, std::nullopt, density, hist, bin_edges);
}

inline std::tuple<at::Tensor, at::Tensor> histogram_bin_ct_configured(
    const at::Tensor& self, int64_t bins, bool has_range,
    rust::Slice<const double> range, bool has_weight,
    const at::Tensor& weight, bool density) {
  const c10::ArrayRef<double> range_ref(range.data(), range.size());
  const auto selected_range = has_range
      ? std::optional<c10::ArrayRef<double>>(range_ref)
      : std::nullopt;
  const auto selected_weight = has_weight
      ? std::optional<at::Tensor>(weight)
      : std::nullopt;
  return at::histogram(self, bins, selected_range, selected_weight, density);
}

inline std::tuple<at::Tensor&, at::Tensor&> histogram_bin_ct_out_configured(
    at::Tensor& hist, at::Tensor& bin_edges, const at::Tensor& self,
    int64_t bins, bool has_range, rust::Slice<const double> range,
    bool has_weight, const at::Tensor& weight, bool density) {
  const c10::ArrayRef<double> range_ref(range.data(), range.size());
  const auto selected_range = has_range
      ? std::optional<c10::ArrayRef<double>>(range_ref)
      : std::nullopt;
  const auto selected_weight = has_weight
      ? std::optional<at::Tensor>(weight)
      : std::nullopt;
  return at::histogram_outf(self, bins, selected_range, selected_weight,
                            density, hist, bin_edges);
}

inline std::tuple<at::Tensor, at::Tensor> histogram_bins_tensor_configured(
    const at::Tensor& self, const at::Tensor& bins, bool has_weight,
    const at::Tensor& weight, bool density) {
  const auto selected_weight = has_weight
      ? std::optional<at::Tensor>(weight)
      : std::nullopt;
  return at::histogram(self, bins, selected_weight, density);
}

inline std::tuple<at::Tensor&, at::Tensor&>
histogram_bins_tensor_out_configured(
    at::Tensor& hist, at::Tensor& bin_edges, const at::Tensor& self,
    const at::Tensor& bins, bool has_weight, const at::Tensor& weight,
    bool density) {
  const auto selected_weight = has_weight
      ? std::optional<at::Tensor>(weight)
      : std::nullopt;
  return at::histogram_outf(self, bins, selected_weight, density, hist,
                            bin_edges);
}

inline at::Tensor& set_source_tensor_in_place(at::Tensor& self, const at::Tensor& source) {
  return at::_ops::set__source_Tensor::call(self, source);
}

inline at::Tensor& set_source_tensor_storage_offset_in_place(
    at::Tensor& self, const at::Tensor& source, int64_t storage_offset,
    at::IntArrayRef size, at::IntArrayRef stride) {
  return at::symint::set_<int64_t>(self, source, storage_offset, size, stride);
}

inline at::Tensor rrelu_default(const at::Tensor& self, double lower, double upper, bool training) {
  return at::rrelu(self, at::Scalar(lower), at::Scalar(upper), training);
}
inline at::Tensor& rrelu_in_place(at::Tensor& self, double lower, double upper, bool training) {
  return at::rrelu_(self, at::Scalar(lower), at::Scalar(upper), training);
}
inline at::Tensor& rrelu_with_noise_in_place(at::Tensor& self, at::Tensor& noise,
                                              double lower, double upper, bool training) {
  return at::rrelu_with_noise_(self, noise, at::Scalar(lower), at::Scalar(upper), training);
}

inline at::Tensor stft_default(const at::Tensor& self, int64_t n_fft) {
  return at::stft(self, n_fft, std::nullopt, std::nullopt, std::nullopt, false,
                  std::nullopt, std::optional<bool>(true), std::nullopt);
}

inline at::Tensor stft_center_default(const at::Tensor& self, int64_t n_fft,
                                      bool center, std::string_view pad_mode) {
  return at::stft(self, n_fft, std::nullopt, std::nullopt, std::nullopt, center,
                  pad_mode, false, std::nullopt, std::optional<bool>(true), std::nullopt);
}

inline at::Tensor istft_default(const at::Tensor& self, int64_t n_fft) {
  return at::istft(self, n_fft, std::nullopt, std::nullopt, std::nullopt, true,
                   false, std::nullopt, std::nullopt, false);
}

inline at::Tensor segment_reduce_lengths(const at::Tensor& data, std::string_view reduce,
                                         const at::Tensor& lengths) {
  return at::segment_reduce(data, reduce, lengths, std::nullopt, std::nullopt, 0, false,
                             std::nullopt);
}

inline at::Tensor& segment_reduce_lengths_out(at::Tensor& out, const at::Tensor& data,
                                              std::string_view reduce,
                                              const at::Tensor& lengths) {
  return at::segment_reduce_out(out, data, reduce, lengths, std::nullopt, std::nullopt, 0,
                                 false, std::nullopt);
}

inline at::Tensor hspmm_default(const at::Tensor& mat1, const at::Tensor& mat2) {
  return at::hspmm(mat1, mat2);
}
inline at::Tensor& hspmm_out_default(at::Tensor& out, const at::Tensor& mat1,
                                     const at::Tensor& mat2) {
  return at::hspmm_out(out, mat1, mat2);
}
inline at::Tensor smm_default(const at::Tensor& self, const at::Tensor& mat2) {
  return at::smm(self, mat2);
}
inline at::Tensor sspaddmm_default(const at::Tensor& self, const at::Tensor& mat1,
                                   const at::Tensor& mat2, double beta, double alpha) {
  return at::sspaddmm(self, mat1, mat2, at::Scalar(beta), at::Scalar(alpha));
}
inline at::Tensor& sspaddmm_out_default(at::Tensor& out, const at::Tensor& self,
                                        const at::Tensor& mat1, const at::Tensor& mat2,
                                        double beta, double alpha) {
  return at::sspaddmm_out(out, self, mat1, mat2, at::Scalar(beta), at::Scalar(alpha));
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> lu_unpack_default(
    const at::Tensor& lu_data, const at::Tensor& lu_pivots, bool unpack_data,
    bool unpack_pivots) {
  return at::lu_unpack(lu_data, lu_pivots, unpack_data, unpack_pivots);
}

inline std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> lu_unpack_out_default(
    at::Tensor& p, at::Tensor& l, at::Tensor& u, const at::Tensor& lu_data,
    const at::Tensor& lu_pivots, bool unpack_data, bool unpack_pivots) {
  return at::lu_unpack_out(p, l, u, lu_data, lu_pivots, unpack_data, unpack_pivots);
}

inline at::Tensor& linalg_svdvals_out_default(at::Tensor& out, const at::Tensor& self) { return at::linalg_svdvals_out(out, self, std::nullopt); }
inline at::Tensor& linalg_tensorsolve_out_default(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) { return at::linalg_tensorsolve_out(out, self, other, std::nullopt); }

inline at::Tensor linalg_cond_default(const at::Tensor& self) { return at::linalg_cond(self, std::nullopt); }
inline at::Tensor linalg_cond_int(const at::Tensor& self, int64_t p) { return at::linalg_cond(self, at::Scalar(p)); }
inline at::Tensor linalg_cond_float(const at::Tensor& self, double p) { return at::linalg_cond(self, at::Scalar(p)); }
inline at::Tensor linalg_cond_string(const at::Tensor& self, std::string_view p) { return at::linalg_cond(self, p); }
inline at::Tensor& linalg_cond_out_default(at::Tensor& out, const at::Tensor& self) { return at::linalg_cond_out(out, self, std::nullopt); }
inline at::Tensor& linalg_cond_out_int(at::Tensor& out, const at::Tensor& self, int64_t p) { return at::linalg_cond_out(out, self, at::Scalar(p)); }
inline at::Tensor& linalg_cond_out_float(at::Tensor& out, const at::Tensor& self, double p) { return at::linalg_cond_out(out, self, at::Scalar(p)); }
inline at::Tensor& linalg_cond_out_string(at::Tensor& out, const at::Tensor& self, std::string_view p) { return at::linalg_cond_out(out, self, p); }

inline at::Tensor linalg_matrix_norm_int(const at::Tensor& self, int64_t p) { return at::linalg_matrix_norm(self, at::Scalar(p), {-2, -1}, false, std::nullopt); }
inline at::Tensor linalg_matrix_norm_float(const at::Tensor& self, double p) { return at::linalg_matrix_norm(self, at::Scalar(p), {-2, -1}, false, std::nullopt); }
inline at::Tensor linalg_matrix_norm_int_configured(const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_matrix_norm(self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor linalg_matrix_norm_float_configured(const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_matrix_norm(self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_matrix_norm_out_int(at::Tensor& out, const at::Tensor& self, int64_t p) { return at::linalg_matrix_norm_out(out, self, at::Scalar(p), {-2, -1}, false, std::nullopt); }
inline at::Tensor& linalg_matrix_norm_out_float(at::Tensor& out, const at::Tensor& self, double p) { return at::linalg_matrix_norm_out(out, self, at::Scalar(p), {-2, -1}, false, std::nullopt); }
inline at::Tensor& linalg_matrix_norm_out_int_configured(at::Tensor& out, const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_matrix_norm_out(out, self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_matrix_norm_out_float_configured(at::Tensor& out, const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_matrix_norm_out(out, self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor linalg_matrix_norm_string(const at::Tensor& self, std::string_view p, c10::ArrayRef<int64_t> dim, bool keepdim) { return at::linalg_matrix_norm(self, p, dim, keepdim, std::nullopt); }
inline at::Tensor linalg_matrix_norm_string_dtype(const at::Tensor& self, std::string_view p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_matrix_norm(self, p, dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_matrix_norm_out_string(at::Tensor& out, const at::Tensor& self, std::string_view p, c10::ArrayRef<int64_t> dim, bool keepdim) { return at::linalg_matrix_norm_out(out, self, p, dim, keepdim, std::nullopt); }
inline at::Tensor& linalg_matrix_norm_out_string_dtype(at::Tensor& out, const at::Tensor& self, std::string_view p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_matrix_norm_out(out, self, p, dim, keepdim, static_cast<at::ScalarType>(dtype)); }

inline at::Tensor linalg_norm_default(const at::Tensor& self) { return at::linalg_norm(self, std::nullopt, std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_norm_int(const at::Tensor& self, int64_t p) { return at::linalg_norm(self, std::optional<at::Scalar>(at::Scalar(p)), std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_norm_float(const at::Tensor& self, double p) { return at::linalg_norm(self, std::optional<at::Scalar>(at::Scalar(p)), std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_norm_int_configured(const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_norm(self, std::optional<at::Scalar>(at::Scalar(p)), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor linalg_norm_float_configured(const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_norm(self, std::optional<at::Scalar>(at::Scalar(p)), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_norm_out_default(at::Tensor& out, const at::Tensor& self) { return at::linalg_norm_out(out, self, std::nullopt, std::nullopt, false, std::nullopt); }
inline at::Tensor& linalg_norm_out_int(at::Tensor& out, const at::Tensor& self, int64_t p) { return at::linalg_norm_out(out, self, std::optional<at::Scalar>(at::Scalar(p)), std::nullopt, false, std::nullopt); }
inline at::Tensor& linalg_norm_out_float(at::Tensor& out, const at::Tensor& self, double p) { return at::linalg_norm_out(out, self, std::optional<at::Scalar>(at::Scalar(p)), std::nullopt, false, std::nullopt); }
inline at::Tensor& linalg_norm_out_int_configured(at::Tensor& out, const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_norm_out(out, self, std::optional<at::Scalar>(at::Scalar(p)), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_norm_out_float_configured(at::Tensor& out, const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_norm_out(out, self, std::optional<at::Scalar>(at::Scalar(p)), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor linalg_norm_string(const at::Tensor& self, std::string_view p, bool keepdim) { return at::linalg_norm(self, p, std::nullopt, keepdim, std::nullopt); }
inline at::Tensor linalg_norm_string_dimensions(const at::Tensor& self, std::string_view p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_norm(self, p, dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_norm_out_string(at::Tensor& out, const at::Tensor& self, std::string_view p, bool keepdim) { return at::linalg_norm_out(out, self, p, std::nullopt, keepdim, std::nullopt); }
inline at::Tensor& linalg_norm_out_string_dimensions(at::Tensor& out, const at::Tensor& self, std::string_view p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_norm_out(out, self, p, dim, keepdim, static_cast<at::ScalarType>(dtype)); }

inline at::Tensor linalg_vector_norm_int(const at::Tensor& self, int64_t p) { return at::linalg_vector_norm(self, at::Scalar(p), std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_vector_norm_float(const at::Tensor& self, double p) { return at::linalg_vector_norm(self, at::Scalar(p), std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_vector_norm_int_configured(const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_vector_norm(self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor linalg_vector_norm_float_configured(const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_vector_norm(self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_vector_norm_out_int(at::Tensor& out, const at::Tensor& self, int64_t p) { return at::linalg_vector_norm_out(out, self, at::Scalar(p), std::nullopt, false, std::nullopt); }
inline at::Tensor& linalg_vector_norm_out_float(at::Tensor& out, const at::Tensor& self, double p) { return at::linalg_vector_norm_out(out, self, at::Scalar(p), std::nullopt, false, std::nullopt); }
inline at::Tensor& linalg_vector_norm_out_int_configured(at::Tensor& out, const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_vector_norm_out(out, self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor& linalg_vector_norm_out_float_configured(at::Tensor& out, const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg_vector_norm_out(out, self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }

inline at::Tensor linalg_powsum_int(const at::Tensor& self, int64_t p) { return at::linalg__powsum(self, at::Scalar(p), std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_powsum_float(const at::Tensor& self, double p) { return at::linalg__powsum(self, at::Scalar(p), std::nullopt, false, std::nullopt); }
inline at::Tensor linalg_powsum_int_configured(const at::Tensor& self, int64_t p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg__powsum(self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }
inline at::Tensor linalg_powsum_float_configured(const at::Tensor& self, double p, c10::ArrayRef<int64_t> dim, bool keepdim, int64_t dtype) { return at::linalg__powsum(self, at::Scalar(p), dim, keepdim, static_cast<at::ScalarType>(dtype)); }

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> linalg_lstsq_default(const at::Tensor& a, const at::Tensor& b) { return at::linalg_lstsq(a, b, std::nullopt, std::nullopt); }
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> linalg_lstsq_configured(const at::Tensor& a, const at::Tensor& b, double rcond, std::string_view driver) { return at::linalg_lstsq(a, b, std::optional<double>(rcond), std::optional<std::string_view>(driver)); }
inline at::Tensor& linalg_lstsq_out_default(at::Tensor& solution, at::Tensor& residuals, at::Tensor& rank, at::Tensor& singular_values, const at::Tensor& a, const at::Tensor& b) { return std::get<0>(at::linalg_lstsq_out(solution, residuals, rank, singular_values, a, b, std::nullopt, std::nullopt)); }
inline at::Tensor& linalg_lstsq_out_configured(at::Tensor& solution, at::Tensor& residuals, at::Tensor& rank, at::Tensor& singular_values, const at::Tensor& a, const at::Tensor& b, double rcond, std::string_view driver) { return std::get<0>(at::linalg_lstsq_out(solution, residuals, rank, singular_values, a, b, std::optional<double>(rcond), std::optional<std::string_view>(driver))); }

inline at::Tensor linalg_matrix_rank_default(const at::Tensor& self, bool hermitian) { return at::linalg_matrix_rank(self, std::optional<double>{}, std::optional<double>{}, hermitian); }
inline at::Tensor linalg_matrix_rank_tolerances(const at::Tensor& self, double atol, double rtol, bool hermitian) { return at::linalg_matrix_rank(self, std::optional<double>(atol), std::optional<double>(rtol), hermitian); }
inline at::Tensor linalg_matrix_rank_tensor_default(const at::Tensor& self, bool hermitian) { return at::linalg_matrix_rank(self, std::optional<at::Tensor>{}, std::optional<at::Tensor>{}, hermitian); }
inline at::Tensor linalg_matrix_rank_tensor_absolute(const at::Tensor& self, const at::Tensor& atol, bool hermitian) { return at::linalg_matrix_rank(self, std::optional<at::Tensor>(atol), std::nullopt, hermitian); }
inline at::Tensor linalg_matrix_rank_tensor_relative(const at::Tensor& self, const at::Tensor& rtol, bool hermitian) { return at::linalg_matrix_rank(self, std::nullopt, std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor linalg_matrix_rank_tensor_tolerances(const at::Tensor& self, const at::Tensor& atol, const at::Tensor& rtol, bool hermitian) { return at::linalg_matrix_rank(self, std::optional<at::Tensor>(atol), std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor linalg_matrix_rank_tolerance(const at::Tensor& self, const at::Tensor& tol, bool hermitian) { return at::linalg_matrix_rank(self, tol, hermitian); }
inline at::Tensor& linalg_matrix_rank_out_default(at::Tensor& out, const at::Tensor& self, bool hermitian) { return at::linalg_matrix_rank_out(out, self, std::optional<double>{}, std::optional<double>{}, hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tolerances(at::Tensor& out, const at::Tensor& self, double atol, double rtol, bool hermitian) { return at::linalg_matrix_rank_out(out, self, std::optional<double>(atol), std::optional<double>(rtol), hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tensor_default(at::Tensor& out, const at::Tensor& self, bool hermitian) { return at::linalg_matrix_rank_out(out, self, std::optional<at::Tensor>{}, std::optional<at::Tensor>{}, hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tensor_absolute(at::Tensor& out, const at::Tensor& self, const at::Tensor& atol, bool hermitian) { return at::linalg_matrix_rank_out(out, self, std::optional<at::Tensor>(atol), std::nullopt, hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tensor_relative(at::Tensor& out, const at::Tensor& self, const at::Tensor& rtol, bool hermitian) { return at::linalg_matrix_rank_out(out, self, std::nullopt, std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tensor_tolerances(at::Tensor& out, const at::Tensor& self, const at::Tensor& atol, const at::Tensor& rtol, bool hermitian) { return at::linalg_matrix_rank_out(out, self, std::optional<at::Tensor>(atol), std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tolerance(at::Tensor& out, const at::Tensor& self, double tol, bool hermitian) { return at::linalg_matrix_rank_out(out, self, tol, hermitian); }
inline at::Tensor& linalg_matrix_rank_out_tensor_tolerance(at::Tensor& out, const at::Tensor& self, const at::Tensor& tol, bool hermitian) { return at::linalg_matrix_rank_out(out, self, tol, hermitian); }

inline at::Tensor linalg_pinv_default(const at::Tensor& self, bool hermitian) { return at::linalg_pinv(self, std::optional<double>{}, std::optional<double>{}, hermitian); }
inline at::Tensor linalg_pinv_tolerances(const at::Tensor& self, double atol, double rtol, bool hermitian) { return at::linalg_pinv(self, std::optional<double>(atol), std::optional<double>(rtol), hermitian); }
inline at::Tensor linalg_pinv_tensor_default(const at::Tensor& self, bool hermitian) { return at::linalg_pinv(self, std::optional<at::Tensor>{}, std::optional<at::Tensor>{}, hermitian); }
inline at::Tensor linalg_pinv_tensor_absolute(const at::Tensor& self, const at::Tensor& atol, bool hermitian) { return at::linalg_pinv(self, std::optional<at::Tensor>(atol), std::nullopt, hermitian); }
inline at::Tensor linalg_pinv_tensor_relative(const at::Tensor& self, const at::Tensor& rtol, bool hermitian) { return at::linalg_pinv(self, std::nullopt, std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor linalg_pinv_tensor_tolerances(const at::Tensor& self, const at::Tensor& atol, const at::Tensor& rtol, bool hermitian) { return at::linalg_pinv(self, std::optional<at::Tensor>(atol), std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor linalg_pinv_rcond(const at::Tensor& self, const at::Tensor& rcond, bool hermitian) { return at::linalg_pinv(self, rcond, hermitian); }
inline at::Tensor& linalg_pinv_out_default(at::Tensor& out, const at::Tensor& self, bool hermitian) { return at::linalg_pinv_out(out, self, std::optional<double>{}, std::optional<double>{}, hermitian); }
inline at::Tensor& linalg_pinv_out_tolerances(at::Tensor& out, const at::Tensor& self, double atol, double rtol, bool hermitian) { return at::linalg_pinv_out(out, self, std::optional<double>(atol), std::optional<double>(rtol), hermitian); }
inline at::Tensor& linalg_pinv_out_tensor_default(at::Tensor& out, const at::Tensor& self, bool hermitian) { return at::linalg_pinv_out(out, self, std::optional<at::Tensor>{}, std::optional<at::Tensor>{}, hermitian); }
inline at::Tensor& linalg_pinv_out_tensor_absolute(at::Tensor& out, const at::Tensor& self, const at::Tensor& atol, bool hermitian) { return at::linalg_pinv_out(out, self, std::optional<at::Tensor>(atol), std::nullopt, hermitian); }
inline at::Tensor& linalg_pinv_out_tensor_relative(at::Tensor& out, const at::Tensor& self, const at::Tensor& rtol, bool hermitian) { return at::linalg_pinv_out(out, self, std::nullopt, std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor& linalg_pinv_out_tensor_tolerances(at::Tensor& out, const at::Tensor& self, const at::Tensor& atol, const at::Tensor& rtol, bool hermitian) { return at::linalg_pinv_out(out, self, std::optional<at::Tensor>(atol), std::optional<at::Tensor>(rtol), hermitian); }
inline at::Tensor& linalg_pinv_out_rcond(at::Tensor& out, const at::Tensor& self, double rcond, bool hermitian) { return at::linalg_pinv_out(out, self, rcond, hermitian); }
inline at::Tensor& linalg_pinv_out_tensor_rcond(at::Tensor& out, const at::Tensor& self, const at::Tensor& rcond, bool hermitian) { return at::linalg_pinv_out(out, self, rcond, hermitian); }

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> linalg_svd_default(const at::Tensor& self, bool full) { return at::linalg_svd(self, full, std::nullopt); }
inline at::Tensor& linalg_svd_out_default(at::Tensor& u, at::Tensor& s, at::Tensor& vh, const at::Tensor& self, bool full) { return std::get<0>(at::linalg_svd_out(u, s, vh, self, full, std::nullopt)); }
inline at::Tensor linalg_svdvals_default(const at::Tensor& self) { return at::linalg_svdvals(self, std::nullopt); }
inline at::Tensor linalg_vander_default(const at::Tensor& self) { return at::linalg_vander(self, std::nullopt); }
inline at::Tensor linalg_vander_columns(const at::Tensor& self, int64_t n) { return at::linalg_vander(self, std::optional<int64_t>(n)); }
inline at::Tensor linalg_tensorsolve_default(const at::Tensor& self, const at::Tensor& other) { return at::linalg_tensorsolve(self, other, std::nullopt); }
inline at::Tensor linalg_tensorsolve_dimensions(const at::Tensor& self, const at::Tensor& other, c10::ArrayRef<int64_t> dim) { return at::linalg_tensorsolve(self, other, dim); }

inline at::Tensor functional_local_response_norm(
    const at::Tensor& input, int64_t size, double alpha, double beta, double k) {
  TORCH_CHECK(
      input.dim() >= 3,
      "Expected 3D or higher dimensionality input (got ",
      input.dim(),
      " dimensions)");
  if (input.numel() == 0) {
    return input;
  }
  return torch::nn::functional::local_response_norm(
      input,
      torch::nn::functional::LocalResponseNormFuncOptions(size)
          .alpha(alpha)
          .beta(beta)
          .k(k));
}

inline at::Tensor functional_multilabel_soft_margin_loss_configured(
    const at::Tensor& input, const at::Tensor& target, bool has_weight,
    const at::Tensor& weight, bool has_size_average, bool size_average,
    bool has_reduce, bool reduce, std::string_view reduction) {
  auto options = torch::nn::functional::MultilabelSoftMarginLossFuncOptions();
  if (has_weight) {
    options.weight(weight);
  }

  auto selected_reduction = reduction;
  if (has_size_average || has_reduce) {
    const bool selected_size_average = has_size_average ? size_average : true;
    const bool selected_reduce = has_reduce ? reduce : true;
    if (!selected_reduce) {
      selected_reduction = "none";
    } else if (selected_size_average) {
      selected_reduction = "mean";
    } else {
      selected_reduction = "sum";
    }
  }

  if (selected_reduction == "none") {
    options.reduction(torch::kNone);
  } else if (selected_reduction == "mean") {
    options.reduction(torch::kMean);
  } else if (selected_reduction == "sum") {
    options.reduction(torch::kSum);
  } else {
    TORCH_CHECK(
        false,
        "",
        selected_reduction,
        " is not valid for multilabel_soft_margin_loss reduction");
  }

  return torch::nn::functional::multilabel_soft_margin_loss(input, target, options);
}

inline at::Tensor functional_lp_pool1d_configured(
    const at::Tensor& input, bool norm_type_is_int, int64_t norm_type_int,
    double norm_type_float, int64_t kernel_size, bool has_stride,
    int64_t stride, bool ceil_mode) {
  const double norm_type =
      norm_type_is_int ? static_cast<double>(norm_type_int) : norm_type_float;
  if (std::isinf(norm_type)) {
    auto options = torch::nn::functional::MaxPool1dFuncOptions(kernel_size)
                       .stride(has_stride ? stride : kernel_size)
                       .padding(0)
                       .dilation(1)
                       .ceil_mode(ceil_mode);
    auto magnitude = at::abs(input);
    if (norm_type > 0) {
      return torch::nn::functional::max_pool1d(magnitude, options);
    }
    return -torch::nn::functional::max_pool1d(-magnitude, options);
  }

  auto options =
      torch::nn::functional::LPPool1dFuncOptions(norm_type, kernel_size)
          .ceil_mode(ceil_mode);
  if (has_stride) {
    options.stride(stride);
  }
  return torch::nn::functional::lp_pool1d(input, options);
}

inline at::Tensor functional_max_unpool1d_configured(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t kernel_size,
    bool has_stride,
    int64_t stride,
    int64_t padding,
    bool has_output_size,
    c10::ArrayRef<int64_t> output_size) {
  auto options = torch::nn::functional::MaxUnpool1dFuncOptions(kernel_size);
  if (has_stride) {
    options.stride(stride);
  }
  options.padding(padding);
  if (has_output_size) {
    options.output_size(
        std::vector<int64_t>(output_size.begin(), output_size.end()));
  }
  return torch::nn::functional::max_unpool1d(input, indices, options);
}

inline at::Tensor functional_lp_pool2d_configured(
    const at::Tensor& input, bool norm_type_is_int, int64_t norm_type_int,
    double norm_type_float, int64_t kernel_size, bool has_stride,
    int64_t stride, bool ceil_mode) {
  const double norm_type =
      norm_type_is_int ? static_cast<double>(norm_type_int) : norm_type_float;
  if (std::isinf(norm_type)) {
    auto options = torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
                       .stride(has_stride ? stride : kernel_size)
                       .padding(0)
                       .dilation(1)
                       .ceil_mode(ceil_mode);
    auto magnitude = at::abs(input);
    if (norm_type > 0) {
      return torch::nn::functional::max_pool2d(magnitude, options);
    }
    return -torch::nn::functional::max_pool2d(-magnitude, options);
  }

  auto options =
      torch::nn::functional::LPPool2dFuncOptions(norm_type, kernel_size)
          .ceil_mode(ceil_mode);
  if (has_stride) {
    options.stride(stride);
  }
  return torch::nn::functional::lp_pool2d(input, options);
}

inline at::Tensor functional_lp_pool3d_configured(
    const at::Tensor& input, bool norm_type_is_int, int64_t norm_type_int,
    double norm_type_float, int64_t kernel_size, bool has_stride,
    int64_t stride, bool ceil_mode) {
  const double norm_type =
      norm_type_is_int ? static_cast<double>(norm_type_int) : norm_type_float;
  if (std::isinf(norm_type)) {
    auto options = torch::nn::functional::MaxPool3dFuncOptions(kernel_size)
                       .stride(has_stride ? stride : kernel_size)
                       .padding(0)
                       .dilation(1)
                       .ceil_mode(ceil_mode);
    auto magnitude = at::abs(input);
    if (norm_type > 0) {
      return torch::nn::functional::max_pool3d(magnitude, options);
    }
    return -torch::nn::functional::max_pool3d(-magnitude, options);
  }

  auto options =
      torch::nn::functional::LPPool3dFuncOptions(norm_type, kernel_size)
          .ceil_mode(ceil_mode);
  if (has_stride) {
    options.stride(stride);
  }
  return torch::nn::functional::lp_pool3d(input, options);
}

inline at::Tensor functional_gumbel_softmax_configured(
    const at::Tensor& logits, double tau, bool hard, double eps, int64_t dim) {
  (void)eps;
  auto options = torch::nn::functional::GumbelSoftmaxFuncOptions()
                     .tau(tau)
                     .hard(hard)
                     .dim(dim);
  return torch::nn::functional::gumbel_softmax(logits, options);
}

inline at::Tensor functional_gaussian_nll_loss_configured(
    const at::Tensor& input, const at::Tensor& target, bool var_is_tensor,
    const at::Tensor& var_tensor, double var_scalar, bool full, double eps,
    std::string_view reduction) {
  auto var = var_is_tensor ? var_tensor.clone() : at::full_like(input, var_scalar);
  if (var_is_tensor && var.dim() + 1 == input.dim()) {
    var = var.unsqueeze(-1);
  }
  var = at::clamp_min(var, eps);
  auto loss = 0.5 * (at::log(var) + (input - target).pow(2) / var);
  if (full) {
    loss = loss + 0.91893853320467274178;
  }
  if (reduction == "none") {
    return loss;
  }
  if (reduction == "mean") {
    return loss.mean();
  }
  if (reduction == "sum") {
    return loss.sum();
  }
  TORCH_CHECK(false, reduction, " is not valid");
}

inline at::Tensor functional_interpolate_configured(
    const at::Tensor& input, bool has_size, c10::ArrayRef<int64_t> size,
    bool has_scale, rust::Slice<const double> scale_factor, std::string_view mode,
    bool has_align_corners, bool align_corners, bool has_recompute,
    bool recompute_scale_factor, bool antialias) {
  auto options = torch::nn::functional::InterpolateFuncOptions();
  if (has_size) {
    options.size(std::vector<int64_t>(size.begin(), size.end()));
  }
  if (has_scale) {
    options.scale_factor(
        std::vector<double>(scale_factor.begin(), scale_factor.end()));
  }
  if (mode == "nearest") {
    options.mode(torch::kNearest);
  } else if (mode == "linear") {
    options.mode(torch::kLinear);
  } else if (mode == "bilinear") {
    options.mode(torch::kBilinear);
  } else if (mode == "bicubic") {
    options.mode(torch::kBicubic);
  } else if (mode == "trilinear") {
    options.mode(torch::kTrilinear);
  } else if (mode == "area") {
    options.mode(torch::kArea);
  } else if (mode == "nearest-exact") {
    options.mode(torch::kNearestExact);
  } else {
    TORCH_CHECK(false, mode, " is not a valid interpolation mode");
  }
  if (has_align_corners) {
    options.align_corners(align_corners);
  }
  if (has_recompute) {
    options.recompute_scale_factor(recompute_scale_factor);
  }
  options.antialias(antialias);
  return torch::nn::functional::interpolate(input, options);
}
}

#include "cuda_stream.hpp"
#include "optimizer.hpp"
#include "tensor_compat.hpp"
#include "tensor_ops.hpp"
#include "tensor_advanced.hpp"
#include "torchscript_compat.hpp"

namespace at::native {

inline std::unique_ptr<std::string> float32_matmul_precision_exact() {
  switch (at::globalContext().float32MatmulPrecision()) {
    case at::Float32MatmulPrecision::HIGHEST:
      return std::make_unique<std::string>("highest");
    case at::Float32MatmulPrecision::HIGH:
      return std::make_unique<std::string>("high");
    case at::Float32MatmulPrecision::MEDIUM:
      return std::make_unique<std::string>("medium");
  }
  throw std::runtime_error("unknown LibTorch float32 matmul precision");
}

inline void set_float32_matmul_precision_code_exact(std::int64_t code) {
  if (code == 0) {
    at::globalContext().setFloat32MatmulPrecision("highest");
  } else if (code == 1) {
    at::globalContext().setFloat32MatmulPrecision("high");
  } else if (code == 2) {
    at::globalContext().setFloat32MatmulPrecision("medium");
  } else {
    throw std::invalid_argument("float32 matmul precision code must be 0, 1, or 2");
  }
}

/** Package-owned float32 matmul precision scope guard with exact restoration. */
struct Float32MatmulPrecisionGuard {
  std::string previous_precision;

  explicit Float32MatmulPrecisionGuard(std::int64_t code)
      : previous_precision(*float32_matmul_precision_exact()) {
    set_float32_matmul_precision_code_exact(code);
  }

  ~Float32MatmulPrecisionGuard() {
    at::globalContext().setFloat32MatmulPrecision(previous_precision);
  }
};

inline std::unique_ptr<Float32MatmulPrecisionGuard>
float32_matmul_precision_enter_exact(std::int64_t code) {
  return std::make_unique<Float32MatmulPrecisionGuard>(code);
}

/** Package-owned deterministic-algorithm scope guard with exact state restoration. */
struct DeterministicAlgorithmsGuard {
  bool previous_enabled;
  bool previous_warn_only;

  DeterministicAlgorithmsGuard(bool enabled, bool warn_only)
      : previous_enabled(at::globalContext().deterministicAlgorithms()),
        previous_warn_only(at::globalContext().deterministicAlgorithmsWarnOnly()) {
    at::globalContext().setDeterministicAlgorithms(enabled, warn_only);
  }

  ~DeterministicAlgorithmsGuard() {
    at::globalContext().setDeterministicAlgorithms(
        previous_enabled, previous_warn_only);
  }
};

inline std::unique_ptr<DeterministicAlgorithmsGuard>
deterministic_algorithms_enter_exact(bool enabled, bool warn_only) {
  return std::make_unique<DeterministicAlgorithmsGuard>(enabled, warn_only);
}

inline bool grad_mode_enabled_exact() {
  return c10::GradMode::is_enabled();
}

inline void set_grad_mode_enabled_exact(bool enabled) {
  c10::GradMode::set_enabled(enabled);
}

inline bool anomaly_mode_enabled_exact() {
  return torch::autograd::AnomalyMode::is_enabled();
}

inline bool anomaly_check_nan_enabled_exact() {
  return torch::autograd::AnomalyMode::should_check_nan();
}

inline void set_anomaly_mode_exact(bool enabled, bool check_nan) {
  torch::autograd::AnomalyMode::set_enabled(enabled, check_nan);
}

/** Package-owned anomaly-detection scope guard with exact state restoration. */
struct AnomalyModeGuard {
  bool previous_enabled;
  bool previous_check_nan;

  AnomalyModeGuard(bool enabled, bool check_nan)
      : previous_enabled(torch::autograd::AnomalyMode::is_enabled()),
        previous_check_nan(torch::autograd::AnomalyMode::should_check_nan()) {
    torch::autograd::AnomalyMode::set_enabled(enabled, check_nan);
  }

  ~AnomalyModeGuard() {
    torch::autograd::AnomalyMode::set_enabled(
        previous_enabled, previous_check_nan);
  }
};

inline std::unique_ptr<AnomalyModeGuard> anomaly_mode_enter_exact(
    bool enabled, bool check_nan) {
  return std::make_unique<AnomalyModeGuard>(enabled, check_nan);
}

inline bool warn_always_enabled_exact() {
  return c10::WarningUtils::get_warnAlways();
}

inline void set_warn_always_exact(bool enabled) {
  c10::WarningUtils::set_warnAlways(enabled);
}

/** Package-owned warning-policy scope guard with exact state restoration. */
struct WarningModeGuard {
  bool previous_enabled;

  explicit WarningModeGuard(bool enabled)
      : previous_enabled(c10::WarningUtils::get_warnAlways()) {
    c10::WarningUtils::set_warnAlways(enabled);
  }

  ~WarningModeGuard() {
    c10::WarningUtils::set_warnAlways(previous_enabled);
  }
};

inline std::unique_ptr<WarningModeGuard> warning_mode_enter_exact(
    bool enabled) {
  return std::make_unique<WarningModeGuard>(enabled);
}

inline std::int64_t get_num_threads_exact() {
  return static_cast<std::int64_t>(at::get_num_threads());
}

inline void set_num_threads_exact(std::int64_t count) {
  if (count <= 0 || count > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("CPU thread count must be positive");
  }
  at::set_num_threads(static_cast<int>(count));
}

inline void init_num_threads_exact() {
  at::init_num_threads();
}

/** Package-owned exact floating-point denormal handling toggle. */
inline bool set_flush_denormal_exact(bool enabled) {
#if defined(__SSE__)
  unsigned int control = _mm_getcsr();
  constexpr unsigned int flush_zero = 1u << 15;
  constexpr unsigned int denormals_zero = 1u << 6;
  if (enabled) {
    control |= flush_zero | denormals_zero;
  } else {
    control &= ~(flush_zero | denormals_zero);
  }
  _mm_setcsr(control);
  return true;
#else
  (void)enabled;
  return false;
#endif
}

inline std::int64_t get_num_interop_threads_exact() {
  return static_cast<std::int64_t>(at::get_num_interop_threads());
}

inline void set_num_interop_threads_exact(std::int64_t count) {
  if (count <= 0 || count > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("CPU interop thread count must be positive");
  }
  at::set_num_interop_threads(static_cast<int>(count));
}

inline bool autocast_cpu_enabled_exact() {
  return at::autocast::is_autocast_enabled(at::kCPU);
}

inline void set_autocast_cpu_enabled_exact(bool enabled) {
  at::autocast::set_autocast_enabled(at::kCPU, enabled);
}

inline std::int64_t autocast_cpu_dtype_exact() {
  return static_cast<std::int64_t>(at::autocast::get_autocast_dtype(at::kCPU));
}

inline void set_autocast_cpu_dtype_exact(std::int64_t dtype) {
  at::autocast::set_autocast_dtype(at::kCPU, static_cast<at::ScalarType>(dtype));
}

inline bool autocast_cache_enabled_exact() {
  return at::autocast::is_autocast_cache_enabled();
}

inline void set_autocast_cache_enabled_exact(bool enabled) {
  at::autocast::set_autocast_cache_enabled(enabled);
}

inline std::int64_t default_dtype_exact() {
  return static_cast<std::int64_t>(c10::get_default_dtype_as_scalartype());
}

inline void set_default_dtype_exact(std::int64_t dtype) {
  c10::set_default_dtype(c10::scalarTypeToTypeMeta(
      static_cast<c10::ScalarType>(dtype)));
}

/** Package-owned default-dtype scope guard with exact restoration. */
struct DefaultDtypeGuard {
  std::int64_t previous_dtype;

  explicit DefaultDtypeGuard(std::int64_t dtype)
      : previous_dtype(static_cast<std::int64_t>(
            c10::get_default_dtype_as_scalartype())) {
    set_default_dtype_exact(dtype);
  }

  ~DefaultDtypeGuard() { set_default_dtype_exact(previous_dtype); }
};

inline std::unique_ptr<DefaultDtypeGuard>
default_dtype_enter_exact(std::int64_t dtype) {
  return std::make_unique<DefaultDtypeGuard>(dtype);
}

inline bool inference_mode_enabled_exact() {
  return c10::InferenceMode::is_enabled();
}

/** Package-owned scope guard for the LibTorch inference-mode state. */
struct InferenceModeGuard {
  explicit InferenceModeGuard(bool enabled) : guard_(enabled) {}

  c10::InferenceMode guard_;
};

inline std::unique_ptr<InferenceModeGuard> inference_mode_enter_exact(
    bool enabled) {
  return std::make_unique<InferenceModeGuard>(enabled);
}

/** Package-owned CPU autocast scope guard with exact state restoration. */
struct AutocastCpuGuard {
  bool previous_enabled;
  at::ScalarType previous_dtype;
  bool previous_cache;

  AutocastCpuGuard(bool enabled, std::int64_t dtype, bool cache_enabled)
      : previous_enabled(at::autocast::is_autocast_enabled(at::kCPU)),
        previous_dtype(at::autocast::get_autocast_dtype(at::kCPU)),
        previous_cache(at::autocast::is_autocast_cache_enabled()) {
    at::autocast::set_autocast_dtype(
        at::kCPU, static_cast<at::ScalarType>(dtype));
    at::autocast::set_autocast_cache_enabled(cache_enabled);
    at::autocast::set_autocast_enabled(at::kCPU, enabled);
  }

  ~AutocastCpuGuard() {
    at::autocast::set_autocast_enabled(at::kCPU, previous_enabled);
    at::autocast::set_autocast_dtype(at::kCPU, previous_dtype);
    at::autocast::set_autocast_cache_enabled(previous_cache);
  }
};

inline std::unique_ptr<AutocastCpuGuard> autocast_cpu_enter_exact(
    bool enabled, std::int64_t dtype, bool cache_enabled) {
  return std::make_unique<AutocastCpuGuard>(enabled, dtype, cache_enabled);
}

inline bool inference_mode_transition_probe_exact() {
  c10::InferenceMode guard(true);
  return c10::InferenceMode::is_enabled();
}

inline at::Tensor chalf_exact(const at::Tensor& input) {
  return input.to(at::kComplexHalf);
}

inline std::vector<at::Tensor> unsafe_chunk_exact(
    const at::Tensor& input,
    std::int64_t chunks,
    std::int64_t dimension) {
  return input.unsafe_chunk(chunks, dimension);
}

inline at::Tensor scalar_tensor_exact(
    bool scalar_is_integer,
    std::int64_t scalar_integer,
    double scalar_floating) {
  const at::Scalar scalar = scalar_is_integer
      ? at::Scalar(scalar_integer)
      : at::Scalar(scalar_floating);
  return at::scalar_tensor(scalar);
}

} // namespace at::native
