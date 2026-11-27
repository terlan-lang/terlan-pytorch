#include "libtorch_bindings.hpp"

#include <torch/nn/functional/loss.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/functional/normalization.h>

namespace at::native {

at::Tensor functional_local_response_norm(
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

at::Tensor functional_multilabel_soft_margin_loss_configured(
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

at::Tensor functional_lp_pool1d_configured(
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

at::Tensor functional_max_unpool1d_configured(
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

at::Tensor functional_lp_pool2d_configured(
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

at::Tensor functional_lp_pool3d_configured(
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

at::Tensor functional_gumbel_softmax_configured(
    const at::Tensor& logits, double tau, bool hard, double eps, int64_t dim) {
  (void)eps;
  auto options = torch::nn::functional::GumbelSoftmaxFuncOptions()
                     .tau(tau)
                     .hard(hard)
                     .dim(dim);
  return torch::nn::functional::gumbel_softmax(logits, options);
}

at::Tensor functional_gaussian_nll_loss_configured(
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

at::Tensor functional_interpolate_configured(
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

} // namespace at::native
