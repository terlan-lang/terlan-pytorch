#include "optimizer.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <torch/torch.h>

namespace at::native {
namespace {

bool valid_adam(double lr, double beta1, double beta2, double epsilon,
               double decay) {
  return std::isfinite(lr) && lr > 0.0 && std::isfinite(beta1) &&
         beta1 >= 0.0 && beta1 < 1.0 && std::isfinite(beta2) && beta2 >= 0.0 &&
         beta2 < 1.0 && std::isfinite(epsilon) && epsilon > 0.0 &&
         std::isfinite(decay) && decay >= 0.0;
}

bool valid_sgd(double lr, double momentum, double dampening, double decay,
               bool nesterov) {
  return std::isfinite(lr) && lr > 0.0 && std::isfinite(momentum) &&
         momentum >= 0.0 && std::isfinite(dampening) && dampening >= 0.0 &&
         std::isfinite(decay) && decay >= 0.0 &&
         (!nesterov || (momentum > 0.0 && dampening == 0.0));
}

}  // namespace

Adam::Adam(double lr, double b1, double b2, double eps, double decay)
    : kind(Kind::Adam),
      learning_rate(lr),
      beta1(b1),
      beta2(b2),
      epsilon(eps),
      weight_decay(decay),
      momentum(0.0),
      dampening(0.0),
      nesterov(false) {
  if (!valid_adam(lr, b1, b2, eps, decay))
    throw std::invalid_argument("invalid Adam hyperparameters");
}

Adam::Adam(double lr, double momentum_value, double dampening_value,
           double decay, bool nesterov_value)
    : kind(Kind::Sgd),
      learning_rate(lr),
      beta1(0.0),
      beta2(0.0),
      epsilon(1.0),
      weight_decay(decay),
      momentum(momentum_value),
      dampening(dampening_value),
      nesterov(nesterov_value) {
  if (!valid_sgd(lr, momentum_value, dampening_value, decay, nesterov_value))
    throw std::invalid_argument("invalid SGD hyperparameters");
}

std::unique_ptr<Adam> adam_create(double lr, double b1, double b2, double eps,
                                  double decay) {
  return std::make_unique<Adam>(lr, b1, b2, eps, decay);
}

std::unique_ptr<Adam> sgd_create(double lr, double momentum, double dampening,
                                 double decay, bool nesterov) {
  return std::make_unique<Adam>(lr, momentum, dampening, decay, nesterov);
}

void adam_add_parameter(Adam& optimizer, const at::Tensor& parameter) {
  if (optimizer.cancelled) throw std::runtime_error("optimizer cancelled");
  if (!parameter.defined() || !parameter.is_floating_point() ||
      !parameter.is_leaf() || !parameter.requires_grad())
    throw std::invalid_argument("parameter must be a floating leaf requiring grad");
  for (const auto& existing : optimizer.parameters)
    if (existing.is_same(parameter))
      throw std::invalid_argument("duplicate optimizer parameter");
  optimizer.parameters.push_back(parameter);
  optimizer.first_moments.push_back(at::zeros_like(parameter));
  optimizer.second_moments.push_back(at::zeros_like(parameter));
}

void adam_zero_grad(Adam& optimizer) {
  if (optimizer.cancelled) throw std::runtime_error("optimizer cancelled");
  torch::NoGradGuard no_grad;
  for (auto& parameter : optimizer.parameters)
    if (parameter.grad().defined()) parameter.mutable_grad().zero_();
}

void adam_step(Adam& optimizer) {
  if (optimizer.cancelled) throw std::runtime_error("optimizer cancelled");
  if (optimizer.parameters.empty()) throw std::runtime_error("optimizer has no parameters");
  for (const auto& parameter : optimizer.parameters)
    if (!parameter.grad().defined()) throw std::runtime_error("parameter gradient unavailable");
  torch::NoGradGuard no_grad;
  const std::int64_t next_step = optimizer.step + 1;
  if (optimizer.kind == Adam::Kind::Sgd) {
    for (std::size_t i = 0; i < optimizer.parameters.size(); ++i) {
      auto gradient = optimizer.parameters[i].grad();
      if (optimizer.weight_decay != 0.0)
        gradient = gradient.add(optimizer.parameters[i], optimizer.weight_decay);
      if (!at::isfinite(gradient).all().item<bool>())
        throw std::runtime_error("non-finite SGD gradient");
      if (optimizer.momentum != 0.0) {
        auto& buffer = optimizer.first_moments[i];
        if (optimizer.step == 0)
          buffer.copy_(gradient);
        else
          buffer.mul_(optimizer.momentum)
              .add_(gradient, 1.0 - optimizer.dampening);
        gradient = optimizer.nesterov
                       ? gradient.add(buffer, optimizer.momentum)
                       : buffer;
      }
      optimizer.parameters[i].add_(gradient, -optimizer.learning_rate);
    }
    optimizer.step = next_step;
    return;
  }
  const double first_correction = 1.0 - std::pow(optimizer.beta1, next_step);
  const double second_correction = 1.0 - std::pow(optimizer.beta2, next_step);
  for (std::size_t i = 0; i < optimizer.parameters.size(); ++i) {
    auto gradient = optimizer.parameters[i].grad();
    if (optimizer.weight_decay != 0.0)
      gradient = gradient.add(optimizer.parameters[i], optimizer.weight_decay);
    if (!at::isfinite(gradient).all().item<bool>())
      throw std::runtime_error("non-finite Adam gradient");
    auto& first = optimizer.first_moments[i];
    auto& second = optimizer.second_moments[i];
    first.mul_(optimizer.beta1).add_(gradient, 1.0 - optimizer.beta1);
    second.mul_(optimizer.beta2)
        .addcmul_(gradient, gradient, 1.0 - optimizer.beta2);
    const auto denominator = second.sqrt().div(std::sqrt(second_correction)).add(optimizer.epsilon);
    optimizer.parameters[i].addcdiv_(first, denominator,
                                     -optimizer.learning_rate / first_correction);
  }
  optimizer.step = next_step;
}

std::int64_t adam_parameter_count(const Adam& optimizer) {
  return static_cast<std::int64_t>(optimizer.parameters.size());
}

std::int64_t adam_step_count(const Adam& optimizer) { return optimizer.step; }

std::unique_ptr<std::vector<std::uint8_t>> adam_checkpoint(
    const Adam& optimizer) {
  if (optimizer.cancelled) throw std::runtime_error("optimizer cancelled");
  torch::serialize::OutputArchive archive;
  archive.write("metadata",
                torch::tensor({std::int64_t{2}, static_cast<std::int64_t>(optimizer.kind),
                               static_cast<std::int64_t>(optimizer.parameters.size()), optimizer.step},
                              torch::TensorOptions().dtype(torch::kInt64)), true);
  archive.write("configuration",
                torch::tensor({optimizer.learning_rate, optimizer.beta1, optimizer.beta2,
                               optimizer.epsilon, optimizer.weight_decay, optimizer.momentum,
                               optimizer.dampening, optimizer.nesterov ? 1.0 : 0.0},
                              torch::TensorOptions().dtype(torch::kFloat64)), true);
  for (std::size_t i = 0; i < optimizer.parameters.size(); ++i) {
    const auto suffix = std::to_string(i);
    archive.write("parameter_" + suffix, optimizer.parameters[i].detach().cpu().clone(), true);
    archive.write("first_" + suffix, optimizer.first_moments[i].detach().cpu().clone(), true);
    archive.write("second_" + suffix, optimizer.second_moments[i].detach().cpu().clone(), true);
  }
  std::ostringstream stream(std::ios::out | std::ios::binary);
  archive.save_to(stream);
  const std::string bytes = stream.str();
  return std::make_unique<std::vector<std::uint8_t>>(bytes.begin(), bytes.end());
}

void adam_restore(Adam& optimizer, rust::Slice<const std::uint8_t> checkpoint) {
  if (optimizer.cancelled) throw std::runtime_error("optimizer cancelled");
  if (checkpoint.empty()) throw std::invalid_argument("empty optimizer checkpoint");
  torch::serialize::InputArchive archive;
  archive.load_from(reinterpret_cast<const char*>(checkpoint.data()), checkpoint.size());
  at::Tensor metadata, configuration;
  archive.read("metadata", metadata, true);
  archive.read("configuration", configuration, true);
  if (metadata.numel() != 4 || configuration.numel() != 8 || metadata[0].item<std::int64_t>() != 2 ||
      metadata[1].item<std::int64_t>() != static_cast<std::int64_t>(optimizer.kind) ||
      metadata[2].item<std::int64_t>() != static_cast<std::int64_t>(optimizer.parameters.size()))
    throw std::runtime_error("optimizer checkpoint mismatch");
  std::vector<at::Tensor> first, second;
  first.reserve(optimizer.parameters.size()); second.reserve(optimizer.parameters.size());
  for (std::size_t i = 0; i < optimizer.parameters.size(); ++i) {
    const auto suffix = std::to_string(i);
    at::Tensor parameter, first_value, second_value;
    archive.read("parameter_" + suffix, parameter, true);
    archive.read("first_" + suffix, first_value, true);
    archive.read("second_" + suffix, second_value, true);
    const auto& target = optimizer.parameters[i];
    if (parameter.sizes() != target.sizes() || first_value.sizes() != target.sizes() ||
        second_value.sizes() != target.sizes() || parameter.scalar_type() != target.scalar_type() ||
        first_value.scalar_type() != target.scalar_type() || second_value.scalar_type() != target.scalar_type())
      throw std::runtime_error("optimizer checkpoint shape mismatch");
    torch::NoGradGuard no_grad;
    target.copy_(parameter.to(target.device()));
    first.push_back(first_value.to(target.device()));
    second.push_back(second_value.to(target.device()));
  }
  const double lr = configuration[0].item<double>(), b1 = configuration[1].item<double>(),
               b2 = configuration[2].item<double>(), eps = configuration[3].item<double>(),
               decay = configuration[4].item<double>(), momentum_value = configuration[5].item<double>(),
               dampening_value = configuration[6].item<double>();
  const double encoded_nesterov = configuration[7].item<double>();
  const bool nesterov_value = encoded_nesterov == 1.0;
  if (optimizer.kind == Adam::Kind::Adam ? !valid_adam(lr, b1, b2, eps, decay)
                                         : ((encoded_nesterov != 0.0 && encoded_nesterov != 1.0) ||
                                            !valid_sgd(lr, momentum_value, dampening_value, decay, nesterov_value)))
    throw std::runtime_error("optimizer checkpoint configuration mismatch");
  optimizer.first_moments = std::move(first); optimizer.second_moments = std::move(second);
  optimizer.learning_rate = lr; optimizer.beta1 = b1; optimizer.beta2 = b2; optimizer.epsilon = eps;
  optimizer.weight_decay = decay; optimizer.momentum = momentum_value; optimizer.dampening = dampening_value;
  optimizer.nesterov = nesterov_value; optimizer.step = metadata[3].item<std::int64_t>();
}

void adam_cancel(Adam& optimizer) { optimizer.cancelled = true; }

}  // namespace at::native
