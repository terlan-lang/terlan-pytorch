#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <ATen/ATen.h>

#if __has_include("rust/cxx.h")
#include "rust/cxx.h"
#else
namespace rust {
inline namespace cxxbridge1 {
template <typename T> class Slice;
}  // namespace cxxbridge1
}  // namespace rust
#endif

namespace at::native {

/** Package-owned, thread-confined Adam/SGD optimizer state. */
class Adam {
 public:
  enum class Kind : std::int64_t { Adam = 0, Sgd = 1 };

  Adam(double learning_rate, double beta1, double beta2, double epsilon,
       double weight_decay);
  Adam(double learning_rate, double momentum, double dampening,
       double weight_decay, bool nesterov);

  Kind kind;
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  double weight_decay;
  double momentum;
  double dampening;
  bool nesterov;
  std::int64_t step = 0;
  bool cancelled = false;
  std::vector<at::Tensor> parameters;
  std::vector<at::Tensor> first_moments;
  std::vector<at::Tensor> second_moments;
};

std::unique_ptr<Adam> adam_create(double learning_rate, double beta1,
                                  double beta2, double epsilon,
                                  double weight_decay);
void adam_add_parameter(Adam& optimizer, const at::Tensor& parameter);
std::unique_ptr<Adam> sgd_create(double learning_rate, double momentum,
                                 double dampening, double weight_decay,
                                 bool nesterov);
void adam_zero_grad(Adam& optimizer);
void adam_step(Adam& optimizer);
std::int64_t adam_parameter_count(const Adam& optimizer);
std::int64_t adam_step_count(const Adam& optimizer);
std::unique_ptr<std::vector<std::uint8_t>> adam_checkpoint(
    const Adam& optimizer);
void adam_restore(Adam& optimizer, rust::Slice<const std::uint8_t> checkpoint);
void adam_cancel(Adam& optimizer);

}  // namespace at::native
