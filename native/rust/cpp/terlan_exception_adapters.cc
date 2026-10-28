#include "include/terlan_exception_adapters.hpp"

#include <stdexcept>
#include <utility>

namespace at::native {

TerlanExceptionEnvelope::TerlanExceptionEnvelope(bool ok, std::int64_t value, double float_value, bool bool_value, std::string code, std::string message) noexcept
    : ok_(ok), value_(value), float_value_(float_value), bool_value_(bool_value), code_(std::move(code)), message_(std::move(message)) {}

bool TerlanExceptionEnvelope::is_ok() const noexcept { return ok_; }
std::int64_t TerlanExceptionEnvelope::value() const noexcept { return value_; }
double TerlanExceptionEnvelope::float_value() const noexcept { return float_value_; }
bool TerlanExceptionEnvelope::bool_value() const noexcept { return bool_value_; }
const std::string& TerlanExceptionEnvelope::code() const noexcept { return code_; }
const std::string& TerlanExceptionEnvelope::message() const noexcept { return message_; }

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_allclose_native(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) noexcept {
  try {
    try {
      const auto result = at::allclose(self, other, rtol, atol, equal_nan);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_equal_native(const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      const auto result = at::equal(self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_same_size_native(const Tensor & self, const Tensor & other) noexcept {
  try {
    try {
      const auto result = at::is_same_size(self, other);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_complex_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_complex();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_conj_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_conj();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_distributed_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_distributed();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_floating_point_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_floating_point();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_inference_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_inference();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_leaf_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_leaf();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch leaf-state inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_neg_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_neg();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_nonzero_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_nonzero();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_signed_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_signed();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_retains_grad_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.retains_grad();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_is_set_to_native(const at::Tensor& value, const Tensor & tensor) noexcept {
  try {
    try {
      const auto result = value.is_set_to(tensor);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_get_device_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.get_device();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_output_nr_native(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.output_nr();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_size_dimension_native(const at::Tensor& value, int64_t dim) noexcept {
  try {
    try {
      const auto result = value.size(dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_stride_dimension_native(const at::Tensor& value, int64_t dim) noexcept {
  try {
    try {
      const auto result = value.stride(dim);
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_can_cast_native(std::int64_t from_, std::int64_t to) noexcept {
  try {
    try {
      if (from_ != static_cast<std::int64_t>(c10::ScalarType::Byte) && from_ != static_cast<std::int64_t>(c10::ScalarType::Char) && from_ != static_cast<std::int64_t>(c10::ScalarType::Short) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int) && from_ != static_cast<std::int64_t>(c10::ScalarType::Long) && from_ != static_cast<std::int64_t>(c10::ScalarType::Half) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float) && from_ != static_cast<std::int64_t>(c10::ScalarType::Double) && from_ != static_cast<std::int64_t>(c10::ScalarType::ComplexHalf) && from_ != static_cast<std::int64_t>(c10::ScalarType::ComplexFloat) && from_ != static_cast<std::int64_t>(c10::ScalarType::ComplexDouble) && from_ != static_cast<std::int64_t>(c10::ScalarType::Bool) && from_ != static_cast<std::int64_t>(c10::ScalarType::QInt8) && from_ != static_cast<std::int64_t>(c10::ScalarType::QUInt8) && from_ != static_cast<std::int64_t>(c10::ScalarType::QInt32) && from_ != static_cast<std::int64_t>(c10::ScalarType::BFloat16) && from_ != static_cast<std::int64_t>(c10::ScalarType::QUInt4x2) && from_ != static_cast<std::int64_t>(c10::ScalarType::QUInt2x4) && from_ != static_cast<std::int64_t>(c10::ScalarType::Bits1x8) && from_ != static_cast<std::int64_t>(c10::ScalarType::Bits2x4) && from_ != static_cast<std::int64_t>(c10::ScalarType::Bits4x2) && from_ != static_cast<std::int64_t>(c10::ScalarType::Bits8) && from_ != static_cast<std::int64_t>(c10::ScalarType::Bits16) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fn) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2fnuz) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fnuz) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt16) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt32) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt64) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt1) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt2) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt3) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt4) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt5) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt6) && from_ != static_cast<std::int64_t>(c10::ScalarType::UInt7) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int1) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int2) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int3) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int4) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int5) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int6) && from_ != static_cast<std::int64_t>(c10::ScalarType::Int7) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float8_e8m0fnu) && from_ != static_cast<std::int64_t>(c10::ScalarType::Float4_e2m1fn_x2)) {
        throw std::invalid_argument("from_ is outside the reviewed enum domain");
      }
      if (to != static_cast<std::int64_t>(c10::ScalarType::Byte) && to != static_cast<std::int64_t>(c10::ScalarType::Char) && to != static_cast<std::int64_t>(c10::ScalarType::Short) && to != static_cast<std::int64_t>(c10::ScalarType::Int) && to != static_cast<std::int64_t>(c10::ScalarType::Long) && to != static_cast<std::int64_t>(c10::ScalarType::Half) && to != static_cast<std::int64_t>(c10::ScalarType::Float) && to != static_cast<std::int64_t>(c10::ScalarType::Double) && to != static_cast<std::int64_t>(c10::ScalarType::ComplexHalf) && to != static_cast<std::int64_t>(c10::ScalarType::ComplexFloat) && to != static_cast<std::int64_t>(c10::ScalarType::ComplexDouble) && to != static_cast<std::int64_t>(c10::ScalarType::Bool) && to != static_cast<std::int64_t>(c10::ScalarType::QInt8) && to != static_cast<std::int64_t>(c10::ScalarType::QUInt8) && to != static_cast<std::int64_t>(c10::ScalarType::QInt32) && to != static_cast<std::int64_t>(c10::ScalarType::BFloat16) && to != static_cast<std::int64_t>(c10::ScalarType::QUInt4x2) && to != static_cast<std::int64_t>(c10::ScalarType::QUInt2x4) && to != static_cast<std::int64_t>(c10::ScalarType::Bits1x8) && to != static_cast<std::int64_t>(c10::ScalarType::Bits2x4) && to != static_cast<std::int64_t>(c10::ScalarType::Bits4x2) && to != static_cast<std::int64_t>(c10::ScalarType::Bits8) && to != static_cast<std::int64_t>(c10::ScalarType::Bits16) && to != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2) && to != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fn) && to != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2fnuz) && to != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fnuz) && to != static_cast<std::int64_t>(c10::ScalarType::UInt16) && to != static_cast<std::int64_t>(c10::ScalarType::UInt32) && to != static_cast<std::int64_t>(c10::ScalarType::UInt64) && to != static_cast<std::int64_t>(c10::ScalarType::UInt1) && to != static_cast<std::int64_t>(c10::ScalarType::UInt2) && to != static_cast<std::int64_t>(c10::ScalarType::UInt3) && to != static_cast<std::int64_t>(c10::ScalarType::UInt4) && to != static_cast<std::int64_t>(c10::ScalarType::UInt5) && to != static_cast<std::int64_t>(c10::ScalarType::UInt6) && to != static_cast<std::int64_t>(c10::ScalarType::UInt7) && to != static_cast<std::int64_t>(c10::ScalarType::Int1) && to != static_cast<std::int64_t>(c10::ScalarType::Int2) && to != static_cast<std::int64_t>(c10::ScalarType::Int3) && to != static_cast<std::int64_t>(c10::ScalarType::Int4) && to != static_cast<std::int64_t>(c10::ScalarType::Int5) && to != static_cast<std::int64_t>(c10::ScalarType::Int6) && to != static_cast<std::int64_t>(c10::ScalarType::Int7) && to != static_cast<std::int64_t>(c10::ScalarType::Float8_e8m0fnu) && to != static_cast<std::int64_t>(c10::ScalarType::Float4_e2m1fn_x2)) {
        throw std::invalid_argument("to is outside the reviewed enum domain");
      }
      const auto result = at::can_cast(static_cast<c10::ScalarType>(from_), static_cast<c10::ScalarType>(to));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_promote_types_native(std::int64_t type1, std::int64_t type2) noexcept {
  try {
    try {
      if (type1 != static_cast<std::int64_t>(c10::ScalarType::Byte) && type1 != static_cast<std::int64_t>(c10::ScalarType::Char) && type1 != static_cast<std::int64_t>(c10::ScalarType::Short) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int) && type1 != static_cast<std::int64_t>(c10::ScalarType::Long) && type1 != static_cast<std::int64_t>(c10::ScalarType::Half) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float) && type1 != static_cast<std::int64_t>(c10::ScalarType::Double) && type1 != static_cast<std::int64_t>(c10::ScalarType::ComplexHalf) && type1 != static_cast<std::int64_t>(c10::ScalarType::ComplexFloat) && type1 != static_cast<std::int64_t>(c10::ScalarType::ComplexDouble) && type1 != static_cast<std::int64_t>(c10::ScalarType::Bool) && type1 != static_cast<std::int64_t>(c10::ScalarType::QInt8) && type1 != static_cast<std::int64_t>(c10::ScalarType::QUInt8) && type1 != static_cast<std::int64_t>(c10::ScalarType::QInt32) && type1 != static_cast<std::int64_t>(c10::ScalarType::BFloat16) && type1 != static_cast<std::int64_t>(c10::ScalarType::QUInt4x2) && type1 != static_cast<std::int64_t>(c10::ScalarType::QUInt2x4) && type1 != static_cast<std::int64_t>(c10::ScalarType::Bits1x8) && type1 != static_cast<std::int64_t>(c10::ScalarType::Bits2x4) && type1 != static_cast<std::int64_t>(c10::ScalarType::Bits4x2) && type1 != static_cast<std::int64_t>(c10::ScalarType::Bits8) && type1 != static_cast<std::int64_t>(c10::ScalarType::Bits16) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fn) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2fnuz) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fnuz) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt16) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt32) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt64) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt1) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt2) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt3) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt4) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt5) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt6) && type1 != static_cast<std::int64_t>(c10::ScalarType::UInt7) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int1) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int2) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int3) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int4) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int5) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int6) && type1 != static_cast<std::int64_t>(c10::ScalarType::Int7) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float8_e8m0fnu) && type1 != static_cast<std::int64_t>(c10::ScalarType::Float4_e2m1fn_x2)) {
        throw std::invalid_argument("type1 is outside the reviewed enum domain");
      }
      if (type2 != static_cast<std::int64_t>(c10::ScalarType::Byte) && type2 != static_cast<std::int64_t>(c10::ScalarType::Char) && type2 != static_cast<std::int64_t>(c10::ScalarType::Short) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int) && type2 != static_cast<std::int64_t>(c10::ScalarType::Long) && type2 != static_cast<std::int64_t>(c10::ScalarType::Half) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float) && type2 != static_cast<std::int64_t>(c10::ScalarType::Double) && type2 != static_cast<std::int64_t>(c10::ScalarType::ComplexHalf) && type2 != static_cast<std::int64_t>(c10::ScalarType::ComplexFloat) && type2 != static_cast<std::int64_t>(c10::ScalarType::ComplexDouble) && type2 != static_cast<std::int64_t>(c10::ScalarType::Bool) && type2 != static_cast<std::int64_t>(c10::ScalarType::QInt8) && type2 != static_cast<std::int64_t>(c10::ScalarType::QUInt8) && type2 != static_cast<std::int64_t>(c10::ScalarType::QInt32) && type2 != static_cast<std::int64_t>(c10::ScalarType::BFloat16) && type2 != static_cast<std::int64_t>(c10::ScalarType::QUInt4x2) && type2 != static_cast<std::int64_t>(c10::ScalarType::QUInt2x4) && type2 != static_cast<std::int64_t>(c10::ScalarType::Bits1x8) && type2 != static_cast<std::int64_t>(c10::ScalarType::Bits2x4) && type2 != static_cast<std::int64_t>(c10::ScalarType::Bits4x2) && type2 != static_cast<std::int64_t>(c10::ScalarType::Bits8) && type2 != static_cast<std::int64_t>(c10::ScalarType::Bits16) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fn) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float8_e5m2fnuz) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float8_e4m3fnuz) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt16) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt32) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt64) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt1) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt2) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt3) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt4) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt5) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt6) && type2 != static_cast<std::int64_t>(c10::ScalarType::UInt7) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int1) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int2) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int3) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int4) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int5) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int6) && type2 != static_cast<std::int64_t>(c10::ScalarType::Int7) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float8_e8m0fnu) && type2 != static_cast<std::int64_t>(c10::ScalarType::Float4_e2m1fn_x2)) {
        throw std::invalid_argument("type2 is outside the reviewed enum domain");
      }
      const auto result = at::promote_types(static_cast<c10::ScalarType>(type1), static_cast<c10::ScalarType>(type2));
      if (result != c10::ScalarType::Byte && result != c10::ScalarType::Char && result != c10::ScalarType::Short && result != c10::ScalarType::Int && result != c10::ScalarType::Long && result != c10::ScalarType::Half && result != c10::ScalarType::Float && result != c10::ScalarType::Double && result != c10::ScalarType::ComplexHalf && result != c10::ScalarType::ComplexFloat && result != c10::ScalarType::ComplexDouble && result != c10::ScalarType::Bool && result != c10::ScalarType::QInt8 && result != c10::ScalarType::QUInt8 && result != c10::ScalarType::QInt32 && result != c10::ScalarType::BFloat16 && result != c10::ScalarType::QUInt4x2 && result != c10::ScalarType::QUInt2x4 && result != c10::ScalarType::Bits1x8 && result != c10::ScalarType::Bits2x4 && result != c10::ScalarType::Bits4x2 && result != c10::ScalarType::Bits8 && result != c10::ScalarType::Bits16 && result != c10::ScalarType::Float8_e5m2 && result != c10::ScalarType::Float8_e4m3fn && result != c10::ScalarType::Float8_e5m2fnuz && result != c10::ScalarType::Float8_e4m3fnuz && result != c10::ScalarType::UInt16 && result != c10::ScalarType::UInt32 && result != c10::ScalarType::UInt64 && result != c10::ScalarType::UInt1 && result != c10::ScalarType::UInt2 && result != c10::ScalarType::UInt3 && result != c10::ScalarType::UInt4 && result != c10::ScalarType::UInt5 && result != c10::ScalarType::UInt6 && result != c10::ScalarType::UInt7 && result != c10::ScalarType::Int1 && result != c10::ScalarType::Int2 && result != c10::ScalarType::Int3 && result != c10::ScalarType::Int4 && result != c10::ScalarType::Int5 && result != c10::ScalarType::Int6 && result != c10::ScalarType::Int7 && result != c10::ScalarType::Float8_e8m0fnu && result != c10::ScalarType::Float4_e2m1fn_x2) {
        throw std::runtime_error("C++ result is outside the reviewed enum domain");
      }
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_tensor_native(const Tensor & tensor, const Tensor & other) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ScalarType (*)(const Tensor &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::result_type);
      const auto result = terlan_selected_overload(tensor, other);
      if (result != c10::ScalarType::Byte && result != c10::ScalarType::Char && result != c10::ScalarType::Short && result != c10::ScalarType::Int && result != c10::ScalarType::Long && result != c10::ScalarType::Half && result != c10::ScalarType::Float && result != c10::ScalarType::Double && result != c10::ScalarType::ComplexHalf && result != c10::ScalarType::ComplexFloat && result != c10::ScalarType::ComplexDouble && result != c10::ScalarType::Bool && result != c10::ScalarType::QInt8 && result != c10::ScalarType::QUInt8 && result != c10::ScalarType::QInt32 && result != c10::ScalarType::BFloat16 && result != c10::ScalarType::QUInt4x2 && result != c10::ScalarType::QUInt2x4 && result != c10::ScalarType::Bits1x8 && result != c10::ScalarType::Bits2x4 && result != c10::ScalarType::Bits4x2 && result != c10::ScalarType::Bits8 && result != c10::ScalarType::Bits16 && result != c10::ScalarType::Float8_e5m2 && result != c10::ScalarType::Float8_e4m3fn && result != c10::ScalarType::Float8_e5m2fnuz && result != c10::ScalarType::Float8_e4m3fnuz && result != c10::ScalarType::UInt16 && result != c10::ScalarType::UInt32 && result != c10::ScalarType::UInt64 && result != c10::ScalarType::UInt1 && result != c10::ScalarType::UInt2 && result != c10::ScalarType::UInt3 && result != c10::ScalarType::UInt4 && result != c10::ScalarType::UInt5 && result != c10::ScalarType::UInt6 && result != c10::ScalarType::UInt7 && result != c10::ScalarType::Int1 && result != c10::ScalarType::Int2 && result != c10::ScalarType::Int3 && result != c10::ScalarType::Int4 && result != c10::ScalarType::Int5 && result != c10::ScalarType::Int6 && result != c10::ScalarType::Int7 && result != c10::ScalarType::Float8_e8m0fnu && result != c10::ScalarType::Float4_e2m1fn_x2) {
        throw std::runtime_error("C++ result is outside the reviewed enum domain");
      }
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_scalar_native(const Tensor & tensor, bool scalar_is_integer, std::int64_t scalar_integer, double scalar_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ScalarType (*)(const Tensor &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::result_type);
      const auto result = terlan_selected_overload(tensor, scalar_is_integer ? c10::Scalar(scalar_integer) : c10::Scalar(scalar_floating));
      if (result != c10::ScalarType::Byte && result != c10::ScalarType::Char && result != c10::ScalarType::Short && result != c10::ScalarType::Int && result != c10::ScalarType::Long && result != c10::ScalarType::Half && result != c10::ScalarType::Float && result != c10::ScalarType::Double && result != c10::ScalarType::ComplexHalf && result != c10::ScalarType::ComplexFloat && result != c10::ScalarType::ComplexDouble && result != c10::ScalarType::Bool && result != c10::ScalarType::QInt8 && result != c10::ScalarType::QUInt8 && result != c10::ScalarType::QInt32 && result != c10::ScalarType::BFloat16 && result != c10::ScalarType::QUInt4x2 && result != c10::ScalarType::QUInt2x4 && result != c10::ScalarType::Bits1x8 && result != c10::ScalarType::Bits2x4 && result != c10::ScalarType::Bits4x2 && result != c10::ScalarType::Bits8 && result != c10::ScalarType::Bits16 && result != c10::ScalarType::Float8_e5m2 && result != c10::ScalarType::Float8_e4m3fn && result != c10::ScalarType::Float8_e5m2fnuz && result != c10::ScalarType::Float8_e4m3fnuz && result != c10::ScalarType::UInt16 && result != c10::ScalarType::UInt32 && result != c10::ScalarType::UInt64 && result != c10::ScalarType::UInt1 && result != c10::ScalarType::UInt2 && result != c10::ScalarType::UInt3 && result != c10::ScalarType::UInt4 && result != c10::ScalarType::UInt5 && result != c10::ScalarType::UInt6 && result != c10::ScalarType::UInt7 && result != c10::ScalarType::Int1 && result != c10::ScalarType::Int2 && result != c10::ScalarType::Int3 && result != c10::ScalarType::Int4 && result != c10::ScalarType::Int5 && result != c10::ScalarType::Int6 && result != c10::ScalarType::Int7 && result != c10::ScalarType::Float8_e8m0fnu && result != c10::ScalarType::Float4_e2m1fn_x2) {
        throw std::runtime_error("C++ result is outside the reviewed enum domain");
      }
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_scalar_tensor_native(const Tensor & tensor, bool scalar_is_integer, std::int64_t scalar_integer, double scalar_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ScalarType (*)(const Scalar &, const Tensor &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::result_type);
      const auto result = terlan_selected_overload(scalar_is_integer ? c10::Scalar(scalar_integer) : c10::Scalar(scalar_floating), tensor);
      if (result != c10::ScalarType::Byte && result != c10::ScalarType::Char && result != c10::ScalarType::Short && result != c10::ScalarType::Int && result != c10::ScalarType::Long && result != c10::ScalarType::Half && result != c10::ScalarType::Float && result != c10::ScalarType::Double && result != c10::ScalarType::ComplexHalf && result != c10::ScalarType::ComplexFloat && result != c10::ScalarType::ComplexDouble && result != c10::ScalarType::Bool && result != c10::ScalarType::QInt8 && result != c10::ScalarType::QUInt8 && result != c10::ScalarType::QInt32 && result != c10::ScalarType::BFloat16 && result != c10::ScalarType::QUInt4x2 && result != c10::ScalarType::QUInt2x4 && result != c10::ScalarType::Bits1x8 && result != c10::ScalarType::Bits2x4 && result != c10::ScalarType::Bits4x2 && result != c10::ScalarType::Bits8 && result != c10::ScalarType::Bits16 && result != c10::ScalarType::Float8_e5m2 && result != c10::ScalarType::Float8_e4m3fn && result != c10::ScalarType::Float8_e5m2fnuz && result != c10::ScalarType::Float8_e4m3fnuz && result != c10::ScalarType::UInt16 && result != c10::ScalarType::UInt32 && result != c10::ScalarType::UInt64 && result != c10::ScalarType::UInt1 && result != c10::ScalarType::UInt2 && result != c10::ScalarType::UInt3 && result != c10::ScalarType::UInt4 && result != c10::ScalarType::UInt5 && result != c10::ScalarType::UInt6 && result != c10::ScalarType::UInt7 && result != c10::ScalarType::Int1 && result != c10::ScalarType::Int2 && result != c10::ScalarType::Int3 && result != c10::ScalarType::Int4 && result != c10::ScalarType::Int5 && result != c10::ScalarType::Int6 && result != c10::ScalarType::Int7 && result != c10::ScalarType::Float8_e8m0fnu && result != c10::ScalarType::Float4_e2m1fn_x2) {
        throw std::runtime_error("C++ result is outside the reviewed enum domain");
      }
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_typerelationsnative_result_type_scalar_scalar_native(bool first_is_integer, std::int64_t first_integer, double first_floating, bool second_is_integer, std::int64_t second_integer, double second_floating) noexcept {
  try {
    try {
    using TerlanSelectedOverload = ScalarType (*)(const Scalar &, const Scalar &);
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&at::result_type);
      const auto result = terlan_selected_overload(first_is_integer ? c10::Scalar(first_integer) : c10::Scalar(first_floating), second_is_integer ? c10::Scalar(second_integer) : c10::Scalar(second_floating));
      if (result != c10::ScalarType::Byte && result != c10::ScalarType::Char && result != c10::ScalarType::Short && result != c10::ScalarType::Int && result != c10::ScalarType::Long && result != c10::ScalarType::Half && result != c10::ScalarType::Float && result != c10::ScalarType::Double && result != c10::ScalarType::ComplexHalf && result != c10::ScalarType::ComplexFloat && result != c10::ScalarType::ComplexDouble && result != c10::ScalarType::Bool && result != c10::ScalarType::QInt8 && result != c10::ScalarType::QUInt8 && result != c10::ScalarType::QInt32 && result != c10::ScalarType::BFloat16 && result != c10::ScalarType::QUInt4x2 && result != c10::ScalarType::QUInt2x4 && result != c10::ScalarType::Bits1x8 && result != c10::ScalarType::Bits2x4 && result != c10::ScalarType::Bits4x2 && result != c10::ScalarType::Bits8 && result != c10::ScalarType::Bits16 && result != c10::ScalarType::Float8_e5m2 && result != c10::ScalarType::Float8_e4m3fn && result != c10::ScalarType::Float8_e5m2fnuz && result != c10::ScalarType::Float8_e4m3fnuz && result != c10::ScalarType::UInt16 && result != c10::ScalarType::UInt32 && result != c10::ScalarType::UInt64 && result != c10::ScalarType::UInt1 && result != c10::ScalarType::UInt2 && result != c10::ScalarType::UInt3 && result != c10::ScalarType::UInt4 && result != c10::ScalarType::UInt5 && result != c10::ScalarType::UInt6 && result != c10::ScalarType::UInt7 && result != c10::ScalarType::Int1 && result != c10::ScalarType::Int2 && result != c10::ScalarType::Int3 && result != c10::ScalarType::Int4 && result != c10::ScalarType::Int5 && result != c10::ScalarType::Int6 && result != c10::ScalarType::Int7 && result != c10::ScalarType::Float8_e8m0fnu && result != c10::ScalarType::Float4_e2m1fn_x2) {
        throw std::runtime_error("C++ result is outside the reviewed enum domain");
      }
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch type-relation operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_scalar_to_int(const c10::Scalar& value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = int64_t (c10::Scalar::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&c10::Scalar::to);
      const auto result = (value.*terlan_selected_overload)();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_scalar_conversion_failed", "LibTorch scalar conversion failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_scalar_to_float(const c10::Scalar& value) noexcept {
  try {
    try {
    using TerlanSelectedOverload = double (c10::Scalar::*)() const;
    const auto terlan_selected_overload = static_cast<TerlanSelectedOverload>(&c10::Scalar::to);
      const auto result = (value.*terlan_selected_overload)();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, result, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_scalar_conversion_failed", "LibTorch scalar conversion failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_defined(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.defined();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch defined-state inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_cpu(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_cpu();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch CPU-device inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_cuda(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_cuda();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch CUDA-device inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_requires_grad(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.requires_grad();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch gradient-state inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_leaf(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_leaf();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch leaf-state inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_has_storage(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.has_storage();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_complex(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_complex();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_floating_point(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_floating_point();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_quantized(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_quantized();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_signed(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_signed();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_sparse(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_sparse();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor property inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_contiguous(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.is_contiguous();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch contiguity inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_contiguous_memory_format_native(const at::Tensor& value, std::int64_t memory_format) noexcept {
  try {
    try {
      if (memory_format != static_cast<std::int64_t>(c10::MemoryFormat::Contiguous) && memory_format != static_cast<std::int64_t>(c10::MemoryFormat::Preserve) && memory_format != static_cast<std::int64_t>(c10::MemoryFormat::ChannelsLast) && memory_format != static_cast<std::int64_t>(c10::MemoryFormat::ChannelsLast3d)) {
        throw std::invalid_argument("memory_format is outside the reviewed enum domain");
      }
      const auto result = value.is_contiguous(static_cast<c10::MemoryFormat>(memory_format));
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch contiguity inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_dim(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.dim();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch rank inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_numel(const at::Tensor& value) noexcept {
  try {
    try {
      const auto result = value.numel();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch element-count inspection failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_tensor_is_pinned_cxx(const Tensor & self, bool has_device, rust::Str device) noexcept {
  try {
    try {
      const auto result = at::native::is_pinned(self, has_device ? ::std::optional<Device>(c10::Device(std::string(device.data(), device.size()))) : std::nullopt);
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_operation_failed", "LibTorch tensor operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_sparse_sparse_dim(const Tensor & self) noexcept {
  try {
    try {
      const auto result = self.sparse_dim();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_sparse_dense_dim(const Tensor & self) noexcept {
  try {
    try {
      const auto result = self.dense_dim();
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_sparse_is_coalesced(const Tensor & self) noexcept {
  try {
    try {
      const auto result = self.is_coalesced();
      return std::make_unique<TerlanExceptionEnvelope>(true, 0, 0.0, result, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_sparse_operation_failed", "LibTorch sparse operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<TerlanExceptionEnvelope> terlan_exception_pytorch_cuda_device_index(const CudaStream & stream) noexcept {
  try {
    try {
      const auto result = at::native::cuda_stream_device_index(stream);
      return std::make_unique<TerlanExceptionEnvelope>(true, static_cast<std::int64_t>(result), 0.0, false, "", "");
    } catch (...) {
      return std::make_unique<TerlanExceptionEnvelope>(false, 0, 0.0, false, "pytorch_cuda_operation_failed", "LibTorch CUDA stream operation failed.");
    }
  } catch (...) {
    return nullptr;
  }
}

}  // namespace at::native
