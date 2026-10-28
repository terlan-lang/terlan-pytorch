#include "include/terlan_enum_adapters.hpp"

namespace at::native {

std::unique_ptr<std::string> terlan_enum_pytorch_tensor_dtype(const at::Tensor& value) noexcept {
  try {
    const auto result = value.scalar_type();
    if (result == c10::ScalarType::Byte) {
      return std::make_unique<std::string>("uint8");
    }
    if (result == c10::ScalarType::Char) {
      return std::make_unique<std::string>("int8");
    }
    if (result == c10::ScalarType::Short) {
      return std::make_unique<std::string>("int16");
    }
    if (result == c10::ScalarType::Int) {
      return std::make_unique<std::string>("int32");
    }
    if (result == c10::ScalarType::Long) {
      return std::make_unique<std::string>("int64");
    }
    if (result == c10::ScalarType::Half) {
      return std::make_unique<std::string>("float16");
    }
    if (result == c10::ScalarType::Float) {
      return std::make_unique<std::string>("float32");
    }
    if (result == c10::ScalarType::Double) {
      return std::make_unique<std::string>("float64");
    }
    if (result == c10::ScalarType::ComplexHalf) {
      return std::make_unique<std::string>("complex32");
    }
    if (result == c10::ScalarType::ComplexFloat) {
      return std::make_unique<std::string>("complex64");
    }
    if (result == c10::ScalarType::ComplexDouble) {
      return std::make_unique<std::string>("complex128");
    }
    if (result == c10::ScalarType::Bool) {
      return std::make_unique<std::string>("bool");
    }
    if (result == c10::ScalarType::QInt8) {
      return std::make_unique<std::string>("qint8");
    }
    if (result == c10::ScalarType::QUInt8) {
      return std::make_unique<std::string>("quint8");
    }
    if (result == c10::ScalarType::QInt32) {
      return std::make_unique<std::string>("qint32");
    }
    if (result == c10::ScalarType::BFloat16) {
      return std::make_unique<std::string>("bfloat16");
    }
    if (result == c10::ScalarType::QUInt4x2) {
      return std::make_unique<std::string>("quint4x2");
    }
    if (result == c10::ScalarType::QUInt2x4) {
      return std::make_unique<std::string>("quint2x4");
    }
    if (result == c10::ScalarType::Bits1x8) {
      return std::make_unique<std::string>("bits1x8");
    }
    if (result == c10::ScalarType::Bits2x4) {
      return std::make_unique<std::string>("bits2x4");
    }
    if (result == c10::ScalarType::Bits4x2) {
      return std::make_unique<std::string>("bits4x2");
    }
    if (result == c10::ScalarType::Bits8) {
      return std::make_unique<std::string>("bits8");
    }
    if (result == c10::ScalarType::Bits16) {
      return std::make_unique<std::string>("bits16");
    }
    if (result == c10::ScalarType::Float8_e5m2) {
      return std::make_unique<std::string>("float8_e5m2");
    }
    if (result == c10::ScalarType::Float8_e4m3fn) {
      return std::make_unique<std::string>("float8_e4m3fn");
    }
    if (result == c10::ScalarType::Float8_e5m2fnuz) {
      return std::make_unique<std::string>("float8_e5m2fnuz");
    }
    if (result == c10::ScalarType::Float8_e4m3fnuz) {
      return std::make_unique<std::string>("float8_e4m3fnuz");
    }
    if (result == c10::ScalarType::UInt16) {
      return std::make_unique<std::string>("uint16");
    }
    if (result == c10::ScalarType::UInt32) {
      return std::make_unique<std::string>("uint32");
    }
    if (result == c10::ScalarType::UInt64) {
      return std::make_unique<std::string>("uint64");
    }
    if (result == c10::ScalarType::UInt1) {
      return std::make_unique<std::string>("uint1");
    }
    if (result == c10::ScalarType::UInt2) {
      return std::make_unique<std::string>("uint2");
    }
    if (result == c10::ScalarType::UInt3) {
      return std::make_unique<std::string>("uint3");
    }
    if (result == c10::ScalarType::UInt4) {
      return std::make_unique<std::string>("uint4");
    }
    if (result == c10::ScalarType::UInt5) {
      return std::make_unique<std::string>("uint5");
    }
    if (result == c10::ScalarType::UInt6) {
      return std::make_unique<std::string>("uint6");
    }
    if (result == c10::ScalarType::UInt7) {
      return std::make_unique<std::string>("uint7");
    }
    if (result == c10::ScalarType::Int1) {
      return std::make_unique<std::string>("int1");
    }
    if (result == c10::ScalarType::Int2) {
      return std::make_unique<std::string>("int2");
    }
    if (result == c10::ScalarType::Int3) {
      return std::make_unique<std::string>("int3");
    }
    if (result == c10::ScalarType::Int4) {
      return std::make_unique<std::string>("int4");
    }
    if (result == c10::ScalarType::Int5) {
      return std::make_unique<std::string>("int5");
    }
    if (result == c10::ScalarType::Int6) {
      return std::make_unique<std::string>("int6");
    }
    if (result == c10::ScalarType::Int7) {
      return std::make_unique<std::string>("int7");
    }
    if (result == c10::ScalarType::Float8_e8m0fnu) {
      return std::make_unique<std::string>("float8_e8m0fnu");
    }
    if (result == c10::ScalarType::Float4_e2m1fn_x2) {
      return std::make_unique<std::string>("float4_e2m1fn_x2");
    }
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<std::string> terlan_enum_pytorch_tensor_layout(const at::Tensor& value) noexcept {
  try {
    const auto result = value.layout();
    if (result == c10::Layout::Strided) {
      return std::make_unique<std::string>("strided");
    }
    if (result == c10::Layout::Sparse) {
      return std::make_unique<std::string>("sparse_coo");
    }
    if (result == c10::Layout::SparseCsr) {
      return std::make_unique<std::string>("sparse_csr");
    }
    if (result == c10::Layout::Mkldnn) {
      return std::make_unique<std::string>("mkldnn");
    }
    if (result == c10::Layout::SparseCsc) {
      return std::make_unique<std::string>("sparse_csc");
    }
    if (result == c10::Layout::SparseBsr) {
      return std::make_unique<std::string>("sparse_bsr");
    }
    if (result == c10::Layout::SparseBsc) {
      return std::make_unique<std::string>("sparse_bsc");
    }
    if (result == c10::Layout::Jagged) {
      return std::make_unique<std::string>("jagged");
    }
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

}  // namespace at::native
