#include "terlan_tensor_packet.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace terlan::tensor_packet {
namespace {

constexpr std::size_t kHeaderBytes = 16;
constexpr std::size_t kMaximumRank = 32;
constexpr std::size_t kMaximumPacketBytes = 16 * 1024 * 1024;

std::uint16_t read_u16_le(const std::uint8_t* input) noexcept {
  return static_cast<std::uint16_t>(input[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(input[1]) << 8);
}

std::uint32_t read_u32_le(const std::uint8_t* input) noexcept {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < 4; ++index) {
    value |= static_cast<std::uint32_t>(input[index]) << (index * 8);
  }
  return value;
}

std::uint64_t read_u64_le(const std::uint8_t* input) noexcept {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(input[index]) << (index * 8);
  }
  return value;
}

void write_u16_le(std::uint8_t* output, std::uint16_t value) noexcept {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8);
}

void write_u32_le(std::uint8_t* output, std::uint32_t value) noexcept {
  for (std::size_t index = 0; index < 4; ++index) {
    output[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

void write_u64_le(std::uint8_t* output, std::uint64_t value) noexcept {
  for (std::size_t index = 0; index < 8; ++index) {
    output[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

std::uint8_t native_endian_code() noexcept {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1 ? 1 : 2;
}

std::pair<at::ScalarType, std::size_t> torch_dtype(std::uint8_t code,
                                                   std::uint8_t bits,
                                                   std::uint16_t lanes) {
  if (lanes != 1) {
    throw std::invalid_argument("TNXP lanes must equal one");
  }
  if (code == 1 && bits == 8) {
    return {at::kByte, 1};
  }
  if (code == 2 && bits == 32) {
    return {at::kFloat, 4};
  }
  if (code == 2 && bits == 64) {
    return {at::kDouble, 8};
  }
  if (code == 0 && bits == 32) {
    return {at::kInt, 4};
  }
  if (code == 0 && bits == 64) {
    return {at::kLong, 8};
  }
  if (code == 6 && bits == 8) {
    return {at::kBool, 1};
  }
  throw std::invalid_argument("TNXP dtype is unsupported by the copied tensor exchange");
}

std::pair<std::uint8_t, std::uint8_t> packet_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kByte:
      return {1, 8};
    case at::kFloat:
      return {2, 32};
    case at::kDouble:
      return {2, 64};
    case at::kInt:
      return {0, 32};
    case at::kLong:
      return {0, 64};
    case at::kBool:
      return {6, 8};
    default:
      throw std::invalid_argument("tensor dtype is unsupported by TNXP");
  }
}

std::size_t checked_metadata_bytes(std::size_t rank) {
  if (rank > kMaximumRank ||
      rank > (std::numeric_limits<std::size_t>::max() - kHeaderBytes) / 16) {
    throw std::invalid_argument("TNXP rank is unsupported");
  }
  return kHeaderBytes + rank * 16;
}

std::size_t checked_storage_elements(const std::vector<std::int64_t>& shape,
                                     const std::vector<std::int64_t>& strides) {
  std::uint64_t storage_elements = 1;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    const auto dimension = shape[index];
    const auto stride = strides[index];
    if (dimension < 0 || stride < 0) {
      throw std::invalid_argument("TNXP dimensions and strides must be non-negative");
    }
    if (dimension == 0) {
      storage_elements = 0;
      continue;
    }
    if (storage_elements == 0) {
      continue;
    }
    const auto extent_dimension = static_cast<std::uint64_t>(dimension - 1);
    const auto extent_stride = static_cast<std::uint64_t>(stride);
    if (extent_dimension != 0 &&
        extent_stride > std::numeric_limits<std::uint64_t>::max() /
                            extent_dimension) {
      throw std::invalid_argument("TNXP storage extent overflows");
    }
    const auto extent = extent_dimension * extent_stride;
    if (extent > std::numeric_limits<std::uint64_t>::max() - storage_elements) {
      throw std::invalid_argument("TNXP storage extent overflows");
    }
    storage_elements += extent;
  }
  if (storage_elements > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("TNXP storage extent is not representable");
  }
  return static_cast<std::size_t>(storage_elements);
}

std::size_t checked_data_bytes(std::size_t elements,
                               std::size_t element_bytes) {
  if (element_bytes == 0 ||
      elements > std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::invalid_argument("TNXP byte size overflows");
  }
  return elements * element_bytes;
}

}  // namespace

std::unique_ptr<at::Tensor> import_tensor_packet_impl(
    rust::Slice<const std::uint8_t> packet) {
  if (packet.size() < kHeaderBytes || packet.size() > kMaximumPacketBytes) {
    throw std::invalid_argument("TNXP packet size is invalid");
  }
  const auto* bytes = packet.data();
  if (!std::equal(bytes, bytes + 4, std::string_view("TNXP").begin()) ||
      bytes[4] != 1 || bytes[5] != native_endian_code()) {
    throw std::invalid_argument("TNXP version or byte order is unsupported");
  }
  if (bytes[10] != 1 || bytes[11] != 0) {
    throw std::invalid_argument("TNXP copied import requires CPU device zero");
  }

  const auto rank = static_cast<std::size_t>(read_u32_le(bytes + 12));
  const auto metadata_bytes = checked_metadata_bytes(rank);
  if (packet.size() < metadata_bytes) {
    throw std::invalid_argument("TNXP metadata is truncated");
  }

  std::vector<std::int64_t> shape(rank);
  std::vector<std::int64_t> strides(rank);
  for (std::size_t index = 0; index < rank; ++index) {
    const auto encoded_dimension = read_u64_le(bytes + kHeaderBytes + index * 8);
    const auto encoded_stride =
        read_u64_le(bytes + kHeaderBytes + rank * 8 + index * 8);
    if (encoded_dimension >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        encoded_stride >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument("TNXP shape or stride is not representable");
    }
    shape[index] = static_cast<std::int64_t>(encoded_dimension);
    strides[index] = static_cast<std::int64_t>(encoded_stride);
  }

  const auto [dtype, element_bytes] =
      torch_dtype(bytes[6], bytes[7], read_u16_le(bytes + 8));
  const auto storage_elements = checked_storage_elements(shape, strides);
  const auto data_bytes = checked_data_bytes(storage_elements, element_bytes);
  if (packet.size() - metadata_bytes != data_bytes) {
    throw std::invalid_argument("TNXP payload size does not match its layout");
  }

  auto tensor = at::empty_strided(
      shape, strides,
      at::TensorOptions().dtype(dtype).layout(at::kStrided).device(at::kCPU));
  if (data_bytes != 0) {
    std::memcpy(tensor.data_ptr(), bytes + metadata_bytes, data_bytes);
  }
  return std::make_unique<at::Tensor>(std::move(tensor));
}

std::unique_ptr<std::vector<std::uint8_t>> export_tensor_packet_impl(
    const at::Tensor& tensor) {
  if (!tensor.defined()) {
    throw std::invalid_argument("cannot export an undefined tensor");
  }
  // Validate the packet budget from metadata before materializing a CPU copy.
  // This keeps a hostile/accidentally huge tensor from forcing an allocation
  // that the packet limit would reject anyway.
  const auto rank_value = tensor.dim();
  if (rank_value < 0) {
    throw std::invalid_argument("tensor rank is invalid");
  }
  const auto rank = static_cast<std::size_t>(rank_value);
  const auto metadata_bytes = checked_metadata_bytes(rank);
  const auto [dtype_code, dtype_bits] = packet_dtype(tensor.scalar_type());
  const auto numel = tensor.numel();
  if (numel < 0) {
    throw std::invalid_argument("tensor element count is invalid");
  }
  const auto data_bytes = checked_data_bytes(
      static_cast<std::size_t>(numel), static_cast<std::size_t>(dtype_bits / 8));
  if (data_bytes > std::numeric_limits<std::size_t>::max() - metadata_bytes ||
      metadata_bytes + data_bytes > kMaximumPacketBytes) {
      throw std::invalid_argument("TNXP packet size is unsupported");
  }

  auto snapshot = tensor.detach().to(at::kCPU).contiguous();

  auto packet = std::make_unique<std::vector<std::uint8_t>>(
      metadata_bytes + data_bytes);
  auto* bytes = packet->data();
  std::memcpy(bytes, "TNXP", 4);
  bytes[4] = 1;
  bytes[5] = native_endian_code();
  bytes[6] = dtype_code;
  bytes[7] = dtype_bits;
  write_u16_le(bytes + 8, 1);
  bytes[10] = 1;
  bytes[11] = 0;
  write_u32_le(bytes + 12, static_cast<std::uint32_t>(rank));
  for (std::size_t index = 0; index < rank; ++index) {
    write_u64_le(bytes + kHeaderBytes + index * 8,
                 static_cast<std::uint64_t>(snapshot.size(index)));
    write_u64_le(bytes + kHeaderBytes + rank * 8 + index * 8,
                 static_cast<std::uint64_t>(snapshot.stride(index)));
  }
  if (data_bytes != 0) {
    std::memcpy(bytes + metadata_bytes, snapshot.const_data_ptr(), data_bytes);
  }
  return packet;
}

std::unique_ptr<at::Tensor> import_tensor_packet(
    rust::Slice<const std::uint8_t> packet) noexcept {
  try {
    return import_tensor_packet_impl(packet);
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<std::vector<std::uint8_t>> export_tensor_packet(
    const at::Tensor& tensor) noexcept {
  try {
    return export_tensor_packet_impl(tensor);
  } catch (...) {
    return nullptr;
  }
}

}  // namespace terlan::tensor_packet
