#include "cuda_stream.hpp"

#include <stdexcept>

#if defined(TERLAN_LIBTORCH_CUDA)
#include <c10/cuda/CUDAStream.h>
#endif

namespace at::native {

std::unique_ptr<CudaStream> cuda_stream_create(std::int64_t device_index,
                                               bool high_priority) {
  if (device_index < 0) {
    throw std::invalid_argument("invalid CUDA device index");
  }
#if !defined(TERLAN_LIBTORCH_CUDA)
  (void)high_priority;
  return nullptr;
#else
  try {
    const auto stream = c10::cuda::getStreamFromPool(
        high_priority, static_cast<c10::DeviceIndex>(device_index));
    return std::make_unique<CudaStream>(
        reinterpret_cast<void*>(stream.stream()), device_index,
        static_cast<std::int64_t>(stream.id()));
  } catch (...) {
    return nullptr;
  }
#endif
}

std::int64_t cuda_stream_device_index(const CudaStream& stream) {
  return stream.device_index_;
}

void cuda_stream_set_current(const CudaStream& stream) {
#if !defined(TERLAN_LIBTORCH_CUDA)
  (void)stream;
  throw std::runtime_error("CUDA support is unavailable in this LibTorch build");
#else
  try {
    const auto native = c10::cuda::CUDAStream::unpack3(
        static_cast<c10::StreamId>(stream.stream_id_),
        static_cast<c10::DeviceIndex>(stream.device_index_),
        c10::DeviceType::CUDA);
    c10::cuda::setCurrentCUDAStream(native);
  } catch (...) {
    throw std::runtime_error(
        "LibTorch C++ CUDA current-stream operation failed");
  }
#endif
}

void cuda_stream_synchronize(const CudaStream& stream) {
#if !defined(TERLAN_LIBTORCH_CUDA)
  (void)stream;
  throw std::runtime_error("CUDA support is unavailable in this LibTorch build");
#else
  try {
    const auto native = c10::cuda::CUDAStream::unpack3(
        static_cast<c10::StreamId>(stream.stream_id_),
        static_cast<c10::DeviceIndex>(stream.device_index_),
        c10::DeviceType::CUDA);
    native.synchronize();
  } catch (...) {
    throw std::runtime_error("LibTorch C++ CUDA stream synchronization failed");
  }
#endif
}

void cuda_stream_record(at::Tensor& tensor, const CudaStream& stream) {
#if !defined(TERLAN_LIBTORCH_CUDA)
  (void)tensor;
  (void)stream;
  throw std::runtime_error("CUDA support is unavailable in this LibTorch build");
#else
  const auto value = c10::Stream(
      c10::Stream::UNSAFE,
      c10::Device(c10::DeviceType::CUDA,
                  static_cast<c10::DeviceIndex>(stream.device_index_)),
      static_cast<c10::StreamId>(stream.stream_id_));
  tensor.record_stream(value);
#endif
}

}  // namespace at::native
