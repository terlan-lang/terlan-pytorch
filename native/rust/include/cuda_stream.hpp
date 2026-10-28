#pragma once

#include <cstdint>
#include <memory>
#include <ATen/ATen.h>

namespace at::native {

class CudaStream {
 public:
  CudaStream(void* value, std::int64_t index, std::int64_t stream_id = 0)
      : handle_(value), device_index_(index), stream_id_(stream_id) {}

  void* handle_;
  std::int64_t device_index_;
  std::int64_t stream_id_;
};

std::unique_ptr<CudaStream> cuda_stream_create(std::int64_t device_index,
                                               bool high_priority);
std::int64_t cuda_stream_device_index(const CudaStream& stream);
void cuda_stream_set_current(const CudaStream& stream);
void cuda_stream_synchronize(const CudaStream& stream);
void cuda_stream_record(at::Tensor& tensor, const CudaStream& stream);

}  // namespace at::native
