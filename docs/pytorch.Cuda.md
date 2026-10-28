# pytorch.Cuda

Generated CXX CUDA stream surface.

## `create_stream`

Creates an owned CUDA stream resource.

- C++ symbol: `cuda_stream_create`
- NativeBoundary operation: `pytorch.cuda.stream.create`
- Ownership: `opaque_handle`

## `device_index`

Returns the stream device index.

- C++ symbol: `cuda_stream_device_index`
- NativeBoundary operation: `pytorch.cuda.stream.device_index`
- Ownership: `borrowed_handle`

## `set_current`

Makes a CUDA stream current.

- C++ symbol: `cuda_stream_set_current`
- NativeBoundary operation: `pytorch.cuda.stream.set_current`
- Ownership: `borrowed_handle`

## `synchronize`

Synchronizes a CUDA stream.

- C++ symbol: `cuda_stream_synchronize`
- NativeBoundary operation: `pytorch.cuda.stream.synchronize`
- Ownership: `borrowed_handle`

## `record_stream`

Records a tensor on an owned CUDA stream for allocator lifetime tracking.

- C++ symbol: `cuda_stream_record`
- NativeBoundary operation: `pytorch.cuda.stream.record`
- Ownership: `mutable_handle`

## `dispose`

Releases an owned CUDA stream resource.

- NativeBoundary operation: `pytorch.cuda.stream.dispose`
- Ownership: `dispose_handle`

## `synchronize_device`

Synchronizes a CUDA device through an owned default-priority stream.

- Implementation: generated Terlan composition
