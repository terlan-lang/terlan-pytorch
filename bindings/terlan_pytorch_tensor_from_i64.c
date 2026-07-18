#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct AtenTensorOpaque* AtenTensorHandle;
typedef int32_t AOTITorchError;

extern int32_t aoti_torch_device_type_cpu(void);
extern int32_t aoti_torch_dtype_int64(void);
extern int32_t aoti_torch_layout_strided(void);
extern AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);
extern AOTITorchError aoti_torch_clone(
    AtenTensorHandle tensor,
    AtenTensorHandle* ret);
extern AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

enum {
  TERLAN_PYTORCH_INVALID_ARGUMENT = 7001,
  TERLAN_PYTORCH_SHAPE_OVERFLOW = 7002,
  TERLAN_PYTORCH_VALUE_COUNT_MISMATCH = 7003,
  TERLAN_PYTORCH_ALLOCATION_FAILED = 7004,
};

AOTITorchError terlan_pytorch_tensor_from_i64(
    const int64_t* values,
    int64_t values_length,
    const int64_t* shape,
    int64_t shape_length,
    AtenTensorHandle* out_tensor) {
  if (out_tensor == NULL) {
    return TERLAN_PYTORCH_INVALID_ARGUMENT;
  }
  *out_tensor = NULL;
  if (values_length < 0 || shape_length < 0 ||
      (values_length > 0 && values == NULL) ||
      (shape_length > 0 && shape == NULL)) {
    return TERLAN_PYTORCH_INVALID_ARGUMENT;
  }

  int has_zero_dimension = 0;
  for (int64_t dimension = 0; dimension < shape_length; ++dimension) {
    if (shape[dimension] < 0) {
      return TERLAN_PYTORCH_INVALID_ARGUMENT;
    }
    if (shape[dimension] == 0) {
      has_zero_dimension = 1;
    }
  }
  int64_t element_count = has_zero_dimension ? 0 : 1;
  if (!has_zero_dimension) {
    for (int64_t dimension = 0; dimension < shape_length; ++dimension) {
      if (shape[dimension] != 0 &&
          element_count > INT64_MAX / shape[dimension]) {
        return TERLAN_PYTORCH_SHAPE_OVERFLOW;
      }
      element_count *= shape[dimension];
    }
  }
  if (element_count != values_length) {
    return TERLAN_PYTORCH_VALUE_COUNT_MISMATCH;
  }
  if ((uint64_t)shape_length > SIZE_MAX / sizeof(int64_t)) {
    return TERLAN_PYTORCH_SHAPE_OVERFLOW;
  }

  int64_t* strides = NULL;
  if (shape_length > 0) {
    strides = malloc((size_t)shape_length * sizeof(int64_t));
    if (strides == NULL) {
      return TERLAN_PYTORCH_ALLOCATION_FAILED;
    }
    int64_t stride = 1;
    for (int64_t dimension = shape_length; dimension-- > 0;) {
      strides[dimension] = stride;
      if (shape[dimension] != 0 && stride > INT64_MAX / shape[dimension]) {
        free(strides);
        return TERLAN_PYTORCH_SHAPE_OVERFLOW;
      }
      stride *= shape[dimension];
    }
  }

  AtenTensorHandle borrowed = NULL;
  AOTITorchError status = aoti_torch_create_tensor_from_blob_v2(
      (void*)values,
      shape_length,
      shape,
      strides,
      0,
      aoti_torch_dtype_int64(),
      aoti_torch_device_type_cpu(),
      -1,
      &borrowed,
      aoti_torch_layout_strided(),
      NULL,
      0);
  free(strides);
  if (status != 0) {
    return status;
  }

  AtenTensorHandle owned = NULL;
  AOTITorchError clone_status = aoti_torch_clone(borrowed, &owned);
  AOTITorchError delete_status = aoti_torch_delete_tensor_object(borrowed);
  if (clone_status != 0) {
    return clone_status;
  }
  if (delete_status != 0) {
    (void)aoti_torch_delete_tensor_object(owned);
    return delete_status;
  }
  *out_tensor = owned;
  return 0;
}
