/*
 * ABI-contract fixture only. This is not a LibTorch implementation and cannot
 * validate PyTorch behavior. It exists to compile, link, own, and execute the
 * generated package when a released LibTorch distribution is unavailable.
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct AtenTensorOpaque {
  double value;
  int64_t rank;
  int64_t sizes[4];
  int32_t dtype;
  int32_t causal_mask;
} AtenTensorOpaque;

typedef struct StableListOpaque {
  uint64_t* values;
  size_t size;
  size_t capacity;
} StableListOpaque;

typedef struct StringOpaque {
  char* data;
  size_t length;
} StringOpaque;

static double fixture_sqrt(double value) {
  if (value <= 0.0) {
    return 0.0;
  }
  double estimate = value > 1.0 ? value : 1.0;
  for (int iteration = 0; iteration < 16; ++iteration) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}

int32_t aoti_torch_device_type_cpu(void) {
  return 0;
}

int32_t aoti_torch_dtype_float64(void) {
  return 7;
}

int32_t aoti_torch_dtype_int64(void) {
  return 4;
}

int32_t aoti_torch_layout_strided(void) {
  return 0;
}

int32_t aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorOpaque** out_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  if (data == NULL || ndim < 0 || ndim > 4 ||
      (ndim > 0 && (sizes == NULL || strides == NULL)) ||
      storage_offset != 0 || (dtype != 7 && dtype != 4) || device_type != 0 ||
      device_index != -1 || out_tensor == NULL || layout != 0 ||
      opaque_metadata != NULL || opaque_metadata_size != 0) {
    return 1;
  }
  int64_t numel = 1;
  for (int64_t dimension = 0; dimension < ndim; ++dimension) {
    if (sizes[dimension] < 0) {
      return 1;
    }
    numel *= sizes[dimension];
  }
  AtenTensorOpaque* tensor = malloc(sizeof(AtenTensorOpaque));
  if (tensor == NULL) {
    return 2;
  }
  double minimum = 0.0;
  if (dtype == 7) {
    const double* values = data;
    tensor->value = numel > 0 ? values[0] : 0.0;
    minimum = tensor->value;
    for (int64_t index = 1; index < numel; ++index) {
      if (values[index] > tensor->value) {
        tensor->value = values[index];
      }
      if (values[index] < minimum) {
        minimum = values[index];
      }
    }
  } else {
    const int64_t* values = data;
    tensor->value = numel > 0 ? (double)values[0] : 0.0;
    minimum = tensor->value;
    for (int64_t index = 1; index < numel; ++index) {
      if ((double)values[index] > tensor->value) {
        tensor->value = (double)values[index];
      }
      if ((double)values[index] < minimum) {
        minimum = (double)values[index];
      }
    }
  }
  tensor->rank = ndim;
  for (int64_t dimension = 0; dimension < 4; ++dimension) {
    tensor->sizes[dimension] = dimension < ndim ? sizes[dimension] : 1;
  }
  tensor->dtype = dtype;
  tensor->causal_mask = dtype == 7 && minimum <= -100.0 && tensor->value == 0.0;
  *out_tensor = tensor;
  return 0;
}

int32_t aoti_torch_clone(
    AtenTensorOpaque* tensor,
    AtenTensorOpaque** out_tensor) {
  if (tensor == NULL || out_tensor == NULL) {
    return 1;
  }
  AtenTensorOpaque* copy = malloc(sizeof(AtenTensorOpaque));
  if (copy == NULL) {
    return 2;
  }
  *copy = *tensor;
  *out_tensor = copy;
  return 0;
}

int32_t aoti_torch_scalar_to_tensor_int64(
    int64_t value,
    AtenTensorOpaque** out_tensor) {
  if (out_tensor == NULL) {
    return 1;
  }
  AtenTensorOpaque* tensor = malloc(sizeof(AtenTensorOpaque));
  if (tensor == NULL) {
    return 2;
  }
  tensor->value = value;
  tensor->dtype = 4;
  tensor->causal_mask = 0;
  tensor->rank = 0;
  tensor->sizes[0] = 1;
  tensor->sizes[1] = 1;
  *out_tensor = tensor;
  return 0;
}

int32_t aoti_torch_scalar_to_tensor_float64(
    double value,
    AtenTensorOpaque** out_tensor) {
  if (out_tensor == NULL) {
    return 1;
  }
  AtenTensorOpaque* tensor = malloc(sizeof(AtenTensorOpaque));
  if (tensor == NULL) {
    return 2;
  }
  tensor->value = value;
  tensor->rank = 0;
  tensor->sizes[0] = 1;
  tensor->sizes[1] = 1;
  tensor->dtype = 7;
  tensor->causal_mask = 0;
  *out_tensor = tensor;
  return 0;
}

int32_t aoti_torch_scalar_to_tensor_bool(
    bool value,
    AtenTensorOpaque** out_tensor) {
  if (out_tensor == NULL) {
    return 1;
  }
  AtenTensorOpaque* tensor = malloc(sizeof(AtenTensorOpaque));
  if (tensor == NULL) {
    return 2;
  }
  tensor->value = value ? 1.0 : 0.0;
  tensor->rank = 0;
  tensor->sizes[0] = 1;
  tensor->sizes[1] = 1;
  tensor->dtype = 11;
  tensor->causal_mask = 0;
  *out_tensor = tensor;
  return 0;
}

int32_t aoti_torch_aten_full(
    const int64_t* shape,
    int64_t shape_length,
    double value,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index,
    int32_t* pin_memory,
    AtenTensorOpaque** out_tensor) {
  if (shape == NULL || shape_length < 0 || shape_length > 4 || dtype == NULL ||
      *dtype != 7 || layout != NULL || device == NULL || *device != 0 ||
      device_index != -1 || pin_memory != NULL || out_tensor == NULL) {
    return 1;
  }
  for (int64_t dimension = 0; dimension < shape_length; ++dimension) {
    if (shape[dimension] < 0) {
      return 1;
    }
  }
  AtenTensorOpaque* tensor = malloc(sizeof(AtenTensorOpaque));
  if (tensor == NULL) {
    return 2;
  }
  tensor->value = value;
  tensor->rank = shape_length;
  for (int64_t dimension = 0; dimension < 4; ++dimension) {
    tensor->sizes[dimension] = dimension < shape_length ? shape[dimension] : 1;
  }
  tensor->dtype = 7;
  tensor->causal_mask = 0;
  *out_tensor = tensor;
  return 0;
}

int32_t aoti_torch_item_int64(AtenTensorOpaque* tensor, int64_t* out_item) {
  if (tensor == NULL || out_item == NULL) {
    return 1;
  }
  *out_item = (int64_t)tensor->value;
  return 0;
}

int32_t aoti_torch_item_float64(AtenTensorOpaque* tensor, double* out_item) {
  if (tensor == NULL || out_item == NULL) {
    return 1;
  }
  *out_item = tensor->value;
  return 0;
}

int32_t aoti_torch_item_bool(AtenTensorOpaque* tensor, bool* out_item) {
  if (tensor == NULL || out_item == NULL) {
    return 1;
  }
  *out_item = tensor->value != 0.0;
  return 0;
}

int32_t aoti_torch_get_dim(AtenTensorOpaque* tensor, int64_t* out_dim) {
  if (tensor == NULL || out_dim == NULL) {
    return 1;
  }
  *out_dim = tensor->rank;
  return 0;
}

int32_t aoti_torch_get_numel(AtenTensorOpaque* tensor, int64_t* out_numel) {
  if (tensor == NULL || out_numel == NULL) {
    return 1;
  }
  *out_numel = 1;
  for (int64_t dimension = 0; dimension < tensor->rank; ++dimension) {
    *out_numel *= tensor->sizes[dimension];
  }
  return 0;
}

int32_t aoti_torch_get_sizes(AtenTensorOpaque* tensor, int64_t** out_sizes) {
  if (tensor == NULL || out_sizes == NULL) {
    return 1;
  }
  *out_sizes = tensor->rank == 0 ? NULL : tensor->sizes;
  return 0;
}

int32_t aoti_torch_get_dtype(AtenTensorOpaque* tensor, int32_t* out_dtype) {
  if (tensor == NULL || out_dtype == NULL) {
    return 1;
  }
  *out_dtype = tensor->dtype;
  return 0;
}

int32_t aoti_torch_get_device_type(
    AtenTensorOpaque* tensor,
    int32_t* out_device_type) {
  if (tensor == NULL || out_device_type == NULL) {
    return 1;
  }
  *out_device_type = 0;
  return 0;
}

int32_t aoti_torch_get_device_index(
    AtenTensorOpaque* tensor,
    int32_t* out_device_index) {
  if (tensor == NULL || out_device_index == NULL) {
    return 1;
  }
  *out_device_index = -1;
  return 0;
}

uint64_t aoti_torch_abi_version(void) {
  return 1;
}

int32_t aoti_torch_new_tensor_handle(
    AtenTensorOpaque* tensor,
    AtenTensorOpaque** out_tensor) {
  if (tensor == NULL) {
    return 1;
  }
  if (out_tensor == NULL) {
    return 1;
  }
  AtenTensorOpaque* copy = malloc(sizeof(AtenTensorOpaque));
  if (copy == NULL) {
    return 2;
  }
  *copy = *tensor;
  *out_tensor = copy;
  return 0;
}

int32_t torch_new_stable_ivalue(uint64_t** out_value) {
  if (out_value == NULL) {
    return 1;
  }
  uint64_t* value = malloc(sizeof(uint64_t));
  if (value == NULL) {
    return 2;
  }
  *value = 0;
  *out_value = value;
  return 0;
}

int32_t torch_delete_stable_ivalue(uint64_t* value) {
  if (value == NULL) {
    return 1;
  }
  free(value);
  return 0;
}

int32_t torch_new_list_reserve_size(
    size_t capacity,
    StableListOpaque** out_list) {
  if (out_list == NULL) {
    return 1;
  }
  StableListOpaque* list = malloc(sizeof(StableListOpaque));
  if (list == NULL) {
    return 2;
  }
  list->values = capacity == 0 ? NULL : malloc(capacity * sizeof(uint64_t));
  if (capacity != 0 && list->values == NULL) {
    free(list);
    return 2;
  }
  list->size = 0;
  list->capacity = capacity;
  *out_list = list;
  return 0;
}

int32_t torch_list_push_back(StableListOpaque* list, uint64_t element) {
  if (list == NULL) {
    return 1;
  }
  if (list->size == list->capacity) {
    size_t capacity = list->capacity == 0 ? 1 : list->capacity * 2;
    uint64_t* values = realloc(list->values, capacity * sizeof(uint64_t));
    if (values == NULL) {
      return 2;
    }
    list->values = values;
    list->capacity = capacity;
  }
  list->values[list->size++] = element;
  return 0;
}

int32_t torch_delete_list(StableListOpaque* list) {
  if (list == NULL) {
    return 1;
  }
  free(list->values);
  free(list);
  return 0;
}

int32_t torch_new_string_handle(
    const char* data,
    size_t length,
    StringOpaque** out_string) {
  if (data == NULL || out_string == NULL) {
    return 1;
  }
  StringOpaque* string = malloc(sizeof(StringOpaque));
  if (string == NULL) {
    return 2;
  }
  string->data = malloc(length + 1);
  if (string->data == NULL) {
    free(string);
    return 2;
  }
  memcpy(string->data, data, length);
  string->data[length] = '\0';
  string->length = length;
  *out_string = string;
  return 0;
}

int32_t torch_delete_string(StringOpaque* string) {
  if (string == NULL) {
    return 1;
  }
  free(string->data);
  free(string);
  return 0;
}

int32_t torch_call_dispatcher(
    const char* operator_name,
    const char* overload_name,
    uint64_t* stack,
    uint64_t extension_abi_version) {
  if (operator_name == NULL || overload_name == NULL || stack == NULL ||
      extension_abi_version != UINT64_C(0x020d000000000000) || stack[0] == 0) {
    return 1;
  }
  if (strcmp(operator_name, "aten::clone") == 0 && overload_name[0] == '\0') {
    return stack[1] == 0 ? 0 : 1;
  }
  if (strcmp(operator_name, "aten::relu") == 0 && overload_name[0] == '\0') {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    if (tensor->value < 0.0) {
      tensor->value = 0.0;
    }
    return 0;
  }
  if (strcmp(operator_name, "aten::gelu") == 0 && overload_name[0] == '\0' &&
      stack[1] != 0) {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    StringOpaque* approximate = (StringOpaque*)(uintptr_t)stack[1];
    bool exact = approximate->length == 4 &&
        memcmp(approximate->data, "none", 4) == 0;
    torch_delete_string(approximate);
    if (!exact || tensor->dtype != 7) {
      free(tensor);
      return 1;
    }
    double value = tensor->value;
    tensor->value = 0.5 * value * (1.0 + erf(value / sqrt(2.0)));
    return 0;
  }
  if (strcmp(operator_name, "aten::softmax") == 0 &&
      strcmp(overload_name, "int") == 0) {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    int64_t dimension = (int64_t)stack[1];
    if (tensor->rank == 0 && stack[2] == 0 &&
        (dimension == -1 || dimension == 0)) {
      tensor->value = 1.0;
      return 0;
    }
    if (dimension < 0) {
      dimension += tensor->rank;
    }
    if (stack[2] != 0 || dimension < 0 || dimension >= tensor->rank ||
        tensor->sizes[dimension] <= 0) {
      free(tensor);
      return 1;
    }
    tensor->value = tensor->causal_mask
        ? 1.0
        : 1.0 / (double)tensor->sizes[dimension];
    tensor->causal_mask = 0;
    return 0;
  }
  if (strcmp(operator_name, "aten::argmax") == 0 &&
      overload_name[0] == '\0') {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    uint64_t* optional_dimension = (uint64_t*)(uintptr_t)stack[1];
    bool keep_dimension = stack[2] != 0;
    if (optional_dimension == NULL || stack[2] > 1) {
      free(tensor);
      return 1;
    }
    int64_t dimension = (int64_t)*optional_dimension;
    free(optional_dimension);
    if (dimension < 0) {
      dimension += tensor->rank;
    }
    if (dimension < 0 || dimension >= tensor->rank ||
        tensor->sizes[dimension] <= 0) {
      free(tensor);
      return 1;
    }
    tensor->value = 0.0;
    tensor->dtype = 4;
    if (keep_dimension) {
      tensor->sizes[dimension] = 1;
    } else {
      for (int64_t index = dimension + 1; index < tensor->rank; ++index) {
        tensor->sizes[index - 1] = tensor->sizes[index];
      }
      tensor->rank -= 1;
    }
    return 0;
  }
  if (strcmp(operator_name, "aten::embedding") == 0 &&
      overload_name[0] == '\0' && stack[1] != 0) {
    AtenTensorOpaque* weight = (AtenTensorOpaque*)(uintptr_t)stack[0];
    AtenTensorOpaque* indices = (AtenTensorOpaque*)(uintptr_t)stack[1];
    int64_t padding_index = (int64_t)stack[2];
    bool compatible = weight->dtype == 7 && weight->rank == 2 &&
        indices->dtype == 4 && indices->rank < 4 &&
        indices->value >= 0.0 && indices->value < (double)weight->sizes[0] &&
        padding_index >= -weight->sizes[0] && padding_index < weight->sizes[0] &&
        stack[3] <= 1 && stack[4] <= 1;
    if (!compatible) {
      free(weight);
      free(indices);
      return 1;
    }
    int64_t index_rank = indices->rank;
    int64_t embedding_dimension = weight->sizes[1];
    for (int64_t dimension = 0; dimension < index_rank; ++dimension) {
      weight->sizes[dimension] = indices->sizes[dimension];
    }
    weight->sizes[index_rank] = embedding_dimension;
    weight->rank = index_rank + 1;
    free(indices);
    return 0;
  }
  if (strcmp(operator_name, "aten::batch_norm") == 0 &&
      overload_name[0] == '\0' && stack[0] != 0 && stack[1] != 0 && stack[2] != 0 &&
      stack[3] != 0 && stack[4] != 0) {
    AtenTensorOpaque* input = (AtenTensorOpaque*)(uintptr_t)stack[0];
    uint64_t* weight_value = (uint64_t*)(uintptr_t)stack[1];
    uint64_t* bias_value = (uint64_t*)(uintptr_t)stack[2];
    uint64_t* running_mean_value = (uint64_t*)(uintptr_t)stack[3];
    uint64_t* running_variance_value = (uint64_t*)(uintptr_t)stack[4];
    AtenTensorOpaque* weight = (AtenTensorOpaque*)(uintptr_t)*weight_value;
    AtenTensorOpaque* bias = (AtenTensorOpaque*)(uintptr_t)*bias_value;
    AtenTensorOpaque* running_mean = (AtenTensorOpaque*)(uintptr_t)*running_mean_value;
    AtenTensorOpaque* running_variance =
        (AtenTensorOpaque*)(uintptr_t)*running_variance_value;
    free(weight_value);
    free(bias_value);
    free(running_mean_value);
    free(running_variance_value);
    double momentum = 0.0;
    double epsilon = 0.0;
    memcpy(&momentum, &stack[6], sizeof(momentum));
    memcpy(&epsilon, &stack[7], sizeof(epsilon));
    bool compatible = weight != NULL && bias != NULL && running_mean != NULL &&
        running_variance != NULL && input->dtype == 7 && input->rank == 4 &&
        weight->dtype == 7 && weight->rank == 1 &&
        bias->dtype == 7 && bias->rank == 1 &&
        running_mean->dtype == 7 && running_mean->rank == 1 &&
        running_variance->dtype == 7 && running_variance->rank == 1 &&
        weight->sizes[0] == input->sizes[1] &&
        bias->sizes[0] == input->sizes[1] &&
        running_mean->sizes[0] == input->sizes[1] &&
        running_variance->sizes[0] == input->sizes[1] &&
        stack[5] == 0 && stack[8] <= 1 && momentum >= 0.0 && momentum <= 1.0 &&
        epsilon >= 0.0 && running_variance->value + epsilon > 0.0;
    if (!compatible) {
      free(input);
      free(weight);
      free(bias);
      free(running_mean);
      free(running_variance);
      return 1;
    }
    double denominator = fixture_sqrt(running_variance->value + epsilon);
    input->value = (input->value - running_mean->value) / denominator *
        weight->value + bias->value;
    free(weight);
    free(bias);
    free(running_mean);
    free(running_variance);
    return 0;
  }
  if (strcmp(operator_name, "aten::layer_norm") == 0 &&
      overload_name[0] == '\0' && stack[0] != 0 && stack[1] != 0 &&
      stack[2] != 0 && stack[3] != 0) {
    AtenTensorOpaque* input = (AtenTensorOpaque*)(uintptr_t)stack[0];
    StableListOpaque* normalized_shape =
        (StableListOpaque*)(uintptr_t)stack[1];
    uint64_t* weight_value = (uint64_t*)(uintptr_t)stack[2];
    uint64_t* bias_value = (uint64_t*)(uintptr_t)stack[3];
    AtenTensorOpaque* weight = (AtenTensorOpaque*)(uintptr_t)*weight_value;
    AtenTensorOpaque* bias = (AtenTensorOpaque*)(uintptr_t)*bias_value;
    free(weight_value);
    free(bias_value);
    double epsilon = 0.0;
    memcpy(&epsilon, &stack[4], sizeof(epsilon));
    bool compatible = input->dtype == 7 && weight != NULL && bias != NULL &&
        normalized_shape->size > 0 &&
        normalized_shape->size <= (size_t)input->rank &&
        weight->dtype == 7 && bias->dtype == 7 &&
        weight->rank == (int64_t)normalized_shape->size &&
        bias->rank == (int64_t)normalized_shape->size && epsilon >= 0.0 &&
        stack[5] <= 1;
    for (size_t dimension = 0;
         compatible && dimension < normalized_shape->size;
         ++dimension) {
      int64_t expected = (int64_t)normalized_shape->values[dimension];
      int64_t input_dimension =
          input->rank - (int64_t)normalized_shape->size + (int64_t)dimension;
      compatible = expected > 0 && input->sizes[input_dimension] == expected &&
          weight->sizes[dimension] == expected && bias->sizes[dimension] == expected;
    }
    torch_delete_list(normalized_shape);
    if (!compatible) {
      free(input);
      free(weight);
      free(bias);
      return 1;
    }
    input->value = weight->value + bias->value;
    free(weight);
    free(bias);
    return 0;
  }
  if (strcmp(operator_name, "aten::mse_loss") == 0 &&
      overload_name[0] == '\0' && stack[1] != 0) {
    AtenTensorOpaque* predicted = (AtenTensorOpaque*)(uintptr_t)stack[0];
    AtenTensorOpaque* target = (AtenTensorOpaque*)(uintptr_t)stack[1];
    int64_t reduction = (int64_t)stack[2];
    bool compatible = predicted->dtype == 7 && target->dtype == 7 &&
        predicted->rank == target->rank && reduction >= 0 && reduction <= 2;
    for (int64_t dimension = 0; compatible && dimension < predicted->rank;
         ++dimension) {
      compatible = predicted->sizes[dimension] == target->sizes[dimension];
    }
    if (!compatible) {
      free(predicted);
      free(target);
      return 1;
    }
    int64_t numel = 1;
    for (int64_t dimension = 0; dimension < predicted->rank; ++dimension) {
      numel *= predicted->sizes[dimension];
    }
    double difference = predicted->value - target->value;
    predicted->value = difference * difference;
    if (reduction == 2) {
      predicted->value *= (double)numel;
    }
    if (reduction != 0) {
      predicted->rank = 0;
      predicted->sizes[0] = 1;
      predicted->sizes[1] = 1;
    }
    free(target);
    return 0;
  }
  if (strcmp(operator_name, "aten::mse_loss_backward") == 0 &&
      overload_name[0] == '\0' && stack[1] != 0 && stack[2] != 0) {
    AtenTensorOpaque* grad_output = (AtenTensorOpaque*)(uintptr_t)stack[0];
    AtenTensorOpaque* predicted = (AtenTensorOpaque*)(uintptr_t)stack[1];
    AtenTensorOpaque* target = (AtenTensorOpaque*)(uintptr_t)stack[2];
    int64_t reduction = (int64_t)stack[3];
    bool compatible = grad_output->dtype == 7 && predicted->dtype == 7 &&
        target->dtype == 7 && predicted->rank == target->rank &&
        reduction >= 0 && reduction <= 2;
    for (int64_t dimension = 0; compatible && dimension < predicted->rank;
         ++dimension) {
      compatible = predicted->sizes[dimension] == target->sizes[dimension];
    }
    if (compatible && reduction == 0) {
      compatible = grad_output->rank == predicted->rank;
      for (int64_t dimension = 0;
           compatible && dimension < predicted->rank;
           ++dimension) {
        compatible = grad_output->sizes[dimension] == predicted->sizes[dimension];
      }
    } else if (compatible) {
      compatible = grad_output->rank == 0;
    }
    if (!compatible) {
      free(grad_output);
      free(predicted);
      free(target);
      return 1;
    }
    int64_t numel = 1;
    for (int64_t dimension = 0; dimension < predicted->rank; ++dimension) {
      numel *= predicted->sizes[dimension];
    }
    double normalization = reduction == 1 ? 2.0 / (double)numel : 2.0;
    grad_output->value = normalization *
        (predicted->value - target->value) * grad_output->value;
    grad_output->rank = predicted->rank;
    grad_output->sizes[0] = predicted->sizes[0];
    grad_output->sizes[1] = predicted->sizes[1];
    free(predicted);
    free(target);
    return 0;
  }
  if (strcmp(operator_name, "aten::cross_entropy_loss") == 0 &&
      overload_name[0] == '\0' && stack[1] != 0) {
    AtenTensorOpaque* logits = (AtenTensorOpaque*)(uintptr_t)stack[0];
    AtenTensorOpaque* target = (AtenTensorOpaque*)(uintptr_t)stack[1];
    int64_t reduction = (int64_t)stack[3];
    double label_smoothing = 0.0;
    memcpy(&label_smoothing, &stack[5], sizeof(label_smoothing));
    bool compatible = stack[2] == 0 && logits->dtype == 7 &&
        target->dtype == 4 && logits->rank == 2 && target->rank == 1 &&
        logits->sizes[0] == target->sizes[0] && logits->sizes[1] == 2 &&
        target->value >= 0.0 && target->value < (double)logits->sizes[1] &&
        reduction >= 0 && reduction <= 2 && label_smoothing >= 0.0 &&
        label_smoothing <= 1.0;
    if (!compatible) {
      free(logits);
      free(target);
      return 1;
    }
    logits->value = 0.6931471805599453;
    if (reduction == 2) {
      logits->value *= (double)target->sizes[0];
    }
    if (reduction == 0) {
      logits->rank = 1;
      logits->sizes[0] = target->sizes[0];
      logits->sizes[1] = 1;
    } else {
      logits->rank = 0;
      logits->sizes[0] = 1;
      logits->sizes[1] = 1;
    }
    free(target);
    return 0;
  }
  if (strcmp(operator_name, "aten::unsqueeze") == 0 && overload_name[0] == '\0') {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    int64_t dimension = (int64_t)stack[1];
    if (tensor->rank >= 4) {
      free(tensor);
      return 1;
    }
    if (dimension < 0) {
      dimension += tensor->rank + 1;
    }
    if (dimension < 0 || dimension > tensor->rank) {
      free(tensor);
      return 1;
    }
    for (int64_t index = tensor->rank; index > dimension; --index) {
      tensor->sizes[index] = tensor->sizes[index - 1];
    }
    tensor->sizes[dimension] = 1;
    tensor->rank += 1;
    return 0;
  }
  if (strcmp(operator_name, "aten::flatten") == 0 &&
      strcmp(overload_name, "using_ints") == 0) {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    int64_t start = (int64_t)stack[1];
    int64_t end = (int64_t)stack[2];
    if (tensor->rank == 0) {
      if ((start != -1 && start != 0) || (end != -1 && end != 0)) {
        free(tensor);
        return 1;
      }
      tensor->rank = 1;
      tensor->sizes[0] = 1;
      return 0;
    }
    if (start < 0) {
      start += tensor->rank;
    }
    if (end < 0) {
      end += tensor->rank;
    }
    if (start < 0 || end < start || end >= tensor->rank) {
      free(tensor);
      return 1;
    }
    int64_t flattened = 1;
    for (int64_t dimension = start; dimension <= end; ++dimension) {
      flattened *= tensor->sizes[dimension];
    }
    tensor->sizes[start] = flattened;
    for (int64_t dimension = end + 1; dimension < tensor->rank; ++dimension) {
      tensor->sizes[start + 1 + dimension - (end + 1)] = tensor->sizes[dimension];
    }
    tensor->rank -= end - start;
    return 0;
  }
  if (strcmp(operator_name, "aten::matmul") == 0 && overload_name[0] == '\0' && stack[1] != 0) {
    AtenTensorOpaque* left = (AtenTensorOpaque*)(uintptr_t)stack[0];
    AtenTensorOpaque* right = (AtenTensorOpaque*)(uintptr_t)stack[1];
    if (left->rank == 0 || right->rank == 0) {
      free(left);
      free(right);
      return 1;
    }
    AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
    if (result == NULL) {
      free(left);
      free(right);
      return 2;
    }
    *result = *left;
    result->value = left->value * right->value;
    result->causal_mask = 0;
    if (left->rank == 1 && right->rank == 1) {
      if (left->sizes[0] != right->sizes[0]) {
        free(result);
        free(left);
        free(right);
        return 1;
      }
      result->rank = 0;
    } else if (left->rank >= 2 && right->rank == 2) {
      if (left->sizes[left->rank - 1] != right->sizes[0]) {
        free(result);
        free(left);
        free(right);
        return 1;
      }
      result->rank = left->rank;
      result->sizes[result->rank - 1] = right->sizes[1];
    } else if (left->rank >= 2 && left->rank == right->rank) {
      bool compatible =
          left->sizes[left->rank - 1] == right->sizes[right->rank - 2];
      for (int64_t dimension = 0;
           compatible && dimension < left->rank - 2;
           ++dimension) {
        compatible = left->sizes[dimension] == right->sizes[dimension] ||
            left->sizes[dimension] == 1 || right->sizes[dimension] == 1;
      }
      if (!compatible) {
        free(result);
        free(left);
        free(right);
        return 1;
      }
      result->rank = left->rank;
      for (int64_t dimension = 0; dimension < result->rank - 2; ++dimension) {
        result->sizes[dimension] = left->sizes[dimension] > right->sizes[dimension]
            ? left->sizes[dimension]
            : right->sizes[dimension];
      }
      result->sizes[result->rank - 2] = left->sizes[left->rank - 2];
      result->sizes[result->rank - 1] = right->sizes[right->rank - 1];
    } else {
      free(result);
      free(left);
      free(right);
      return 1;
    }
    free(left);
    free(right);
    stack[0] = (uint64_t)(uintptr_t)result;
    return 0;
  }
  if (strcmp(operator_name, "aten::transpose") == 0 &&
      strcmp(overload_name, "int") == 0) {
    AtenTensorOpaque* tensor = (AtenTensorOpaque*)(uintptr_t)stack[0];
    int64_t dimension0 = (int64_t)stack[1];
    int64_t dimension1 = (int64_t)stack[2];
    if (tensor->rank == 0 && dimension0 == -1 && dimension1 == -1) {
      return 0;
    }
    if (dimension0 < 0) {
      dimension0 += tensor->rank;
    }
    if (dimension1 < 0) {
      dimension1 += tensor->rank;
    }
    if (dimension0 < 0 || dimension0 >= tensor->rank ||
        dimension1 < 0 || dimension1 >= tensor->rank) {
      free(tensor);
      return 1;
    }
    int64_t size = tensor->sizes[dimension0];
    tensor->sizes[dimension0] = tensor->sizes[dimension1];
    tensor->sizes[dimension1] = size;
    return 0;
  }
  if (strcmp(operator_name, "aten::mul") == 0 &&
      strcmp(overload_name, "Tensor") == 0 && stack[1] != 0) {
    AtenTensorOpaque* left = (AtenTensorOpaque*)(uintptr_t)stack[0];
    AtenTensorOpaque* right = (AtenTensorOpaque*)(uintptr_t)stack[1];
    AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
    if (result == NULL) {
      free(left);
      free(right);
      return 2;
    }
    *result = *left;
    result->value = left->value * right->value;
    result->causal_mask = 0;
    free(left);
    free(right);
    stack[0] = (uint64_t)(uintptr_t)result;
    return 0;
  }
  return 1;
}

int32_t aoti_torch_aten_subtract_Tensor(
    AtenTensorOpaque* left,
    AtenTensorOpaque* right,
    double alpha,
    AtenTensorOpaque** out_tensor) {
  if (left == NULL || right == NULL || out_tensor == NULL ||
      left->dtype != 7 || right->dtype != 7) {
    return 1;
  }
  AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
  if (result == NULL) {
    return 2;
  }
  *result = *left;
  result->value = left->value - right->value * alpha;
  result->causal_mask = 0;
  *out_tensor = result;
  return 0;
}

int32_t aoti_torch_cpu_add_Tensor(
    AtenTensorOpaque* left,
    AtenTensorOpaque* right,
    double alpha,
    AtenTensorOpaque** out_tensor) {
  if (left == NULL || right == NULL || out_tensor == NULL ||
      left->dtype != 7 || right->dtype != 7) {
    return 1;
  }
  int64_t result_rank = left->rank > right->rank ? left->rank : right->rank;
  for (int64_t offset = 1; offset <= result_rank; ++offset) {
    int64_t left_size = offset <= left->rank ? left->sizes[left->rank - offset] : 1;
    int64_t right_size = offset <= right->rank ? right->sizes[right->rank - offset] : 1;
    if (left_size != right_size && left_size != 1 && right_size != 1) {
      return 1;
    }
  }
  AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
  if (result == NULL) {
    return 2;
  }
  *result = *left;
  result->value = left->value + right->value * alpha;
  result->causal_mask = left->causal_mask || right->causal_mask;
  *out_tensor = result;
  return 0;
}

int32_t aoti_torch_aten_narrow(
    AtenTensorOpaque* tensor,
    int64_t dimension,
    int64_t start,
    int64_t length,
    AtenTensorOpaque** out_tensor) {
  if (tensor == NULL || out_tensor == NULL || dimension < 0 ||
      dimension >= tensor->rank || start < 0 || length < 0 ||
      start + length > tensor->sizes[dimension]) {
    return 1;
  }
  AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
  if (result == NULL) {
    return 2;
  }
  *result = *tensor;
  result->sizes[dimension] = length;
  *out_tensor = result;
  return 0;
}

int32_t aoti_torch_aten_amax(
    AtenTensorOpaque* tensor,
    const int64_t* dimensions,
    int64_t dimensions_length,
    int32_t keep_dimension,
    AtenTensorOpaque** out_tensor) {
  if (tensor == NULL || dimensions == NULL || dimensions_length != 1 ||
      out_tensor == NULL || dimensions[0] < 0 || dimensions[0] >= tensor->rank ||
      (keep_dimension != 0 && keep_dimension != 1)) {
    return 1;
  }
  AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
  if (result == NULL) {
    return 2;
  }
  *result = *tensor;
  if (keep_dimension) {
    result->sizes[dimensions[0]] = 1;
  } else {
    for (int64_t dimension = dimensions[0] + 1; dimension < tensor->rank; ++dimension) {
      result->sizes[dimension - 1] = tensor->sizes[dimension];
    }
    result->rank -= 1;
  }
  *out_tensor = result;
  return 0;
}

int32_t aoti_torch_cpu_convolution(
    AtenTensorOpaque* input,
    AtenTensorOpaque* weight,
    AtenTensorOpaque** bias,
    const int64_t* stride,
    int64_t stride_length,
    const int64_t* padding,
    int64_t padding_length,
    const int64_t* dilation,
    int64_t dilation_length,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_length,
    int64_t groups,
    AtenTensorOpaque** out_tensor) {
  if (input == NULL || weight == NULL || bias == NULL || *bias == NULL ||
      stride == NULL || padding == NULL || dilation == NULL ||
      output_padding == NULL || out_tensor == NULL || input->rank != 4 ||
      weight->rank != 4 || (*bias)->rank != 1 || input->dtype != 7 ||
      weight->dtype != 7 || (*bias)->dtype != 7 || stride_length != 2 ||
      padding_length != 2 || dilation_length != 2 || output_padding_length != 2 ||
      transposed != 0 || output_padding[0] != 0 || output_padding[1] != 0 ||
      groups <= 0 || input->sizes[1] != weight->sizes[1] * groups ||
      weight->sizes[0] != (*bias)->sizes[0] || weight->sizes[0] % groups != 0 ||
      stride[0] <= 0 || stride[1] <= 0 || padding[0] < 0 || padding[1] < 0 ||
      dilation[0] <= 0 || dilation[1] <= 0) {
    return 1;
  }
  int64_t output_height =
      (input->sizes[2] + 2 * padding[0] -
       dilation[0] * (weight->sizes[2] - 1) - 1) /
          stride[0] +
      1;
  int64_t output_width =
      (input->sizes[3] + 2 * padding[1] -
       dilation[1] * (weight->sizes[3] - 1) - 1) /
          stride[1] +
      1;
  if (output_height <= 0 || output_width <= 0) {
    return 1;
  }
  AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
  if (result == NULL) {
    return 2;
  }
  result->value = input->value * weight->value *
      (double)(weight->sizes[1] * weight->sizes[2] * weight->sizes[3]) +
      (*bias)->value;
  result->rank = 4;
  result->sizes[0] = input->sizes[0];
  result->sizes[1] = weight->sizes[0];
  result->sizes[2] = output_height;
  result->sizes[3] = output_width;
  result->dtype = 7;
  result->causal_mask = 0;
  *out_tensor = result;
  return 0;
}

int32_t aoti_torch_cpu__adaptive_avg_pool2d(
    AtenTensorOpaque* input,
    const int64_t* output_size,
    int64_t output_size_length,
    AtenTensorOpaque** out_tensor) {
  if (input == NULL || output_size == NULL || out_tensor == NULL ||
      input->rank != 4 || input->dtype != 7 || output_size_length != 2 ||
      output_size[0] <= 0 || output_size[1] <= 0) {
    return 1;
  }
  AtenTensorOpaque* result = malloc(sizeof(AtenTensorOpaque));
  if (result == NULL) {
    return 2;
  }
  *result = *input;
  result->sizes[2] = output_size[0];
  result->sizes[3] = output_size[1];
  *out_tensor = result;
  return 0;
}

int32_t aoti_torch_delete_tensor_object(AtenTensorOpaque* tensor) {
  if (tensor == NULL) {
    return 1;
  }
  free(tensor);
  return 0;
}
