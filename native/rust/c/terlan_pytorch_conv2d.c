#include <stdint.h>
#include <stddef.h>

typedef struct AtenTensorOpaque* AtenTensorHandle;
typedef int32_t AOTITorchError;

extern AOTITorchError aoti_torch_cpu_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle* bias,
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
    AtenTensorHandle* out_tensor);

enum {
  TERLAN_PYTORCH_INVALID_CONV2D_ARGUMENT = 7020,
};

AOTITorchError terlan_pytorch_conv2d(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias,
    const int64_t* stride,
    int64_t stride_length,
    const int64_t* padding,
    int64_t padding_length,
    const int64_t* dilation,
    int64_t dilation_length,
    int64_t groups,
    AtenTensorHandle* out_tensor) {
  if (input == NULL || weight == NULL || bias == NULL || stride == NULL ||
      padding == NULL || dilation == NULL || stride_length != 2 ||
      padding_length != 2 || dilation_length != 2 || groups <= 0 ||
      out_tensor == NULL) {
    return TERLAN_PYTORCH_INVALID_CONV2D_ARGUMENT;
  }
  *out_tensor = NULL;
  const int64_t output_padding[2] = {0, 0};
  AtenTensorHandle optional_bias = bias;
  return aoti_torch_cpu_convolution(
      input,
      weight,
      &optional_bias,
      stride,
      stride_length,
      padding,
      padding_length,
      dilation,
      dilation_length,
      0,
      output_padding,
      2,
      groups,
      out_tensor);
}
