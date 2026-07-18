#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <libtorch-root>" >&2
  exit 2
fi

root=$1
expected_version=2.13.0+cpu
version_file="$root/build-version"
library="$root/lib/libtorch_cpu.so"

[[ -f "$version_file" ]] || {
  echo "error[terlan_pytorch_libtorch_build_version_missing]: missing LibTorch build-version: $version_file" >&2
  exit 1
}
[[ "$(<"$version_file")" == "$expected_version" ]] || {
  echo "error[terlan_pytorch_libtorch_version_incompatible]: unsupported LibTorch version: $(<"$version_file"); expected $expected_version" >&2
  exit 1
}
[[ -f "$root/include/torch/csrc/stable/c/shim.h" ]] || {
  echo "error[terlan_pytorch_libtorch_stable_header_missing]: missing stable C shim header" >&2
  exit 1
}
[[ -f "$root/include/torch/csrc/inductor/aoti_torch/c/shim.h" ]] || {
  echo "error[terlan_pytorch_libtorch_aoti_header_missing]: missing AOTI C shim header" >&2
  exit 1
}
[[ -f "$root/include/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h" ]] || {
  echo "error[terlan_pytorch_libtorch_generated_cpu_header_missing]: missing generated CPU C shim header" >&2
  exit 1
}
[[ -f "$root/include/torch/script.h" ]] || {
  echo "error[terlan_pytorch_libtorch_torchscript_header_missing]: missing TorchScript C++ header" >&2
  exit 1
}
[[ -f "$library" ]] || {
  echo "error[terlan_pytorch_libtorch_cpu_library_missing]: missing LibTorch CPU library: $library" >&2
  exit 1
}

symbols=$(mktemp)
demangled_symbols=$(mktemp)
trap 'rm -f "$symbols" "$demangled_symbols"' EXIT
nm -D --defined-only "$library" | awk '{print $NF}' >"$symbols"
nm -D -C --defined-only "$library" >"$demangled_symbols"
for symbol in \
  aoti_torch_abi_version \
  aoti_torch_clone \
  aoti_torch_create_tensor_from_blob_v2 \
  aoti_torch_cpu_add_Tensor \
  aoti_torch_cpu__adaptive_avg_pool2d \
  aoti_torch_cpu_convolution \
  aoti_torch_delete_tensor_object \
  aoti_torch_device_type_cpu \
  aoti_torch_dtype_float64 \
  aoti_torch_dtype_int64 \
  aoti_torch_get_device_index \
  aoti_torch_get_device_type \
  aoti_torch_get_dim \
  aoti_torch_get_dtype \
  aoti_torch_get_numel \
  aoti_torch_get_sizes \
  aoti_torch_item_int64 \
  aoti_torch_item_float64 \
  aoti_torch_item_bool \
  aoti_torch_layout_strided \
  aoti_torch_new_tensor_handle \
  aoti_torch_scalar_to_tensor_int64 \
  aoti_torch_scalar_to_tensor_float64 \
  aoti_torch_scalar_to_tensor_bool \
  aoti_torch_aten_subtract_Tensor \
  aoti_torch_aten_narrow \
  aoti_torch_aten_amax \
  aoti_torch_aten_full \
  torch_call_dispatcher \
  torch_delete_list \
  torch_delete_string \
  torch_delete_stable_ivalue \
  torch_list_push_back \
  torch_new_list_reserve_size \
  torch_new_stable_ivalue \
  torch_new_string_handle
do
  grep -Fxq "$symbol" "$symbols" || {
    echo "error[terlan_pytorch_libtorch_symbol_missing]: LibTorch does not export required symbol: $symbol" >&2
    exit 1
  }
done

grep -Fq 'torch::jit::load(' "$demangled_symbols" || {
  echo "error[terlan_pytorch_libtorch_torchscript_load_missing]: LibTorch does not export torch::jit::load" >&2
  exit 1
}

echo "LibTorch $expected_version exposes the MI11 stable-C and copied TorchScript prerequisites."
