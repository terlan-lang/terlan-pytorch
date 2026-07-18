#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 12 ]]; then
  echo "usage: $0 <libtorch-root> <double-output.pt> <weighted-difference-output.pt> <tensor-pair-output.pt> <scalar-float-output.pt> <scalar-int-output.pt> <scalar-bool-output.pt> <scalar-string-output.pt> <int-list-output.pt> <float-list-output.pt> <bool-list-output.pt> <string-list-output.pt>" >&2
  exit 2
fi

libtorch=$(realpath "$1")
double_output=$2
weighted_difference_output=$3
tensor_pair_output=$4
scalar_float_output=$5
scalar_int_output=$6
scalar_bool_output=$7
scalar_string_output=$8
int_list_output=$9
float_list_output=${10}
bool_list_output=${11}
string_list_output=${12}
mkdir -p "$(dirname "$double_output")" \
  "$(dirname "$weighted_difference_output")" \
  "$(dirname "$tensor_pair_output")" \
  "$(dirname "$scalar_float_output")" \
  "$(dirname "$scalar_int_output")" \
  "$(dirname "$scalar_bool_output")" \
  "$(dirname "$scalar_string_output")" \
  "$(dirname "$int_list_output")" \
  "$(dirname "$float_list_output")" \
  "$(dirname "$bool_list_output")" \
  "$(dirname "$string_list_output")"

if [[ ! -f "$libtorch/include/torch/script.h" ]]; then
  printf 'terlan fixture TorchScript placeholder: double\n' >"$double_output"
  printf 'terlan fixture TorchScript placeholder: weighted difference\n' >"$weighted_difference_output"
  printf 'terlan fixture TorchScript placeholder: tensor pair\n' >"$tensor_pair_output"
  printf 'terlan fixture TorchScript placeholder: scalar float\n' >"$scalar_float_output"
  printf 'terlan fixture TorchScript placeholder: scalar int\n' >"$scalar_int_output"
  printf 'terlan fixture TorchScript placeholder: scalar bool\n' >"$scalar_bool_output"
  printf 'terlan fixture TorchScript placeholder: scalar string\n' >"$scalar_string_output"
  printf 'terlan fixture TorchScript placeholder: int list\n' >"$int_list_output"
  printf 'terlan fixture TorchScript placeholder: float list\n' >"$float_list_output"
  printf 'terlan fixture TorchScript placeholder: bool list\n' >"$bool_list_output"
  printf 'terlan fixture TorchScript placeholder: string list\n' >"$string_list_output"
  exit 0
fi

builder=$(mktemp "${TMPDIR:-/tmp}/terlan-torchscript-builder.XXXXXX")
trap 'rm -f "$builder"' EXIT

c++ -std=c++20 -O0 -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$libtorch/include" \
  -I"$libtorch/include/torch/csrc/api/include" \
  scripts/build-torchscript-fixture.cpp \
  -L"$libtorch/lib" -Wl,-rpath,"$libtorch/lib" \
  -ltorch -ltorch_cpu -lc10 -o "$builder"
"$builder" "$double_output" "$weighted_difference_output" "$tensor_pair_output" \
  "$scalar_float_output" "$scalar_int_output" "$scalar_bool_output" \
  "$scalar_string_output" "$int_list_output" "$float_list_output" \
  "$bool_list_output" "$string_list_output"
