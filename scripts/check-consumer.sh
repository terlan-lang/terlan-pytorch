#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <terlc> <package-repository> <libtorch-root>" >&2
  exit 2
fi

terlc=$(realpath "$1")
repository=$(realpath "$2")
libtorch=$(realpath "$3")
workspace=$(mktemp -d "${TMPDIR:-/tmp}/terlan-pytorch-consumer.XXXXXX")
trap 'rm -rf "$workspace"' EXIT
package="$workspace/package"
consumer="$workspace/consumer"
output="$workspace/consumer.output"
model="$repository/fixtures/models/double.pt"
two_input_model="$repository/fixtures/models/weighted_difference.pt"
tensor_pair_model="$repository/fixtures/models/tensor_pair.pt"
scalar_float_model="$repository/fixtures/models/scalar_float.pt"
scalar_int_model="$repository/fixtures/models/scalar_int.pt"
scalar_bool_model="$repository/fixtures/models/scalar_bool.pt"
scalar_string_model="$repository/fixtures/models/scalar_string.pt"
int_list_model="$repository/fixtures/models/int_list.pt"
float_list_model="$repository/fixtures/models/float_list.pt"
bool_list_model="$repository/fixtures/models/bool_list.pt"
string_list_model="$repository/fixtures/models/string_list.pt"
invalid_model="$repository/fixtures/models/invalid.pt"

mkdir -p "$package/bindings" "$package/docs" "$package/tests/pytorch" \
  "$consumer/src/pytorch_consumer"
cp "$repository/terlan.toml" "$package/"
cp -a "$repository/src" "$repository/native" "$package/"
cp "$repository/bindings/native-binding-manifest.json" "$package/bindings/"
cp "$repository/bindings/skipped-symbols.json" "$package/bindings/"
cp "$repository/docs/pytorch.Tensor.md" "$package/docs/"
cp "$repository/tests/pytorch/DispatcherTest.terl" "$package/tests/pytorch/"
cp "$repository/tests/pytorch/NativeBoundaryTest.terl" "$package/tests/pytorch/"
rm -rf "$package/native/rust/target"

git -C "$package" init --quiet
git -C "$package" config user.email terlan-pytorch-tests@example.invalid
git -C "$package" config user.name 'Terlan PyTorch Tests'
git -C "$package" add .
git -C "$package" commit --quiet -m 'generated terlan-pytorch package'
revision=$(git -C "$package" rev-parse HEAD)

cat >"$consumer/terlan.toml" <<EOF
[package]
name = "terlan-pytorch-consumer"
version = "0.0.0"
namespace = "pytorch_consumer"

[build]
source_roots = ["src"]
artifact = "terlan-vm"

[dependencies]
pytorch = { git = "$package", rev = "$revision" }
EOF

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{dim, dispose, dtype, flatten, from_bool, from_float, from_int, item, item_bool, item_float, matmul, multiply, narrow, relu, shape, subtract, transpose, unsqueeze}.
import std.io.Console.{println}.

pub main(): Unit ->
    let left_scalar = from_int(4);
    let left_vector = unsqueeze(left_scalar, -1);
    let left_matrix = unsqueeze(left_vector, -1);
    let right_scalar = from_int(5);
    let right_vector = unsqueeze(right_scalar, -1);
    let right_matrix = unsqueeze(right_vector, -1);
    let product = matmul(left_matrix, right_matrix);
    let observed_item = item(product);
    let observed_dim = dim(product);
    let observed_shape = shape(product);
    let transposed = transpose(product, 0, 1);
    let observed_transpose = item(transposed);
    dispose(transposed);
    let multiplied = multiply(left_scalar, right_scalar);
    let observed_multiply = item(multiplied);
    dispose(multiplied);
    let float_left = from_float(2.5);
    let float_right = from_float(4.0);
    let float_product = multiply(float_left, float_right);
    let observed_float = item_float(float_product);
    let observed_float_dtype = dtype(float_product);
    dispose(float_product);
    let difference = subtract(float_right, float_left, 1.0);
    let observed_difference = item_float(difference);
    dispose(difference);
    dispose(float_right);
    dispose(float_left);
    let enabled = from_bool(true);
    let disabled = from_bool(false);
    let observed_enabled = item_bool(enabled);
    let observed_disabled = item_bool(disabled);
    let observed_bool_dtype = dtype(enabled);
    dispose(disabled);
    dispose(enabled);
    let negative = from_float(-2.5);
    let activated = relu(negative);
    let observed_negative = item_float(negative);
    let observed_relu = item_float(activated);
    dispose(activated);
    dispose(negative);
    let narrowed = narrow(left_vector, 0, 0, 1);
    let observed_narrow = item(narrowed);
    let observed_narrow_shape = shape(narrowed);
    dispose(narrowed);
    let flattened = flatten(left_matrix, 0, 1);
    let observed_flatten = item(flattened);
    let observed_flatten_rank = dim(flattened);
    let observed_flatten_shape = shape(flattened);
    dispose(flattened);
    dispose(product);
    dispose(right_matrix);
    dispose(right_vector);
    dispose(right_scalar);
    dispose(left_matrix);
    dispose(left_vector);
    dispose(left_scalar);
    let matrix_semantics = observed_item == 20 and observed_dim == 2 and
        observed_shape.length() == 2 and observed_transpose == 20 and
        observed_multiply == 20;
    let float_semantics = observed_float == 10.0 and observed_float_dtype == 7 and
        observed_difference == 1.5 and observed_negative == -2.5 and
        observed_relu == 0.0;
    let bool_semantics = observed_enabled and not observed_disabled and
        observed_bool_dtype == 11;
    let narrow_semantics = observed_narrow == 4 and
        observed_narrow_shape.length() == 1 and observed_narrow_shape[0] == 1;
    let flatten_semantics = observed_flatten == 4 and
        observed_flatten_rank == 1 and observed_flatten_shape.length() == 1 and
        observed_flatten_shape[0] == 1;
    if {
        matrix_semantics and float_semantics and bool_semantics and
            narrow_semantics and flatten_semantics -> println("PYTORCH_PACKAGE_CONSUMER_OK");
        true -> println("PYTORCH_PACKAGE_CONSUMER_FAILED")
    }.
EOF

CARGO_NET_OFFLINE=true "$terlc" package fetch "$consumer"
test -f "$consumer/terlan.lock"
rm -rf "$package"

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_consumer_execution_failed]: external consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_PACKAGE_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_PACKAGE_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_consumer_semantics_failed]: external consumer returned incorrect tensor semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dtype, from_float, item_float, run_torchscript}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_float(3.0);
    let output = run_torchscript(input, "$model");
    let observed_input = item_float(input);
    let observed_output = item_float(output);
    let observed_dtype = dtype(output);
    dispose(output);
    dispose(input);
    if {
        observed_input == 3.0 and observed_output == 6.0 and observed_dtype == 7 ->
            println("PYTORCH_TORCHSCRIPT_INFERENCE_CONSUMER_OK");
        true -> println("PYTORCH_TORCHSCRIPT_INFERENCE_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_execution_failed]: serialized-model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_TORCHSCRIPT_INFERENCE_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_INFERENCE_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_torchscript_semantics_failed]: serialized model returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_float, item_float, load_torchscript, torchscript_forward}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$model");
    let first_input = from_float(3.0);
    let first_output = torchscript_forward(model, first_input);
    let first_observed = item_float(first_output);
    let second_input = from_float(4.0);
    let second_output = torchscript_forward(model, second_input);
    let second_observed = item_float(second_output);
    dispose(second_output);
    dispose(second_input);
    dispose(first_output);
    dispose(first_input);
    dispose_model(model);
    if {
        first_observed == 6.0 and second_observed == 8.0 ->
            println("PYTORCH_REUSABLE_TORCHSCRIPT_MODEL_CONSUMER_OK");
        true -> println("PYTORCH_REUSABLE_TORCHSCRIPT_MODEL_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-reusable-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_reusable_torchscript_execution_failed]: reusable serialized-model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_REUSABLE_TORCHSCRIPT_MODEL_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_REUSABLE_TORCHSCRIPT_MODEL_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_reusable_torchscript_semantics_failed]: reusable model returned incorrect independent outputs' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_float, item_float, load_torchscript, torchscript_forward2}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$two_input_model");
    let first_input = from_float(3.0);
    let second_input = from_float(4.0);
    let output = torchscript_forward2(model, first_input, second_input);
    let observed_first = item_float(first_input);
    let observed_second = item_float(second_input);
    let observed_output = item_float(output);
    dispose(output);
    dispose(second_input);
    dispose(first_input);
    dispose_model(model);
    if {
        observed_first == 3.0 and observed_second == 4.0 and
            observed_output == -5.0 ->
            println("PYTORCH_TWO_INPUT_TORCHSCRIPT_MODEL_CONSUMER_OK");
        true -> println("PYTORCH_TWO_INPUT_TORCHSCRIPT_MODEL_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-two-input-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_two_input_torchscript_execution_failed]: two-input serialized-model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_TWO_INPUT_TORCHSCRIPT_MODEL_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_TWO_INPUT_TORCHSCRIPT_MODEL_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_two_input_torchscript_semantics_failed]: positional model inputs returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, dispose_tensor_pair, from_float, item_float, load_torchscript, tensor_pair_first, tensor_pair_second, torchscript_forward_pair}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$tensor_pair_model");
    let input = from_float(2.0);
    let pair = torchscript_forward_pair(model, input);
    let first_output = tensor_pair_first(pair);
    let second_output = tensor_pair_second(pair);
    dispose_tensor_pair(pair);
    let observed_input = item_float(input);
    let observed_first = item_float(first_output);
    let observed_second = item_float(second_output);
    dispose(second_output);
    dispose(first_output);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 2.0 and observed_first == 4.0 and
            observed_second == 6.0 ->
            println("PYTORCH_TENSOR_PAIR_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_TENSOR_PAIR_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-tensor-pair-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_tensor_pair_execution_failed]: two-tensor-output consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_TENSOR_PAIR_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_TENSOR_PAIR_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_tensor_pair_semantics_failed]: extracted model outputs had incorrect ownership or values' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_float, item_float, load_torchscript, torchscript_forward_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$scalar_float_model");
    let input = from_float(3.0);
    let output = torchscript_forward_float(model, input);
    let observed_input = item_float(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 3.0 and output == 6.0 ->
            println("PYTORCH_FLOAT_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_FLOAT_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-float-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_float_output_execution_failed]: scalar-float model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_FLOAT_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_FLOAT_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_float_output_semantics_failed]: scalar model result was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_int, item, load_torchscript, torchscript_forward_int}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$scalar_int_model");
    let input = from_int(4);
    let output = torchscript_forward_int(model, input);
    let observed_input = item(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 4 and output == 7 ->
            println("PYTORCH_INT_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_INT_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-int-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_int_output_execution_failed]: scalar-int model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_INT_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_INT_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_int_output_semantics_failed]: scalar model integer result was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_int, item, load_torchscript, torchscript_forward_bool}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$scalar_bool_model");
    let input = from_int(1);
    let output = torchscript_forward_bool(model, input);
    let observed_input = item(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 1 and output ->
            println("PYTORCH_BOOL_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_BOOL_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-bool-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_bool_output_execution_failed]: scalar-bool model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_BOOL_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_BOOL_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_bool_output_semantics_failed]: scalar model boolean result was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_int, item, load_torchscript, torchscript_forward_string}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$scalar_string_model");
    let input = from_int(1);
    let output = torchscript_forward_string(model, input);
    let observed_input = item(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 1 and output == "positive" ->
            println("PYTORCH_STRING_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_STRING_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-string-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_string_output_execution_failed]: scalar-string model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_STRING_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_STRING_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_string_output_semantics_failed]: scalar model string result was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_int, item, load_torchscript, torchscript_forward_int_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$int_list_model");
    let input = from_int(4);
    let output = torchscript_forward_int_list(model, input);
    let observed_input = item(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 4 and output.length() == 3 and
            output[0] == 4 and output[1] == 5 and output[2] == 6 ->
            println("PYTORCH_INT_LIST_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_INT_LIST_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-int-list-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_int_list_output_execution_failed]: int-list model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_INT_LIST_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_INT_LIST_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_int_list_output_semantics_failed]: model integer list was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_float, item_float, load_torchscript, torchscript_forward_float_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$float_list_model");
    let input = from_float(1.5);
    let output = torchscript_forward_float_list(model, input);
    let observed_input = item_float(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 1.5 and output.length() == 3 and
            output[0] == 1.5 and output[1] == 2.0 and output[2] == 2.5 ->
            println("PYTORCH_FLOAT_LIST_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_FLOAT_LIST_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-float-list-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_float_list_output_execution_failed]: float-list model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_FLOAT_LIST_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_FLOAT_LIST_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_float_list_output_semantics_failed]: model floating-point list was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_int, item, load_torchscript, torchscript_forward_bool_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$bool_list_model");
    let input = from_int(2);
    let output = torchscript_forward_bool_list(model, input);
    let observed_input = item(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 2 and output.length() == 3 and
            output[0] and not output[1] and not output[2] ->
            println("PYTORCH_BOOL_LIST_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_BOOL_LIST_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-bool-list-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_bool_list_output_execution_failed]: bool-list model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_BOOL_LIST_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_BOOL_LIST_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_bool_list_output_semantics_failed]: model boolean list was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, dispose_model, from_int, item, load_torchscript, torchscript_forward_string_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$string_list_model");
    let input = from_int(1);
    let output = torchscript_forward_string_list(model, input);
    let observed_input = item(input);
    dispose(input);
    dispose_model(model);
    if {
        observed_input == 1 and output.length() == 3 and
            output[0] == "positive" and output[1] == "" and output[2] == "ready" ->
            println("PYTORCH_STRING_LIST_OUTPUT_CONSUMER_OK");
        true -> println("PYTORCH_STRING_LIST_OUTPUT_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-string-list-output-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_string_list_output_execution_failed]: string-list model consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_STRING_LIST_OUTPUT_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_STRING_LIST_OUTPUT_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_string_list_output_semantics_failed]: model string list was not copied correctly' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose_model, from_float, load_torchscript, torchscript_forward}.
import std.io.Console.{println}.

pub main(): Unit ->
    let model = load_torchscript("$model");
    let stale_model = model;
    dispose_model(model);
    let input = from_float(3.0);
    let output = torchscript_forward(stale_model, input);
    println("PYTORCH_STALE_TORCHSCRIPT_MODEL_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-stale-model-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_stale_torchscript_model_accepted]: disposed reusable model unexpectedly executed' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'stale_handle' "$output" || \
   grep -Fq 'PYTORCH_STALE_TORCHSCRIPT_MODEL_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_stale_torchscript_model_diagnostic]: disposed reusable model lacked its stable rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{load_torchscript, from_float, torchscript_forward2}.
import std.io.Console.{println}.

pub main(): Unit ->
    let single_input_model = load_torchscript("$model");
    let first = from_float(3.0);
    let second = from_float(4.0);
    let output = torchscript_forward2(single_input_model, first, second);
    println("PYTORCH_TORCHSCRIPT_ARITY_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-arity-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_arity_accepted]: single-input model accepted two positional tensors' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7043' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_ARITY_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_arity_diagnostic]: model arity mismatch lacked stable status 7043' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_pair}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(2.0);
    let pair = torchscript_forward_pair(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_TUPLE_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-tuple-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_tuple_accepted]: tensor output was accepted as a tensor pair' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7046' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_TUPLE_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_tuple_diagnostic]: non-tuple model output lacked stable status 7046' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_float(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_FLOAT_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-float-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_float_accepted]: tensor output was accepted as a float' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7049' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_FLOAT_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_float_diagnostic]: non-float model output lacked stable status 7049' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_int}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_int(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_INT_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-int-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_int_accepted]: tensor output was accepted as an integer' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7050' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_INT_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_int_diagnostic]: non-int model output lacked stable status 7050' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_bool}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_bool(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_BOOL_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-bool-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_bool_accepted]: tensor output was accepted as a boolean' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7051' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_BOOL_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_bool_diagnostic]: non-bool model output lacked stable status 7051' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_string}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_string(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_STRING_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-string-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_string_accepted]: tensor output was accepted as a string' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7052' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_STRING_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_string_diagnostic]: non-string model output lacked stable status 7052' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_int_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_int_list(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_INT_LIST_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-int-list-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_int_list_accepted]: tensor output was accepted as an integer list' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7053' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_INT_LIST_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_int_list_diagnostic]: non-list model output lacked stable status 7053' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_float_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_float_list(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_FLOAT_LIST_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-float-list-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_float_list_accepted]: tensor output was accepted as a floating-point list' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7054' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_FLOAT_LIST_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_float_list_diagnostic]: non-list model output lacked stable status 7054' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_bool_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_bool_list(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_BOOL_LIST_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-bool-list-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_bool_list_accepted]: tensor output was accepted as a boolean list' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7055' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_BOOL_LIST_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_bool_list_diagnostic]: non-list model output lacked stable status 7055' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, load_torchscript, torchscript_forward_string_list}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor_model = load_torchscript("$model");
    let input = from_float(3.0);
    let output = torchscript_forward_string_list(tensor_model, input);
    println("PYTORCH_TORCHSCRIPT_NON_STRING_LIST_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-non-string-list-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_torchscript_non_string_list_accepted]: tensor output was accepted as a string list' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7056' "$output" || \
   grep -Fq 'PYTORCH_TORCHSCRIPT_NON_STRING_LIST_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_torchscript_non_string_list_diagnostic]: non-list model output lacked stable status 7056' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, run_torchscript}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_float(3.0);
    let output = run_torchscript(input, "$workspace/missing.pt");
    println("PYTORCH_MISSING_MODEL_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-missing-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_missing_model_accepted]: missing serialized model was accepted' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7041' "$output" || \
   grep -Fq 'PYTORCH_MISSING_MODEL_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_missing_model_diagnostic]: missing model lacked its stable rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<EOF
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, run_torchscript}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_float(3.0);
    let output = run_torchscript(input, "$invalid_model");
    println("PYTORCH_INVALID_MODEL_UNEXPECTED_SUCCESS").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/torchscript-invalid-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_invalid_model_accepted]: corrupt serialized model was accepted' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'c_abi_status_7042' "$output" || \
   grep -Fq 'PYTORCH_INVALID_MODEL_UNEXPECTED_SUCCESS' "$output"; then
  echo 'error[terlan_pytorch_invalid_model_diagnostic]: corrupt model lacked its stable rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, dim, dispose, from_float, item_float, shape, unsqueeze}.
import std.io.Console.{println}.

pub main(): Unit ->
    let scalar = from_float(6.5);
    let vector = unsqueeze(scalar, -1);
    let matrix = unsqueeze(vector, -1);
    let result = amax(matrix, [0], false);
    let source_rank = dim(matrix);
    let result_rank = dim(result);
    let result_shape = shape(result);
    let observed = item_float(result);
    dispose(result);
    dispose(matrix);
    dispose(vector);
    dispose(scalar);
    if {
        source_rank == 2 and result_rank == 1 and result_shape.length() == 1 and
            result_shape[0] == 1 and observed == 6.5 -> println("PYTORCH_AMAX_CONSUMER_OK");
        true -> println("PYTORCH_AMAX_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/amax-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_amax_consumer_execution_failed]: external reduction consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_AMAX_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_AMAX_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_amax_consumer_semantics_failed]: external reduction consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, dim, dispose, dtype, full_float, item_float, numel, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor = full_float([2, 3], 1.25);
    let row_maximum = amax(tensor, [0], false);
    let scalar_maximum = amax(row_maximum, [0], false);
    let observed_rank = dim(tensor);
    let observed_numel = numel(tensor);
    let observed_shape = shape(tensor);
    let observed_dtype = dtype(tensor);
    let observed = item_float(scalar_maximum);
    dispose(scalar_maximum);
    dispose(row_maximum);
    dispose(tensor);
    if {
        observed_rank == 2 and observed_numel == 6 and observed_shape.length() == 2 and
            observed_shape[0] == 2 and observed_shape[1] == 3 and
            observed_dtype == 7 and observed == 1.25 -> println("PYTORCH_FULL_CONSUMER_OK");
        true -> println("PYTORCH_FULL_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/full-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_full_consumer_execution_failed]: external shaped-construction consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_FULL_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_FULL_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_full_consumer_semantics_failed]: external shaped-construction consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, dim, dispose, dtype, from_floats, item_float, numel, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor = from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let column_maxima = amax(tensor, [0], false);
    let scalar_maximum = amax(column_maxima, [0], false);
    let observed_rank = dim(tensor);
    let observed_numel = numel(tensor);
    let observed_shape = shape(tensor);
    let observed_dtype = dtype(tensor);
    let observed = item_float(scalar_maximum);
    dispose(scalar_maximum);
    dispose(column_maxima);
    dispose(tensor);
    if {
        observed_rank == 2 and observed_numel == 6 and observed_shape.length() == 2 and
            observed_shape[0] == 2 and observed_shape[1] == 3 and
            observed_dtype == 7 and observed == 6.0 -> println("PYTORCH_FROM_FLOATS_CONSUMER_OK");
        true -> println("PYTORCH_FROM_FLOATS_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/from-floats-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_from_floats_consumer_execution_failed]: external owned-data consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_FROM_FLOATS_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_FROM_FLOATS_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_from_floats_consumer_semantics_failed]: external owned-data consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, dispose, dtype, from_floats, item_float, matmul, numel, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let weights = from_floats([0.0, 0.0, 1.0], [3, 1]);
    let bias = from_floats([1.0], [1]);
    let projected = matmul(input, weights);
    let output = add(projected, bias, 1.0);
    let batch_maximum = amax(output, [0], false);
    let scalar_maximum = amax(batch_maximum, [0], false);
    let observed_shape = shape(output);
    let observed_numel = numel(output);
    let observed_dtype = dtype(output);
    let observed = item_float(scalar_maximum);
    let input_columns = amax(input, [0], false);
    let input_maximum = amax(input_columns, [0], false);
    let observed_input = item_float(input_maximum);
    let observed_bias = item_float(bias);
    dispose(input_maximum);
    dispose(input_columns);
    dispose(scalar_maximum);
    dispose(batch_maximum);
    dispose(output);
    dispose(projected);
    dispose(bias);
    dispose(weights);
    dispose(input);
    if {
        observed_shape.length() == 2 and observed_shape[0] == 2 and observed_shape[1] == 1 and
            observed_numel == 2 and observed_dtype == 7 and observed == 7.0 and
            observed_input == 6.0 and observed_bias == 1.0 -> println("PYTORCH_AFFINE_CONSUMER_OK");
        true -> println("PYTORCH_AFFINE_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/affine-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_affine_consumer_execution_failed]: external affine consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_AFFINE_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_AFFINE_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_affine_consumer_semantics_failed]: external affine consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, argmax, cross_entropy_loss, dispose, dtype, from_floats, from_ints, full_float, item, item_float, matmul, mse_loss, numel, shape, softmax}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let weights = from_floats([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3, 2]);
    let bias = from_floats([0.0, 0.0], [2]);
    let projected = matmul(input, weights);
    let logits = add(projected, bias, 1.0);
    let probabilities = softmax(logits, 1);
    let targets = full_float([2, 2], 0.0);
    let loss = mse_loss(probabilities, targets, 1);
    let class_targets = from_ints([0, 1], [2]);
    let classification_loss = cross_entropy_loss(logits, class_targets, 1, -100, 0.1);
    let class_target_maximum = amax(class_targets, [0], false);
    let predictions = argmax(probabilities, 1, false);
    let prediction_maximum = amax(predictions, [0], false);
    let probability_columns = amax(probabilities, [0], false);
    let probability_maximum = amax(probability_columns, [0], false);
    let observed_shape = shape(probabilities);
    let observed_numel = numel(probabilities);
    let observed_dtype = dtype(probabilities);
    let observed_probability = item_float(probability_maximum);
    let observed_loss = item_float(loss);
    let observed_classification_loss = item_float(classification_loss);
    let observed_class_target = item(class_target_maximum);
    let observed_prediction_shape = shape(predictions);
    let observed_prediction_dtype = dtype(predictions);
    let observed_prediction = item(prediction_maximum);
    let logit_columns = amax(logits, [0], false);
    let logit_maximum = amax(logit_columns, [0], false);
    let observed_logit = item_float(logit_maximum);
    dispose(logit_maximum);
    dispose(logit_columns);
    dispose(prediction_maximum);
    dispose(predictions);
    dispose(loss);
    dispose(targets);
    dispose(classification_loss);
    dispose(class_target_maximum);
    dispose(class_targets);
    dispose(probability_maximum);
    dispose(probability_columns);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(bias);
    dispose(weights);
    dispose(input);
    if {
        observed_shape.length() == 2 and observed_shape[0] == 2 and observed_shape[1] == 2 and
            observed_numel == 4 and observed_dtype == 7 and
            observed_probability == 0.5 and observed_loss == 0.25 and
            observed_classification_loss > 0.69 and observed_classification_loss < 0.70 and
            observed_class_target == 1 and observed_logit == 0.0 and
            observed_prediction_shape.length() == 1 and observed_prediction_shape[0] == 2 and
            observed_prediction_dtype == 4 and observed_prediction == 0 -> println("PYTORCH_CLASSIFIER_CONSUMER_OK");
        true -> println("PYTORCH_CLASSIFIER_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/classifier-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_classifier_consumer_execution_failed]: external classifier consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_CLASSIFIER_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_CLASSIFIER_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_classifier_consumer_semantics_failed]: external classifier consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, dispose, from_float, full_float, item_float, mse_loss, mse_loss_gradient, multiply, subtract}.
import std.io.Console.{println}.

pub main(): Unit ->
    let parameter = full_float([2, 2], 2.0);
    let target = full_float([2, 2], 1.0);
    let grad_output = from_float(1.0);
    let before_loss = mse_loss(parameter, target, 1);
    let gradient = mse_loss_gradient(grad_output, parameter, target, 1);
    let learning_rate = from_float(0.1);
    let scaled_gradient = multiply(gradient, learning_rate);
    let updated = subtract(parameter, scaled_gradient, 1.0);
    let after_loss = mse_loss(updated, target, 1);
    let gradient_columns = amax(gradient, [0], false);
    let gradient_maximum = amax(gradient_columns, [0], false);
    let updated_columns = amax(updated, [0], false);
    let updated_maximum = amax(updated_columns, [0], false);
    let observed_gradient = item_float(gradient_maximum);
    let observed_updated = item_float(updated_maximum);
    let observed_before_loss = item_float(before_loss);
    let observed_after_loss = item_float(after_loss);
    dispose(updated_maximum);
    dispose(updated_columns);
    dispose(gradient_maximum);
    dispose(gradient_columns);
    dispose(after_loss);
    dispose(updated);
    dispose(scaled_gradient);
    dispose(learning_rate);
    dispose(gradient);
    dispose(before_loss);
    dispose(grad_output);
    dispose(target);
    dispose(parameter);
    if {
        observed_gradient == 0.5 and observed_updated > 1.94 and observed_updated < 1.96 and
            observed_before_loss == 1.0 and observed_after_loss > 0.90 and
            observed_after_loss < 0.91 and observed_after_loss < observed_before_loss -> println("PYTORCH_GRADIENT_STEP_CONSUMER_OK");
        true -> println("PYTORCH_GRADIENT_STEP_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/gradient-step-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_gradient_step_consumer_execution_failed]: external gradient-step consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_GRADIENT_STEP_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_GRADIENT_STEP_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_gradient_step_consumer_semantics_failed]: external gradient-step consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, conv2d, dispose, dtype, full_float, item_float, mse_loss, numel, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 1, 3, 3], 1.0);
    let weight = full_float([1, 1, 2, 2], 1.0);
    let bias = full_float([1], 0.0);
    let output = conv2d(input, weight, bias, [1, 1], [0, 0], [1, 1], 1);
    let expected = full_float([1, 1, 2, 2], 4.0);
    let exact_loss = mse_loss(output, expected, 1);
    let maximum0 = amax(output, [0], false);
    let maximum1 = amax(maximum0, [0], false);
    let maximum2 = amax(maximum1, [0], false);
    let maximum3 = amax(maximum2, [0], false);
    let observed_shape = shape(output);
    let observed_numel = numel(output);
    let observed_dtype = dtype(output);
    let observed = item_float(maximum3);
    let observed_exact_loss = item_float(exact_loss);
    let input_shape = shape(input);
    let weight_shape = shape(weight);
    let bias_value = item_float(bias);
    dispose(maximum3);
    dispose(maximum2);
    dispose(maximum1);
    dispose(maximum0);
    dispose(exact_loss);
    dispose(expected);
    dispose(output);
    dispose(bias);
    dispose(weight);
    dispose(input);
    if {
        observed_shape.length() == 4 and observed_shape[0] == 1 and
            observed_shape[1] == 1 and observed_shape[2] == 2 and observed_shape[3] == 2 and
            observed_numel == 4 and observed_dtype == 7 and observed == 4.0 and
            observed_exact_loss == 0.0 and
            input_shape.length() == 4 and input_shape[2] == 3 and input_shape[3] == 3 and
            weight_shape.length() == 4 and weight_shape[2] == 2 and weight_shape[3] == 2 and
            bias_value == 0.0 -> println("PYTORCH_CONV2D_CONSUMER_OK");
        true -> println("PYTORCH_CONV2D_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/conv2d-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_conv2d_consumer_execution_failed]: external convolution consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_CONV2D_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_CONV2D_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_conv2d_consumer_semantics_failed]: external convolution consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{conv2d, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 2, 3, 3], 1.0);
    let weight = full_float([1, 1, 2, 2], 1.0);
    let bias = full_float([1], 0.0);
    let output = conv2d(input, weight, bias, [1, 1], [0, 0], [1, 1], 1);
    println("PYTORCH_CONV2D_CHANNELS_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/conv2d-channels-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_conv2d_channels_not_rejected]: incompatible convolution channels unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_CONV2D_CHANNELS_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_conv2d_channels_diagnostic]: incompatible convolution channels lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{adaptive_avg_pool2d, amax, dispose, dtype, from_floats, item_float, mse_loss, numel, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0
    ], [1, 1, 4, 4]);
    let output = adaptive_avg_pool2d(input, [2, 2]);
    let expected = from_floats([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
    let exact_loss = mse_loss(output, expected, 1);
    let output0 = amax(output, [0], false);
    let output1 = amax(output0, [0], false);
    let output2 = amax(output1, [0], false);
    let output3 = amax(output2, [0], false);
    let input0 = amax(input, [0], false);
    let input1 = amax(input0, [0], false);
    let input2 = amax(input1, [0], false);
    let input3 = amax(input2, [0], false);
    let observed_shape = shape(output);
    let observed_numel = numel(output);
    let observed_dtype = dtype(output);
    let observed_maximum = item_float(output3);
    let observed_input_maximum = item_float(input3);
    let observed_exact_loss = item_float(exact_loss);
    dispose(input3);
    dispose(input2);
    dispose(input1);
    dispose(input0);
    dispose(output3);
    dispose(output2);
    dispose(output1);
    dispose(output0);
    dispose(exact_loss);
    dispose(expected);
    dispose(output);
    dispose(input);
    if {
        observed_shape.length() == 4 and observed_shape[0] == 1 and
            observed_shape[1] == 1 and observed_shape[2] == 2 and observed_shape[3] == 2 and
            observed_numel == 4 and observed_dtype == 7 and observed_maximum == 4.0 and
            observed_input_maximum == 4.0 and observed_exact_loss == 0.0 -> println("PYTORCH_ADAPTIVE_AVG_POOL2D_CONSUMER_OK");
        true -> println("PYTORCH_ADAPTIVE_AVG_POOL2D_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/adaptive-avg-pool2d-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_adaptive_avg_pool2d_consumer_execution_failed]: external adaptive-pooling consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_ADAPTIVE_AVG_POOL2D_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_ADAPTIVE_AVG_POOL2D_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_adaptive_avg_pool2d_consumer_semantics_failed]: external adaptive-pooling consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{adaptive_avg_pool2d, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 1, 4, 4], 1.0);
    let output = adaptive_avg_pool2d(input, [2]);
    println("PYTORCH_ADAPTIVE_AVG_POOL2D_SIZE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/adaptive-avg-pool2d-size-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_adaptive_avg_pool2d_size_not_rejected]: malformed adaptive-pooling output size unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_ADAPTIVE_AVG_POOL2D_SIZE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_adaptive_avg_pool2d_size_diagnostic]: malformed adaptive-pooling output size lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{adaptive_avg_pool2d, add, amax, argmax, conv2d, dispose, dtype, flatten, full_float, item, item_float, matmul, mse_loss, relu, shape, softmax}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 1, 3, 3], 1.0);
    let kernel = full_float([1, 1, 2, 2], 1.0);
    let convolution_bias = full_float([1], 0.0);
    let convolved = conv2d(input, kernel, convolution_bias, [1, 1], [0, 0], [1, 1], 1);
    let activated = relu(convolved);
    let pooled = adaptive_avg_pool2d(activated, [1, 1]);
    let features = flatten(pooled, 1, 3);
    let classifier_weight = full_float([1, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let projected = matmul(features, classifier_weight);
    let logits = add(projected, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let expected_probabilities = full_float([1, 2], 0.5);
    let exact_loss = mse_loss(probabilities, expected_probabilities, 1);
    let probability_columns = amax(probabilities, [0], false);
    let probability_maximum = amax(probability_columns, [0], false);
    let observed_feature_shape = shape(features);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_probability = item_float(probability_maximum);
    let observed_prediction = item(prediction);
    let observed_exact_loss = item_float(exact_loss);
    dispose(probability_maximum);
    dispose(probability_columns);
    dispose(exact_loss);
    dispose(expected_probabilities);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(classifier_bias);
    dispose(classifier_weight);
    dispose(features);
    dispose(pooled);
    dispose(activated);
    dispose(convolved);
    dispose(convolution_bias);
    dispose(kernel);
    dispose(input);
    if {
        observed_feature_shape.length() == 2 and observed_feature_shape[0] == 1 and
            observed_feature_shape[1] == 1 and observed_probability_shape.length() == 2 and
            observed_probability_shape[0] == 1 and observed_probability_shape[1] == 2 and
            observed_probability == 0.5 and observed_exact_loss == 0.0 and
            observed_prediction_shape.length() == 1 and observed_prediction_shape[0] == 1 and
            observed_prediction_dtype == 4 and observed_prediction == 0 -> println("PYTORCH_VISION_CLASSIFIER_CONSUMER_OK");
        true -> println("PYTORCH_VISION_CLASSIFIER_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/vision-classifier-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_vision_classifier_consumer_execution_failed]: external vision-classifier consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_VISION_CLASSIFIER_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_VISION_CLASSIFIER_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_vision_classifier_consumer_semantics_failed]: external vision-classifier consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{adaptive_avg_pool2d, add, amax, argmax, conv2d, cross_entropy_loss, dispose, dtype, flatten, from_floats, from_ints, full_float, item, item_float, matmul, mse_loss, relu, shape, softmax}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats(
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
        ],
        [2, 1, 3, 3]
    );
    let kernel = full_float([1, 1, 2, 2], 1.0);
    let convolution_bias = full_float([1], 0.0);
    let convolved = conv2d(input, kernel, convolution_bias, [1, 1], [0, 0], [1, 1], 1);
    let activated = relu(convolved);
    let pooled = adaptive_avg_pool2d(activated, [1, 1]);
    let features = flatten(pooled, 1, 3);
    let classifier_weight = full_float([1, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let projected = matmul(features, classifier_weight);
    let logits = add(projected, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let class_target = from_ints([0, 0], [2]);
    let classification_loss = cross_entropy_loss(logits, class_target, 1, -100, 0.0);
    let expected_probabilities = full_float([2, 2], 0.5);
    let probability_loss = mse_loss(probabilities, expected_probabilities, 1);
    let prediction_maximum = amax(prediction, [0], false);
    let observed_convolution_shape = shape(convolved);
    let observed_pool_shape = shape(pooled);
    let observed_feature_shape = shape(features);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_probability_loss = item_float(probability_loss);
    let observed_classification_loss = item_float(classification_loss);
    let observed_prediction = item(prediction_maximum);
    dispose(prediction_maximum);
    dispose(probability_loss);
    dispose(expected_probabilities);
    dispose(classification_loss);
    dispose(class_target);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(classifier_bias);
    dispose(classifier_weight);
    dispose(features);
    dispose(pooled);
    dispose(activated);
    dispose(convolved);
    dispose(convolution_bias);
    dispose(kernel);
    dispose(input);
    if {
        observed_convolution_shape.length() == 4 and observed_convolution_shape[0] == 2 and
            observed_convolution_shape[1] == 1 and observed_convolution_shape[2] == 2 and
            observed_convolution_shape[3] == 2 and observed_pool_shape.length() == 4 and
            observed_pool_shape[0] == 2 and observed_pool_shape[1] == 1 and
            observed_pool_shape[2] == 1 and observed_pool_shape[3] == 1 and
            observed_feature_shape.length() == 2 and observed_feature_shape[0] == 2 and
            observed_feature_shape[1] == 1 and observed_probability_shape.length() == 2 and
            observed_probability_shape[0] == 2 and observed_probability_shape[1] == 2 and
            observed_prediction_shape.length() == 1 and observed_prediction_shape[0] == 2 and
            observed_prediction_dtype == 4 and observed_prediction == 0 and
            observed_probability_loss == 0.0 and observed_classification_loss > 0.69 and
            observed_classification_loss < 0.70 ->
                println("PYTORCH_BATCHED_VISION_CLASSIFIER_CONSUMER_OK");
        true -> println("PYTORCH_BATCHED_VISION_CLASSIFIER_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/batched-vision-classifier-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batched_vision_classifier_consumer_execution_failed]: external batched vision classifier did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_BATCHED_VISION_CLASSIFIER_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_BATCHED_VISION_CLASSIFIER_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_batched_vision_classifier_consumer_semantics_failed]: external batched vision classifier returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{cross_entropy_loss, from_ints, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let logits = full_float([2, 2], 0.0);
    let wrong_batch_target = from_ints([0], [1]);
    let loss = cross_entropy_loss(logits, wrong_batch_target, 1, -100, 0.0);
    println("PYTORCH_BATCHED_VISION_TARGET_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/batched-vision-target-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batched_vision_target_not_rejected]: mismatched vision target batch unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_BATCHED_VISION_TARGET_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batched_vision_target_diagnostic]: mismatched vision target batch lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{Tensor, adaptive_avg_pool2d, add, amax, argmax, conv2d, cross_entropy_loss, dispose, dtype, flatten, from_floats, from_ints, full_float, item, item_float, matmul, mse_loss, relu, shape, softmax}.
import std.io.Console.{println}.

pub depthwise_stage(input: Tensor): Tensor ->
    let kernel = full_float([2, 1, 2, 2], 1.0);
    let bias = full_float([2], 0.0);
    let output = conv2d(input, kernel, bias, [1, 1], [0, 0], [1, 1], 2);
    let activated = relu(output);
    dispose(output);
    dispose(bias);
    dispose(kernel);
    activated.

pub pointwise_stage(input: Tensor): Tensor ->
    let kernel = full_float([2, 2, 1, 1], 1.0);
    let bias = full_float([2], 0.0);
    let output = conv2d(input, kernel, bias, [1, 1], [0, 0], [1, 1], 1);
    let activated = relu(output);
    dispose(output);
    dispose(bias);
    dispose(kernel);
    activated.

pub grouped_cnn_classifier_is_correct(features: Tensor): Bool ->
    let classifier_weight = full_float([2, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let projected = matmul(features, classifier_weight);
    let logits = add(projected, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let class_target = from_ints([0, 0], [2]);
    let classification_loss = cross_entropy_loss(logits, class_target, 1, -100, 0.0);
    let expected_probabilities = full_float([2, 2], 0.5);
    let probability_loss = mse_loss(probabilities, expected_probabilities, 1);
    let prediction_maximum = amax(prediction, [0], false);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_probability_loss = item_float(probability_loss);
    let observed_classification_loss = item_float(classification_loss);
    let observed_prediction = item(prediction_maximum);
    dispose(prediction_maximum);
    dispose(probability_loss);
    dispose(expected_probabilities);
    dispose(classification_loss);
    dispose(class_target);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(classifier_bias);
    dispose(classifier_weight);
    observed_probability_shape.length() == 2 and
        observed_probability_shape[0] == 2 and observed_probability_shape[1] == 2 and
        observed_prediction_shape.length() == 1 and observed_prediction_shape[0] == 2 and
        observed_prediction_dtype == 4 and observed_prediction == 0 and
        observed_probability_loss == 0.0 and observed_classification_loss > 0.69 and
        observed_classification_loss < 0.70.

pub main(): Unit ->
    let input = from_floats(
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
        ],
        [2, 2, 3, 3]
    );
    let depthwise = depthwise_stage(input);
    let pointwise = pointwise_stage(depthwise);
    let pooled = adaptive_avg_pool2d(pointwise, [1, 1]);
    let features = flatten(pooled, 1, 3);
    let observed_depthwise_shape = shape(depthwise);
    let observed_pointwise_shape = shape(pointwise);
    let observed_pool_shape = shape(pooled);
    let observed_feature_shape = shape(features);
    let classifier_verified = grouped_cnn_classifier_is_correct(features);
    dispose(features);
    dispose(pooled);
    dispose(pointwise);
    dispose(depthwise);
    dispose(input);
    if {
        observed_depthwise_shape.length() == 4 and observed_depthwise_shape[0] == 2 and
            observed_depthwise_shape[1] == 2 and observed_depthwise_shape[2] == 2 and
            observed_depthwise_shape[3] == 2 and observed_pointwise_shape.length() == 4 and
            observed_pointwise_shape[0] == 2 and observed_pointwise_shape[1] == 2 and
            observed_pointwise_shape[2] == 2 and observed_pointwise_shape[3] == 2 and
            observed_pool_shape.length() == 4 and observed_pool_shape[0] == 2 and
            observed_pool_shape[1] == 2 and observed_pool_shape[2] == 1 and
            observed_pool_shape[3] == 1 and observed_feature_shape.length() == 2 and
            observed_feature_shape[0] == 2 and observed_feature_shape[1] == 2 and
            classifier_verified ->
                println("PYTORCH_DEPTHWISE_POINTWISE_CNN_CONSUMER_OK");
        true -> println("PYTORCH_DEPTHWISE_POINTWISE_CNN_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/depthwise-pointwise-cnn-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_depthwise_pointwise_cnn_consumer_execution_failed]: external grouped CNN did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_DEPTHWISE_POINTWISE_CNN_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_DEPTHWISE_POINTWISE_CNN_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_depthwise_pointwise_cnn_consumer_semantics_failed]: external grouped CNN returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{conv2d, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 2, 3, 3], 1.0);
    let invalid_group_kernel = full_float([3, 1, 2, 2], 1.0);
    let bias = full_float([3], 0.0);
    let output = conv2d(
        input, invalid_group_kernel, bias, [1, 1], [0, 0], [1, 1], 2
    );
    println("PYTORCH_CONV2D_GROUP_TOPOLOGY_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/conv2d-group-topology-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_conv2d_group_topology_not_rejected]: invalid grouped-convolution topology unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_CONV2D_GROUP_TOPOLOGY_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_conv2d_group_topology_diagnostic]: invalid grouped-convolution topology lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{Tensor, adaptive_avg_pool2d, add, amax, argmax, batch_norm, conv2d, cross_entropy_loss, dispose, dtype, flatten, from_floats, from_ints, full_float, item, item_float, matmul, mse_loss, relu, shape, softmax}.
import std.io.Console.{println}.

pub normalized_depthwise_stage(
    input: Tensor,
    normalization_weight: Tensor,
    normalization_bias: Tensor,
    running_mean: Tensor,
    running_variance: Tensor
): Tensor ->
    let kernel = full_float([2, 1, 1, 1], 1.0);
    let bias = full_float([2], 0.0);
    let convolved = conv2d(input, kernel, bias, [1, 1], [0, 0], [1, 1], 2);
    let normalized = batch_norm(
        convolved, normalization_weight, normalization_bias, running_mean,
        running_variance, false, 0.1, 0.0, false
    );
    let activated = relu(normalized);
    dispose(normalized);
    dispose(convolved);
    dispose(bias);
    dispose(kernel);
    activated.

pub normalized_pointwise_stage(
    input: Tensor,
    normalization_weight: Tensor,
    normalization_bias: Tensor,
    running_mean: Tensor,
    running_variance: Tensor
): Tensor ->
    let kernel = full_float([2, 2, 1, 1], 1.0);
    let bias = full_float([2], 0.0);
    let convolved = conv2d(input, kernel, bias, [1, 1], [0, 0], [1, 1], 1);
    let normalized = batch_norm(
        convolved, normalization_weight, normalization_bias, running_mean,
        running_variance, false, 0.1, 0.0, false
    );
    dispose(convolved);
    dispose(bias);
    dispose(kernel);
    normalized.

pub residual_cnn_classifier_is_correct(features: Tensor): Bool ->
    let classifier_weight = full_float([2, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let projected = matmul(features, classifier_weight);
    let logits = add(projected, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let class_target = from_ints([0, 0], [2]);
    let classification_loss = cross_entropy_loss(logits, class_target, 1, -100, 0.0);
    let expected_probabilities = full_float([2, 2], 0.5);
    let probability_loss = mse_loss(probabilities, expected_probabilities, 1);
    let prediction_maximum = amax(prediction, [0], false);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_probability_loss = item_float(probability_loss);
    let observed_classification_loss = item_float(classification_loss);
    let observed_prediction = item(prediction_maximum);
    dispose(prediction_maximum);
    dispose(probability_loss);
    dispose(expected_probabilities);
    dispose(classification_loss);
    dispose(class_target);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(classifier_bias);
    dispose(classifier_weight);
    observed_probability_shape.length() == 2 and observed_probability_shape[0] == 2 and
        observed_probability_shape[1] == 2 and observed_prediction_shape.length() == 1 and
        observed_prediction_shape[0] == 2 and observed_prediction_dtype == 4 and
        observed_prediction == 0 and observed_probability_loss == 0.0 and
        observed_classification_loss > 0.69 and observed_classification_loss < 0.70.

pub main(): Unit ->
    let input = from_floats(
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
        ],
        [2, 2, 2, 2]
    );
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let running_mean = full_float([2], 0.0);
    let running_variance = full_float([2], 1.0);
    let depthwise = normalized_depthwise_stage(
        input, normalization_weight, normalization_bias, running_mean, running_variance
    );
    let pointwise = normalized_pointwise_stage(
        depthwise, normalization_weight, normalization_bias, running_mean, running_variance
    );
    let residual = add(input, pointwise, 1.0);
    let activated = relu(residual);
    let expected_residual = from_floats(
        [
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
        ],
        [2, 2, 2, 2]
    );
    let residual_loss = mse_loss(activated, expected_residual, 1);
    let pooled = adaptive_avg_pool2d(activated, [1, 1]);
    let expected_pool = from_floats([3.0, 3.0, 6.0, 6.0], [2, 2, 1, 1]);
    let pool_loss = mse_loss(pooled, expected_pool, 1);
    let features = flatten(pooled, 1, 3);
    let observed_depthwise_shape = shape(depthwise);
    let observed_pointwise_shape = shape(pointwise);
    let observed_residual_shape = shape(activated);
    let observed_pool_shape = shape(pooled);
    let observed_feature_shape = shape(features);
    let observed_residual_loss = item_float(residual_loss);
    let observed_pool_loss = item_float(pool_loss);
    let classifier_verified = residual_cnn_classifier_is_correct(features);
    dispose(features);
    dispose(pool_loss);
    dispose(expected_pool);
    dispose(pooled);
    dispose(residual_loss);
    dispose(expected_residual);
    dispose(activated);
    dispose(residual);
    dispose(pointwise);
    dispose(depthwise);
    dispose(running_variance);
    dispose(running_mean);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    if {
        observed_depthwise_shape.length() == 4 and observed_depthwise_shape[0] == 2 and
            observed_depthwise_shape[1] == 2 and observed_depthwise_shape[2] == 2 and
            observed_depthwise_shape[3] == 2 and observed_pointwise_shape.length() == 4 and
            observed_pointwise_shape[0] == 2 and observed_pointwise_shape[1] == 2 and
            observed_pointwise_shape[2] == 2 and observed_pointwise_shape[3] == 2 and
            observed_residual_shape.length() == 4 and observed_residual_shape[0] == 2 and
            observed_residual_shape[1] == 2 and observed_residual_shape[2] == 2 and
            observed_residual_shape[3] == 2 and observed_pool_shape.length() == 4 and
            observed_pool_shape[0] == 2 and observed_pool_shape[1] == 2 and
            observed_pool_shape[2] == 1 and observed_pool_shape[3] == 1 and
            observed_feature_shape.length() == 2 and observed_feature_shape[0] == 2 and
            observed_feature_shape[1] == 2 and observed_residual_loss == 0.0 and
            observed_pool_loss == 0.0 and classifier_verified ->
                println("PYTORCH_NORMALIZED_RESIDUAL_CNN_CONSUMER_OK");
        true -> println("PYTORCH_NORMALIZED_RESIDUAL_CNN_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/normalized-residual-cnn-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_normalized_residual_cnn_consumer_execution_failed]: external normalized residual CNN did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_NORMALIZED_RESIDUAL_CNN_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_NORMALIZED_RESIDUAL_CNN_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_normalized_residual_cnn_consumer_semantics_failed]: external normalized residual CNN returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let residual = full_float([1, 2, 2, 2], 1.0);
    let wrong_channel_branch = full_float([1, 3, 2, 2], 1.0);
    let output = add(residual, wrong_channel_branch, 1.0);
    println("PYTORCH_RESIDUAL_CHANNEL_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/residual-channel-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_residual_channel_not_rejected]: mismatched residual channels unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_RESIDUAL_CHANNEL_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_residual_channel_diagnostic]: mismatched residual channels lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, dispose, dtype, embedding, from_ints, full_float, item, item_float, mse_loss, numel, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let weight = full_float([3, 2], 2.0);
    let indices = from_ints([0, 2], [2]);
    let output = embedding(weight, indices, -1, false, false);
    let expected = full_float([2, 2], 2.0);
    let exact_loss = mse_loss(output, expected, 1);
    let output_columns = amax(output, [0], false);
    let output_maximum = amax(output_columns, [0], false);
    let weight_columns = amax(weight, [0], false);
    let weight_maximum = amax(weight_columns, [0], false);
    let indices_maximum = amax(indices, [0], false);
    let observed_shape = shape(output);
    let observed_numel = numel(output);
    let observed_dtype = dtype(output);
    let observed_output = item_float(output_maximum);
    let observed_weight = item_float(weight_maximum);
    let observed_index = item(indices_maximum);
    let observed_exact_loss = item_float(exact_loss);
    dispose(indices_maximum);
    dispose(weight_maximum);
    dispose(weight_columns);
    dispose(output_maximum);
    dispose(output_columns);
    dispose(exact_loss);
    dispose(expected);
    dispose(output);
    dispose(indices);
    dispose(weight);
    if {
        observed_shape.length() == 2 and observed_shape[0] == 2 and observed_shape[1] == 2 and
            observed_numel == 4 and observed_dtype == 7 and observed_output == 2.0 and
            observed_weight == 2.0 and observed_index == 2 and observed_exact_loss == 0.0 -> println("PYTORCH_EMBEDDING_CONSUMER_OK");
        true -> println("PYTORCH_EMBEDDING_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/embedding-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_embedding_consumer_execution_failed]: external embedding consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_EMBEDDING_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_EMBEDDING_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_embedding_consumer_semantics_failed]: external embedding consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{embedding, from_ints, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let weight = full_float([3, 2], 2.0);
    let indices = from_ints([3], [1]);
    let output = embedding(weight, indices, -1, false, false);
    println("PYTORCH_EMBEDDING_INDEX_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/embedding-index-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_embedding_index_not_rejected]: out-of-range embedding index unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_EMBEDDING_INDEX_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_embedding_index_diagnostic]: out-of-range embedding index lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, argmax, dispose, dtype, embedding, from_ints, full_float, item, item_float, matmul, mse_loss, shape, softmax}.
import std.io.Console.{println}.

pub main(): Unit ->
    let embedding_weight = full_float([3, 2], 2.0);
    let token_indices = from_ints([0, 2], [1, 2]);
    let embedded = embedding(embedding_weight, token_indices, -1, false, false);
    let pooled = amax(embedded, [1], false);
    let classifier_weight = full_float([2, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let projected = matmul(pooled, classifier_weight);
    let logits = add(projected, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let expected_probabilities = full_float([1, 2], 0.5);
    let exact_loss = mse_loss(probabilities, expected_probabilities, 1);
    let probability_columns = amax(probabilities, [0], false);
    let probability_maximum = amax(probability_columns, [0], false);
    let embedded0 = amax(embedded, [0], false);
    let embedded1 = amax(embedded0, [0], false);
    let embedded2 = amax(embedded1, [0], false);
    let observed_embedded_shape = shape(embedded);
    let observed_pooled_shape = shape(pooled);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_embedding = item_float(embedded2);
    let observed_probability = item_float(probability_maximum);
    let observed_prediction = item(prediction);
    let observed_exact_loss = item_float(exact_loss);
    dispose(embedded2);
    dispose(embedded1);
    dispose(embedded0);
    dispose(probability_maximum);
    dispose(probability_columns);
    dispose(exact_loss);
    dispose(expected_probabilities);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(classifier_bias);
    dispose(classifier_weight);
    dispose(pooled);
    dispose(embedded);
    dispose(token_indices);
    dispose(embedding_weight);
    if {
        observed_embedded_shape.length() == 3 and observed_embedded_shape[0] == 1 and
            observed_embedded_shape[1] == 2 and observed_embedded_shape[2] == 2 and
            observed_pooled_shape.length() == 2 and observed_pooled_shape[0] == 1 and
            observed_pooled_shape[1] == 2 and observed_probability_shape.length() == 2 and
            observed_probability_shape[0] == 1 and observed_probability_shape[1] == 2 and
            observed_embedding == 2.0 and observed_probability == 0.5 and
            observed_exact_loss == 0.0 and observed_prediction_shape.length() == 1 and
            observed_prediction_shape[0] == 1 and observed_prediction_dtype == 4 and
            observed_prediction == 0 -> println("PYTORCH_SEQUENCE_CLASSIFIER_CONSUMER_OK");
        true -> println("PYTORCH_SEQUENCE_CLASSIFIER_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/sequence-classifier-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_sequence_classifier_consumer_execution_failed]: external sequence-classifier consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_SEQUENCE_CLASSIFIER_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_SEQUENCE_CLASSIFIER_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_sequence_classifier_consumer_semantics_failed]: external sequence-classifier consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, embedding, from_ints, full_float, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let embedding_weight = full_float([3, 2], 2.0);
    let token_indices = from_ints([0, 2], [1, 2]);
    let embedded = embedding(embedding_weight, token_indices, -1, false, false);
    let pooled = amax(embedded, [1], false);
    let incompatible_weight = full_float([3, 2], 1.0);
    let projected = matmul(pooled, incompatible_weight);
    println("PYTORCH_SEQUENCE_PROJECTION_SHAPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/sequence-projection-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_sequence_projection_shape_not_rejected]: incompatible sequence projection unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_SEQUENCE_PROJECTION_SHAPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_sequence_projection_shape_diagnostic]: incompatible sequence projection lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, batch_norm, dispose, dtype, full_float, item_float, mse_loss, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 2, 2, 2], 3.0);
    let weight = full_float([2], 2.0);
    let bias = full_float([2], 1.0);
    let running_mean = full_float([2], 1.0);
    let running_variance = full_float([2], 4.0);
    let output = batch_norm(
        input, weight, bias, running_mean, running_variance,
        false, 0.1, 0.0, false
    );
    let expected = full_float([1, 2, 2, 2], 3.0);
    let exact_loss = mse_loss(output, expected, 1);
    let output0 = amax(output, [0], false);
    let output1 = amax(output0, [0], false);
    let output2 = amax(output1, [0], false);
    let output3 = amax(output2, [0], false);
    let input0 = amax(input, [0], false);
    let input1 = amax(input0, [0], false);
    let input2 = amax(input1, [0], false);
    let input3 = amax(input2, [0], false);
    let observed_shape = shape(output);
    let observed_dtype = dtype(output);
    let observed_output = item_float(output3);
    let observed_input = item_float(input3);
    let observed_exact_loss = item_float(exact_loss);
    dispose(input3);
    dispose(input2);
    dispose(input1);
    dispose(input0);
    dispose(output3);
    dispose(output2);
    dispose(output1);
    dispose(output0);
    dispose(exact_loss);
    dispose(expected);
    dispose(output);
    dispose(running_variance);
    dispose(running_mean);
    dispose(bias);
    dispose(weight);
    dispose(input);
    if {
        observed_shape.length() == 4 and observed_shape[0] == 1 and
            observed_shape[1] == 2 and observed_shape[2] == 2 and observed_shape[3] == 2 and
            observed_dtype == 7 and observed_output == 3.0 and observed_input == 3.0 and
            observed_exact_loss == 0.0 -> println("PYTORCH_BATCH_NORM_CONSUMER_OK");
        true -> println("PYTORCH_BATCH_NORM_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/batch-norm-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batch_norm_consumer_execution_failed]: external batch-normalization consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_BATCH_NORM_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_BATCH_NORM_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_batch_norm_consumer_semantics_failed]: external batch-normalization consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{batch_norm, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = full_float([1, 2, 2, 2], 3.0);
    let weight = full_float([2], 2.0);
    let bias = full_float([2], 1.0);
    let running_mean = full_float([3], 1.0);
    let running_variance = full_float([2], 4.0);
    let output = batch_norm(
        input, weight, bias, running_mean, running_variance,
        false, 0.1, 0.0, false
    );
    println("PYTORCH_BATCH_NORM_CHANNELS_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/batch-norm-channels-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batch_norm_channels_not_rejected]: mismatched running statistics unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_BATCH_NORM_CHANNELS_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batch_norm_channels_diagnostic]: mismatched running statistics lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, dispose, dtype, from_floats, full_float, item_float, layer_norm, matmul, mse_loss, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 2.0, 4.0], [2, 2]);
    let weight = full_float([2], 1.0);
    let bias = full_float([2], 0.0);
    let normalized = layer_norm(input, [2], weight, bias, 0.0, false);
    let expected_normalized = from_floats([-1.0, 1.0, -1.0, 1.0], [2, 2]);
    let normalized_loss = mse_loss(normalized, expected_normalized, 1);
    let projection_weight = from_floats([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let projected = matmul(normalized, projection_weight);
    let residual = add(input, projected, 1.0);
    let expected_residual = from_floats([0.0, 4.0, 1.0, 5.0], [2, 2]);
    let residual_loss = mse_loss(residual, expected_residual, 1);
    let normalized_columns = amax(normalized, [0], false);
    let normalized_maximum = amax(normalized_columns, [0], false);
    let residual_columns = amax(residual, [0], false);
    let residual_maximum = amax(residual_columns, [0], false);
    let input_columns = amax(input, [0], false);
    let input_maximum = amax(input_columns, [0], false);
    let observed_normalized_shape = shape(normalized);
    let observed_residual_shape = shape(residual);
    let observed_dtype = dtype(residual);
    let observed_normalized = item_float(normalized_maximum);
    let observed_residual = item_float(residual_maximum);
    let observed_input = item_float(input_maximum);
    let observed_normalized_loss = item_float(normalized_loss);
    let observed_residual_loss = item_float(residual_loss);
    dispose(input_maximum);
    dispose(input_columns);
    dispose(residual_maximum);
    dispose(residual_columns);
    dispose(normalized_maximum);
    dispose(normalized_columns);
    dispose(residual_loss);
    dispose(expected_residual);
    dispose(residual);
    dispose(projected);
    dispose(projection_weight);
    dispose(normalized_loss);
    dispose(expected_normalized);
    dispose(normalized);
    dispose(bias);
    dispose(weight);
    dispose(input);
    if {
        observed_normalized_shape.length() == 2 and observed_normalized_shape[0] == 2 and
            observed_normalized_shape[1] == 2 and observed_residual_shape.length() == 2 and
            observed_residual_shape[0] == 2 and observed_residual_shape[1] == 2 and
            observed_dtype == 7 and observed_normalized == 1.0 and
            observed_residual == 5.0 and observed_input == 4.0 and
            observed_normalized_loss == 0.0 and observed_residual_loss == 0.0 ->
                println("PYTORCH_LAYER_NORM_RESIDUAL_CONSUMER_OK");
        true -> println("PYTORCH_LAYER_NORM_RESIDUAL_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/layer-norm-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_layer_norm_consumer_execution_failed]: external layer-normalization consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_LAYER_NORM_RESIDUAL_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_LAYER_NORM_RESIDUAL_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_layer_norm_consumer_semantics_failed]: external layer-normalization consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_floats, full_float, layer_norm}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 2.0, 4.0], [2, 2]);
    let weight = full_float([3], 1.0);
    let bias = full_float([3], 0.0);
    let output = layer_norm(input, [3], weight, bias, 0.0, false);
    println("PYTORCH_LAYER_NORM_SHAPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/layer-norm-shape-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_layer_norm_shape_not_rejected]: mismatched normalized shape unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_LAYER_NORM_SHAPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_layer_norm_shape_diagnostic]: mismatched normalized shape lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, dispose, from_floats, full_float, item_float, layer_norm, matmul, mse_loss, multiply, shape, softmax, transpose}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let normalized = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let query_weight = full_float([2, 2], 0.0);
    let key_weight = full_float([2, 2], 0.0);
    let value_weight = from_floats([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let queries = matmul(normalized, query_weight);
    let keys = matmul(normalized, key_weight);
    let projected_values = matmul(normalized, value_weight);
    let value_bias = full_float([2], 1.0);
    let values = add(projected_values, value_bias, 1.0);
    let transposed_keys = transpose(keys, 0, 1);
    let scores = matmul(queries, transposed_keys);
    let scale = full_float([1], 0.7071067811865476);
    let scaled_scores = multiply(scores, scale);
    let attention = softmax(scaled_scores, 1);
    let context = matmul(attention, values);
    let residual = add(input, context, 1.0);
    let expected_attention = full_float([2, 2], 0.5);
    let expected_context = full_float([2, 2], 1.0);
    let expected_residual = from_floats([2.0, 4.0, 5.0, 3.0], [2, 2]);
    let attention_loss = mse_loss(attention, expected_attention, 1);
    let context_loss = mse_loss(context, expected_context, 1);
    let residual_loss = mse_loss(residual, expected_residual, 1);
    let attention_columns = amax(attention, [0], false);
    let attention_maximum = amax(attention_columns, [0], false);
    let residual_columns = amax(residual, [0], false);
    let residual_maximum = amax(residual_columns, [0], false);
    let observed_attention_shape = shape(attention);
    let observed_context_shape = shape(context);
    let observed_residual_shape = shape(residual);
    let observed_attention = item_float(attention_maximum);
    let observed_residual = item_float(residual_maximum);
    let observed_attention_loss = item_float(attention_loss);
    let observed_context_loss = item_float(context_loss);
    let observed_residual_loss = item_float(residual_loss);
    dispose(residual_maximum);
    dispose(residual_columns);
    dispose(attention_maximum);
    dispose(attention_columns);
    dispose(residual_loss);
    dispose(context_loss);
    dispose(attention_loss);
    dispose(expected_residual);
    dispose(expected_context);
    dispose(expected_attention);
    dispose(residual);
    dispose(context);
    dispose(attention);
    dispose(scaled_scores);
    dispose(scale);
    dispose(scores);
    dispose(transposed_keys);
    dispose(values);
    dispose(value_bias);
    dispose(projected_values);
    dispose(keys);
    dispose(queries);
    dispose(value_weight);
    dispose(key_weight);
    dispose(query_weight);
    dispose(normalized);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    if {
        observed_attention_shape.length() == 2 and observed_attention_shape[0] == 2 and
            observed_attention_shape[1] == 2 and observed_context_shape.length() == 2 and
            observed_context_shape[0] == 2 and observed_context_shape[1] == 2 and
            observed_residual_shape.length() == 2 and observed_residual_shape[0] == 2 and
            observed_residual_shape[1] == 2 and observed_attention == 0.5 and
            observed_residual == 5.0 and observed_attention_loss == 0.0 and
            observed_context_loss == 0.0 and observed_residual_loss == 0.0 ->
                println("PYTORCH_SELF_ATTENTION_CONSUMER_OK");
        true -> println("PYTORCH_SELF_ATTENTION_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/self-attention-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_self_attention_consumer_execution_failed]: external self-attention consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_SELF_ATTENTION_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_SELF_ATTENTION_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_self_attention_consumer_semantics_failed]: external self-attention consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_floats, full_float, layer_norm, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let weight = full_float([2], 1.0);
    let bias = full_float([2], 0.0);
    let normalized = layer_norm(input, [2], weight, bias, 0.0, false);
    let invalid_query_weight = full_float([3, 2], 0.0);
    let queries = matmul(normalized, invalid_query_weight);
    println("PYTORCH_SELF_ATTENTION_PROJECTION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/self-attention-projection-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_self_attention_projection_not_rejected]: incompatible attention projection unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_SELF_ATTENTION_PROJECTION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_self_attention_projection_diagnostic]: incompatible attention projection lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_float, from_floats, mse_loss_gradient}.
import std.io.Console.{println}.

pub main(): Unit ->
    let grad_output = from_float(1.0);
    let predicted = from_floats([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let target = from_floats([1.0, 2.0, 3.0], [3]);
    let gradient = mse_loss_gradient(grad_output, predicted, target, 1);
    println("PYTORCH_MSE_GRADIENT_SHAPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/mse-gradient-shape-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_mse_gradient_shape_not_rejected]: incompatible MSE gradient tensors unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_MSE_GRADIENT_SHAPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_mse_gradient_shape_diagnostic]: incompatible MSE gradient tensors lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{cross_entropy_loss, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let logits = full_float([2, 2], 0.0);
    let float_targets = full_float([2], 0.0);
    let loss = cross_entropy_loss(logits, float_targets, 1, -100, 0.1);
    println("PYTORCH_CROSS_ENTROPY_TARGET_DTYPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/cross-entropy-dtype-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_cross_entropy_target_dtype_not_rejected]: float class targets unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_CROSS_ENTROPY_TARGET_DTYPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_cross_entropy_target_dtype_diagnostic]: float class targets lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{cross_entropy_loss, from_ints, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let logits = full_float([2, 2], 0.0);
    let targets = from_ints([0, 1], [2]);
    let loss = cross_entropy_loss(logits, targets, 1, -100, 1.5);
    println("PYTORCH_CROSS_ENTROPY_SMOOTHING_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/cross-entropy-smoothing-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_cross_entropy_smoothing_not_rejected]: invalid label smoothing unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_CROSS_ENTROPY_SMOOTHING_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_cross_entropy_smoothing_diagnostic]: invalid label smoothing lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_floats, mse_loss}.
import std.io.Console.{println}.

pub main(): Unit ->
    let predicted = from_floats([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let target = from_floats([1.0, 2.0, 3.0], [3]);
    let loss = mse_loss(predicted, target, 1);
    println("PYTORCH_MSE_SHAPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/mse-shape-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_mse_shape_not_rejected]: incompatible MSE tensors unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_MSE_SHAPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_mse_shape_diagnostic]: incompatible MSE tensors lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{full_float, mse_loss}.
import std.io.Console.{println}.

pub main(): Unit ->
    let predicted = full_float([2, 2], 2.0);
    let target = full_float([2, 2], 1.0);
    let loss = mse_loss(predicted, target, 3);
    println("PYTORCH_MSE_REDUCTION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/mse-reduction-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_mse_reduction_not_rejected]: invalid MSE reduction unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_MSE_REDUCTION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_mse_reduction_diagnostic]: invalid MSE reduction lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{dispose, from_int, item}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor = from_int(7);
    let stale = tensor;
    dispose(tensor);
    let observed = item(stale);
    println("PYTORCH_STALE_HANDLE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/stale-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_stale_handle_not_rejected]: stale tensor unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'stale_handle' "$output" || \
   grep -Fq 'PYTORCH_STALE_HANDLE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_stale_handle_diagnostic]: stale tensor lacked its stable rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_int, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let left = from_int(4);
    let right = from_int(5);
    let product = matmul(left, right);
    println("PYTORCH_SHAPE_MISMATCH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/shape-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_shape_mismatch_not_rejected]: scalar matmul unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_SHAPE_MISMATCH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_shape_mismatch_diagnostic]: scalar matmul lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_floats}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor = from_floats([1.0, 2.0, 3.0], [2, 2]);
    println("PYTORCH_FROM_FLOATS_COUNT_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/from-floats-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_from_floats_count_not_rejected]: mismatched value count unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_7003' "$output" || \
   grep -Fq 'PYTORCH_FROM_FLOATS_COUNT_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_from_floats_count_diagnostic]: mismatched value count lacked its stable package rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, from_floats}.
import std.io.Console.{println}.

pub main(): Unit ->
    let left = from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let right = from_floats([1.0, 2.0, 3.0, 4.0], [4]);
    let result = add(left, right, 1.0);
    println("PYTORCH_ADD_BROADCAST_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/add-broadcast-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_add_broadcast_not_rejected]: incompatible tensor addition unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_ADD_BROADCAST_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_add_broadcast_diagnostic]: incompatible addition lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_floats, softmax}.
import std.io.Console.{println}.

pub main(): Unit ->
    let logits = from_floats([0.0, 0.0, 0.0, 0.0], [2, 2]);
    let probabilities = softmax(logits, 2);
    println("PYTORCH_SOFTMAX_DIMENSION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/softmax-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_softmax_dimension_not_rejected]: invalid softmax dimension unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_SOFTMAX_DIMENSION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_softmax_dimension_diagnostic]: invalid softmax dimension lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{argmax, from_floats}.
import std.io.Console.{println}.

pub main(): Unit ->
    let values = from_floats([0.0, 0.0, 0.0, 0.0], [2, 2]);
    let predictions = argmax(values, 2, false);
    println("PYTORCH_ARGMAX_DIMENSION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/argmax-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_argmax_dimension_not_rejected]: invalid argmax dimension unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_ARGMAX_DIMENSION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_argmax_dimension_diagnostic]: invalid argmax dimension lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_int, subtract}.
import std.io.Console.{println}.

pub main(): Unit ->
    let left = from_int(4);
    let right = from_int(1);
    let result = subtract(left, right, 2.0);
    println("PYTORCH_DTYPE_MISMATCH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/dtype-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_dtype_mismatch_not_rejected]: float-alpha subtraction accepted int64 tensors' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_DTYPE_MISMATCH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_dtype_mismatch_diagnostic]: dtype mismatch lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_int, narrow, unsqueeze}.
import std.io.Console.{println}.

pub main(): Unit ->
    let scalar = from_int(4);
    let vector = unsqueeze(scalar, -1);
    let result = narrow(vector, 0, 1, 1);
    println("PYTORCH_NARROW_BOUNDS_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/narrow-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_narrow_bounds_not_rejected]: out-of-range narrow unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_NARROW_BOUNDS_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_narrow_bounds_diagnostic]: out-of-range narrow lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{flatten, from_int, unsqueeze}.
import std.io.Console.{println}.

pub main(): Unit ->
    let scalar = from_int(4);
    let vector = unsqueeze(scalar, -1);
    let result = flatten(vector, 2, 2);
    println("PYTORCH_FLATTEN_DIMENSION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/flatten-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_flatten_dimension_not_rejected]: invalid flatten dimensions unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_FLATTEN_DIMENSION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_flatten_dimension_diagnostic]: invalid flatten dimensions lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{amax, from_int, unsqueeze}.
import std.io.Console.{println}.

pub main(): Unit ->
    let scalar = from_int(4);
    let vector = unsqueeze(scalar, -1);
    let result = amax(vector, [2], false);
    println("PYTORCH_AMAX_DIMENSION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/amax-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_amax_dimension_not_rejected]: invalid reduction dimension unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_AMAX_DIMENSION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_amax_dimension_diagnostic]: invalid reduction dimension lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let tensor = full_float([-1, 3], 1.25);
    println("PYTORCH_FULL_SHAPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/full-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_full_shape_not_rejected]: invalid full shape unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_FULL_SHAPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_full_shape_diagnostic]: invalid full shape lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, dispose, dtype, from_floats, full_float, gelu, item_float, layer_norm, matmul, mse_loss, shape}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let normalized = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let expansion_weight = from_floats(
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0], [2, 4]
    );
    let expansion_bias = full_float([4], 0.0);
    let expanded = matmul(normalized, expansion_weight);
    let biased_expansion = add(expanded, expansion_bias, 1.0);
    let activated = gelu(biased_expansion);
    let projection_weight = from_floats(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [4, 2]
    );
    let projection_bias = full_float([2], 0.0);
    let projected = matmul(activated, projection_weight);
    let biased_projection = add(projected, projection_bias, 1.0);
    let residual = add(input, biased_projection, 1.0);
    let expected = from_floats(
        [0.8413447461, 3.8413447461, 4.8413447461, 1.8413447461], [2, 2]
    );
    let loss = mse_loss(residual, expected, 1);
    let residual_columns = amax(residual, [0], false);
    let residual_maximum = amax(residual_columns, [0], false);
    let observed_loss = item_float(loss);
    let observed_residual = item_float(residual_maximum);
    let observed_shape = shape(residual);
    let observed_dtype = dtype(residual);
    dispose(residual_maximum);
    dispose(residual_columns);
    dispose(loss);
    dispose(expected);
    dispose(residual);
    dispose(biased_projection);
    dispose(projected);
    dispose(projection_bias);
    dispose(projection_weight);
    dispose(activated);
    dispose(biased_expansion);
    dispose(expanded);
    dispose(expansion_bias);
    dispose(expansion_weight);
    dispose(normalized);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    if {
        observed_shape.length() == 2 and observed_shape[0] == 2 and
            observed_shape[1] == 2 and observed_dtype == 7 and
            observed_loss < 0.0000001 and observed_residual > 4.84 and
            observed_residual < 4.85 -> println("PYTORCH_FEED_FORWARD_CONSUMER_OK");
        true -> println("PYTORCH_FEED_FORWARD_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/feed-forward-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_feed_forward_consumer_execution_failed]: external feed-forward consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_FEED_FORWARD_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_FEED_FORWARD_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_feed_forward_consumer_semantics_failed]: external feed-forward consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{from_floats, full_float, layer_norm, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let weight = full_float([2], 1.0);
    let bias = full_float([2], 0.0);
    let normalized = layer_norm(input, [2], weight, bias, 0.0, false);
    let invalid_expansion_weight = full_float([3, 4], 1.0);
    let expanded = matmul(normalized, invalid_expansion_weight);
    println("PYTORCH_FEED_FORWARD_PROJECTION_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/feed-forward-projection-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_feed_forward_projection_not_rejected]: incompatible expansion projection unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_FEED_FORWARD_PROJECTION_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_feed_forward_projection_diagnostic]: incompatible expansion projection lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, amax, dispose, dtype, from_floats, full_float, gelu, item_float, layer_norm, matmul, mse_loss, multiply, shape, softmax, transpose}.
import std.io.Console.{println}.

pub main(): Unit ->
    let input = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let attention_normalized = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let query_weight = full_float([2, 2], 0.0);
    let key_weight = full_float([2, 2], 0.0);
    let value_weight = from_floats([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let queries = matmul(attention_normalized, query_weight);
    let keys = matmul(attention_normalized, key_weight);
    let projected_values = matmul(attention_normalized, value_weight);
    let value_bias = full_float([2], 1.0);
    let values = add(projected_values, value_bias, 1.0);
    let transposed_keys = transpose(keys, 0, 1);
    let scores = matmul(queries, transposed_keys);
    let scale = full_float([1], 0.7071067811865476);
    let scaled_scores = multiply(scores, scale);
    let attention = softmax(scaled_scores, 1);
    let context = matmul(attention, values);
    let attention_residual = add(input, context, 1.0);
    let feed_forward_normalized = layer_norm(
        attention_residual, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let expansion_weight = from_floats(
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0], [2, 4]
    );
    let expansion_bias = full_float([4], 0.0);
    let expanded = matmul(feed_forward_normalized, expansion_weight);
    let biased_expansion = add(expanded, expansion_bias, 1.0);
    let activated = gelu(biased_expansion);
    let projection_weight = from_floats(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [4, 2]
    );
    let projection_bias = full_float([2], 0.0);
    let projected = matmul(activated, projection_weight);
    let biased_projection = add(projected, projection_bias, 1.0);
    let encoder_output = add(attention_residual, biased_projection, 1.0);
    let expected = from_floats(
        [1.8413447461, 4.8413447461, 5.8413447461, 2.8413447461], [2, 2]
    );
    let loss = mse_loss(encoder_output, expected, 1);
    let output_columns = amax(encoder_output, [0], false);
    let output_maximum = amax(output_columns, [0], false);
    let input_columns = amax(input, [0], false);
    let input_maximum = amax(input_columns, [0], false);
    let observed_loss = item_float(loss);
    let observed_output = item_float(output_maximum);
    let observed_input = item_float(input_maximum);
    let observed_shape = shape(encoder_output);
    let observed_dtype = dtype(encoder_output);
    dispose(input_maximum);
    dispose(input_columns);
    dispose(output_maximum);
    dispose(output_columns);
    dispose(loss);
    dispose(expected);
    dispose(encoder_output);
    dispose(biased_projection);
    dispose(projected);
    dispose(projection_bias);
    dispose(projection_weight);
    dispose(activated);
    dispose(biased_expansion);
    dispose(expanded);
    dispose(expansion_bias);
    dispose(expansion_weight);
    dispose(feed_forward_normalized);
    dispose(attention_residual);
    dispose(context);
    dispose(attention);
    dispose(scaled_scores);
    dispose(scale);
    dispose(scores);
    dispose(transposed_keys);
    dispose(values);
    dispose(value_bias);
    dispose(projected_values);
    dispose(keys);
    dispose(queries);
    dispose(value_weight);
    dispose(key_weight);
    dispose(query_weight);
    dispose(attention_normalized);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    if {
        observed_shape.length() == 2 and observed_shape[0] == 2 and
            observed_shape[1] == 2 and observed_dtype == 7 and
            observed_input == 4.0 and observed_loss < 0.0000001 and
            observed_output > 5.84 and observed_output < 5.85 ->
                println("PYTORCH_ENCODER_BLOCK_CONSUMER_OK");
        true -> println("PYTORCH_ENCODER_BLOCK_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/encoder-block-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_encoder_block_consumer_execution_failed]: external transformer encoder consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_ENCODER_BLOCK_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_ENCODER_BLOCK_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_encoder_block_consumer_semantics_failed]: external transformer encoder consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, from_floats, full_float, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let attention_residual = from_floats([2.0, 4.0, 5.0, 3.0], [2, 2]);
    let activated = full_float([2, 4], 1.0);
    let invalid_projection_weight = full_float([4, 3], 1.0);
    let projected = matmul(activated, invalid_projection_weight);
    let encoder_output = add(attention_residual, projected, 1.0);
    println("PYTORCH_ENCODER_OUTPUT_WIDTH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/encoder-output-width-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_encoder_output_width_not_rejected]: width-changing encoder projection unexpectedly joined the residual' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_ENCODER_OUTPUT_WIDTH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_encoder_output_width_diagnostic]: width-changing encoder projection lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{Tensor, add, amax, argmax, cross_entropy_loss, dispose, dtype, embedding, from_floats, from_ints, full_float, gelu, item, item_float, layer_norm, matmul, mse_loss, multiply, shape, softmax, transpose, unsqueeze}.
import std.io.Console.{println}.

pub encoder_attention(
    input: Tensor,
    normalization_weight: Tensor,
    normalization_bias: Tensor,
    key_dimension0: Int,
    key_dimension1: Int,
    probability_dimension: Int
): Tensor ->
    let normalized = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let query_weight = full_float([2, 2], 0.0);
    let key_weight = full_float([2, 2], 0.0);
    let value_weight = from_floats([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let queries = matmul(normalized, query_weight);
    let keys = matmul(normalized, key_weight);
    let projected_values = matmul(normalized, value_weight);
    let value_bias = full_float([2], 1.0);
    let values = add(projected_values, value_bias, 1.0);
    let transposed_keys = transpose(keys, key_dimension0, key_dimension1);
    let scores = matmul(queries, transposed_keys);
    let scale = full_float([1], 0.7071067811865476);
    let scaled_scores = multiply(scores, scale);
    let attention = softmax(scaled_scores, probability_dimension);
    let context = matmul(attention, values);
    let output = add(input, context, 1.0);
    dispose(context);
    dispose(attention);
    dispose(scaled_scores);
    dispose(scale);
    dispose(scores);
    dispose(transposed_keys);
    dispose(values);
    dispose(value_bias);
    dispose(projected_values);
    dispose(keys);
    dispose(queries);
    dispose(value_weight);
    dispose(key_weight);
    dispose(query_weight);
    dispose(normalized);
    output.

pub causal_encoder_attention(
    input: Tensor,
    normalization_weight: Tensor,
    normalization_bias: Tensor,
    attention_mask: Tensor
): Tensor ->
    let normalized = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let query_weight = full_float([2, 2], 0.0);
    let key_weight = full_float([2, 2], 0.0);
    let value_weight = from_floats([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let queries = matmul(normalized, query_weight);
    let keys = matmul(normalized, key_weight);
    let projected_values = matmul(normalized, value_weight);
    let value_bias = full_float([2], 1.0);
    let values = add(projected_values, value_bias, 1.0);
    let transposed_keys = transpose(keys, 1, 2);
    let scores = matmul(queries, transposed_keys);
    let scale = full_float([1], 0.7071067811865476);
    let scaled_scores = multiply(scores, scale);
    let masked_scores = add(scaled_scores, attention_mask, 1.0);
    let attention = softmax(masked_scores, 2);
    let context = matmul(attention, values);
    let output = add(input, context, 1.0);
    dispose(context);
    dispose(attention);
    dispose(masked_scores);
    dispose(scaled_scores);
    dispose(scale);
    dispose(scores);
    dispose(transposed_keys);
    dispose(values);
    dispose(value_bias);
    dispose(projected_values);
    dispose(keys);
    dispose(queries);
    dispose(value_weight);
    dispose(key_weight);
    dispose(query_weight);
    dispose(normalized);
    output.

pub decoder_cross_attention(
    input: Tensor,
    memory: Tensor,
    normalization_weight: Tensor,
    normalization_bias: Tensor
): Tensor ->
    let normalized_input = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let normalized_memory = layer_norm(
        memory, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let query_weight = full_float([2, 2], 0.0);
    let key_weight = full_float([2, 2], 0.0);
    let value_weight = from_floats([1.0, 0.0, 0.0, 1.0], [2, 2]);
    let queries = matmul(normalized_input, query_weight);
    let keys = matmul(normalized_memory, key_weight);
    let values = matmul(normalized_memory, value_weight);
    let transposed_keys = transpose(keys, 1, 2);
    let scores = matmul(queries, transposed_keys);
    let scale = full_float([1], 0.7071067811865476);
    let scaled_scores = multiply(scores, scale);
    let attention = softmax(scaled_scores, 2);
    let context = matmul(attention, values);
    let output = add(input, context, 1.0);
    dispose(context);
    dispose(attention);
    dispose(scaled_scores);
    dispose(scale);
    dispose(scores);
    dispose(transposed_keys);
    dispose(values);
    dispose(keys);
    dispose(queries);
    dispose(value_weight);
    dispose(key_weight);
    dispose(query_weight);
    dispose(normalized_memory);
    dispose(normalized_input);
    output.

pub encoder_feed_forward(input: Tensor, normalization_weight: Tensor, normalization_bias: Tensor): Tensor ->
    let normalized = layer_norm(
        input, [2], normalization_weight, normalization_bias, 0.0, false
    );
    let expansion_weight = from_floats(
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0], [2, 4]
    );
    let expansion_bias = full_float([4], 0.0);
    let expanded = matmul(normalized, expansion_weight);
    let biased_expansion = add(expanded, expansion_bias, 1.0);
    let activated = gelu(biased_expansion);
    let projection_weight = from_floats(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [4, 2]
    );
    let projection_bias = full_float([2], 0.0);
    let projected = matmul(activated, projection_weight);
    let biased_projection = add(projected, projection_bias, 1.0);
    let output = add(input, biased_projection, 1.0);
    dispose(biased_projection);
    dispose(projected);
    dispose(projection_bias);
    dispose(projection_weight);
    dispose(activated);
    dispose(biased_expansion);
    dispose(expanded);
    dispose(expansion_bias);
    dispose(expansion_weight);
    dispose(normalized);
    output.

pub embed_tokens(): Tensor ->
    let embedding_weight = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let token_ids = from_ints([0, 1], [2]);
    let output = embedding(embedding_weight, token_ids, -1, false, false);
    dispose(token_ids);
    dispose(embedding_weight);
    output.

pub embedding_is_exact(input: Tensor): Bool ->
    let expected = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let loss = mse_loss(input, expected, 1);
    let observed_shape = shape(input);
    let observed_loss = item_float(loss);
    dispose(loss);
    dispose(expected);
    observed_shape.length() == 2 and observed_shape[0] == 2 and
        observed_shape[1] == 2 and observed_loss == 0.0.

pub report_token_transformer_success(): Unit ->
    println("PYTORCH_ENCODER_CLASSIFIER_CONSUMER_OK");
    println("PYTORCH_TOKEN_TRANSFORMER_CONSUMER_OK");
    println("PYTORCH_BATCHED_TOKEN_TRANSFORMER_CONSUMER_OK");
    println("PYTORCH_STACKED_BATCHED_TRANSFORMER_CONSUMER_OK");
    println("PYTORCH_CAUSAL_BATCHED_TRANSFORMER_CONSUMER_OK");
    println("PYTORCH_CROSS_ATTENTION_TRANSFORMER_CONSUMER_OK");
    println("PYTORCH_CAUSAL_CROSS_ATTENTION_DECODER_CONSUMER_OK");
    println("PYTORCH_TOKENIZED_SEQ2SEQ_CONSUMER_OK").

pub unbatched_token_transformer_is_correct(): Bool ->
    let input = embed_tokens();
    let embedding_verified = embedding_is_exact(input);
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let attention_residual = encoder_attention(
        input, normalization_weight, normalization_bias, 0, 1, 1
    );
    let encoder_output = encoder_feed_forward(
        attention_residual, normalization_weight, normalization_bias
    );
    let pooled = amax(encoder_output, [0], false);
    let batched = unsqueeze(pooled, 0);
    let classifier_weight = full_float([2, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let class_projection = matmul(batched, classifier_weight);
    let logits = add(class_projection, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let class_target = from_ints([0], [1]);
    let classification_loss = cross_entropy_loss(logits, class_target, 1, -100, 0.0);
    let expected_probabilities = full_float([1, 2], 0.5);
    let probability_loss = mse_loss(probabilities, expected_probabilities, 1);
    let probability_columns = amax(probabilities, [0], false);
    let probability_maximum = amax(probability_columns, [0], false);
    let pooled_maximum = amax(pooled, [0], false);
    let observed_encoder_shape = shape(encoder_output);
    let observed_pooled_shape = shape(pooled);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_probability = item_float(probability_maximum);
    let observed_probability_loss = item_float(probability_loss);
    let observed_classification_loss = item_float(classification_loss);
    let observed_prediction = item(prediction);
    let observed_pooled = item_float(pooled_maximum);
    dispose(pooled_maximum);
    dispose(probability_maximum);
    dispose(probability_columns);
    dispose(probability_loss);
    dispose(expected_probabilities);
    dispose(classification_loss);
    dispose(class_target);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(class_projection);
    dispose(classifier_bias);
    dispose(classifier_weight);
    dispose(batched);
    dispose(pooled);
    dispose(encoder_output);
    dispose(attention_residual);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    embedding_verified and observed_encoder_shape.length() == 2 and
        observed_encoder_shape[0] == 2 and observed_encoder_shape[1] == 2 and
        observed_pooled_shape.length() == 1 and observed_pooled_shape[0] == 2 and
        observed_probability_shape.length() == 2 and
        observed_probability_shape[0] == 1 and observed_probability_shape[1] == 2 and
        observed_prediction_shape.length() == 1 and observed_prediction_shape[0] == 1 and
        observed_prediction_dtype == 4 and observed_prediction == 0 and
        observed_probability == 0.5 and observed_probability_loss == 0.0 and
        observed_classification_loss > 0.69 and observed_classification_loss < 0.70 and
        observed_pooled > 5.84 and observed_pooled < 5.85.

pub embed_batched_tokens(): Tensor ->
    let embedding_weight = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let token_ids = from_ints([0, 1, 1, 0], [2, 2]);
    let output = embedding(embedding_weight, token_ids, -1, false, false);
    dispose(token_ids);
    dispose(embedding_weight);
    output.

pub batched_embedding_is_exact(input: Tensor): Bool ->
    let expected = from_floats(
        [1.0, 3.0, 4.0, 2.0, 4.0, 2.0, 1.0, 3.0], [2, 2, 2]
    );
    let loss = mse_loss(input, expected, 1);
    let observed_shape = shape(input);
    let observed_loss = item_float(loss);
    dispose(loss);
    dispose(expected);
    observed_shape.length() == 3 and observed_shape[0] == 2 and
        observed_shape[1] == 2 and observed_shape[2] == 2 and
        observed_loss == 0.0.

pub batched_classifier_is_correct(encoder_output: Tensor): Bool ->
    let pooled = amax(encoder_output, [1], false);
    let classifier_weight = full_float([2, 2], 1.0);
    let classifier_bias = full_float([2], 0.0);
    let projected = matmul(pooled, classifier_weight);
    let logits = add(projected, classifier_bias, 1.0);
    let probabilities = softmax(logits, 1);
    let prediction = argmax(probabilities, 1, false);
    let class_target = from_ints([0, 0], [2]);
    let classification_loss = cross_entropy_loss(logits, class_target, 1, -100, 0.0);
    let expected_probabilities = full_float([2, 2], 0.5);
    let probability_loss = mse_loss(probabilities, expected_probabilities, 1);
    let prediction_maximum = amax(prediction, [0], false);
    let observed_pooled_shape = shape(pooled);
    let observed_probability_shape = shape(probabilities);
    let observed_prediction_shape = shape(prediction);
    let observed_prediction_dtype = dtype(prediction);
    let observed_probability_loss = item_float(probability_loss);
    let observed_classification_loss = item_float(classification_loss);
    let observed_prediction = item(prediction_maximum);
    dispose(prediction_maximum);
    dispose(probability_loss);
    dispose(expected_probabilities);
    dispose(classification_loss);
    dispose(class_target);
    dispose(prediction);
    dispose(probabilities);
    dispose(logits);
    dispose(projected);
    dispose(classifier_bias);
    dispose(classifier_weight);
    dispose(pooled);
    observed_pooled_shape.length() == 2 and observed_pooled_shape[0] == 2 and
        observed_pooled_shape[1] == 2 and observed_probability_shape.length() == 2 and
        observed_probability_shape[0] == 2 and observed_probability_shape[1] == 2 and
        observed_prediction_shape.length() == 1 and observed_prediction_shape[0] == 2 and
        observed_prediction_dtype == 4 and observed_prediction == 0 and
        observed_probability_loss == 0.0 and observed_classification_loss > 0.69 and
        observed_classification_loss < 0.70.

pub batched_token_transformer_is_correct(): Bool ->
    let input = embed_batched_tokens();
    let embedding_verified = batched_embedding_is_exact(input);
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let attention_residual = encoder_attention(
        input, normalization_weight, normalization_bias, 1, 2, 2
    );
    let encoder_output = encoder_feed_forward(
        attention_residual, normalization_weight, normalization_bias
    );
    let expected_encoder = from_floats(
        [
            1.8413447461, 4.8413447461, 5.8413447461, 2.8413447461,
            5.8413447461, 2.8413447461, 1.8413447461, 4.8413447461
        ],
        [2, 2, 2]
    );
    let encoder_loss = mse_loss(encoder_output, expected_encoder, 1);
    let observed_encoder_shape = shape(encoder_output);
    let observed_encoder_loss = item_float(encoder_loss);
    let classifier_verified = batched_classifier_is_correct(encoder_output);
    dispose(encoder_loss);
    dispose(expected_encoder);
    dispose(encoder_output);
    dispose(attention_residual);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    embedding_verified and observed_encoder_shape.length() == 3 and
        observed_encoder_shape[0] == 2 and observed_encoder_shape[1] == 2 and
        observed_encoder_shape[2] == 2 and observed_encoder_loss < 0.0000001 and
        classifier_verified.

pub stacked_batched_transformer_is_correct(): Bool ->
    let input = embed_batched_tokens();
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let first_attention = encoder_attention(
        input, normalization_weight, normalization_bias, 1, 2, 2
    );
    let first_encoder = encoder_feed_forward(
        first_attention, normalization_weight, normalization_bias
    );
    let second_attention = encoder_attention(
        first_encoder, normalization_weight, normalization_bias, 1, 2, 2
    );
    let second_encoder = encoder_feed_forward(
        second_attention, normalization_weight, normalization_bias
    );
    let expected_encoder = from_floats(
        [
            2.6826894921, 6.6826894921, 7.6826894921, 3.6826894921,
            7.6826894921, 3.6826894921, 2.6826894921, 6.6826894921
        ],
        [2, 2, 2]
    );
    let encoder_loss = mse_loss(second_encoder, expected_encoder, 1);
    let observed_first_shape = shape(first_encoder);
    let observed_second_shape = shape(second_encoder);
    let observed_encoder_loss = item_float(encoder_loss);
    let classifier_verified = batched_classifier_is_correct(second_encoder);
    dispose(encoder_loss);
    dispose(expected_encoder);
    dispose(second_encoder);
    dispose(second_attention);
    dispose(first_encoder);
    dispose(first_attention);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    observed_first_shape.length() == 3 and observed_first_shape[0] == 2 and
        observed_first_shape[1] == 2 and observed_first_shape[2] == 2 and
        observed_second_shape.length() == 3 and observed_second_shape[0] == 2 and
        observed_second_shape[1] == 2 and observed_second_shape[2] == 2 and
        observed_encoder_loss < 0.0000001 and classifier_verified.

pub causal_batched_transformer_is_correct(): Bool ->
    let input = embed_batched_tokens();
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let attention_mask = from_floats([0.0, -1000.0, 0.0, 0.0], [2, 2]);
    let attention_residual = causal_encoder_attention(
        input, normalization_weight, normalization_bias, attention_mask
    );
    let expected_residual = from_floats(
        [1.0, 5.0, 5.0, 3.0, 6.0, 2.0, 2.0, 4.0], [2, 2, 2]
    );
    let residual_loss = mse_loss(attention_residual, expected_residual, 1);
    let encoder_output = encoder_feed_forward(
        attention_residual, normalization_weight, normalization_bias
    );
    let expected_encoder = from_floats(
        [
            0.8413447461, 5.8413447461, 5.8413447461, 2.8413447461,
            6.8413447461, 1.8413447461, 1.8413447461, 4.8413447461
        ],
        [2, 2, 2]
    );
    let encoder_loss = mse_loss(encoder_output, expected_encoder, 1);
    let observed_residual_shape = shape(attention_residual);
    let observed_encoder_shape = shape(encoder_output);
    let observed_residual_loss = item_float(residual_loss);
    let observed_encoder_loss = item_float(encoder_loss);
    let classifier_verified = batched_classifier_is_correct(encoder_output);
    dispose(encoder_loss);
    dispose(expected_encoder);
    dispose(encoder_output);
    dispose(residual_loss);
    dispose(expected_residual);
    dispose(attention_residual);
    dispose(attention_mask);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(input);
    observed_residual_shape.length() == 3 and observed_residual_shape[0] == 2 and
        observed_residual_shape[1] == 2 and observed_residual_shape[2] == 2 and
        observed_encoder_shape.length() == 3 and observed_encoder_shape[0] == 2 and
        observed_encoder_shape[1] == 2 and observed_encoder_shape[2] == 2 and
        observed_residual_loss == 0.0 and observed_encoder_loss < 0.0000001 and
        classifier_verified.

pub cross_attention_transformer_is_correct(): Bool ->
    let decoder_input = from_floats(
        [1.0, 3.0, 2.0, 5.0, 3.0, 1.0, 5.0, 2.0], [2, 2, 2]
    );
    let memory = from_floats(
        [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0],
        [2, 3, 2]
    );
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let attention_residual = decoder_cross_attention(
        decoder_input, memory, normalization_weight, normalization_bias
    );
    let expected_residual = from_floats(
        [
            0.6666666666666667, 3.3333333333333335,
            1.6666666666666667, 5.333333333333333,
            3.3333333333333335, 0.6666666666666667,
            5.333333333333333, 1.6666666666666667
        ],
        [2, 2, 2]
    );
    let residual_loss = mse_loss(attention_residual, expected_residual, 1);
    let encoder_output = encoder_feed_forward(
        attention_residual, normalization_weight, normalization_bias
    );
    let expected_encoder = from_floats(
        [
            0.5080114128, 4.1746780794, 1.5080114128, 6.1746780794,
            4.1746780794, 0.5080114128, 6.1746780794, 1.5080114128
        ],
        [2, 2, 2]
    );
    let encoder_loss = mse_loss(encoder_output, expected_encoder, 1);
    let observed_residual_shape = shape(attention_residual);
    let observed_encoder_shape = shape(encoder_output);
    let observed_residual_loss = item_float(residual_loss);
    let observed_encoder_loss = item_float(encoder_loss);
    let classifier_verified = batched_classifier_is_correct(encoder_output);
    dispose(encoder_loss);
    dispose(expected_encoder);
    dispose(encoder_output);
    dispose(residual_loss);
    dispose(expected_residual);
    dispose(attention_residual);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(memory);
    dispose(decoder_input);
    observed_residual_shape.length() == 3 and observed_residual_shape[0] == 2 and
        observed_residual_shape[1] == 2 and observed_residual_shape[2] == 2 and
        observed_encoder_shape.length() == 3 and observed_encoder_shape[0] == 2 and
        observed_encoder_shape[1] == 2 and observed_encoder_shape[2] == 2 and
        observed_residual_loss < 0.0000001 and observed_encoder_loss < 0.0000001 and
        classifier_verified.

pub causal_cross_attention_decoder_is_correct(): Bool ->
    let decoder_input = embed_batched_tokens();
    let memory = from_floats(
        [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0],
        [2, 3, 2]
    );
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let causal_mask = from_floats([0.0, -1000.0, 0.0, 0.0], [2, 2]);
    let self_attention_residual = causal_encoder_attention(
        decoder_input, normalization_weight, normalization_bias, causal_mask
    );
    let expected_self_attention_residual = from_floats(
        [1.0, 5.0, 5.0, 3.0, 6.0, 2.0, 2.0, 4.0], [2, 2, 2]
    );
    let self_attention_loss = mse_loss(
        self_attention_residual, expected_self_attention_residual, 1
    );
    let cross_attention_residual = decoder_cross_attention(
        self_attention_residual, memory, normalization_weight, normalization_bias
    );
    let expected_cross_attention_residual = from_floats(
        [
            0.6666666666666667, 5.333333333333333,
            4.666666666666667, 3.3333333333333335,
            6.333333333333333, 1.6666666666666667,
            2.3333333333333335, 3.6666666666666665
        ],
        [2, 2, 2]
    );
    let cross_attention_loss = mse_loss(
        cross_attention_residual, expected_cross_attention_residual, 1
    );
    let decoder_output = encoder_feed_forward(
        cross_attention_residual, normalization_weight, normalization_bias
    );
    let expected_decoder_output = from_floats(
        [
            0.5080114128, 6.1746780794, 5.5080114128, 3.1746780794,
            7.1746780794, 1.5080114128, 2.1746780794, 4.5080114128
        ],
        [2, 2, 2]
    );
    let decoder_loss = mse_loss(decoder_output, expected_decoder_output, 1);
    let observed_self_shape = shape(self_attention_residual);
    let observed_cross_shape = shape(cross_attention_residual);
    let observed_decoder_shape = shape(decoder_output);
    let observed_self_loss = item_float(self_attention_loss);
    let observed_cross_loss = item_float(cross_attention_loss);
    let observed_decoder_loss = item_float(decoder_loss);
    let classifier_verified = batched_classifier_is_correct(decoder_output);
    dispose(decoder_loss);
    dispose(expected_decoder_output);
    dispose(decoder_output);
    dispose(cross_attention_loss);
    dispose(expected_cross_attention_residual);
    dispose(cross_attention_residual);
    dispose(self_attention_loss);
    dispose(expected_self_attention_residual);
    dispose(self_attention_residual);
    dispose(causal_mask);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(memory);
    dispose(decoder_input);
    observed_self_shape.length() == 3 and observed_self_shape[0] == 2 and
        observed_self_shape[1] == 2 and observed_self_shape[2] == 2 and
        observed_cross_shape.length() == 3 and observed_cross_shape[0] == 2 and
        observed_cross_shape[1] == 2 and observed_cross_shape[2] == 2 and
        observed_decoder_shape.length() == 3 and observed_decoder_shape[0] == 2 and
        observed_decoder_shape[1] == 2 and observed_decoder_shape[2] == 2 and
        observed_self_loss == 0.0 and observed_cross_loss < 0.0000001 and
        observed_decoder_loss < 0.0000001 and classifier_verified.

pub tokenized_seq2seq_is_correct(): Bool ->
    let source_embedding_weight = from_floats([0.0, 2.0, 2.0, 0.0], [2, 2]);
    let source_token_ids = from_ints([0, 0, 1, 1, 1, 0], [2, 3]);
    let memory = embedding(
        source_embedding_weight, source_token_ids, -1, false, false
    );
    let expected_memory = from_floats(
        [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0],
        [2, 3, 2]
    );
    let memory_loss = mse_loss(memory, expected_memory, 1);
    let target_embedding_weight = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let target_token_ids = from_ints([0, 1, 1, 0], [2, 2]);
    let decoder_input = embedding(
        target_embedding_weight, target_token_ids, -1, false, false
    );
    let expected_decoder_input = from_floats(
        [1.0, 3.0, 4.0, 2.0, 4.0, 2.0, 1.0, 3.0], [2, 2, 2]
    );
    let decoder_input_loss = mse_loss(decoder_input, expected_decoder_input, 1);
    let normalization_weight = full_float([2], 1.0);
    let normalization_bias = full_float([2], 0.0);
    let causal_mask = from_floats([0.0, -1000.0, 0.0, 0.0], [2, 2]);
    let self_attention_residual = causal_encoder_attention(
        decoder_input, normalization_weight, normalization_bias, causal_mask
    );
    let cross_attention_residual = decoder_cross_attention(
        self_attention_residual, memory, normalization_weight, normalization_bias
    );
    let decoder_output = encoder_feed_forward(
        cross_attention_residual, normalization_weight, normalization_bias
    );
    let expected_decoder_output = from_floats(
        [
            0.5080114128, 6.1746780794, 5.5080114128, 3.1746780794,
            7.1746780794, 1.5080114128, 2.1746780794, 4.5080114128
        ],
        [2, 2, 2]
    );
    let decoder_loss = mse_loss(decoder_output, expected_decoder_output, 1);
    let observed_memory_shape = shape(memory);
    let observed_decoder_input_shape = shape(decoder_input);
    let observed_decoder_shape = shape(decoder_output);
    let observed_memory_loss = item_float(memory_loss);
    let observed_decoder_input_loss = item_float(decoder_input_loss);
    let observed_decoder_loss = item_float(decoder_loss);
    let classifier_verified = batched_classifier_is_correct(decoder_output);
    dispose(decoder_loss);
    dispose(expected_decoder_output);
    dispose(decoder_output);
    dispose(cross_attention_residual);
    dispose(self_attention_residual);
    dispose(causal_mask);
    dispose(normalization_bias);
    dispose(normalization_weight);
    dispose(decoder_input_loss);
    dispose(expected_decoder_input);
    dispose(decoder_input);
    dispose(target_token_ids);
    dispose(target_embedding_weight);
    dispose(memory_loss);
    dispose(expected_memory);
    dispose(memory);
    dispose(source_token_ids);
    dispose(source_embedding_weight);
    observed_memory_shape.length() == 3 and observed_memory_shape[0] == 2 and
        observed_memory_shape[1] == 3 and observed_memory_shape[2] == 2 and
        observed_decoder_input_shape.length() == 3 and observed_decoder_input_shape[0] == 2 and
        observed_decoder_input_shape[1] == 2 and observed_decoder_input_shape[2] == 2 and
        observed_decoder_shape.length() == 3 and observed_decoder_shape[0] == 2 and
        observed_decoder_shape[1] == 2 and observed_decoder_shape[2] == 2 and
        observed_memory_loss == 0.0 and observed_decoder_input_loss == 0.0 and
        observed_decoder_loss < 0.0000001 and classifier_verified.

pub main(): Unit ->
    let unbatched_verified = unbatched_token_transformer_is_correct();
    let batched_verified = batched_token_transformer_is_correct();
    let stacked_verified = stacked_batched_transformer_is_correct();
    let causal_verified = causal_batched_transformer_is_correct();
    let cross_attention_verified = cross_attention_transformer_is_correct();
    let decoder_verified = causal_cross_attention_decoder_is_correct();
    let seq2seq_verified = tokenized_seq2seq_is_correct();
    if {
        unbatched_verified and batched_verified and stacked_verified and causal_verified and
            cross_attention_verified and decoder_verified and seq2seq_verified ->
            report_token_transformer_success();
        true -> println("PYTORCH_TOKENIZED_SEQ2SEQ_CONSUMER_FAILED")
    }.
EOF

if ! CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/encoder-classifier-positive-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_encoder_classifier_consumer_execution_failed]: external transformer classifier consumer did not execute' >&2
  exit 1
fi
cat "$output"
if ! grep -Fq 'PYTORCH_ENCODER_CLASSIFIER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_TOKEN_TRANSFORMER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_BATCHED_TOKEN_TRANSFORMER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_STACKED_BATCHED_TRANSFORMER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_CAUSAL_BATCHED_TRANSFORMER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_CROSS_ATTENTION_TRANSFORMER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_CAUSAL_CROSS_ATTENTION_DECODER_CONSUMER_OK' "$output" || \
   ! grep -Fq 'PYTORCH_TOKENIZED_SEQ2SEQ_CONSUMER_OK' "$output" || \
   grep -Fq 'PYTORCH_TOKENIZED_SEQ2SEQ_CONSUMER_FAILED' "$output"; then
  echo 'error[terlan_pytorch_tokenized_seq2seq_consumer_semantics_failed]: external tokenized seq2seq consumer returned incorrect semantics' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{cross_entropy_loss, from_ints, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let logits = full_float([1, 2], 0.0);
    let out_of_range_target = from_ints([2], [1]);
    let loss = cross_entropy_loss(logits, out_of_range_target, 1, -100, 0.0);
    println("PYTORCH_ENCODER_CLASS_TARGET_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/encoder-class-target-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_encoder_class_target_not_rejected]: out-of-range transformer class target unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_ENCODER_CLASS_TARGET_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_encoder_class_target_diagnostic]: out-of-range transformer class target lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{embedding, from_floats, from_ints}.
import std.io.Console.{println}.

pub main(): Unit ->
    let embedding_weight = from_floats([1.0, 3.0, 4.0, 2.0], [2, 2]);
    let out_of_vocabulary_token = from_ints([2], [1]);
    let embedded = embedding(
        embedding_weight, out_of_vocabulary_token, -1, false, false
    );
    println("PYTORCH_TOKEN_ID_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/token-id-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_token_id_not_rejected]: out-of-vocabulary token unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_TOKEN_ID_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_token_id_diagnostic]: out-of-vocabulary token lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{full_float, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let left = full_float([2, 2, 2], 1.0);
    let incompatible_batches = full_float([3, 2, 2], 1.0);
    let result = matmul(left, incompatible_batches);
    println("PYTORCH_BATCHED_MATMUL_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/batched-matmul-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batched_matmul_not_rejected]: incompatible batch dimensions unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_BATCHED_MATMUL_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_batched_matmul_diagnostic]: incompatible batch dimensions lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, full_float, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let layer_state = full_float([2, 2, 2], 1.0);
    let width_changing_projection = full_float([2, 3], 1.0);
    let projected = matmul(layer_state, width_changing_projection);
    let next_layer_state = add(layer_state, projected, 1.0);
    println("PYTORCH_STACKED_TRANSFORMER_WIDTH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/stacked-transformer-width-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_stacked_transformer_width_not_rejected]: width-changing stacked transformer state unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_STACKED_TRANSFORMER_WIDTH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_stacked_transformer_width_diagnostic]: width-changing stacked transformer state lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, from_floats, full_float}.
import std.io.Console.{println}.

pub main(): Unit ->
    let attention_scores = full_float([2, 2, 2], 0.0);
    let malformed_mask = from_floats(
        [0.0, -1000.0, -1000.0, 0.0, 0.0, -1000.0, 0.0, 0.0, 0.0], [3, 3]
    );
    let masked_scores = add(attention_scores, malformed_mask, 1.0);
    println("PYTORCH_CAUSAL_MASK_SHAPE_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/causal-mask-shape-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_causal_mask_shape_not_rejected]: malformed causal mask unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_CAUSAL_MASK_SHAPE_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_causal_mask_shape_diagnostic]: malformed causal mask lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{full_float, matmul, transpose}.
import std.io.Console.{println}.

pub main(): Unit ->
    let decoder_queries = full_float([2, 2, 2], 0.0);
    let wrong_batch_memory_keys = full_float([3, 3, 2], 0.0);
    let transposed_keys = transpose(wrong_batch_memory_keys, 1, 2);
    let scores = matmul(decoder_queries, transposed_keys);
    println("PYTORCH_CROSS_ATTENTION_BATCH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/cross-attention-batch-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_cross_attention_batch_not_rejected]: mismatched decoder and memory batches unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_CROSS_ATTENTION_BATCH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_cross_attention_batch_diagnostic]: mismatched decoder and memory batches lacked their stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{add, full_float, matmul}.
import std.io.Console.{println}.

pub main(): Unit ->
    let decoder_state = full_float([2, 2, 2], 1.0);
    let attention = full_float([2, 2, 3], 0.3333333333333333);
    let width_three_memory_values = full_float([2, 3, 3], 1.0);
    let width_three_context = matmul(attention, width_three_memory_values);
    let invalid_decoder_state = add(decoder_state, width_three_context, 1.0);
    println("PYTORCH_DECODER_CONTEXT_WIDTH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/decoder-context-width-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_decoder_context_width_not_rejected]: width-changing cross-attention context unexpectedly joined the decoder residual' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_DECODER_CONTEXT_WIDTH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_decoder_context_width_diagnostic]: width-changing cross-attention context lacked its stable C ABI rejection' >&2
  exit 1
fi

cat >"$consumer/src/pytorch_consumer/Main.terl" <<'EOF'
module pytorch_consumer.Main.

import pytorch.Tensor.{embedding, from_floats, from_ints, full_float, matmul, transpose}.
import std.io.Console.{println}.

pub main(): Unit ->
    let source_embedding_weight = from_floats([0.0, 2.0, 2.0, 0.0], [2, 2]);
    let three_batch_source_ids = from_ints(
        [0, 0, 1, 1, 1, 0, 0, 1, 0], [3, 3]
    );
    let three_batch_memory = embedding(
        source_embedding_weight, three_batch_source_ids, -1, false, false
    );
    let key_weight = full_float([2, 2], 0.0);
    let memory_keys = matmul(three_batch_memory, key_weight);
    let transposed_keys = transpose(memory_keys, 1, 2);
    let two_batch_target_queries = full_float([2, 2, 2], 0.0);
    let scores = matmul(two_batch_target_queries, transposed_keys);
    println("PYTORCH_SEQ2SEQ_BATCH_NOT_REJECTED").
EOF

if CARGO_NET_OFFLINE=true LIBTORCH="$libtorch" \
  "$terlc" run "$consumer" --target terlan-vm --out-dir "$workspace/seq2seq-batch-negative-build" \
  >"$output" 2>&1; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_seq2seq_batch_not_rejected]: incompatible source and target batches unexpectedly executed' >&2
  exit 1
fi
if ! grep -Fq 'c_abi_status_' "$output" || \
   grep -Fq 'PYTORCH_SEQ2SEQ_BATCH_NOT_REJECTED' "$output"; then
  cat "$output" >&2
  echo 'error[terlan_pytorch_seq2seq_batch_diagnostic]: mismatched source and target batches lacked their stable C ABI rejection' >&2
  exit 1
fi

echo 'Fresh Git-package consumer executed through the verified package cache.'
