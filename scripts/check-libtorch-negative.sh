#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
check="$root/scripts/check-libtorch.sh"
workspace=$(mktemp -d "${TMPDIR:-/tmp}/terlan-pytorch-libtorch-negative.XXXXXX")
trap 'rm -rf "$workspace"' EXIT

expect_rejection() {
  local expected=$1
  local candidate=$2
  local output="$workspace/output"

  if "$check" "$candidate" >"$output" 2>&1; then
    echo "error[terlan_pytorch_libtorch_negative_not_rejected]: candidate unexpectedly passed: $candidate" >&2
    exit 1
  fi
  if ! grep -Fq "$expected" "$output"; then
    echo "error[terlan_pytorch_libtorch_negative_diagnostic]: expected $expected" >&2
    cat "$output" >&2
    exit 1
  fi
}

expect_rejection \
  'error[terlan_pytorch_libtorch_build_version_missing]' \
  "$workspace/missing"

mkdir -p "$workspace/incompatible"
printf '%s\n' '0.0.0+incompatible' >"$workspace/incompatible/build-version"
expect_rejection \
  'error[terlan_pytorch_libtorch_version_incompatible]' \
  "$workspace/incompatible"

echo 'LibTorch missing and incompatible distributions are rejected stably.'
