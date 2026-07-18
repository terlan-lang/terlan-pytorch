#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <generated-package> <repository-root>" >&2
  exit 2
fi

generated=$(realpath "$1")
repository=$(realpath "$2")

for path in terlan.toml src native bindings/native-binding-manifest.json \
  bindings/skipped-symbols.json docs/pytorch.Tensor.md \
  tests/pytorch/NativeBoundaryTest.terl; do
  test -e "$generated/$path" || {
    echo "error[terlan_pytorch_generated_artifact_missing]: $path" >&2
    exit 1
  }
done

rm -rf "$repository/src" "$repository/native"
cp -a "$generated/src" "$repository/src"
cp -a "$generated/native" "$repository/native"
cp "$generated/terlan.toml" "$repository/terlan.toml"
cp "$generated/bindings/native-binding-manifest.json" "$repository/bindings/"
cp "$generated/bindings/skipped-symbols.json" "$repository/bindings/"
cp "$generated/docs/pytorch.Tensor.md" "$repository/docs/"
cp "$generated/tests/pytorch/NativeBoundaryTest.terl" "$repository/tests/pytorch/"
rm -rf "$repository/native/rust/target"

while IFS= read -r -d '' path; do
  sed -i '${/^$/d;}' "$path"
done < <(
  find "$repository/src" "$repository/native" \
    "$repository/docs/pytorch.Tensor.md" \
    "$repository/tests/pytorch/NativeBoundaryTest.terl" \
    -type f -print0
)
