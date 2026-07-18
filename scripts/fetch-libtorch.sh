#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=2.13.0+cpu
archive="$root/vendor/libtorch-${version}.zip"
destination="$root/vendor/libtorch-${version}"
url='https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.13.0%2Bcpu.zip'
sha256=edbf4cbed78433d803e90a65f1752e57783d164bce66c95c0872b2ab8f5c159e

mkdir -p "$root/vendor"
if [[ ! -f "$archive" ]] || ! printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status; then
  rm -f "$archive"
  curl --fail --location --retry 3 "$url" --output "$archive"
fi
printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status

if [[ ! -f "$destination/libtorch/build-version" ]] || \
   [[ "$(<"$destination/libtorch/build-version")" != "$version" ]]; then
  rm -rf "$destination"
  mkdir -p "$destination"
  unzip -q "$archive" -d "$destination"
fi

echo "$destination/libtorch"
