#!/usr/bin/env python3
"""Reject drift between the canonical native C++ tree and generated output."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PAIRS = (
    ("native/rust/build.rs", "native/rust/build.rs"),
    ("native/rust/src/lib.rs", "native/rust/src/lib.rs"),
    ("native/rust/cpp/cuda_stream.cc", "native/rust/cpp/cuda_stream.cc"),
    ("native/rust/cpp/libtorch_metadata.cc", "native/rust/cpp/libtorch_metadata.cc"),
    ("native/rust/cpp/optimizer.cc", "native/rust/cpp/optimizer.cc"),
    ("native/rust/cpp/tensor_advanced.cc", "native/rust/cpp/tensor_advanced.cc"),
    ("native/rust/cpp/tensor_compat.cc", "native/rust/cpp/tensor_compat.cc"),
    ("native/rust/cpp/tensor_ops.cc", "native/rust/cpp/tensor_ops.cc"),
    ("native/rust/cpp/terlan_tensor_packet.cc", "native/rust/cpp/terlan_tensor_packet.cc"),
    ("native/rust/cpp/torchscript_compat.cc", "native/rust/cpp/torchscript_compat.cc"),
    ("native/rust/include/cuda_stream.hpp", "native/rust/include/cuda_stream.hpp"),
    ("native/rust/include/libtorch_bindings.hpp", "native/rust/include/libtorch_bindings.hpp"),
    ("native/rust/include/optimizer.hpp", "native/rust/include/optimizer.hpp"),
    ("native/rust/include/tensor_advanced.hpp", "native/rust/include/tensor_advanced.hpp"),
    ("native/rust/include/tensor_compat.hpp", "native/rust/include/tensor_compat.hpp"),
    ("native/rust/include/tensor_ops.hpp", "native/rust/include/tensor_ops.hpp"),
    ("native/rust/include/terlan_tensor_packet.hpp", "native/rust/include/terlan_tensor_packet.hpp"),
    ("native/rust/include/torchscript_compat.hpp", "native/rust/include/torchscript_compat.hpp"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check-cxx-source-ownership.py REPOSITORY GENERATED", file=sys.stderr)
        return 2
    repository = Path(sys.argv[1])
    generated = Path(sys.argv[2])
    errors: list[str] = []
    for canonical, generated_relative in PAIRS:
        source = repository / canonical
        target = generated / generated_relative
        if not source.is_file() or not target.is_file():
            errors.append(f"missing canonical/generated source: {canonical} -> {generated_relative}")
        elif digest(source) != digest(target):
            errors.append(f"generated CXX source drift: {canonical} -> {generated_relative}")
    if errors:
        for error in errors:
            print(f"error[cxx_source_ownership]: {error}", file=sys.stderr)
        return 1
    print("canonical CXX source ownership is synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
