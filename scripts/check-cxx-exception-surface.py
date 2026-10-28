#!/usr/bin/env python3
"""Reject direct CXX declarations or calls that can bypass exception translation."""

from __future__ import annotations

import re
import sys
from pathlib import Path


SKIP = {
    "terlan_resource_list_pytorch_tensor_tensor_new",
    "terlan_resource_list_pytorch_tensor_tensor_push",
    "terlan_resource_pytorch_cuda_stream_copy",
    "terlan_resource_pytorch_randomcompletion_generator_copy",
    "terlan_resource_pytorch_storagenative_storagehandle_copy",
    "terlan_resource_pytorch_tensor_tensor_copy",
    "q_per_channel_axis",
    "q_scale",
    "q_zero_point",
}


def direct_names(source: str) -> list[str]:
    prefix = source.split("fn terlan_owned_value_pytorch_tensor_ones", 1)[0]
    return [
        name
        for name in re.findall(r"\bfn\s+([A-Za-z0-9_]+)\s*\(", prefix)
        if name not in SKIP
    ]


def declaration(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\s*\([^;]*?;", source)
    if not match:
        raise SystemExit(f"missing direct CXX declaration: {name}")
    return match.group(0)


def check(root: Path) -> None:
    lib = root / "native/rust/src/lib.rs"
    helper = root / "native/rust/src/bin/native_boundary_helper.rs"
    source = lib.read_text(encoding="utf-8")
    names = direct_names(source)
    missing = [name for name in names if "Result<" not in declaration(source, name)]
    if missing:
        raise SystemExit("direct CXX declarations without Result translation: " + ", ".join(missing))

    helper_source = helper.read_text(encoding="utf-8")
    unwrapped = []
    for match in re.finditer(r"ffi::([A-Za-z0-9_]+)\s*\(", helper_source):
        if match.group(1) in names and not helper_source[max(0, match.start() - 14) : match.start()].endswith(
            "ffi_result!("
        ):
            unwrapped.append(match.group(1))
    if unwrapped:
        raise SystemExit("direct CXX calls without ffi_result!: " + ", ".join(sorted(set(unwrapped))))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    check(root)
    if len(sys.argv) > 1:
        generated = Path(sys.argv[1])
        if (generated / "native/rust/src/lib.rs").exists():
            check(generated)
    print("CXX exception surface is fully Result-wrapped")


if __name__ == "__main__":
    main()
