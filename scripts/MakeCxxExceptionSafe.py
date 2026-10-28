#!/usr/bin/env python3
"""Make the direct CXX compatibility surface exception-safe.

The generated owned/mutation adapters already return contained envelopes.  The
small direct compatibility surface is different: cxx must be told explicitly
which functions may throw so it can translate C++ exceptions to Rust Results.
This script keeps the checked-in helper and generated snapshot in sync without
hand-editing thousands of generated lines.
"""

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
    names = re.findall(r"\bfn\s+([A-Za-z0-9_]+)\s*\(", prefix)
    return [name for name in names if name not in SKIP]


def resultify_declarations(source: str, names: list[str]) -> str:
    for name in names:
        pattern = re.compile(rf"(\bfn\s+{re.escape(name)}\s*\([^;]*?)(?:\s*->\s*([^;]+))?;")

        def replace(match: re.Match[str]) -> str:
            prefix, return_type = match.groups()
            if return_type and return_type.strip().startswith("Result<"):
                return match.group(0)
            ok_type = return_type.strip() if return_type else "()"
            return f"{prefix} -> Result<{ok_type}>;"

        source, count = pattern.subn(replace, source, count=1)
        if count != 1:
            raise RuntimeError(f"could not resultify CXX declaration: {name}")
    return source


def call_span(source: str, start: int) -> tuple[int, int]:
    open_at = source.find("(", start)
    if open_at < 0:
        raise RuntimeError("missing call opening parenthesis")
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(open_at, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise RuntimeError("unterminated CXX call")


def occurrences(source: str, names: set[str]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for match in re.finditer(r"ffi::([A-Za-z0-9_]+)\s*\(", source):
        if match.group(1) in names:
            result.append(call_span(source, match.start()))
    return result


def patch_helper(path: Path, names: set[str]) -> None:
    source = path.read_text(encoding="utf-8")
    if "macro_rules! ffi_result" not in source:
        insert = '''fn native_exception_failure(error: cxx::Exception) -> String {
    protocol_error("native_cpp_exception", error.what())
}

macro_rules! ffi_result {
    ($call:expr) => {
        match $call {
            Ok(value) => value,
            Err(error) => return native_exception_failure(error),
        }
    };
}

'''
        marker = "const MAX_ADAPTER_MESSAGE_BYTES: usize ="
        marker_start = source.index(marker)
        position = source.index("\n", source.index(";", marker_start)) + 1
        source = source[:position] + insert + source[position:]
    spans = occurrences(source, names)
    for start, end in reversed(spans):
        expression = source[start:end]
        if source[max(0, start - 14) : start].endswith("ffi_result!("):
            continue
        source = source[:start] + f"ffi_result!({expression})" + source[end:]
    path.write_text(source, encoding="utf-8")


def patch_tests(root: Path, names: set[str]) -> None:
    test_roots = [root / "native/rust/tests", root / "tests/native", root / "tests/cxx"]
    for test_root in test_roots:
        if not test_root.exists():
            continue
        for path in test_root.glob("*.rs"):
            if path.name == "exception_boundary_cxx.rs":
                continue
            source = path.read_text(encoding="utf-8")
            spans = occurrences(source, names)
            for start, end in reversed(spans):
                if re.match(r"\s*\.expect\(", source[end:]):
                    continue
                source = source[:end] + '.expect("CXX exception")' + source[end:]
            path.write_text(source, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "native/rust/src/lib.rs"
    source = source_path.read_text(encoding="utf-8")
    names = direct_names(source)
    updated = resultify_declarations(source, names)
    source_path.write_text(updated, encoding="utf-8")

    generated_root = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "generated"
    generated = generated_root / "native/rust/src/lib.rs"
    if generated.exists():
        generated.write_text(
            resultify_declarations(generated.read_text(encoding="utf-8"), names),
            encoding="utf-8",
        )
    patch_helper(root / "native/rust/src/bin/native_boundary_helper.rs", set(names))
    generated_helper = generated_root / "native/rust/src/bin/native_boundary_helper.rs"
    if generated_helper.exists():
        patch_helper(generated_helper, set(names))
    patch_tests(root, set(names))


if __name__ == "__main__":
    main()
