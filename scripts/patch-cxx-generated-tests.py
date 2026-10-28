#!/usr/bin/env python3
"""Adapt legacy CXX behavior fixtures to the generated integer dtype ABI.

The fixtures intentionally share the package behavior corpus.  Their short
dtype names are atom aliases in the legacy package, while CXX adapters carry
the corresponding c10::ScalarType values as integers.  This keeps the fixture
logic identical while making the CXX target explicit about the ABI values.
"""

import argparse
import re
from pathlib import Path


DTYPES = {
    "UInt8": "0",
    "Int8": "1",
    "Int16": "2",
    "Int32": "3",
    "Int64": "4",
    "Float16": "5",
    "Float32": "6",
    "Float64": "7",
    "Complex32": "8",
    "Complex64": "9",
    "Complex128": "10",
}


def patch_imports(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        names = [name.strip() for name in match.group(1).split(",")]
        # The generated Tensor module contains dtype atoms (including Bool).
        # Do not import them: fixtures use the CXX bridge's integer codes and
        # retain Terlan's built-in Bool type for logical assertions.
        names = [
            name
            for name in names
            if name and name not in DTYPES and name not in {"Bool", "Strided"}
        ]
        return "import pytorch.Tensor.{" + ", ".join(names) + "}"

    return re.sub(r"import pytorch\.Tensor\.\{([^}]*)\}", replace, text)


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    text = patch_imports(text)
    protected = {}
    for index, short_name in enumerate([*DTYPES.keys(), "Strided"]):
        marker = f"__TERLAN_CXX_PATH_{index}__"
        protected[marker] = f"pytorch.{short_name}"
        text = text.replace(protected[marker], marker)
    # Keep module paths intact (for example ``pytorch.Strided``); only
    # rewrite aliases in expressions and declarations below import lines.
    lines = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("import "):
            lines.append(line)
            continue
        for short_name, enum_name in DTYPES.items():
            line = re.sub(rf"\b{re.escape(short_name)}\b", enum_name, line)
        line = re.sub(r"\bStrided\b", "0", line)
        lines.append(line)
    text = "".join(lines)
    # Bool is also a language type, so only rewrite it where it is a dtype
    # argument or a dtype comparison.
    text = re.sub(
        r"\b(ones_(?:dtype|device))\(([^\n]*?,\s*)Bool\b",
        r"\1(\g<2>11",
        text,
    )
    text = re.sub(r"(dtype\([^\n]*?\)\s*==\s*)Bool\b", r"\g<1>11", text)
    # The generic C++ ownership smoke generator currently samples every
    # floating constructor parameter as `2.0`.  Adam rejects beta values and
    # epsilon outside their valid ranges, so keep this generated lifecycle
    # test executable with the canonical stable hyperparameters.
    text = text.replace(
        "adam(2.0, 2.0, 2.0, 2.0, 2.0)",
        "adam(0.001, 0.9, 0.999, 0.00000001, 0.0)",
    )
    for marker, value in protected.items():
        text = text.replace(marker, value)
    if text != original:
        path.write_text(text, encoding="utf-8")


def patch_runtime_tensor_test(path: Path) -> None:
    """Keep the generated ownership smoke test on the typed CXX surface.

    The generic runtime fixture predates the CXX migration and passes a
    Tensor directly to ``scalar_to_int`` while comparing symbolic dtype and
    layout atoms.  The generated bridge intentionally exposes an owned
    Scalar and integer enum projections, so express the same lifecycle check
    through those public signatures.
    """
    if path.name != "TensorTest.terl":
        return
    path.write_text(
        """module pytorch.TensorTest.

import pytorch.Tensor.{abi_version, dispose, dispose_scalar, dtype, item, layout, ones, scalar_to_int}.
import std.core.Result.{Err, Ok}.

@test
pub generated_cpp_resource_executes(): Bool ->
    let boundary = ones([1]);
    let scalar = item(boundary);
    let observed = scalar_to_int(scalar);
    let copied_dtype = dtype(boundary);
    let copied_layout = layout(boundary);
    dispose_scalar(scalar);
    dispose(boundary);
    case observed {
        Ok(value) -> value == 1 and copied_dtype == 6 and copied_layout == 0 and abi_version() > 0;
        Err(_reason) -> false
    }.
""",
        encoding="utf-8",
    )


def patch_runtime_cuda_test(path: Path) -> None:
    """Keep the generated CUDA stream smoke test on the available device.

    The generic ownership-smoke generator uses ``1`` as its sample integer
    argument.  For CUDA stream constructors that selects device 1, while the
    validation host intentionally exposes a single GPU at device 0.  Use the
    first device so the smoke test validates the CXX resource lifecycle rather
    than depending on a multi-GPU machine.
    """
    if path.name != "CudaTest.terl":
        return
    text = path.read_text(encoding="utf-8")
    patched = text.replace("create_stream(1, true)", "create_stream(0, true)")
    if patched != text:
        path.write_text(patched, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="_build/libtorch-cxx-generated")
    args = parser.parse_args()
    root = Path(args.root)
    for path in root.glob("tests/**/*.terl"):
        patch_file(path)
        patch_runtime_tensor_test(path)
        patch_runtime_cuda_test(path)
    for path in root.glob("src/pytorch/*Support.terl"):
        patch_file(path)


if __name__ == "__main__":
    main()
