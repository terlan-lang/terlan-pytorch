#!/usr/bin/env python3
"""Freeze the callable public API exported by the pinned PyTorch runtime.

Python is used only as the upstream oracle. The committed JSON is consumed by
the Terlan-owned parity checker and package gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import re
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import torch


VERSION = "2.13.0+cpu"
BUILD_HASH = "cf30153c4c131c8164ee7798e5022d810682e2cb"
NAMESPACES = (
    # Root classes are inventoried as constructors. Tensor is expanded below;
    # expanding every root-exported C++ metatype would incorrectly treat
    # implementation helpers such as AliasDb as user-facing class APIs.
    ("torch", False),
    ("torch.nn", True),
    ("torch.nn.functional", False),
    ("torch.autograd", True),
    ("torch.optim", True),
    ("torch.optim.lr_scheduler", True),
    ("torch.linalg", False),
    ("torch.fft", False),
    ("torch.special", False),
    ("torch.distributions", True),
    ("torch.sparse", True),
    ("torch.ao.quantization", True),
    ("torch.ao.nn.quantized", True),
    ("torch.ao.nn.qat", True),
    ("torch.ao.nn.intrinsic", True),
    ("torch.amp", True),
    ("torch.cuda", True),
    ("torch.distributed", True),
    ("torch.export", True),
    ("torch.serialization", True),
    ("torch.utils.data", True),
)


def stable_signature(value: Any) -> tuple[str | None, str]:
    try:
        signature = str(inspect.signature(value, follow_wrapped=False))
    except (TypeError, ValueError):
        return None, "unavailable"
    return re.sub(r" at 0x[0-9a-fA-F]+", "", signature), "inspect"


def callable_kind(value: Any, member: bool = False) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isbuiltin(value):
        return "builtin_method" if member else "builtin_function"
    if inspect.ismethod(value):
        return "method"
    if inspect.isfunction(value):
        return "method" if member else "function"
    return "callable_object"


def entry(export: str, namespace: str, value: Any, member: bool = False) -> dict[str, Any]:
    signature, signature_source = stable_signature(value)
    name = export.rsplit(".", 1)[-1]
    identity = export + (signature if signature is not None else "(?)")
    return {
        "id": identity,
        "export": export,
        "namespace": namespace,
        "callable_kind": callable_kind(value, member),
        "signature": signature,
        "signature_source": signature_source,
        "overload_family": name.rstrip("_") or name,
        "device_support": "unclassified",
        "mutability": "in_place" if name.endswith("_") and not name.endswith("__") else "unclassified",
        "source_module": getattr(value, "__module__", None),
        "source_qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        "reference": f"https://docs.pytorch.org/docs/2.13/generated/{export}.html",
    }


def exported_names(module: ModuleType) -> list[str]:
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return sorted({name for name in declared if isinstance(name, str)})
    return sorted(name for name in dir(module) if not name.startswith("_"))


def class_members(export: str, namespace: str, cls: type[Any]) -> list[dict[str, Any]]:
    result = []
    # Inherited behavior is frozen once on the class that defines it. Repeating
    # every Module/Distribution base method for every subclass would inflate
    # the inventory with aliases rather than distinct callables.
    for name in sorted(name for name in cls.__dict__ if not name.startswith("_")):
        try:
            value = getattr(cls, name)
        except Exception:
            continue
        if callable(value):
            result.append(entry(f"{export}.{name}", namespace, value, member=True))
    return result


def collect() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for namespace, expand_classes in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in exported_names(module):
            try:
                value = getattr(module, name)
            except Exception:
                continue
            if not callable(value):
                continue
            exported = f"{namespace}.{name}"
            item = entry(exported, namespace, value)
            by_id[item["id"]] = item
            is_tensor_alias = inspect.isclass(value) and (
                issubclass(value, torch.Tensor)
                or (namespace == "torch.cuda" and name.endswith("Tensor"))
            )
            if expand_classes and inspect.isclass(value) and not is_tensor_alias:
                for member in class_members(exported, namespace, value):
                    by_id[member["id"]] = member

    for name in sorted(name for name in dir(torch.Tensor) if not name.startswith("_")):
        try:
            value = getattr(torch.Tensor, name)
        except Exception:
            continue
        if callable(value):
            item = entry(f"torch.Tensor.{name}", "torch.Tensor", value, member=True)
            by_id[item["id"]] = item
    return [by_id[key] for key in sorted(by_id)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if torch.__version__ != VERSION or torch.version.git_version != BUILD_HASH:
        raise SystemExit(
            f"expected PyTorch {VERSION} at {BUILD_HASH}, got "
            f"{torch.__version__} at {torch.version.git_version}"
        )
    entries = collect()
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    document = {
        "schema": "terlan.pytorch.public-api.v1",
        "pytorch_version": VERSION,
        "build_hash": BUILD_HASH,
        "source": "pinned Python runtime exports and public class members",
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "schemas": [item["id"] for item in entries],
        "entries": entries,
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"froze {len(entries)} public PyTorch callables at {args.output}")


if __name__ == "__main__":
    main()
