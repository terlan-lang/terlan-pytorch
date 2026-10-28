#!/usr/bin/env python3
"""Require migrated public modules to exactly match the frozen CXX contract."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_functions(manifest: dict[str, Any], module_name: str) -> list[dict[str, Any]]:
    matches = [module for module in manifest["modules"] if module["module"] == module_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one module `{module_name}`, found {len(matches)}")
    module = matches[0]
    functions = list(module["functions"])
    for family in module.get("function_families", []):
        prefix = family["operation_prefix"]
        for member in family["members"]:
            name = member.get("name")
            if name is None:
                raise ValueError(
                    f"migration contract requires an explicit public name for family member "
                    f"`{member['cpp_symbol']}` in `{module_name}`"
                )
            functions.append(
                {
                    "name": name,
                    "operation": f"{prefix}.{member.get('operation_suffix', name)}",
                    "args": family.get("args", []),
                    "returns": family["returns"],
                    "visibility": family.get("visibility"),
                }
            )
    return functions


def public_contract(function: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": function["name"],
        "operation": function["operation"],
        "args": [
            {
                "name": argument["name"],
                "ty": argument["ty"],
                "default": argument.get("default"),
            }
            for argument in function.get("args", [])
        ],
        "returns": function["returns"],
    }


def snapshot_contracts(snapshot: dict[str, Any], module_name: str) -> list[dict[str, Any]]:
    matches = [module for module in snapshot["modules"] if module["module"] == module_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one module `{module_name}`, found {len(matches)}")
    return sorted(matches[0]["functions"], key=lambda function: (function["name"], function["operation"]))


def contracts(
    manifest: dict[str, Any],
    module_name: str,
    expected: list[dict[str, Any]] | None = None,
    allowed_extras: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    functions = module_functions(manifest, module_name)
    if expected is not None:
        # The CXX package may add reviewed Terlan compositions while retaining
        # an exact contract for every migrated C-ABI operation. Native extras
        # still fail the audit unless explicitly reviewed in the migration
        # ledger; composition bodies are package-owned surface.
        expected_keys = {(function["name"], function["operation"]) for function in expected}
        functions = [
            function
            for function in functions
            if (function["name"], function["operation"]) in expected_keys
            or (
                allowed_extras is not None
                and (function["name"], function["operation"]) in allowed_extras
            )
            or (
                function.get("visibility") != "private"
                and "terlan_body" not in function
            )
        ]
        if allowed_extras:
            functions = [
                function
                for function in functions
                if (function["name"], function["operation"]) not in allowed_extras
            ]
    return sorted(
        (public_contract(function) for function in functions),
        key=lambda function: (function["name"], function["operation"]),
    )


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: check-cxx-migration.py LEDGER CONTRACT_SNAPSHOT CXX_MANIFEST",
            file=sys.stderr,
        )
        return 2
    ledger = load(sys.argv[1])
    snapshot = load(sys.argv[2])
    generated = load(sys.argv[3])
    if ledger.get("schema") != "terlan.pytorch.cxx-migration.v1":
        raise ValueError("unsupported CXX migration ledger schema")
    if snapshot.get("schema") != "terlan.pytorch.cxx-contract-snapshot.v1":
        raise ValueError("unsupported CXX contract snapshot schema")
    expected_snapshot_hash = ledger.get("snapshot_sha256")
    if not isinstance(expected_snapshot_hash, str) or not expected_snapshot_hash:
        raise ValueError("migration ledger must pin snapshot_sha256")
    actual_snapshot_hash = sha256(sys.argv[2])
    if actual_snapshot_hash != expected_snapshot_hash:
        raise ValueError(
            "CXX contract snapshot provenance mismatch: "
            f"expected {expected_snapshot_hash}, got {actual_snapshot_hash}"
        )
    if generated["package"]["adapter"] != "cxx":
        raise ValueError("migration target must use the cxx adapter")

    snapshot_operations = snapshot["operations"]
    counted_snapshot_operations = sum(
        len(module.get("functions", [])) for module in snapshot["modules"]
    )
    if counted_snapshot_operations != snapshot_operations:
        raise ValueError(
            "CXX contract snapshot operation count does not match its module contents"
        )
    if snapshot_operations > ledger["maximum_snapshot_operations"]:
        raise ValueError(
            f"CXX contract snapshot grew to {snapshot_operations} operations; "
            f"maximum is {ledger['maximum_snapshot_operations']}"
        )

    migrated_operations = 0
    for module_name in ledger["migrated_modules"]:
        expected = snapshot_contracts(snapshot, module_name)
        allowed_extras = {
            (entry["name"], entry["operation"])
            for entry in ledger.get("allowed_native_extras", {}).get(module_name, [])
        }
        actual = contracts(generated, module_name, expected, allowed_extras)
        if actual != expected:
            expected_by_name = {entry["name"]: entry for entry in expected}
            actual_by_name = {entry["name"]: entry for entry in actual}
            missing = sorted(expected_by_name.keys() - actual_by_name.keys())
            extra = sorted(actual_by_name.keys() - expected_by_name.keys())
            changed = sorted(
                name
                for name in expected_by_name.keys() & actual_by_name.keys()
                if expected_by_name[name] != actual_by_name[name]
            )
            raise ValueError(
                f"migrated module `{module_name}` changed public contract: "
                f"missing={missing}, extra={extra}, changed={changed}"
            )
        migrated_operations += len(expected)

    report = {
        "schema": ledger["schema"],
        "snapshot_operations": snapshot_operations,
        "migrated_modules": ledger["migrated_modules"],
        "migrated_operations": migrated_operations,
        "remaining_operations": snapshot_operations - migrated_operations,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error[cxx_migration_contract]: {error}", file=sys.stderr)
        raise SystemExit(1)
