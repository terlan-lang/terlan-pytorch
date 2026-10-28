#!/usr/bin/env python3
"""Seed exact one-schema public callables from reviewed ATen evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ADMITTED_NAMESPACES = {"torch", "torch.Tensor", "torch.nn.functional"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aten-reference", required=True, type=Path)
    parser.add_argument("--aten-policy", required=True, type=Path)
    parser.add_argument("--public-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    aten_reference = json.loads(args.aten_reference.read_text(encoding="utf-8"))
    aten_policy = json.loads(args.aten_policy.read_text(encoding="utf-8"))
    public_reference = json.loads(args.public_reference.read_text(encoding="utf-8"))

    schemas_by_family: dict[str, list[str]] = defaultdict(list)
    for schema in aten_reference["schemas"]:
        operator = schema.split("(", 1)[0]
        schemas_by_family[operator.split(".", 1)[0]].append(schema)

    exact_families = {}
    for family, schemas in schemas_by_family.items():
        if len(schemas) == 1 and schemas[0] in aten_policy["implemented"]:
            exact_families[family] = aten_policy["implemented"][schemas[0]]

    implemented = {}
    for item in public_reference["entries"]:
        namespace = item["namespace"]
        if namespace not in ADMITTED_NAMESPACES:
            continue
        export = item["export"]
        prefix = namespace + "."
        if not export.startswith(prefix):
            continue
        leaf = export[len(prefix) :]
        if "." in leaf or leaf not in exact_families:
            continue
        implemented[item["id"]] = exact_families[leaf]

    document = {
        "schema": "terlan.pytorch.api-policy.v1",
        "pytorch_version": public_reference["pytorch_version"],
        "implemented": dict(sorted(implemented.items())),
        "equivalent": {},
        "non_applicable": {},
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"seeded {len(implemented)} exact public-callable classifications")


if __name__ == "__main__":
    main()
