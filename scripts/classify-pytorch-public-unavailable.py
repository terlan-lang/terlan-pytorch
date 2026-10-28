#!/usr/bin/env python3
"""Classify public inventory entries with no frozen callable signature.

The PyTorch export inventory includes classes and descriptors for which the
reference interpreter cannot provide a signature.  They remain in the frozen
inventory, but there is no executable callable contract to implement or
validate until PyTorch publishes one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REASON = (
    "Frozen PyTorch public export has no callable signature; there is no "
    "executable Terlan contract to implement or validate."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    implemented = policy.setdefault("implemented", {})
    equivalent = policy.setdefault("equivalent", {})
    non_applicable = policy.setdefault("non_applicable", {})
    changed = 0
    for entry in reference["entries"]:
        identifier = entry["id"]
        if entry.get("signature_source") != "unavailable":
            continue
        if identifier in implemented or identifier in equivalent:
            continue
        if identifier not in non_applicable:
            non_applicable[identifier] = REASON
            changed += 1
    args.policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"classified {changed} public exports without signatures")


if __name__ == "__main__":
    main()
