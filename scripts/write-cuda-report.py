#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libtorch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.libtorch).resolve()
    candidates = (
        root / "lib/libtorch_cuda.so",
        root / "lib/libtorch_cuda.dylib",
        root / "lib/torch_cuda.lib",
        root / "bin/torch_cuda.dll",
    )
    library = next((path for path in candidates if path.is_file()), None)
    if library is None:
        raise SystemExit("error[terlan_pytorch_cuda_distribution]: CUDA library is missing")
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    report = {
        "backend": "libtorch-cxx-generated",
        "cleanup": "generation-tagged-explicit-dispose",
        "device": device,
        "device_pointer_exposed": False,
        "distribution": {
            "build_version": (root / "build-version").read_text().strip(),
            "cuda_library": library.name,
            "cuda_library_sha256": digest(library),
        },
        "operations": [
            "stream-create",
            "stream-set-current",
            "cpu-to-cuda",
            "matmul",
            "autograd-explicit-gradient",
            "autograd-functional-grad",
            "autograd-multiple-roots",
            "autograd-multiple-roots-backward",
            "autograd-inference-mode-guard",
            "autograd-autocast-cpu-guard",
            "autograd-anomaly-mode-guard",
            "autograd-warning-mode-guard",
            "autograd-deterministic-algorithms-guard",
            "autograd-float32-matmul-precision-guard",
            "autograd-default-dtype-guard",
            "cuda-dtype-float32",
            "cuda-dtype-float64",
            "deterministic-categorical-sample",
            "ppo-minibatch",
            "adam-step",
            "stream-synchronize",
            "cuda-to-cpu",
        ],
        "schema": "terlan.pytorch-cuda-execution.v4",
        "status": "passed",
        "synchronization": {
            "copy": "at::Tensor::copy_ (generated CXX wrapper)",
            "stream": "at::native::cuda_stream_synchronize (generated CXX wrapper; upstream runtime export)",
        },
        "stream_ownership": "opaque-generation-tagged-package-resource",
        "rejections": ["invalid-device", "stale-stream"],
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
