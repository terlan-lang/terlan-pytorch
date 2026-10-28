#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check-cuda-report.py REPORT")
    report = json.loads(pathlib.Path(sys.argv[1]).read_text())
    if report.get("schema") != "terlan.pytorch-cuda-execution.v4":
        raise SystemExit("error[terlan_pytorch_cuda_report_schema]: invalid schema")
    if report.get("status") != "passed":
        raise SystemExit("error[terlan_pytorch_cuda_report_status]: CUDA execution did not pass")
    if report.get("backend") != "libtorch-cxx-generated":
        raise SystemExit("error[terlan_pytorch_cuda_backend]: CUDA report is not backed by generated CXX")
    if report.get("operations") != [
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
    ]:
        raise SystemExit("error[terlan_pytorch_cuda_report_operations]: operation proof drifted")
    if report.get("device_pointer_exposed") is not False:
        raise SystemExit("error[terlan_pytorch_cuda_pointer_exposure]: pointer exposure is forbidden")
    distribution = report.get("distribution", {})
    if not distribution.get("build_version", "").startswith("2.13.0+cu"):
        raise SystemExit("error[terlan_pytorch_cuda_version]: CUDA distribution version drifted")
    if distribution.get("cuda_library") != "libtorch_cuda.so":
        raise SystemExit("error[terlan_pytorch_cuda_library]: CUDA library identity drifted")
    digest = distribution.get("cuda_library_sha256", "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit("error[terlan_pytorch_cuda_digest]: CUDA library digest is invalid")
    if report.get("synchronization") != {
        "copy": "at::Tensor::copy_ (generated CXX wrapper)",
        "stream": "at::native::cuda_stream_synchronize (generated CXX wrapper; upstream runtime export)",
    }:
        raise SystemExit("error[terlan_pytorch_cuda_synchronization]: synchronization proof drifted")
    if report.get("stream_ownership") != "opaque-generation-tagged-package-resource":
        raise SystemExit("error[terlan_pytorch_cuda_stream_ownership]: stream ownership drifted")
    if report.get("rejections") != ["invalid-device", "stale-stream"]:
        raise SystemExit("error[terlan_pytorch_cuda_rejections]: adversarial proof drifted")
    print(
        f"[terlan-pytorch-cuda] status=passed backend={report['backend']} "
        f"operations={','.join(report['operations'])}"
    )


if __name__ == "__main__":
    main()
