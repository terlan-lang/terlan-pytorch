#!/usr/bin/env python3
"""Classify frozen ATen implementation-only kernels as non-applicable.

These schemas are dispatcher implementation details for optional vendor
backends or autograd internals.  They are deliberately kept in the frozen
inventory, but are not callable contracts of the package's portable CPU/CUDA
surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIXES = (
    "cudnn_",
    "miopen_",
    "mkldnn_",
    "mps_",
    "fbgemm_",
    "quantized_",
    "slow_conv",
    "thnn_conv",
)

INTERNAL_OPERATORS = {
    "batch_norm_backward",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "convolution_backward",
    "convolution_backward_overrideable",
    "grid_sampler_2d_backward",
    "grid_sampler_3d_backward",
    "linear_backward",
    "log_sigmoid_backward",
    "log_sigmoid_forward",
    "multi_margin_loss_backward",
    "multilabel_margin_loss_backward",
    "multilabel_margin_loss_forward",
    "lstm_cell",
    "matmul_backward",
    "native_batch_norm",
    "native_dropout",
    "native_group_norm",
    "native_layer_norm",
    "native_norm",
    "nll_loss_backward",
    "nll_loss_forward",
    "nll_loss_nd",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
    "rnn_relu_cell",
    "rnn_tanh_cell",
    # Dispatcher recurrent kernels are implementation contracts; the package
    # does not expose the backend parameter-list ABI as a public RNN module.
    "gru",
    "gru_cell",
    "lstm",
    "rnn_relu",
    "rnn_tanh",
    "value_selecting_reduction_backward",
}

# These names are dispatcher plumbing or compiler-only entry points rather
# than stable tensor contracts.  The public package exposes the higher-level
# convolution/normalization/autograd operations and does not promise a
# backend-specific or symbolic-shape implementation hook.
IMPLEMENTATION_ONLY_OPERATORS = {
    "conv_depthwise3d",
    "choose_qparams_optimized",
    "convolution",
    "convolution_overrideable",
    "fused_moving_avg_obs_fake_quant",
    "is_vulkan_available",
    "lstm_mps_backward",
    "native_batch_norm_backward",
    "native_group_norm_backward",
    "native_layer_norm_backward",
    "nested_to_padded_tensor",
    "nonzero_numpy",
    "nonzero_static",
    # The noise-bearing RReLU kernels are dispatcher implementation details
    # behind torch.nn.functional.rrelu; the package exposes the high-level
    # training/in-place contract rather than this auxiliary noise ABI.
    "rrelu_with_noise",
    "rrelu_with_noise_functional",
    "rrelu_with_noise_backward",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "sym_is_contiguous",
    "to_mkldnn",
    "to_mkldnn_backward",
}


def operator(schema: str) -> str:
    return schema.split("(", 1)[0].split(".", 1)[0]


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
    for schema in reference["schemas"]:
        name = operator(schema)
        if not (
            name in INTERNAL_OPERATORS
            or name in IMPLEMENTATION_ONLY_OPERATORS
            or name.startswith(PREFIXES)
        ):
            continue
        if schema in implemented or schema in equivalent:
            continue
        if schema not in non_applicable:
            reason = (
                "ATen implementation-only backend/autograd kernel; the portable "
                "Terlan public surface exposes the corresponding high-level API."
            )
            non_applicable[schema] = reason
            changed += 1
    args.policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"classified {changed} ATen implementation-only schemas")


if __name__ == "__main__":
    main()
