#!/usr/bin/env python3
"""Preserve the established PyTorch surface over generated CXX modules.

The CXX metadata intentionally transports symbolic enums as atoms.  The
public PyTorch package has historically exposed the reviewed dtype constants
as an integer enum and a structured device helper API, so those small pure
Terlan conveniences are applied after the generated module is emitted.
"""

import argparse
import re
from pathlib import Path


DTYPE = """pub type DType: Int =
    UINT8 = 0 | INT8 = 1 | INT16 = 2 | INT32 = 3 | INT64 = 4
    | FLOAT16 = 5 | FLOAT32 = 6 | FLOAT64 = 7 | COMPLEX32 = 8
    | COMPLEX64 = 9 | COMPLEX128 = 10 | BOOL = 11 | QINT8 = 12
    | QUINT8 = 13 | QINT32 = 14 | BFLOAT16 = 15 | QUINT4X2 = 16
    | QUINT2X4 = 17 | BITS1X8 = 18 | BITS2X4 = 19 | BITS4X2 = 20
    | BITS8 = 21 | BITS16 = 22 | FLOAT8_E5M2 = 23
    | FLOAT8_E4M3FN = 24 | FLOAT8_E5M2FNUZ = 25
    | FLOAT8_E4M3FNUZ = 26 | UINT16 = 27 | UINT32 = 28 | UINT64 = 29
    | UINT1 = 30 | UINT2 = 31 | UINT3 = 32 | UINT4 = 33 | UINT5 = 34
    | UINT6 = 35 | UINT7 = 36 | INT1 = 37 | INT2 = 38 | INT3 = 39
    | INT4 = 40 | INT5 = 41 | INT6 = 42 | INT7 = 43
    | FLOAT8_E8M0FNU = 44 | FLOAT4_E2M1FN_X2 = 45.
"""

CONVENIENCES = """
/** Returns the stable c10 scalar-type code used by native adapters. */
pub dtype_code(dtype: DType): Int ->
    if {
        dtype == DType.UINT8 -> 0;
        dtype == DType.INT8 -> 1;
        dtype == DType.INT16 -> 2;
        dtype == DType.INT32 -> 3;
        dtype == DType.INT64 -> 4;
        dtype == DType.FLOAT16 -> 5;
        dtype == DType.FLOAT32 -> 6;
        dtype == DType.FLOAT64 -> 7;
        dtype == DType.COMPLEX32 -> 8;
        dtype == DType.COMPLEX64 -> 9;
        dtype == DType.COMPLEX128 -> 10;
        dtype == DType.BOOL -> 11;
        dtype == DType.QINT8 -> 12;
        dtype == DType.QUINT8 -> 13;
        dtype == DType.QINT32 -> 14;
        dtype == DType.BFLOAT16 -> 15;
        dtype == DType.QUINT4X2 -> 16;
        dtype == DType.QUINT2X4 -> 17;
        dtype == DType.BITS1X8 -> 18;
        dtype == DType.BITS2X4 -> 19;
        dtype == DType.BITS4X2 -> 20;
        dtype == DType.BITS8 -> 21;
        dtype == DType.BITS16 -> 22;
        dtype == DType.FLOAT8_E5M2 -> 23;
        dtype == DType.FLOAT8_E4M3FN -> 24;
        dtype == DType.FLOAT8_E5M2FNUZ -> 25;
        dtype == DType.FLOAT8_E4M3FNUZ -> 26;
        dtype == DType.UINT16 -> 27;
        dtype == DType.UINT32 -> 28;
        dtype == DType.UINT64 -> 29;
        dtype == DType.UINT1 -> 30;
        dtype == DType.UINT2 -> 31;
        dtype == DType.UINT3 -> 32;
        dtype == DType.UINT4 -> 33;
        dtype == DType.UINT5 -> 34;
        dtype == DType.UINT6 -> 35;
        dtype == DType.UINT7 -> 36;
        dtype == DType.INT1 -> 37;
        dtype == DType.INT2 -> 38;
        dtype == DType.INT3 -> 39;
        dtype == DType.INT4 -> 40;
        dtype == DType.INT5 -> 41;
        dtype == DType.INT6 -> 42;
        dtype == DType.INT7 -> 43;
        dtype == DType.FLOAT8_E8M0FNU -> 44;
        true -> 45
    }.

/** Returns the canonical CPU device spelling. */
pub cpu(): Device -> "cpu".

/** Returns the numeric backend kind for a canonical device spelling. */
pub device_kind(device: Device): Int ->
    if {
        device == "cpu" -> 0;
        device.starts_with("cuda:") -> 1;
        true -> 0
    }.

/** Returns the CUDA ordinal, or -1 for CPU. */
pub device_index_for(device: Device): Int ->
    if {
        device == "cpu" -> -1;
        device.starts_with("cuda:") ->
            case std.core.Int.from_string(device.utf8_slice(5, device.length() - 5)) {
                Some(index) -> index;
                None -> -1
            };
        true -> -1
    }.

/** Returns a canonical CUDA device spelling. */
pub cuda(index: Int): Device ->
    "cuda:" + std.core.Int.to_string(index).
"""

# The reviewed package tests intentionally exercise the small compatibility
# surface that historically lived beside the generated Tensor declarations.
# These names are backed by generated CXX leaves already present in the
# manifest; publishing them here keeps the CXX-generated package callable by
# the same behavior fixtures without introducing handwritten ABI glue.
PUBLIC_COMPATIBILITY_FUNCTIONS = (
    "ones",
    "ones_dtype",
    "ones_layout",
    "ones_device",
    "ones_options",
    "item",
    "broadcast_tensors",
    "meshgrid",
    "meshgrid_indexing",
    "mul_int",
    "abs",
    "ceil",
    "floor",
    "reciprocal",
    "round",
    "sign",
    "square",
    "trunc",
    "device",
    "layout",
    "is_defined",
    "is_cpu",
    "is_cuda",
    "requires_grad",
    "is_leaf",
    "has_storage",
    "is_complex",
    "is_floating_point",
    "is_quantized",
    "is_signed",
    "is_sparse",
    "dispose_scalar",
    "scalar_to_int",
    "scalar_to_float",
)

COMPATIBILITY_ALIASES = """
/* The compatibility aliases call these generated CXX leaves directly.  Keeping
 * the declarations local avoids cyclic imports through modules that also
 * consume the canonical Tensor handle. */
@compiler.native {pytorch.dense.type_as}
tensor_type_as(tensor: Tensor, other: Tensor): Tensor ->
    native.

@compiler.native {pytorch.type_relations.get_device}
tensor_get_device(tensor: Tensor): Int ->
    native.

@compiler.native {pytorch.view.adjoint}
tensor_adjoint(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.conj}
tensor_conj(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.detach}
tensor_detach(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.positive}
tensor_positive(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.ravel}
tensor_ravel(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.resolve_conj}
tensor_resolve_conj(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.resolve_neg}
tensor_resolve_neg(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.squeeze}
tensor_squeeze(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.view.t}
tensor_t(tensor: Tensor): Tensor ->
    native.

@compiler.native {pytorch.sparse.to_sparse_configured}
tensor_to_sparse_configured(
    tensor: Tensor,
    layout: Int,
    has_layout: Bool,
    block_size: List[Int],
    has_block_size: Bool,
    dense_dimensions: Int,
    has_dense_dimensions: Bool,
): Tensor ->
    native.

@compiler.native {pytorch.sparse.to_dense_configured}
tensor_to_dense_configured(
    tensor: Tensor,
    dtype: Int,
    has_dtype: Bool,
    masked_grad: Bool,
    has_masked_grad: Bool,
): Tensor ->
    native.

@compiler.native {pytorch.clamp.cxx.clamp_scalar_min_max}
tensor_clip_float(
    tensor: Tensor,
    min_is_int: Bool,
    min_int: Int,
    min_float: Float,
    max_is_int: Bool,
    max_int: Int,
    max_float: Float,
): Tensor ->
    native.

@compiler.native {pytorch.scaled_arithmetic.cxx.scaled_add_tensor_in_place_cxx}
tensor_add_in_place(
    left: Tensor,
    right: Tensor,
    alpha_is_int: Bool,
    alpha_int: Int,
    alpha_float: Float,
): Unit ->
    native.

@compiler.native {pytorch.scaled_arithmetic.cxx.scaled_sub_tensor_in_place_cxx}
tensor_subtract_in_place(
    left: Tensor,
    right: Tensor,
    alpha_is_int: Bool,
    alpha_int: Int,
    alpha_float: Float,
): Unit ->
    native.

@compiler.native {pytorch.binary_scalar_native.cxx.binary_tensor_in_mul}
tensor_multiply_in_place(left: Tensor, right: Tensor): Unit ->
    native.

/** Compatibility alias for the floating-point scalar multiplication leaf. */
pub mul_float(tensor: Tensor, value: Float): Tensor ->
    multiply_scalar(tensor, value).

/** Compatibility alias for the tensor rank projection. */
pub rank(tensor: Tensor): Int ->
    dim(tensor).

/** Compatibility alias for the tensor element-count projection. */
pub element_count(tensor: Tensor): Int ->
    numel(tensor).

/** Returns the storage width in bytes for the tensor's scalar dtype. */
pub element_size(tensor: Tensor): Int ->
    let code = dtype(tensor);
    if {
        code == 0 or code == 1 or code == 11 or code == 12 or code == 13
            or code >= 16 and code <= 26 -> 1;
        code == 2 or code == 5 or code == 15 -> 2;
        code == 3 or code == 8 -> 4;
        code == 4 or code == 6 or code == 7 or code == 9 or code == 14 -> 8;
        code == 10 -> 16;
        true -> 1
    }.

/** Zero-filled tensor compatibility constructor for legacy Tensor imports. */
pub zeros(size: List[Int]): Tensor ->
    multiply_scalar(ones(size), 0.0).

/** Typed zero-filled tensor compatibility constructor. */
pub zeros(size: List[Int], dtype: Int): Tensor ->
    to_dtype(zeros(size), dtype).

/** Typed range constructor with the reviewed CXX range leaf. */
pub arange(start: Float, end: Float, step: Float, dtype: DType): Tensor ->
    arange_typed_native(start, end, step, dtype_code(dtype), 0, -1, false).

/** Typed evenly-spaced constructor with the reviewed CXX range leaf. */
pub linspace(start: Float, end: Float, steps: Int, dtype: DType): Tensor ->
    linspace_typed_native(start, end, steps, dtype_code(dtype), 0, -1, false).

/** Typed rectangular identity constructor with the reviewed CXX range leaf. */
pub eye(rows: Int, columns: Int, dtype: DType): Tensor ->
    eye_typed_native(rows, columns, dtype_code(dtype), 0, -1, false).

/** Tensor CPU placement convenience alias matching torch.Tensor.cpu(). */
pub cpu(tensor: Tensor): Tensor ->
    to_device(tensor, 0, -1).

/** Rank compatibility alias matching torch.Tensor.ndimension(). */
pub ndimension(tensor: Tensor): Int ->
    dim(tensor).

/** Element-count compatibility alias matching torch.Tensor.nelement(). */
pub nelement(tensor: Tensor): Int ->
    numel(tensor).

/** Dtype-copy convenience alias matching torch.Tensor.type_as(). */
pub type_as(tensor: Tensor, other: Tensor): Tensor ->
    tensor_type_as(tensor, other).

/** Default pinned-state convenience alias matching torch.Tensor.is_pinned(). */
pub is_pinned(tensor: Tensor): Bool ->
    is_pinned(tensor, false, cpu()).

/** Device-index convenience alias matching torch.Tensor.get_device(). */
pub get_device(tensor: Tensor): Int ->
    tensor_get_device(tensor).

/** Creates a zero-filled tensor on an explicit CPU or CUDA device. */
pub zeros_on(size: List[Int], dtype: DType, device: Device): Tensor ->
    full_typed_native(
        size,
        0.0,
        dtype_code(dtype),
        device_kind(device),
        device_index_for(device),
    ).

/** Tensor view convenience aliases matching PyTorch Tensor methods. */
pub adjoint(tensor: Tensor): Tensor ->
    tensor_adjoint(tensor).

pub conj(tensor: Tensor): Tensor ->
    tensor_conj(tensor).

pub detach(tensor: Tensor): Tensor ->
    tensor_detach(tensor).

pub positive(tensor: Tensor): Tensor ->
    tensor_positive(tensor).

pub ravel(tensor: Tensor): Tensor ->
    tensor_ravel(tensor).

pub t(tensor: Tensor): Tensor ->
    tensor_t(tensor).

pub resolve_conj(tensor: Tensor): Tensor ->
    tensor_resolve_conj(tensor).

pub resolve_neg(tensor: Tensor): Tensor ->
    tensor_resolve_neg(tensor).

pub squeeze(tensor: Tensor): Tensor ->
    tensor_squeeze(tensor).

/** Tensor CUDA placement convenience alias matching torch.Tensor.cuda(). */
pub cuda(tensor: Tensor): Tensor ->
    to_device(tensor, 1, 0).

/** Tensor CUDA placement convenience alias with an explicit ordinal. */
pub cuda(tensor: Tensor, index: Int): Tensor ->
    to_device(tensor, 1, index).

/** Explicit-gradient overload matching torch.Tensor.backward(gradient). */
pub backward(loss: Tensor, gradient: Tensor): Unit ->
    backward_with_gradient(loss, gradient, false, false).

/** Explicit-gradient overload with graph-retention controls. */
pub backward(
    loss: Tensor,
    gradient: Tensor,
    retain_graph: Bool,
    create_graph: Bool,
): Unit ->
    backward_with_gradient(loss, gradient, retain_graph, create_graph).

/** Multiple-root backward alias with the public Unit-shaped contract. */
pub backward_many(
    outputs: List[Tensor],
    grad_outputs: List[Tensor],
    retain_graph: Bool = false,
    create_graph: Bool = false,
): Unit ->
    let token = backward_many_token(outputs, grad_outputs, retain_graph, create_graph);
    dispose(token).

/**
 * Common dtype overload of torch.Tensor.to().  PyTorch exposes additional
 * Python-only overloads (including memory-format and copy controls); these
 * reviewed CXX overloads preserve the ownership and conversion semantics
 * available in the generated manifest without reintroducing ABI glue.
 */
pub to(tensor: Tensor, dtype: Int): Tensor ->
    to_dtype(tensor, dtype).

/** Device overload of torch.Tensor.to(). */
pub to(tensor: Tensor, device_type: Int, device_index: Int): Tensor ->
    to_device(tensor, device_type, device_index).

/** Tensor clip alias matching the common floating-point torch.Tensor.clip overload. */
pub clip(tensor: Tensor, minimum: Float, maximum: Float): Tensor ->
    tensor_clip_float(tensor, false, 0, minimum, false, 0, maximum).

/** Tensor in-place addition alias matching torch.Tensor.add_. */
pub add_in_place(left: Tensor, right: Tensor, alpha: Float): Unit ->
    tensor_add_in_place(left, right, false, 0, alpha).

/** Tensor in-place subtraction alias matching torch.Tensor.sub_ / subtract_. */
pub subtract_in_place(left: Tensor, right: Tensor, alpha: Float): Unit ->
    tensor_subtract_in_place(left, right, false, 0, alpha).

/** Tensor in-place multiplication alias matching torch.Tensor.multiply_. */
pub multiply_in_place(left: Tensor, right: Tensor): Unit ->
    tensor_multiply_in_place(left, right).

/** Dense-to-COO convenience alias matching torch.Tensor.to_sparse_coo(). */
pub to_sparse_coo(tensor: Tensor): Tensor ->
    tensor_to_sparse_configured(tensor, 1, true, [], false, 0, false).

/** Sparse-to-dense convenience alias matching torch.Tensor.to_dense(). */
pub to_dense(tensor: Tensor): Tensor ->
    tensor_to_dense_configured(tensor, 0, false, false, false).

/** Complex-half conversion matching torch.chalf's default policy. */
pub chalf(tensor: Tensor): Tensor ->
    to_dtype(tensor, 8).

/** Tensor dtype convenience aliases matching PyTorch's no-argument methods. */
pub float(tensor: Tensor): Tensor ->
    to_dtype(tensor, 6).

pub double(tensor: Tensor): Tensor ->
    to_dtype(tensor, 7).

pub half(tensor: Tensor): Tensor ->
    to_dtype(tensor, 5).

pub int(tensor: Tensor): Tensor ->
    to_dtype(tensor, 3).

pub long(tensor: Tensor): Tensor ->
    to_dtype(tensor, 4).

pub bool(tensor: Tensor): Tensor ->
    to_dtype(tensor, 11).

pub bfloat16(tensor: Tensor): Tensor ->
    to_dtype(tensor, 15).

pub byte(tensor: Tensor): Tensor ->
    to_dtype(tensor, 0).

pub cdouble(tensor: Tensor): Tensor ->
    to_dtype(tensor, 10).

pub cfloat(tensor: Tensor): Tensor ->
    to_dtype(tensor, 9).

pub char(tensor: Tensor): Tensor ->
    to_dtype(tensor, 1).

pub short(tensor: Tensor): Tensor ->
    to_dtype(tensor, 2).
"""

SHAPE_PARTITION_COMPATIBILITY = """
/** Splits the depth dimension into a section count. */
@compiler.native {pytorch.shape_partition.dsplit_sections}
pub dsplit_sections(input: Tensor, sections: Int): List[Tensor] -> native.

/** Splits the depth dimension at explicit indices. */
@compiler.native {pytorch.shape_partition.dsplit_indices}
pub dsplit_indices(input: Tensor, indices: List[Int]): List[Tensor] -> native.

/** Splits the horizontal dimension into a section count. */
@compiler.native {pytorch.shape_partition.hsplit_sections}
pub hsplit_sections(input: Tensor, sections: Int): List[Tensor] -> native.

/** Splits the horizontal dimension at explicit indices. */
@compiler.native {pytorch.shape_partition.hsplit_indices}
pub hsplit_indices(input: Tensor, indices: List[Int]): List[Tensor] -> native.

/** Splits the vertical dimension into a section count. */
@compiler.native {pytorch.shape_partition.vsplit_sections}
pub vsplit_sections(input: Tensor, sections: Int): List[Tensor] -> native.

/** Splits the vertical dimension at explicit indices. */
@compiler.native {pytorch.shape_partition.vsplit_indices}
pub vsplit_indices(input: Tensor, indices: List[Int]): List[Tensor] -> native.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="generated")
    args = parser.parse_args()
    functional_completion = Path(args.root) / "src/pytorch/FunctionalCompletion.terl"
    if functional_completion.exists():
        text = functional_completion.read_text(encoding="utf-8")
        # Generic Option[Tensor] helpers must resolve Tensor through the
        # canonical module import.  The raw binder's `pytorch.Tensor.Tensor`
        # spelling is sufficient for direct leaves but creates a distinct
        # generic argument identity for Some(value)/None call sites.
        text = text.replace(
            "import type pytorch.Tensor.Tensor.",
            "import type pytorch.Tensor.",
        )
        text = text.replace("Option[Tensor]", "Option[pytorch.Tensor.Tensor]")
        option_import = "import std.core.Option.{None, Option, Some}."
        if "loss_optional_tensor" in text and option_import not in text:
            marker = "module pytorch.FunctionalCompletion.\n\n"
            if marker not in text:
                raise SystemExit("generated FunctionalCompletion module header was not found")
            text = text.replace(marker, marker + option_import + "\n\n", 1)
        # The generated surface does not contain Python's legacy upsample
        # aliases.  Keep the compatibility shims finite by routing them to
        # the configured interpolation leaf instead of emitting accidental
        # self-recursive definitions.
        text = re.sub(
            r'(?ms)pub upsample_bilinear\(input: Tensor, has_size: Bool, size: List\[Int\], has_scale: Bool, scale_factor: List\[Float\]\): Tensor ->\s*upsample_bilinear\(input, has_size, size, has_scale, scale_factor\)\.',
            'pub upsample_bilinear(input: Tensor, has_size: Bool, size: List[Int], has_scale: Bool, scale_factor: List[Float]): Tensor ->\n'
            '    interpolate(input, has_size, size, has_scale, scale_factor, "bilinear", true, false, false, false, false).',
            text,
        )
        text = re.sub(
            r'(?ms)pub upsample_nearest\(input: Tensor, has_size: Bool, size: List\[Int\], has_scale: Bool, scale_factor: List\[Float\]\): Tensor ->\s*upsample_nearest\(input, has_size, size, has_scale, scale_factor\)\.',
            'pub upsample_nearest(input: Tensor, has_size: Bool, size: List[Int], has_scale: Bool, scale_factor: List[Float]): Tensor ->\n'
            '    interpolate(input, has_size, size, has_scale, scale_factor, "nearest", false, false, false, false, false).',
            text,
        )
        # These projection helpers are part of the public completion surface
        # (the parity fixtures import them directly), even though the CXX
        # binder emits them without an explicit visibility marker.
        text = re.sub(r'(?m)^(lp_pool[13]d_(?:int|float)\()', r'pub \1', text)
        # Keep the public overloads self-contained.  The native application
        # closure admits generated leaves and exported overloads, but does not
        # retain an otherwise-unreferenced pure projection helper as a resume
        # target; inline the eight finite projection forms at their callsite.
        projection_calls = {
            "lp_pool1d_int(input, norm_type, kernel_size, false, kernel_size, false)":
                "lp_pool1d_configured(input, true, norm_type, 0.0, kernel_size, false, kernel_size, false)",
            "lp_pool1d_float(input, norm_type, kernel_size, false, kernel_size, false)":
                "lp_pool1d_configured(input, false, 0, norm_type, kernel_size, false, kernel_size, false)",
            "lp_pool1d_int(input, norm_type, kernel_size, false, kernel_size, ceil_mode)":
                "lp_pool1d_configured(input, true, norm_type, 0.0, kernel_size, false, kernel_size, ceil_mode)",
            "lp_pool1d_float(input, norm_type, kernel_size, false, kernel_size, ceil_mode)":
                "lp_pool1d_configured(input, false, 0, norm_type, kernel_size, false, kernel_size, ceil_mode)",
            "lp_pool1d_int(input, norm_type, kernel_size, true, stride, false)":
                "lp_pool1d_configured(input, true, norm_type, 0.0, kernel_size, true, stride, false)",
            "lp_pool1d_float(input, norm_type, kernel_size, true, stride, false)":
                "lp_pool1d_configured(input, false, 0, norm_type, kernel_size, true, stride, false)",
            "lp_pool1d_int(input, norm_type, kernel_size, true, stride, ceil_mode)":
                "lp_pool1d_configured(input, true, norm_type, 0.0, kernel_size, true, stride, ceil_mode)",
            "lp_pool1d_float(input, norm_type, kernel_size, true, stride, ceil_mode)":
                "lp_pool1d_configured(input, false, 0, norm_type, kernel_size, true, stride, ceil_mode)",
            "lp_pool3d_int(input, norm_type, kernel_size, false, kernel_size, false)":
                "lp_pool3d_configured(input, true, norm_type, 0.0, kernel_size, false, kernel_size, false)",
            "lp_pool3d_float(input, norm_type, kernel_size, false, kernel_size, false)":
                "lp_pool3d_configured(input, false, 0, norm_type, kernel_size, false, kernel_size, false)",
            "lp_pool3d_int(input, norm_type, kernel_size, false, kernel_size, ceil_mode)":
                "lp_pool3d_configured(input, true, norm_type, 0.0, kernel_size, false, kernel_size, ceil_mode)",
            "lp_pool3d_float(input, norm_type, kernel_size, false, kernel_size, ceil_mode)":
                "lp_pool3d_configured(input, false, 0, norm_type, kernel_size, false, kernel_size, ceil_mode)",
            "lp_pool3d_int(input, norm_type, kernel_size, true, stride, false)":
                "lp_pool3d_configured(input, true, norm_type, 0.0, kernel_size, true, stride, false)",
            "lp_pool3d_float(input, norm_type, kernel_size, true, stride, false)":
                "lp_pool3d_configured(input, false, 0, norm_type, kernel_size, true, stride, false)",
            "lp_pool3d_int(input, norm_type, kernel_size, true, stride, ceil_mode)":
                "lp_pool3d_configured(input, true, norm_type, 0.0, kernel_size, true, stride, ceil_mode)",
            "lp_pool3d_float(input, norm_type, kernel_size, true, stride, ceil_mode)":
                "lp_pool3d_configured(input, false, 0, norm_type, kernel_size, true, stride, ceil_mode)",
        }
        for source_call, configured_call in projection_calls.items():
            text = text.replace(source_call, configured_call)
        text = text.replace(
            "gaussian_nll_loss_tensor(input, target, variance, full, eps, reduction)",
            "gaussian_nll_loss_configured(input, target, true, variance, 0.0, full, eps, reduction)",
        )
        text = text.replace(
            "gaussian_nll_loss_scalar(input, target, variance, full, eps, reduction)",
            "gaussian_nll_loss_configured(input, target, false, input, variance, full, eps, reduction)",
        )
        functional_completion.write_text(text, encoding="utf-8")
    unique_completion = Path(args.root) / "src/pytorch/UniqueCompletion.terl"
    if unique_completion.exists():
        text = unique_completion.read_text(encoding="utf-8")
        marker = "module pytorch.UniqueCompletion.\n\n"
        imports = (
            "import pytorch.Tensor.{dispose, flatten}.\n"
            "import std.core.Option.{None, Option, Some}.\n\n"
        )
        if marker in text and "import std.core.Option." not in text:
            text = text.replace(marker, marker + imports, 1)
        unique_completion.write_text(text, encoding="utf-8")
    tensor = Path(args.root) / "src/pytorch/Tensor.terl"
    text = tensor.read_text(encoding="utf-8")
    # The generated Tensor module also contains a private direct detach leaf.
    # Keep that leaf available for native callers under a distinct name so the
    # public Tensor.detach compatibility alias can route through the reviewed
    # View contract without creating a duplicate Terlan definition.
    text = re.sub(
        r"(?m)^detach\(tensor: Tensor\): Tensor ->(?:\s+native\.)?$",
        "detach_native(tensor: Tensor): Tensor -> native.",
        text,
        count=1,
    )
    if "pub type DType = Atom[" in text:
        dtype_start = text.index("pub type DType = Atom[")
        dtype_end = text.index("\n/** Dense strided tensor layout. */", dtype_start)
        text = text[:dtype_start] + DTYPE.rstrip() + text[dtype_end:]
    marker = "pub type Device = String."
    if "pub device_index_for(device: Device)" not in text:
        text = text.replace(marker, marker + "\n" + CONVENIENCES.rstrip(), 1)
    # CXX bridge adapters transport c10::ScalarType and c10::Layout as their
    # integer representation. Keep the generated enum declarations, but
    # make native leaf signatures match the bridge ABI exactly.
    text = text.replace("dtype: DType", "dtype: Int")
    text = text.replace("layout: Layout", "layout: Int")
    text = text.replace("dtype(tensor: Tensor): DType", "dtype(tensor: Tensor): Int")
    text = text.replace("layout(tensor: Tensor): Layout", "layout(tensor: Tensor): Int")
    # The broad ABI spelling above also sees the pure dtype-code helper.  It
    # remains a typed public projection over DType, rather than a bridge leaf.
    text = text.replace(
        "pub dtype_code(dtype: Int): Int",
        "pub dtype_code(dtype: DType): Int",
        1,
    )
    text = re.sub(
        r"/\*\*\n"
        r" \* Returns the tensor dtype as a reviewed symbolic value\.\n"
        r" \*/\n"
        r"@compiler\.native \{pytorch\.cxx_migration\.tensor\.dtype\}\n"
        r"(?:pub )?dtype\(tensor: Tensor\): (?:DType|Int) ->\n"
        r"\s+native\.\n\n",
        "",
        text,
    )

    # The generated dense-creation dispatcher predates the integer CXX
    # compatibility surface.  Its private selector helpers still returned
    # typed enum atoms even though their callers and native leaves transport
    # reviewed integer codes.  Normalize only this generated CXX module; the
    # package-authored Python-facing source keeps its stronger enum types.
    creation = Path(args.root) / "src/pytorch/CreationCompletionNative.terl"
    if creation.exists():
        creation_text = creation.read_text(encoding="utf-8")
        creation_text = creation_text.replace(
            "import type pytorch.Tensor.{DType, Device, Layout}.",
            "import type pytorch.Tensor.{Device}.",
        )
        creation_text = creation_text.replace("dtype: DType", "dtype: Int")
        creation_text = creation_text.replace("layout: Layout", "layout: Int")
        creation_text = creation_text.replace(
            "memory_format: MemoryFormat", "memory_format: Int"
        )
        creation_text = creation_text.replace(
            "creation_dtype(code: Int): DType", "creation_dtype(code: Int): Int"
        )
        creation_text = creation_text.replace(
            "creation_layout(code: Int): Layout", "creation_layout(code: Int): Int"
        )
        creation_text = creation_text.replace(
            "creation_memory_format(code: Int): MemoryFormat",
            "creation_memory_format(code: Int): Int",
        )
        enum_codes = {
            "UInt8": 0,
            "Int8": 1,
            "Int16": 2,
            "Int32": 3,
            "Int64": 4,
            "Float16": 5,
            "Float32": 6,
            "Float64": 7,
            "Complex32": 8,
            "Complex64": 9,
            "Complex128": 10,
            "Bool": 11,
            "QInt8": 12,
            "QUInt8": 13,
            "QInt32": 14,
            "BFloat16": 15,
            "QUInt4x2": 16,
            "QUInt2x4": 17,
            "Bits1x8": 18,
            "Bits2x4": 19,
            "Bits4x2": 20,
            "Bits8": 21,
            "Bits16": 22,
            "Float8E5M2": 23,
            "Float8E4M3Fn": 24,
            "Float8E5M2Fnuz": 25,
            "Float8E4M3Fnuz": 26,
            "UInt16": 27,
            "UInt32": 28,
            "UInt64": 29,
            "UInt1": 30,
            "UInt2": 31,
            "UInt3": 32,
            "UInt4": 33,
            "UInt5": 34,
            "UInt6": 35,
            "UInt7": 36,
            "Int1": 37,
            "Int2": 38,
            "Int3": 39,
            "Int4": 40,
            "Int5": 41,
            "Int6": 42,
            "Int7": 43,
            "Float8E8M0Fnu": 44,
            "Float4E2M1FnX2": 45,
            "Strided": 0,
            "SparseCoo": 1,
            "SparseCsr": 2,
            "Mkldnn": 3,
            "SparseCsc": 4,
            "SparseBsr": 5,
            "SparseBsc": 6,
            "Jagged": 7,
            "Contiguous": 0,
            "Preserve": 1,
            "ChannelsLast": 2,
            "ChannelsLast3d": 3,
        }
        for name, code in enum_codes.items():
            creation_text = re.sub(
                rf"(code == \d+ -> ){re.escape(name)}([;\n])",
                rf"\g<1>{code}\2",
                creation_text,
            )
            creation_text = re.sub(
                rf"(true -> let _failed = [^\n]+; ){re.escape(name)}([ .])",
                rf"\g<1>{code}\2",
                creation_text,
            )
        creation.write_text(creation_text, encoding="utf-8")

    # The range and window native dispatchers use the same reviewed enum
    # selectors.  Keep their CXX-facing leaves on integer codes as well; this
    # prevents typed enum atoms from leaking into the integer compatibility
    # surface when a behavior fixture calls them directly.
    for module_name in ["RangeNative", "WindowNative"]:
        module = Path(args.root) / f"src/pytorch/{module_name}.terl"
        if not module.exists():
            continue
        module_text = module.read_text(encoding="utf-8")
        module_text = module_text.replace("dtype: DType", "dtype: Int")
        module_text = module_text.replace("layout: Layout", "layout: Int")
        module_text = module_text.replace(") : DType", ") : Int")
        module_text = module_text.replace("): DType", "): Int")
        module_text = module_text.replace("): Layout", "): Int")
        module_codes = {
            "UInt8": 0, "Int8": 1, "Int16": 2, "Int32": 3, "Int64": 4,
            "Float16": 5, "Float32": 6, "Float64": 7, "Complex32": 8,
            "Complex64": 9, "Complex128": 10, "QInt8": 12,
            "QUInt8": 13, "QInt32": 14, "BFloat16": 15, "QUInt4x2": 16,
            "QUInt2x4": 17, "Bits1x8": 18, "Bits2x4": 19, "Bits4x2": 20,
            "Bits8": 21, "Bits16": 22, "Float8E5M2": 23,
            "Float8E4M3Fn": 24, "Float8E5M2Fnuz": 25,
            "Float8E4M3Fnuz": 26, "UInt16": 27, "UInt32": 28, "UInt64": 29,
            "UInt1": 30, "UInt2": 31, "UInt3": 32, "UInt4": 33,
            "UInt5": 34, "UInt6": 35, "UInt7": 36, "Int1": 37,
            "Int2": 38, "Int3": 39, "Int4": 40, "Int5": 41, "Int6": 42,
            "Int7": 43, "Float8E8M0Fnu": 44, "Float4E2M1FnX2": 45,
            "Strided": 0, "SparseCoo": 1, "SparseCsr": 2, "Mkldnn": 3,
            "SparseCsc": 4, "SparseBsr": 5, "SparseBsc": 6, "Jagged": 7,
        }
        for name, code in module_codes.items():
            module_text = re.sub(rf"\b{re.escape(name)}\b", str(code), module_text)
        module_text = module_text.replace("code == 11 -> Bool", "code == 11 -> 11")
        module.write_text(module_text, encoding="utf-8")

    # The sparse declarations use the generated Tensor handle in their
    # signatures.  Older bind metadata emitted that name without an explicit
    # semantic type import, which makes it a distinct unresolved handle at
    # runtime and rejects a normal pytorch.Tensor.Tensor value.  Keep the
    # generated module's handle identity tied to the canonical Tensor type.
    sparse = Path(args.root) / "src/pytorch/Sparse.terl"
    if sparse.exists():
        sparse_text = sparse.read_text(encoding="utf-8")
        if "import type pytorch.Tensor.Tensor." not in sparse_text:
            sparse_text = sparse_text.replace(
                "module pytorch.Sparse.\n",
                "module pytorch.Sparse.\nimport type pytorch.Tensor.Tensor.\n",
                1,
            )
            sparse.write_text(sparse_text, encoding="utf-8")
    # Generated doc comments vary slightly across compiler revisions; the
    # annotation and declaration are the stable parts of this duplicate ABI
    # leaf, so remove them independently as a defensive second pass.
    text = re.sub(
        r"@compiler\.native \{pytorch\.cxx_migration\.tensor\.dtype\}\n"
        r"(?:pub )?dtype\(tensor: Tensor\): (?:DType|Int) ->\n"
        r"\s+native\.\n\n",
        "",
        text,
    )
    text = re.sub(
        r"@compiler\.native \{pytorch\.cxx_migration\.tensor\.dtype\}\n"
        r"(?:pub )?dtype\(tensor: Tensor\): (?:DType|Int) ->\s*"
        r"native\.\n\n",
        "",
        text,
    )
    # Keep device parsing in the generated public surface so typed CUDA
    # constructors do not silently collapse every ordinal to -1.
    if "import std.core.Option.{Some}." not in text:
        text = text.replace(
            "module pytorch.Tensor.\n",
            "module pytorch.Tensor.\n\nimport std.core.Option.{Some}.\n",
            1,
        )
    text = re.sub(
        r"/\*\* Returns the CUDA ordinal, or -1 for CPU\. \*/\n"
        r"pub device_index_for\(device: Device\): Int ->.*?\n    \}\.\n\n",
        "/** Returns the CUDA ordinal, or -1 for CPU. */\n"
        "pub device_index_for(device: Device): Int ->\n"
        "    if {\n"
        '        device == "cpu" -> -1;\n'
        '        device.starts_with("cuda:") ->\n'
        "            case std.core.Int.from_string(device.utf8_slice(5, device.length() - 5)) {\n"
        "                Some(index) -> index;\n"
        "                _ -> -1\n"
        "            };\n"
        "        true -> -1\n"
        "    }.\n\n",
        text,
        flags=re.DOTALL,
    )
    # The CXX manifest exposes both the migration Scalar-returning item leaf
    # and the legacy Int convenience overload.  Behavior fixtures use the
    # Scalar form (followed by scalar_to_*), so keep that name unambiguous and
    # retain the Int form under an explicit name for direct callers.
    text = text.replace(
        "@compiler.native {pytorch.tensor.item}\npub item(tensor: Tensor): Int -> native.",
        "@compiler.native {pytorch.tensor.item}\nitem_int(tensor: Tensor): Int -> native.",
        1,
    )
    for name in PUBLIC_COMPATIBILITY_FUNCTIONS:
        text = text.replace(f"\n{name}(", f"\npub {name}(")
    if "pub mul_float(tensor: Tensor" not in text:
        text += "\n" + COMPATIBILITY_ALIASES.rstrip() + "\n"
    tensor.write_text(text, encoding="utf-8")

    shape_partition = Path(args.root) / "src/pytorch/ShapePartition.terl"
    if shape_partition.exists():
        shape_text = shape_partition.read_text(encoding="utf-8")
        if "pub dsplit_sections(input: Tensor" not in shape_text:
            shape_partition.write_text(
                shape_text.rstrip() + "\n\n" + SHAPE_PARTITION_COMPATIBILITY.rstrip() + "\n",
                encoding="utf-8",
            )

    # LibTorch exposes quantized metadata queries as Tensor member methods in
    # addition to the generated ATen free-function declarations.  The CXX
    # bridge therefore emits these three leaves as methods; adjust the
    # protocol helper's call site to use the generated method bindings rather
    # than introducing handwritten ABI adapters.
    helper = Path(args.root) / "native/rust/src/bin/native_boundary_helper.rs"
    if helper.exists():
        helper_text = helper.read_text(encoding="utf-8")
        # CXX constructors return null for unavailable optional backends.  The
        # CUDA stream contract retains the reviewed status code used by the
        # adversarial suite rather than leaking a generic null-handle code.
        helper_text = helper_text.replace(
            'if value.is_null() {\n                    return native_null_failure("constructor returned null");\n                }\n                self.next_id += 1;\n                let id = self.next_id;\n                self.handles.insert(id, HandleEntry { generation: 1, type_name: "pytorch.Cuda.Stream", value: HandleValue::PytorchCudaStream(value) });',
            'if value.is_null() {\n                    return protocol_error("c_abi_status_7071", "CUDA backend is unavailable");\n                }\n                self.next_id += 1;\n                let id = self.next_id;\n                self.handles.insert(id, HandleEntry { generation: 1, type_name: "pytorch.Cuda.Stream", value: HandleValue::PytorchCudaStream(value) });',
            1,
        )
        helper_text = helper_text.replace(
            'let result = ffi::terlan_owned_value_pytorch_tensor_to_dtype(arg_0_ref, *arg_1);\n                if result.is_null() { return native_null_failure("native operation returned null"); }',
            'let result = ffi::terlan_owned_value_pytorch_tensor_to_dtype(arg_0_ref, *arg_1);\n                if result.is_null() { return protocol_error("c_abi_status_7064", "unsupported dtype code"); }',
            1,
        )
        helper_text = helper_text.replace(
            'let result = ffi::terlan_owned_value_pytorch_tensor_full_typed_native(arg_ints(arg_0), *arg_1, *arg_2, *arg_3, *arg_4);\n                if result.is_null() { return native_null_failure("native operation returned null"); }',
            'let result = ffi::terlan_owned_value_pytorch_tensor_full_typed_native(arg_ints(arg_0), *arg_1, *arg_2, *arg_3, *arg_4);\n                if result.is_null() {\n                    let code = if *arg_3 == 1 && *arg_4 < 0 { "c_abi_status_7060" } else { "c_abi_status_7061" };\n                    return protocol_error(code, "tensor construction failed");\n                }',
            1,
        )
        helper_text = helper_text.replace(
            'let value = ffi::terlan_owned_value_pytorch_tensor_ones(arg_ints(arg_0));\n                if value.is_null() {\n                    return native_null_failure("constructor returned null");\n                }',
            'let value = ffi::terlan_owned_value_pytorch_tensor_ones(arg_ints(arg_0));\n                if value.is_null() {\n                    return protocol_error("c_abi_status_7061", "tensor construction failed");\n                }',
            1,
        )
        helper_text = helper_text.replace(
            'let result = ffi::terlan_owned_value_pytorch_tensor_arange_typed_native(*arg_0, *arg_1, *arg_2, *arg_3, *arg_4, *arg_5, *arg_6);\n                if result.is_null() { return native_null_failure("native operation returned null"); }',
            'let result = ffi::terlan_owned_value_pytorch_tensor_arange_typed_native(*arg_0, *arg_1, *arg_2, *arg_3, *arg_4, *arg_5, *arg_6);\n                if result.is_null() { return protocol_error("c_abi_status_7090", "range construction failed"); }',
            1,
        )
        helper_text = helper_text.replace(
            'let result = ffi::terlan_owned_value_pytorch_tensor_linspace_typed_native(*arg_0, *arg_1, *arg_2, *arg_3, *arg_4, *arg_5, *arg_6);\n                if result.is_null() { return native_null_failure("native operation returned null"); }',
            'let result = ffi::terlan_owned_value_pytorch_tensor_linspace_typed_native(*arg_0, *arg_1, *arg_2, *arg_3, *arg_4, *arg_5, *arg_6);\n                if result.is_null() { return protocol_error("c_abi_status_7090", "range construction failed"); }',
            1,
        )
        helper_text = helper_text.replace(
            'let result = ffi::terlan_owned_value_pytorch_tensor_eye_typed_native(*arg_0, *arg_1, *arg_2, *arg_3, *arg_4, *arg_5);\n                if result.is_null() { return native_null_failure("native operation returned null"); }',
            'let result = ffi::terlan_owned_value_pytorch_tensor_eye_typed_native(*arg_0, *arg_1, *arg_2, *arg_3, *arg_4, *arg_5);\n                if result.is_null() { return protocol_error("c_abi_status_7090", "range construction failed"); }',
            1,
        )
        dtype_result = (
            'let code = match result.to_string_lossy().as_ref() { '
            '"uint8" => 0_i64, "int8" => 1_i64, "int16" => 2_i64, '
            '"int32" => 3_i64, "int64" => 4_i64, "float16" => 5_i64, '
            '"float32" => 6_i64, "float64" => 7_i64, "complex32" => 8_i64, '
            '"complex64" => 9_i64, "complex128" => 10_i64, "bool" => 11_i64, '
            '"qint8" => 12_i64, "quint8" => 13_i64, "qint32" => 14_i64, '
            '"bfloat16" => 15_i64, "quint4x2" => 16_i64, "quint2x4" => 17_i64, '
            '"bits1x8" => 18_i64, "bits2x4" => 19_i64, "bits4x2" => 20_i64, '
            '"bits8" => 21_i64, "bits16" => 22_i64, "float8_e5m2" => 23_i64, '
            '"float8_e4m3fn" => 24_i64, "float8_e5m2fnuz" => 25_i64, '
            '"float8_e4m3fnuz" => 26_i64, "uint16" => 27_i64, "uint32" => 28_i64, '
            '"uint64" => 29_i64, "uint1" => 30_i64, "uint2" => 31_i64, '
            '"uint3" => 32_i64, "uint4" => 33_i64, "uint5" => 34_i64, '
            '"uint6" => 35_i64, "uint7" => 36_i64, "int1" => 37_i64, '
            '"int2" => 38_i64, "int3" => 39_i64, "int4" => 40_i64, '
            '"int5" => 41_i64, "int6" => 42_i64, "int7" => 43_i64, '
            '"float8_e8m0fnu" => 44_i64, "float4_e2m1fn_x2" => 45_i64, '
            '_ => return protocol_error("invalid_enum_value", "dtype returned an unselected enum value") }; '
            'format!("ok_int {code}")'
        )
        layout_result = (
            'let code = match result.to_string_lossy().as_ref() { "strided" => 0_i64, '
            '"sparse_coo" => 1_i64, "sparse_csr" => 2_i64, "mkldnn" => 3_i64, '
            '"sparse_csc" => 4_i64, "sparse_bsr" => 5_i64, "sparse_bsc" => 6_i64, '
            '"jagged" => 7_i64, _ => return protocol_error("invalid_enum_value", '
            '"layout returned an unselected enum value") }; format!("ok_int {code}")'
        )
        enum_atom_result = 'format!("ok_atom {}", STANDARD.encode(result.as_bytes()))'
        helper_text = helper_text.replace(enum_atom_result, dtype_result, 1)
        helper_text = helper_text.replace(enum_atom_result, layout_result, 1)
        helper_text = helper_text.replace(
            "result.as_str()", "result.to_string_lossy().as_ref()"
        )
        helper_text = helper_text.replace("ffi::q_scale(arg_0_ref)", "arg_0_ref.q_scale()")
        helper_text = helper_text.replace("ffi::q_zero_point(arg_0_ref)", "arg_0_ref.q_zero_point()")
        helper_text = helper_text.replace("ffi::q_per_channel_axis(arg_0_ref)", "arg_0_ref.q_per_channel_axis()")
        helper.write_text(helper_text, encoding="utf-8")

    # A small set of sparse projections are exposed as Tensor member methods
    # by the LibTorch distribution.  The generated ATen namespace shims for
    # these leaves compile, but dispatch through an unavailable composite
    # kernel on the shipped CPU build.  Keep the CXX-generated ABI intact and
    # call the public Tensor methods in the generated adapter bodies.
    exception_adapters = (
        Path(args.root) / "native/rust/cpp/terlan_exception_adapters.cc"
    )
    if exception_adapters.exists():
        exception_text = exception_adapters.read_text(encoding="utf-8")
        for name in ("sparse_dim", "dense_dim", "is_coalesced"):
            exception_text = exception_text.replace(
                f"at::compositeexplicitautograd::{name}(self)",
                f"self.{name}()",
            )
        exception_adapters.write_text(exception_text, encoding="utf-8")

    owned_adapters = Path(args.root) / "native/rust/cpp/terlan_owned_value_adapters.cc"
    if owned_adapters.exists():
        owned_text = owned_adapters.read_text(encoding="utf-8")
        for name in (
            "indices",
            "values",
            "crow_indices",
            "col_indices",
            "ccol_indices",
            "row_indices",
        ):
            owned_text = owned_text.replace(
                f"at::compositeexplicitautograd::{name}(self)",
                f"self.{name}()",
            )
        owned_adapters.write_text(owned_text, encoding="utf-8")


if __name__ == "__main__":
    main()
