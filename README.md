# terlan-pytorch

Terlan bindings for PyTorch's stable LibTorch C ABI.

This repository contains PyTorch-specific metadata and package checks. Generic
C ABI generation remains in the Terlan compiler repository.

The repository root is also the consumable Terlan package. `make generate`
rebuilds the package-owned `terlan.toml`, `src/`, `native/`, generated binding
manifests, API documentation, and NativeBoundary test from
`bindings/libtorch-c.json`. A consumer can therefore pin this repository as an
ordinary immutable Git dependency without copying `generated/` or changing the
compiler.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the CPU-to-CUDA implementation order
and [docs/TARGET_SURFACE.md](docs/TARGET_SURFACE.md) for the accepted ABI surface.

## First compatibility slice

The initial CPU-only slice binds an owned `AtenTensorHandle` through:

- `aoti_torch_scalar_to_tensor_int64`
- `aoti_torch_scalar_to_tensor_float64`
- `aoti_torch_scalar_to_tensor_bool`
- `aoti_torch_item_int64`
- `aoti_torch_item_float64`
- `aoti_torch_item_bool`
- `aoti_torch_get_dim`
- `aoti_torch_get_numel`
- `aoti_torch_get_sizes`
- `aoti_torch_get_dtype`
- `aoti_torch_get_device_type`
- `aoti_torch_get_device_index`
- `aoti_torch_abi_version`
- `aoti_torch_new_tensor_handle`
- `torch_call_dispatcher` for reviewed `aten::clone` and `aten::unsqueeze` stacks
- `torch_call_dispatcher` for two-input CPU `aten::matmul`
- `torch_call_dispatcher` for named-overload `aten::transpose.int`
- `torch_call_dispatcher` for named-overload `aten::mul.Tensor`
- `torch_call_dispatcher` for nonlinear `aten::relu`
- `torch_new_string_handle` / `torch_delete_string` plus
  `torch_call_dispatcher` for exact-default `aten::gelu`
- `torch_call_dispatcher` for exact `aten::softmax.int` probability normalization
- `torch_new_stable_ivalue` / `torch_delete_stable_ivalue` for owned optional
  dispatcher arguments
- `torch_call_dispatcher` for exact `aten::argmax` class-index selection
- `torch_call_dispatcher` for exact `aten::mse_loss` model evaluation
- `torch_call_dispatcher` for exact `aten::mse_loss_backward` explicit gradients
- `torch_call_dispatcher` for exact `aten::cross_entropy_loss` classification
  evaluation with float label smoothing
- `torch_call_dispatcher` for rank-normalizing `aten::flatten.using_ints`
- `aoti_torch_aten_subtract_Tensor` for float64 scalar arithmetic
- `aoti_torch_aten_narrow` for bounded indexing and owned views
- `aoti_torch_aten_amax` for dimension-list reductions
- `aoti_torch_aten_full` for owned shaped float64 tensors
- `aoti_torch_cpu_add_Tensor` for scaled tensor addition and broadcast bias
- `aoti_torch_cpu_convolution` behind a package-owned `conv2d` adapter that
  contains LibTorch's optional-bias pointer encoding and fixes ordinary 2D
  convolution rather than the transposed form
- `aoti_torch_cpu__adaptive_avg_pool2d` for exact NCHW spatial output sizing
- `aoti_torch_create_tensor_from_blob_v2` plus `aoti_torch_clone` behind the
  package-owned `from_floats(values, shape)` and `from_ints(values, shape)`
  stable-C adapters
- `torch::jit::load` behind package-owned C++ adapters for stateless
  `run_torchscript(input, path)`, reusable `TorchScriptModel` inference, and
  fixed positional-input and two-tensor-output forwarding
- `aoti_torch_delete_tensor_object`

Run the pinned released-distribution gate:

```sh
make release-check
```

This downloads and verifies the official `2.13.0+cpu` archive under ignored
`vendor/`, checks its headers and exported symbols, and runs the generated Rust
and Terlan consumers. The gate also fetches the generated output as an
immutable Git dependency into a fresh consumer, removes the source package,
executes from the verified cache, checks stable distribution/shape/stale-handle
rejections, and writes `generated/package-execution-report.json`. Alternatively,
set `LIBTORCH` explicitly and run
`make package-check LIBTORCH=/path/to/libtorch`.

`TERLC` may override the compiler executable. No package gate builds the
PyTorch source checkout under `upstream/`.

## Serialized model inference

The first post-composition capability is
`run_torchscript(input: Tensor, path: String): Tensor`. It loads a serialized
TorchScript archive, runs its single-tensor `forward`, and returns a separately
owned tensor while leaving the input borrowed and unchanged. The pinned fixture
is a real C++-generated module implementing `input * 2.0`; package and fresh Git
consumers require exact float64 `3.0 -> 6.0` execution. Missing files and corrupt
archives are rejected as stable statuses `7041` and `7042`, and no C++ exception
crosses the C boundary.

MI1 also exposes a thread-confined, independently owned `TorchScriptModel`:
`load_torchscript(path)` loads once, `torchscript_forward(model, input)` may
borrow it repeatedly, and `dispose_model(model)` destroys it exactly once. The
package and fresh Git consumer require independent `3.0 -> 6.0` and
`4.0 -> 8.0` results from one model and reject use after disposal as
`stale_handle`. The generic C ABI binder owns the multi-resource tagged-handle
machinery; no PyTorch rule lives in the compiler. Variable-arity inputs and
non-tensor output values remain deferred.

MI2 adds `torchscript_forward2(model, first, second)` for models with two
positional tensor arguments. The pinned order-sensitive fixture computes
`left - right * 2.0`, requires exact `3.0, 4.0 -> -5.0` execution with both
inputs unchanged, and rejects model-schema arity mismatch as stable status
`7043`. Variable-arity tensor lists and arbitrary structured outputs remain
deferred to the generic native value model.

MI3 accepts a TorchScript tuple of exactly two tensors through
`torchscript_forward_pair`. It returns an owned `TensorPair`; extracting its
first and second elements produces independently owned tensors that remain
valid after disposing the pair. The pinned fixture requires
`2.0 -> (4.0, 6.0)` and rejects a plain tensor output as stable status `7046`.
Wrong tuple arity and non-tensor elements have reserved statuses `7047` and
`7048`. This fixed container is an ownership bridge, not general Terlan tuple
support. At MI3 the generated public surface contained 48 functions.

MI4 adds `torchscript_forward_float(model, input): Float`. The C++ adapter
accepts only an IValue double and copies it into Terlan, so the result has no
native handle or disposal obligation. The pinned fixture requires unchanged
input `3.0`, exact output `6.0`, and stable status `7049` when a tensor-returning
model is invoked through the float surface. The current generated public
surface contained 49 functions.

MI5 adds `torchscript_forward_int(model, input): Int` through the already
generic fallible `int64_t` output lowering. The adapter copies only an exact
TorchScript integer result, with pinned unchanged input `4`, exact output `7`,
and stable status `7050` for tensor or other non-integer results. It adds no
native result resource or compiler special case. The current generated public
surface contained 50 functions.

MI6 adds `torchscript_forward_bool(model, input): Bool`, copying only an exact
TorchScript boolean through the generic fallible `bool` output lowering. The
pinned model requires unchanged integer input `1`, exact result `true`, and
stable status `7051` for tensor or other non-boolean results. The current
generated public surface contains 51 functions.

MI7 adds `torchscript_forward_string(model, input): String` through Terlan's
generic owned UTF-8 C-ABI output contract. Structured metadata names the byte
length and destructor; generated Rust copies the bytes, destroys native
storage exactly once, and returns no pointer or lifetime. The pinned model
requires unchanged input `1`, exact result `"positive"`, and stable status
`7052` for tensor or other non-string results. The generated public surface
contains 52 functions.

MI8 adds `torchscript_forward_int_list(model, input): List[Int]` through a
generic owned `int64_t` array contract. Generated Rust copies the declared
element count, calls the named destructor once, and transports the Rust-owned
vector through the existing `ok_ints` protocol. The pinned model requires
unchanged input `4`, exact result `[4,5,6]`, and stable status `7053` for tensor
or other non-list outputs. The generated public surface contains 53 functions.

MI9 generalizes that owned-array contract with an explicit element kind and
adds `torchscript_forward_float_list(model, input): List[Float]`. Its `double`
elements are copied before the named destructor runs, then returned through the
existing `ok_floats` protocol. Package and immutable consumers require unchanged
input `1.5`, exact result `[1.5,2.0,2.5]`, and stable status `7054` for tensor or
other non-floating-point-list outputs. The generated public surface contains 54
functions.

MI10 adds `torchscript_forward_bool_list(model, input): List[Bool]`. Its generic
owned-array contract uses portable `uint8_t` storage, validates every copied
element as exactly `0` or `1`, destroys native storage, and returns through the
new generic `ok_bools` protocol. Package and immutable consumers require
unchanged input `2`, exact result `[true,false,false]`, and stable status `7055`
for tensor or other non-boolean-list outputs. The generated public surface
contains 55 functions.

MI11 adds `torchscript_forward_string_list(model, input): List[String]`. The
generic owned string-array contract carries `char ***` values, per-element byte
lengths, and a count into generated Rust, which copies and destroys all native
storage before UTF-8 validation. The pinned model preserves an empty element in
exact output `["positive","","ready"]`; wrong output shapes reject with stable
status `7056`. The generated public surface contains 56 functions.

`from_floats` accepts non-constant row-major `List[Float]` data. Its adapter
validates shape and value count, creates a call-scoped LibTorch view, clones
it, and destroys the temporary view before returning the owned tensor. No
caller buffer or callback outlives the native call.

The current broad CPU slice composes `from_floats`, `matmul`, and `add` into an
affine forward pass with non-constant input, weights, and broadcast bias. The
package and fresh consumer require exact `[2, 1]` output shape and maximum
`7.0`, preserve the inputs, and reject incompatible broadcast shapes.

The classifier slice extends that composition to `[2, 2]` logits,
dimension-1 softmax, dimension-1 argmax, and mean squared error against an
equal-shape target. Batched int64 labels also feed smoothed cross-entropy. Zero
logits must produce uniform probability maximum `0.5`, deterministic int64
class indices `[0, 0]`, MSE `0.25`, and cross-entropy between `0.69` and `0.70`;
invalid dimensions, shapes, reductions, target dtypes, and smoothing values are
rejected stably.

The first training-oriented slice computes the exact MSE gradient, scales it by
a learning-rate tensor, applies a functional parameter update, and verifies the
new loss is lower. Graph autograd remains deferred: the released stable C ABI
can set `requires_grad`, but currently has no stable gradient getter, while
`aten::_backward` also requires Tensor-list and unit-output dispatcher support.

The first vision-oriented slice executes real CPU NCHW `conv2d` over rank-4
input and weight tensors. A `[1, 1, 3, 3]` tensor of ones convolved with a
`[1, 1, 2, 2]` kernel of ones and zero bias produces independently owned
`[1, 1, 2, 2]` output containing `4.0`. The package adapter targets the current
generated `aoti_torch_cpu_convolution` shim; it does not use the deprecated
device-generic `aoti_torch_convolution` export. Incompatible input/weight
channels are rejected through a stable C ABI status.

The next vision slice applies adaptive average pooling directly through the
generated CPU shim. A non-constant `[1, 1, 4, 4]` quadrant image reduces to
exact `[1.0, 2.0, 3.0, 4.0]` values at shape `[1, 1, 2, 2]`; exact MSE `0.0`
proves the complete result, and malformed one-element output sizes are rejected
stably.

The composed vision-classifier slice then runs `conv2d`, ReLU, global adaptive
average pooling, flattening, affine projection, softmax, and argmax as one
Terlan program. Its `[1, 2]` probabilities are exactly `[0.5, 0.5]`, the int64
prediction is `[0]`, and every intermediate native handle is disposed.

The first sequence/recommendation slice binds exact dispatcher
`aten::embedding`. Batched int64 indices `[0,2]` select vectors from a `[3,2]`
table and return owned `[2,2]` float features while both sources remain
readable; an index equal to the vocabulary size is rejected stably.

The composed sequence-classifier slice accepts batched token indices at
`[1,2]`, embeds them to `[1,2,2]`, max-pools tokens to `[1,2]`, and performs
affine softmax classification with int64 prediction `[0]`. An incompatible
projection matrix is rejected before classification.

The first normalization slice binds inference-mode `aten::batch_norm` with
explicit affine parameters and running statistics. Five public tensors remain
readable after normalization; its four present optional tensors use generic
allocator-backed StableIValue ownership, and mismatched channel statistics are
rejected through the dispatcher status boundary.

The transformer-normalization slice binds exact `aten::layer_norm` through a
generic owned integer-list StableIValue plus allocator-backed optional affine
tensors. It also composes normalization, matrix projection, and residual
addition in ordinary Terlan; incompatible normalized shapes are rejected
before a result handle is returned.

The next broad composition builds normalized scaled dot-product self-attention
from existing tensor functions: Q/K/V projections, transposed score
multiplication, scaling, softmax, context multiplication, and residual
addition. Exact attention, context, and residual tensors are checked without a
fused native attention wrapper.

The transformer feed-forward slice adds generic owned fixed-string dispatcher
lowering and binds `aten::gelu` with `approximate="none"`. Ordinary Terlan then
composes layer normalization, `[2,2] -> [2,4]` expansion, GELU, `[2,4] ->
[2,2]` projection, bias, and residual addition. Package and immutable consumers
require the expected `[2,2]` result within `1e-7` MSE and reject an incompatible
expansion projection.

The next composition joins normalized self-attention and normalized GELU
feed-forward into a complete pre-normalized transformer encoder block. It keeps
the public surface at 39 generated functions, produces a `[2,2]` encoder result
within `1e-7` MSE, preserves the source, and rejects a width-changing final
projection before residual addition.

The encoder-classifier slice then max-pools tokens, inserts a batch dimension,
and runs affine logits, softmax, argmax, and cross-entropy as one Terlan model
graph. It produces exact two-class uniform probabilities, prediction `[0]`, and
the expected `0.69–0.70` loss range while rejecting an out-of-range class
target. No new generated function is introduced.

The token-transformer slice closes the complete text path: int64 token IDs
select exact embeddings, which flow through the reusable Terlan encoder and
classifier into probabilities, prediction, and cross-entropy. Both fixture and
pinned LibTorch consumers verify the embedded values and final results, reject
an out-of-vocabulary ID through the stable C ABI status, and keep the public
surface at 39 generated functions.

The batched token-transformer slice promotes that path to `[B,T,D]` execution.
Rank-3 attention uses batched matrix multiplication and a real key-dimension
transpose, token pooling returns one feature row per sample, and the classifier
produces predictions `[0,0]` for a two-sample batch. Incompatible leading batch
dimensions are rejected stably; no new generated operation is required.

The batched vision slice likewise preserves two distinct images through
convolution, activation, global pooling, feature flattening, classification,
prediction, and cross-entropy. Complete convolution and pooled tensors are
checked exactly, and a mismatched target batch is rejected before evaluation.

The grouped-CNN slice adds a depthwise `groups=2` spatial convolution followed
by a pointwise `1×1` channel-mixing convolution. It validates exact intermediate
feature maps, global features, and two-sample classification while rejecting
invalid group/output-channel topology.

The normalized residual-CNN slice adds inference batch normalization around
the depthwise and pointwise stages, joins the branch back to an unchanged-width
input, and validates exact residual features before pooled classification. A
channel-changing branch is rejected at the residual boundary.

The stacked-transformer slice runs two reusable pre-normalized encoder layers
over a batched token tensor, verifies the complete second-layer state, and
classifies pooled sequences. Layer parameters are safely reused while all
intermediate NativeBoundary handles remain independently disposed.

The causal-transformer slice broadcasts an additive triangular mask over
batched attention scores. It verifies the complete masked attention, context,
residual, encoder, pooled, and classifier tensors and rejects incompatible mask
shapes without introducing a mask-specific native operation.

The cross-attention slice separates a two-token decoder state from a
three-token memory sequence. Q is derived from the decoder while K/V are
derived from independently normalized memory, producing exact `[2,2,3]`
attention before residual, feed-forward, pooled classification, prediction,
and loss. Incompatible decoder/memory batches are rejected without adding a
fused attention operation or compiler special case.

The decoder-layer slice composes both attention modes: embedded target tokens
first pass through causal self-attention, then attend to a separately owned
three-token memory sequence, and finally pass through normalized GELU
feed-forward and batched classification. Exact residuals are validated at both
attention boundaries, and a context projection with the wrong decoder width is
rejected before residual construction.

The final broad CPU composition makes that decoder fully token-to-token:
source IDs are embedded through a source-owned table into encoder memory, while
independent target IDs use a distinct target table before causal decoding.
Exact source memory and target state flow through the complete decoder and
classifier, and incompatible source/target batches are rejected before
cross-attention.
