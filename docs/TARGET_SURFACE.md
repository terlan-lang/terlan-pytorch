# LibTorch target surface

The active package targets LibTorch through generated CXX bindings. It does
not require PyTorch's stable C/AOTI exports and does not build the PyTorch
source tree. The CXX surface is pinned to the released `2.13.0` distribution
at build commit `cf30153c4c131c8164ee7798e5022d810682e2cb`; admission validates
the C++ headers, TorchScript support, exact build version, and exact build
hash. The PT0–PT39 sections below retain the historical migration record; any
reference to Stable-C describes the retired compatibility target, not a runtime
dependency of the current package. CUDA admission resolves the repository's
bundled `libcusparseLt` and `nvshmem` SONAME directories before validating
LibTorch, so the runtime check is reproducible without a system CUDA package.

## PT0: scalar tensor lifecycle

PT0 proves external dynamic linking, ABI aliases, status translation, and sole
ownership of an opaque tensor handle. An `Int` scalar is enough to construct a
defined tensor without exposing a caller-owned buffer or deleter callback.

The accepted surface is lifecycle, scalar item, dimension, element count,
owned shape lists copied from borrowed size arrays, dtype, device type/index,
and ABI version. CPU is the pinned release baseline. CUDA execution is admitted
only by the separate opt-in gate with a CUDA-enabled LibTorch distribution.

## PT1: copied shape arrays

PT1 binds `aoti_torch_get_sizes` only with structured metadata naming the
tensor owner, `aoti_torch_get_dim` length source, and immediate-copy policy.
Generated Rust rejects negative lengths and non-empty null pointers, then
copies the borrowed values before releasing the call-scoped tensor borrow.

## PT2: versioned StableIValue dispatcher

PT2 describes `aten::clone` as data: the exact operator and overload names,
the `0x020d000000000000` extension ABI version, one independently owned tensor
handle, one null optional argument, and one owned tensor result slot. Generated
Rust calls `aoti_torch_new_tensor_handle` before `torch_call_dispatcher`; the
dispatcher consumes that copied input and transfers the result back into a new
Terlan `Tensor` handle. The released-package gate reads and disposes both the
source and clone through the NativeBoundary helper.

The compiler intentionally rejects other dispatcher stack values, ambiguous
or embedded-NUL overload names, malformed versions, and mismatched result
slots under `native_bindgen.c_dispatcher_contract`. This is a bounded first
operation, not a claim that arbitrary IValues are supported.

## PT3: signed integer StableIValue

PT3 adds `aten::unsqueeze` with a metadata-linked Terlan `Int` argument. The
generator preserves its signed 64-bit representation in the StableIValue slot;
the package consumer deliberately passes `-1`, reads the returned tensor, and
disposes it independently. A PyTorch-owned semantic test also requires the
returned rank to increase by one and its copied shape length to match that rank.
Metadata must encode every declared integer exactly once, and unknown or
repeated argument names fail under the dispatcher contract.

## PT4: multiple tensor inputs

PT4 adds `aten::matmul` with two independently duplicated tensor handles on the
owning StableIValue stack. Generated Rust guards each duplicate until the whole
stack takes ownership, so failure while preparing a later input releases every
earlier copy. The NativeBoundary helper validates both live handles before it
borrows either value for the immutable operation.

The package test constructs deterministic `[1, 1]` tensors from scalars, checks
that `4 × 5` returns item `20` with rank and shape length `2`, and disposes every
source, intermediate, and result handle. The compiler fixture additionally
rejects stale and wrong-type secondary handles before native dispatch.

## PT6: mixed dispatcher stack and named overload

PT6 adds `aten::transpose.int` with one independently duplicated tensor handle
and two signed dimensions in one three-slot StableIValue stack. It proves exact
named-overload selection rather than relying on the empty default overload.
Both the package test and fresh Git consumer transpose a rank-2 tensor, require
the exact value and rank to survive, then dispose the input and independent
result. Scalar arithmetic is not represented as a StableIValue integer because
LibTorch's stable API routes `Scalar` parameters through generated C shims.

## PT7: named tensor arithmetic overload

PT7 adds `aten::mul.Tensor`, selecting the exact tensor/tensor overload and
transferring two independently duplicated handles through the dispatcher.
Package-owned and fresh consumers require `multiply(4, 5)` to return `20` and
independently dispose all three handles. PT8 and PT9 subsequently add the
compatible float constructor/dtype surface needed to publish the released
`aoti_torch_aten_subtract_Tensor` shim.

## PT8: float64 scalar lifecycle

PT8 extends the generic C ABI helper protocol with Terlan `Float` arguments and
results and permits multiple constructors for one opaque resource type. The
package binds LibTorch's stable float64 scalar constructor and item readback,
then requires `multiply(2.5, 4.0)` to return `10.0` with dtype identifier `7`.
The fixture, package tests, and fresh Git consumer independently dispose both
inputs and the result.

## PT9: stable C scalar arithmetic and dtype rejection

PT9 publishes `aoti_torch_aten_subtract_Tensor` now that the package can create
compatible float64 tensors. The generic direct C wrapper maps both borrowed
handles, converts Terlan `Float` alpha to C `double`, and owns the returned
tensor. Positive consumers require `subtract(4.0, 1.5, 2.0) = 1.0`; the
adversarial consumer passes int64 tensors and requires a stable C ABI status
rejection instead of silently changing dtype semantics.

## PT10: bool scalar lifecycle

PT10 extends the generic C ABI adapter and NativeBoundary helper protocol with
Terlan `Bool` arguments and results. The package binds LibTorch's stable
`aoti_torch_scalar_to_tensor_bool` and `aoti_torch_item_bool` symbols. Both the
package-owned tests and fresh Git consumer round-trip `true` and `false`, assert
bool dtype identifier `11`, and independently dispose the owned tensors.

## PT11: nonlinear activation

PT11 adds the exact default `aten::relu` operation through the versioned
StableIValue dispatcher. Generated Rust duplicates the borrowed source handle
before dispatcher ownership transfer and returns a new Terlan-owned tensor.
Package-owned and fresh Git consumers require `relu(-2.5) = 0.0`, verify that
the original tensor still reads `-2.5`, and dispose source and result
independently. This is the first nonlinear ML operation in the public surface.

## PT12: bounded indexing and owned views

PT12 binds LibTorch's generated stable C `aoti_torch_aten_narrow` operation.
The public method borrows its source and accepts signed dimension, start, and
length values; generated Rust owns the returned tensor handle independently.
Both consumer paths narrow a one-element vector to shape `[1]`, read the source
afterward, and dispose both handles. An adversarial consumer requests the
out-of-range interval `[1, 2)` and requires a stable C ABI status rejection.
The generic smoke-test generator skips validation-dependent direct operations
instead of guessing package-specific legal index values.

## PT13: dimension-range flattening

PT13 adds the exact named `aten::flatten.using_ints` overload through a
three-slot StableIValue stack: one independently owned tensor copy plus signed
start and end dimensions. Package-owned and fresh Git consumers flatten a
rank-2 tensor across dimensions `0..1`, require rank 1 and shape `[1]`, and
preserve exact scalar readback. The source remains independently owned. A
negative consumer requests dimension `2` from a rank-1 tensor and requires a
stable status rejection.

## PT14: dimension-list reduction

PT14 extends Terlan's generic C binder with structured borrowed input arrays.
An `input_array.length_parameter` link maps one Terlan `List[Int]` to a
call-scoped `const int64_t*` plus checked `int64_t` length. The native protocol
parses the existing `li:` list encoding into owned Rust integers and borrows
their slice only for the C call. The package uses this machinery for
`aoti_torch_aten_amax`, reducing a rank-2 tensor over `[0]` to rank 1 while
leaving the source independently readable. An invalid dimension `[2]` must
produce a stable C ABI status rejection.

## PT15: owned shaped tensor construction

PT15 adds reviewed fixed C inputs to the generic binder. Metadata may supply a
typed `int32_t` scalar, a call-scoped `int32_t*`, or a null optional pointer;
the generated public Terlan signature does not expose those ABI details. The
package uses this with `aoti_torch_aten_full` to publish
`full_float(shape, value)`, fixed to CPU float64 for the baseline. Package and
fresh Git consumers construct a `[2, 3]` tensor, require rank `2`, six elements,
dtype `7`, exact copied shape, and repeated value `1.25` observed after complete
`amax` reduction. A negative dimension is rejected stably.

## PT16: owned non-constant float64 data

The generic C binder accepts typed `List[Float]` input arrays and can compile
package-owned C adapter sources while linking an external stable C
distribution. `from_floats(values, shape)` validates non-negative dimensions,
checked shape multiplication, and exact value count before calling LibTorch.

The package adapter creates a temporary tensor view over the call-scoped
Terlan values with `aoti_torch_create_tensor_from_blob_v2`, immediately clones
it with `aoti_torch_clone`, and destroys the temporary tensor object before the
call returns. Only the clone crosses into the generated `Tensor` handle, so no
Terlan list pointer, deleter callback, or borrowed lifetime survives the call.
Package-owned and fresh Git consumers construct values `1.0` through `6.0` at
shape `[2, 3]`, observe maximum `6.0`, and dispose every tensor. A `[2, 2]`
shape paired with three values is rejected as `c_abi_status_7003`.
An empty Terlan list is resolved against the declared `List[Float]` parameter;
`from_floats([], [0, 3])` produces an owned rank-2 tensor with zero elements.

## PT17: affine forward composition

The CPU baseline binds `aoti_torch_cpu_add_Tensor` as
`add(left, right, alpha)`. Both tensor inputs are borrowed for the call and the
result is a separate owned handle. Standard LibTorch broadcasting applies; an
incompatible `[2, 3] + [4]` request must fail through a stable C ABI status.

Package-owned and fresh Git consumers now execute a small affine layer:
`[2, 3]` non-constant input multiplied by `[3, 1]` weights, followed by a
broadcast `[1]` bias. The result has shape `[2, 1]`, float64 dtype, and maximum
`7.0`. Inputs remain independently readable and all
five source/intermediate/result handles are disposed separately. CUDA may
later select its generated add shim behind the same public Terlan operation.

## PT18: classifier forward and softmax

`softmax(tensor, dimension)` selects `aten::softmax.int` exactly. Its
StableIValue stack owns a duplicated tensor handle, carries the signed
dimension, and passes null for the optional dtype so the input float64 dtype is
preserved. The returned probability tensor is independently owned and the
source logits remain readable.

The classifier fixture composes `[2, 3]` input, `[3, 2]` zero weights, `[2]`
bias, `matmul`, broadcast `add`, and dimension-1 `softmax`. It requires `[2,
2]` probabilities, four elements, dtype `7`, unchanged zero logits, and
uniform probability maximum `0.5`. Dimension `2` is rejected for these rank-2
logits. Softmax is marked `generated_smoke: package_owned` because the generic
smoke starts from a scalar; package and fresh-consumer execution remain
mandatory.

## PT19: classifier predictions and argmax

`argmax(tensor, dimension, keep_dimension)` selects the exact default
`aten::argmax` schema. Its `int?` dimension is not encoded as a raw integer:
the generated adapter allocates one StableIValue, stores the signed dimension,
and transfers that backing allocation to the dispatcher for consumption. The
boolean keep-dimension value is encoded directly. Allocation and destruction
symbols are declared and signature-checked by structured metadata.

The classifier fixture reduces `[2, 2]` probabilities over dimension 1 into
an int64 `[2]` prediction tensor with indices `[0, 0]`. A separate retained-
dimension case requires `[2, 1]`. Dimension `2` is rejected for rank-2 input,
and source probabilities remain independently owned.

## PT20: model evaluation and MSE loss

`mse_loss(predicted, target, reduction)` selects exact default
`aten::mse_loss`. Reduction values follow LibTorch's stable schema: `0` keeps
elementwise loss, `1` computes the mean, and `2` computes the sum. The
dispatcher owns separate copies of predicted and target tensors and returns a
new owned result, leaving both public source handles readable.

The direct loss fixture compares uniform `[2, 2]` tensors containing `2.0` and
`1.0`, requiring a scalar float64 mean loss of `1.0`. The classifier fixture
compares uniform `0.5` probabilities with an equal-shape zero target and
requires loss `0.25`. Incompatible `[2, 2]` and `[3]` operands are rejected
through a stable C ABI status, as are reduction values outside `0`, `1`, and
`2`.

## PT21: batched labels and cross-entropy

`from_ints(values, shape)` mirrors the owned-copy contract of `from_floats` for
row-major int64 data. It creates a call-scoped LibTorch blob view, clones the
tensor, destroys the temporary view, and returns storage independent of the
Terlan list. `[0, 1]` with shape `[2]` therefore supplies real batched class
labels rather than treating labels as floating-point features.

`cross_entropy_loss(logits, target, reduction, ignore_index,
label_smoothing)` selects exact default `aten::cross_entropy_loss` with null
class weights. Signed reduction and ignore-index values use integer
StableIValues; label smoothing uses the exact IEEE-754 bits of the Terlan
Float. The first classifier case uses `[2, 2]` zero logits, labels `[0, 1]`,
mean reduction, ignore index `-100`, and smoothing `0.1`, requiring an owned
scalar float64 loss between `0.69` and `0.70`. Both inputs remain readable.
Float labels and smoothing above `1.0` are rejected stably.

## PT22: explicit MSE gradient and parameter update

`mse_loss_gradient(grad_output, predicted, target, reduction)` selects exact
default `aten::mse_loss_backward`. Three duplicated tensor handles transfer
into the dispatcher and the returned gradient is independently owned. For
uniform `[2, 2]` prediction `2.0`, target `1.0`, scalar upstream gradient
`1.0`, and mean reduction, every gradient element is `0.5`.

The training fixture multiplies this gradient by scalar learning rate `0.1`,
subtracts the scaled gradient from the original parameter, and requires the
updated maximum to lie between `1.94` and `1.96`. Mean loss must decrease from
`1.0` into `(0.90, 0.91)`. Incompatible backward operands are rejected stably.

This is an explicit differentiable-operator path, not graph autograd. Released
LibTorch stable C exposes `torch_set_requires_grad`, but not a stable gradient
getter. The dispatcher-only `aten::_backward` schema also requires an owned
`Tensor[]` input and unit output. Those generic boundary shapes must be closed
before Terlan claims graph-based backward execution.

## PT23: NCHW 2D convolution

`conv2d(input, weight, bias, stride, padding, dilation, groups)` is the first
rank-4 vision kernel. Its public contract requires NCHW input, OIHW weight, a
one-dimensional bias tensor, exactly two values in each spatial parameter list,
and a positive group count. The returned tensor owns its handle independently;
input, weight, and bias remain borrowed and readable after the call.

The released distribution still exports a device-generic
`aoti_torch_convolution`, but marks it deprecated. PT23 instead selects the
current generated CPU symbol `aoti_torch_cpu_convolution`. That C signature
represents optional bias as `AtenTensorHandle*`, so a small package-owned C
adapter contains the representation, supplies a required public bias, fixes
`transposed = false` and `output_padding = [0, 0]`, and leaves the generic
Terlan binder unchanged.

The executable case convolves a float64 `[1, 1, 3, 3]` tensor of ones with a
float64 `[1, 1, 2, 2]` kernel of ones and `[1]` zero bias. Stride and dilation
are `[1, 1]`, padding is `[0, 0]`, and groups is `1`; the required output is
`[1, 1, 2, 2]`, numel `4`, with every value `4.0`. A mismatched two-channel
input and one-channel weight must fail through a stable C ABI status. Exact
mean squared error `0.0` against a `[1, 1, 2, 2]` tensor filled with `4.0`
proves the complete output contents rather than only its maximum.

## PT24: adaptive average pooling

`adaptive_avg_pool2d(input, output_size)` binds the generated CPU
`aoti_torch_cpu__adaptive_avg_pool2d` symbol directly. The public contract
requires a rank-4 NCHW float tensor and exactly two positive spatial output
dimensions. It returns one independently owned tensor and leaves the input
readable. This surface uses only the generic borrowed opaque-handle and copied
`List[Int]` lowering already present in the compiler.

The executable input is a `[1, 1, 4, 4]` image whose four 2×2 quadrants are
constant `1.0`, `2.0`, `3.0`, and `4.0`. Pooling to `[2, 2]` must return shape
`[1, 1, 2, 2]` with row-major values `[1.0, 2.0, 3.0, 4.0]`, maximum `4.0`,
and exact MSE `0.0` against the expected tensor. A one-element output-size list
must be rejected through a stable C ABI status. Adaptive max pooling remains
deferred because its generated shim returns both values and indices, requiring
generic multiple-result ownership.

## PT25: composed vision classifier

PT25 introduces no native symbol. It proves that the independently bound
operators form a useful inference graph in ordinary Terlan source:

`conv2d -> relu -> adaptive_avg_pool2d -> flatten -> matmul -> add -> softmax -> argmax`

The fixture starts with `[1,1,3,3]` input, produces `[1,1]` global pooled
features, projects to `[1,2]` logits, and returns `[1,2]` probabilities plus a
`[1]` int64 prediction. Symmetric classifier weights require exact probabilities
`[0.5,0.5]`, MSE `0.0` against the expected tensor, and class index `0`. Every
source, parameter, intermediate, result, and validation tensor has independent
opaque-handle ownership and is disposed explicitly.

## PT26: embedding lookup

`embedding(weight, indices, padding_index, scale_gradient_by_frequency,
sparse)` selects exact default `aten::embedding` through the versioned
StableIValue dispatcher. Weight and indices are duplicated into independently
owned dispatcher inputs; padding uses a signed integer slot and both mode flags
use reviewed boolean slots. The returned tensor transfers back as a new opaque
handle, leaving the public sources readable.

The executable table is float64 shape `[3,2]`, the int64 indices are `[0,2]` at
shape `[2]`, and the required output is float64 `[2,2]`. A uniform table value
`2.0` permits complete-content validation with MSE `0.0`. Index `3`, equal to
the row count, must fail through a stable C ABI dispatcher status. This opens
sequence and recommendation workloads without introducing another native shim
or any PyTorch-specific compiler branch.

## PT27: composed sequence classifier

PT27 composes `embedding -> amax -> matmul -> add -> softmax -> argmax` without
adding a native operation. Batched token indices `[1,2]` become embeddings
`[1,2,2]`; maximum reduction over the token dimension produces `[1,2]`
features, followed by `[1,2]` probabilities and a `[1]` int64 prediction.
Symmetric parameters require probabilities `[0.5,0.5]`, exact MSE `0.0`, and
prediction `0`. A projection whose input dimension is `3` must be rejected for
the two-feature pooled tensor.

PT27 deferred layer normalization because exact `aten::layer_norm` needed a
dispatcher integer-list value and optional public weight/bias tensors. PT29
closes that single-result route generically. `aten::native_layer_norm` still
returns output, mean, and reciprocal standard deviation, and embedding bag
returns four tensors; those remain behind generic multiple-result ownership.

## PT28: inference batch normalization

`batch_norm(input, weight, bias, running_mean, running_variance, training,
momentum, epsilon, cudnn_enabled)` selects exact default `aten::batch_norm`.
PT28 requires all four optional schema tensors to be present and executes with
`training = false`. The input is duplicated directly, while each affine or
running-stat tensor is duplicated into an allocator-backed nested StableIValue
through generic `owned_optional_handle_copy` lowering. Both booleans and both
IEEE-754 floats use the existing generic stack encodings. The single output
transfers back independently.

The executable NCHW input is uniform `3.0` at shape `[1,2,2,2]`. Per-channel
weight `2.0`, bias `1.0`, running mean `1.0`, running variance `4.0`, and zero
epsilon yield exact output `3.0` with identical shape and float64 dtype. MSE
`0.0` proves complete contents while all public sources remain readable. A
running-mean shape `[3]` must be rejected for two input channels. Training mode
is exposed separately through `pytorch.Training.batch_norm_training`. The
wrapper clones both running statistics before selecting `training = true`, then
returns an ordinary struct owning the output and updated statistic tensors.
This functional contract does not mutate caller-owned statistics.

## PT29: layer normalization and transformer residual

`layer_norm(input, normalized_shape, weight, bias, epsilon, cudnn_enabled)`
selects exact default `aten::layer_norm`. The normalized shape is copied into an
owned stable list through generic `owned_int_list_argument` lowering. Weight
and bias occupy present `Tensor?` positions through allocator-backed nested
StableIValues. The list and optional backing storage are consumed by dispatch;
the public tensors remain readable and the output returns independently.

For input `[1.0,3.0,2.0,4.0]` at shape `[2,2]`, trailing normalization over
`[2]` with unit weight, zero bias, and zero epsilon produces exact
`[-1.0,1.0,-1.0,1.0]`. An identity projection followed by residual addition
then produces `[0.0,4.0,1.0,5.0]`. Both complete outputs require MSE `0.0` in
fixture and pinned consumers. Normalized shape `[3]` is rejected for trailing
width `2`.

## PT30: normalized self-attention block

PT30 adds no public native operation. Ordinary Terlan composes `layer_norm`,
three Q/K/V matrix projections, key transpose, score matrix multiplication,
tensor scaling, softmax, context multiplication, and residual addition. Zero
query/key projections make the exact attention matrix uniformly `0.5`; an
identity value projection plus unit bias over normalized
`[1.0,3.0;4.0,2.0]` produces unit context and residual
`[2.0,4.0;5.0,3.0]`.

Package and immutable consumers require exact MSE `0.0` for attention, context,
and residual tensors while the source remains readable. A query projection
whose input width is `3` is rejected for normalized width `2`. The slice proves
an attention graph using reusable generated primitives rather than a fused
package helper.

## PT31: GELU and normalized feed-forward block

`gelu(tensor)` selects exact default `aten::gelu` with fixed schema string
`approximate="none"`. Generic `owned_string_literal` metadata validates the
stable string allocator and destructor, copies the literal under a generated
cleanup guard, and transfers its ownership to the dispatcher. The input tensor
is still independently readable and disposable.

Ordinary Terlan composes layer normalization, `[2,2] -> [2,4]` affine
expansion, GELU, `[2,4] -> [2,2]` affine projection, and residual addition. The
result approximates `[0.8413447461,3.8413447461;4.8413447461,1.8413447461]`
within `1e-7` MSE. Fixture and pinned immutable consumers preserve shape and
float64 dtype, dispose every handle, and reject a `[3,4]` expansion matrix for
normalized width `2`. No fused feed-forward helper or compiler operator case is
introduced.

## PT32: complete pre-normalized transformer encoder

PT32 introduces no native operation. One Terlan graph combines the PT30
normalized scaled dot-product attention residual with a second layer
normalization and the PT31 GELU feed-forward residual. For input
`[1.0,3.0;4.0,2.0]`, attention produces `[2.0,4.0;5.0,3.0]`; the `[2,4]`
feed-forward hidden state then produces final output
`[1.8413447461,4.8413447461;5.8413447461,2.8413447461]` within `1e-7` MSE.

Package-owned and immutable consumers preserve the original input, retain
`[2,2]` float64 output shape, and independently dispose every handle. A
width-`3` feed-forward output cannot join the width-`2` attention residual and
is rejected through the stable C ABI status. The slice proves the reusable
surface can express a full transformer encoder block without a fused binding.

## PT33: transformer encoder classifier and evaluation

PT33 extends the complete encoder without adding a native operation. Token-wise
maximum pooling reduces its `[2,2]` output to `[2]`; `unsqueeze(..., 0)` restores
an explicit `[1,2]` batch before an affine two-class head. Symmetric classifier
weights produce exact `[0.5,0.5]` probabilities, deterministic int64 prediction
`[0]`, and cross-entropy between `0.69` and `0.70` for target `[0]`.

The ABI fixture now models generic valid `aten::unsqueeze` insertion dimensions
rather than only appending `-1`; this corrects fixture behavior but changes no
public binding. The immutable model expresses attention and feed-forward as
reusable Terlan functions, bounding VM artifact nesting while keeping native
cleanup local to each returned tensor boundary. Package-owned and immutable
consumers preserve the original input and dispose the complete encoder,
pooling, classifier, prediction, and loss graph. Class index `2` is rejected
for two-class logits through a stable C ABI status.

## PT34: token IDs through transformer classification

PT34 closes the first end-to-end text-model path using only the existing 39
generated functions. Int64 IDs `[0,1]` index a `[2,2]` embedding table to exact
features `[1.0,3.0;4.0,2.0]`; those features pass through the complete
pre-normalized transformer encoder, token maximum pooling, an affine two-class
head, softmax, argmax, and cross-entropy evaluation.

Package-owned and immutable consumers verify the embedding at MSE `0.0`, exact
probabilities `[0.5,0.5]`, prediction `[0]`, and loss between `0.69` and `0.70`
against both the ABI fixture and pinned LibTorch. Attention and feed-forward
remain ordinary Terlan functions, not fused bindings. Token ID `2` is rejected
for the two-row table through the stable C ABI status, and the slice adds no
native symbol, adapter, or compiler special case.

## PT35: batched token-transformer inference

PT35 proves that the existing generic tensor operations preserve a leading
batch dimension through the complete text model. Token IDs
`[[0,1],[1,0]]` embed to `[2,2,2]`; Q/K/V projections use rank-3 by rank-2
matrix multiplication, attention scores and context use batched rank-3 matrix
multiplication, keys transpose dimensions `1` and `2`, and softmax normalizes
the final attention dimension `2`.

Token pooling over dimension `1` yields `[2,2]` classifier features. Both
samples produce exact uniform probabilities and int64 predictions `[0,0]`,
with mean cross-entropy between `0.69` and `0.70`. The ABI fixture now models
the required batched matrix shapes and actual transpose size swaps; real
semantics remain validated by pinned LibTorch. Incompatible batch dimensions
`[2,2,2] × [3,2,2]` are rejected through the stable C ABI status. No public
function, native symbol, package adapter, or compiler branch is added.

## PT36: batched vision-classifier inference

PT36 promotes the complete vision path to two-image NCHW execution without a
new binding. A unit image and a doubled image at `[2,1,3,3]` pass together
through `conv2d`, ReLU, global adaptive average pooling, and
`flatten(..., 1, 3)`. Their convolution tensors contain exact values `4.0` and
`8.0`; global pooling retains exact features `[4.0,8.0]` at `[2,1,1,1]`, with
MSE `0.0` for both stages.

The `[2,1]` feature matrix is classified in one operation, producing exact
uniform `[2,2]` probabilities, int64 predictions `[0,0]`, and mean
cross-entropy between `0.69` and `0.70`. A class-target batch of length `1` is
rejected for two logit rows through the stable C ABI status. Fixture and pinned
immutable consumers dispose every handle and retain the same 39 generated
functions.

## PT37: grouped depthwise-pointwise CNN

PT37 exercises the grouped-convolution contract as a useful two-stage visual
backbone. A `[2,2,3,3]` batch enters a depthwise `[2,1,2,2]` kernel with
`groups=2`; each channel remains independent and produces exact `4.0`/`8.0`
features at `[2,2,2,2]`. A pointwise `[2,2,1,1]` kernel with `groups=1` then
mixes channels to exact `8.0`/`16.0` features without changing spatial shape.

Global pooling yields `[8.0,8.0;16.0,16.0]`, followed by batched classification,
uniform probabilities, predictions `[0,0]`, and mean cross-entropy between
`0.69` and `0.70`. Complete depthwise, pointwise, and pooled tensors require
MSE `0.0`. A grouped kernel with three output channels and `groups=2` is
rejected through the stable C ABI status. This composes the existing 39
functions and changes no fixture, native adapter, or compiler path.

## PT38: normalized residual CNN block

PT38 composes the existing grouped-convolution and inference-normalization
surfaces into a residual visual block. A `[2,2,2,2]` batch passes through
depthwise `groups=2` convolution, batch normalization, ReLU, pointwise channel
mixing, and a second batch normalization. The unchanged-width branch then
joins the original input through residual addition and ReLU.

Normalization preserves exact depthwise values `1.0`/`2.0` and pointwise
values `2.0`/`4.0`; the residual is exactly `3.0`/`6.0`, and global features
are `[3.0,3.0;6.0,6.0]`, all at MSE `0.0`. Batched classification produces
uniform probabilities, predictions `[0,0]`, and mean cross-entropy in
`0.69–0.70`. A three-channel branch cannot join a two-channel residual and is
rejected stably. Immutable consumers use bounded Terlan stage functions; the
public surface remains 39 functions with no fixture or compiler change.

## PT39: stacked batched transformer encoders

PT39 proves model depth by invoking the reusable Terlan attention and
feed-forward functions twice over `[2,2,2]` token state. The same immutable
layer-normalization parameters are safely reused across both pre-normalized
encoder layers, while every layer result remains an independently owned opaque
handle.

The complete second-layer tensor matches
`[2.6826894921,6.6826894921;7.6826894921,3.6826894921]` and its reversed second
batch within `1e-7` MSE. Token pooling yields identical
`[7.6826894921,6.6826894921]` rows before uniform classification and predictions
`[0,0]`. A width-`3` projection cannot become the state of a width-`2` next
layer and is rejected through the stable C ABI boundary. No binding, fixture,
adapter, or compiler change is required.

## PT40: additive causal attention mask

PT40 inserts a caller-owned additive mask between scaled attention scores and
softmax. Mask `[0.0,-1000.0;0.0,0.0]` broadcasts over the two-sequence batch,
yielding exact attention rows `[1.0,0.0]` for the first token and `[0.5,0.5]`
for the second. This proves future-token exclusion rather than merely accepting
a mask-shaped argument.

The complete context is `[0,2;1,1]` and `[2,0;1,1]`; residual, feed-forward,
pooled, probability, prediction, and cross-entropy outputs are then validated
through fixture and pinned immutable consumers. The fixture records a reviewed
marker for copied large-negative causal-mask data so its representative
softmax can distinguish this path, while real LibTorch validates every tensor
element. A `[3,3]` mask is rejected for `[2,2,2]` scores. The public surface
remains 39 functions.

## PT41: batched encoder-decoder cross-attention

PT41 separates the decoder query state from encoder-style memory. Decoder
shape `[2,2,2]` supplies Q while independently owned memory `[2,3,2]` supplies
K and V. Their distinct lengths produce attention `[2,2,3]`, with every row
exactly `[1/3,1/3,1/3]`; this cannot be mistaken for the existing two-token
self-attention path.

Asymmetric memory rows yield signed context values `[-1/3,1/3]` for the first
batch and `[1/3,-1/3]` for the second. Exact residuals then pass through the
existing feed-forward, token-pooling, classification, prediction, and
cross-entropy composition. Mismatched decoder/memory batch dimensions are
rejected before context construction. Both sources and all intermediates are
owned and disposed independently. The public surface remains 39 functions,
with no fixture, adapter, or compiler change.

## PT42: complete causal cross-attention decoder layer

PT42 joins the preceding causal and cross-attention capabilities into a real
decoder-layer topology. Batched embedded tokens first use the `[2,2]` additive
causal mask, producing exact self-attention residual
`[1,5;5,3]` and `[6,2;2,4]`. That state supplies Q for attention over the
separately owned `[2,3,2]` memory from PT41.

The resulting cross-attention residual is checked element-by-element before
the normalized GELU feed-forward path. Token pooling produces
`[5.5080114128,6.1746780794]` and
`[7.1746780794,4.5080114128]`, followed by uniform classification and
predictions `[0,0]`. A width-`3` context cannot join width-`2` decoder state and
is rejected stably. The public surface remains 39 functions, with no new
fixture behavior, native adapter, or compiler change.

## PT43: tokenized source-target sequence model

PT43 removes the last caller-constructed model-state shortcut from the broad
sequence path. Source IDs `[[0,0,1],[1,1,0]]` use a source-owned embedding table
to create exact `[2,3,2]` encoder memory. Independent target IDs
`[[0,1],[1,0]]` use a distinct target table to create the `[2,2,2]` causal
decoder input.

Those two token streams then execute the complete PT42 decoder, pooling, and
classification graph with exact embeddings, decoder output, and pooled
features. A three-batch source stream cannot cross-attend with a two-batch target
stream and is rejected stably. The public surface remains 39 functions with no
fixture, adapter, or compiler change. This is the final planned broad CPU model
composition; later work closes deferred runtime capabilities.

## MI0: stateless serialized TorchScript inference

MI0 adds `run_torchscript(input: Tensor, path: String): Tensor` as the 40th
generated function. The package-owned C++ adapter loads the archive with
`torch::jit::load`, invokes a single-tensor `forward`, verifies that the result
is a tensor, and returns a newly owned `AtenTensorHandle`. The caller retains
ownership of the input; the scoped module and all C++ exceptions remain inside
the adapter.

The pinned model implements `input * 2.0`, so both package and immutable Git
consumers require scalar float64 input `3.0`, output `6.0`, unchanged input, and
independent disposal. A missing file returns stable status `7041`; an existing
invalid archive returns `7042`. Paths containing an interior NUL are rejected
by the generated Rust boundary before C++ invocation. This initial surface is
stateless by design: reusable model handles, multiple inputs, tuples, and
non-tensor outputs remain outside MI0.

## MI1: reusable TorchScript model handle

MI1 adds an independently owned, thread-confined `TorchScriptModel` opaque
resource beside `Tensor`. `load_torchscript(path)` constructs the resource,
`torchscript_forward(model, input)` borrows both model and tensor for one call
and returns a new owned tensor, and `dispose_model(model)` destroys the module
exactly once. The MI0 `run_torchscript` convenience remains available.

Loading one archive and forwarding scalar float64 inputs `3.0` and `4.0`
produces independently owned outputs `6.0` and `8.0`. Wire handles are checked
against a generated tagged resource store, so tensor/model confusion and use
after model disposal cannot reach C++; a disposed model is rejected as
`stale_handle`. The surface contains 43 generated functions. Multiple model
inputs, tuples, lists, dictionaries, optional values, and non-tensor outputs
remain deferred.

## MI2: two positional tensor inputs

`torchscript_forward2(model, first, second)` borrows one loaded model and two
independently owned tensors for a single forward call, returning one new owned
tensor. The pinned model implements the order-sensitive expression
`left - right * 2.0`; inputs `3.0` and `4.0` must remain readable and produce
exact output `-5.0`. Passing those two inputs to the single-input fixture is a
schema-arity error contained as stable C status `7043`.

The surface contains 44 generated functions. This is fixed positional arity,
matching ordinary Python calls into a known model signature. Variable-length
`List[Tensor]`, keyword arguments, nested tuples/dictionaries, and multiple
outputs remain deferred until the generic native value protocol owns those
shapes and their handle lifetimes.

## MI3: owned two-tensor output container

`torchscript_forward_pair(model, input)` accepts only a TorchScript tuple of
exactly two tensors and returns an owned `TensorPair`. Its `tensor_pair_first`
and `tensor_pair_second` operations each return a new independently owned
`Tensor`; both extracted handles remain readable after `dispose_tensor_pair`.
The pinned model maps input `2.0` to `(4.0, 6.0)` while preserving the input.

A non-tuple result is status `7046`, wrong tuple arity is `7047`, and a
non-tensor element is `7048`. The surface contains 48 generated functions and
three resource classes: `Tensor`, `TorchScriptModel`, and `TensorPair`.
`TensorPair` is an explicit native ownership container rather than a Terlan
tuple; arbitrary nested output structures remain deferred to generic
NativeBoundary value lowering.

## MI4: copied float model result

`torchscript_forward_float(model, input)` accepts only a TorchScript `float`
result and copies its IValue double directly into an ordinary Terlan `Float`.
No native result handle or disposal operation exists. The pinned model computes
`float(input.item()) * 2.0`, requiring unchanged input `3.0` and exact output
`6.0`. A tensor or other non-float result is stable status `7049`.

At MI4 the surface contained 49 generated functions.

## MI5: copied integer model result

`torchscript_forward_int(model, input)` accepts only a TorchScript `int` result
and copies its signed 64-bit value directly into an ordinary Terlan `Int`. The
pinned model computes `int(input.item()) + 3`, requiring unchanged integer input
`4` and exact output `7`. A tensor or other non-integer result is stable status
`7050`; no native result handle or disposal operation exists.

At MI5 the surface contained 50 generated functions.

## MI6: copied boolean model result

`torchscript_forward_bool(model, input)` accepts only a TorchScript `bool`
result and copies it directly into an ordinary Terlan `Bool`. The pinned model
computes `input.item() > 0`, requiring unchanged integer input `1` and exact
output `true`. A tensor or other non-boolean result is stable status `7051`; no
native result handle or disposal operation exists.

At MI6 the surface contained 51 generated functions.

## MI7: copied owned string model result

`torchscript_forward_string(model, input)` accepts only a TorchScript `str`
result. Structured C metadata ties an owned `char **` output to a `size_t`
byte-length output, immediate UTF-8 copying, and an infallible consuming
destructor. The generated wrapper copies the bytes and destroys native storage
before returning an ordinary Terlan `String`; no native pointer or lifetime is
observable.

The pinned conditional model returns `"positive"` for integer input `1` while
leaving that input readable. A tensor or other non-string result is stable
status `7052`. At MI7 the surface contained 52 generated functions. Optional,
list, dictionary, and nested mixed results remained deferred to their own
reviewed value-model slices.

## MI8: copied owned integer-list model result

`torchscript_forward_int_list(model, input)` accepts only a TorchScript
`List[int]` result. Structured metadata ties an owned `int64_t **` output to a
`size_t` element count, immediate copy, and an infallible consuming destructor.
The generated wrapper copies every element and destroys native storage before
returning an ordinary Terlan `List[Int]`.

The pinned model maps integer input `4` to exact list `[4,5,6]` while preserving
the input. A tensor or other non-integer-list result is stable status `7053`.
The surface contains 53 generated functions. Optional, dictionary, and nested
mixed results remain deferred.

## MI9: copied owned floating-point-list model result

`torchscript_forward_float_list(model, input)` accepts only a TorchScript
`List[float]` result. The generic owned-array metadata explicitly selects
`float64`, tying a `double **` output to a `size_t` element count, immediate
copy, and an infallible `double *` consuming destructor. Generated Rust returns
an ordinary Terlan `List[Float]` through the existing `ok_floats` protocol.

The pinned model maps floating input `1.5` to exact list `[1.5,2.0,2.5]` while
preserving the input. A tensor or other non-floating-point-list result is stable
status `7054`. The surface contains 54 generated functions.

## MI10: validated copied boolean-list model result

`torchscript_forward_bool_list(model, input)` accepts only a TorchScript
`List[bool]` result. The adapter materializes LibTorch's specialized BoolList as
owned `uint8_t` values, while generic metadata ties the `uint8_t **` output to
its length and consuming destructor. Generated Rust copies and destroys native
storage, rejects every byte other than `0` or `1`, converts to `Vec<bool>`, and
returns an ordinary Terlan `List[Bool]` through `ok_bools`.

The pinned model maps integer input `2` to exact list `[true,false,false]` while
preserving the input. A tensor or other non-boolean-list result is stable status
`7055`. The surface contains 55 generated functions.

## MI11: copied owned string-list model result

`torchscript_forward_string_list(model, input)` accepts only a generic
TorchScript list whose every element is a string. Structured metadata ties
owned `char ***` values and owned `size_t **` byte lengths to one element count
and one consuming destructor. Generated Rust copies all bytes, destroys all
native allocations exactly once, validates UTF-8, and returns `List[String]`
through `ok_strings`. Empty elements are preserved without NUL termination.

The pinned model maps integer input `1` to exact list
`["positive","","ready"]` while preserving the input. A tensor or other
non-string-list result is stable status `7056`. The surface contains 56
generated functions.

## MI12: optional owned tensor model result

`torchscript_forward_optional_tensor(model, input)` requires an exact
TorchScript `Optional[Tensor]` result. A null output pointer becomes ordinary
Terlan `None`; a non-null output becomes `Some(Tensor)` with independent native
ownership and a VM resource identity. No placeholder tensor is allocated for
the absent case.

The pinned model maps positive input `3.0` to `Some(6.0)` and input `0.0` to
`None` while preserving both inputs. A model whose schema returns an ordinary
tensor is rejected as stable status `7057`.

## MI13: independently owned tensor-list model result

`torchscript_forward_tensor_list(model, input)` requires an exact TorchScript
`List[Tensor]` result. The C++ adapter allocates one independent tensor object
per list element and transfers their identities in a temporary `uint64_t`
array. Generated Rust copies and destroys that array, rejects null elements
while cleaning every valid transferred tensor, and returns `List[Tensor]`
through the generic `ok_handles` NativeBoundary reply. Every element receives
its own VM resource identity and can be disposed independently.

The pinned model maps input `3.0` through `[input * 2.0, input]` to exact values
`[6.0,3.0]` while preserving the input. A tensor or other non-tensor-list result
is rejected as stable status `7058`.

## Deferred surfaces

- additional dispatcher-list element types, dictionaries, and nested mixed
  model results.
- zero-copy caller-buffer construction and deleter callback ownership; the
  current shaped-data constructor deliberately performs an owned copy.
- zero-copy tensor exchange; `Tensor.from_packet` and `Tensor.to_packet` now
  provide bidirectional pointer-free copied TNXP exchange for Bool, UInt8,
  Int32, Int64, Float32, and Float64. PyTorch remains independent of Polars;
  Polars and ndarray own their packet conversion APIs.
- synchronous CUDA placement, CPU/CUDA operation parity, and opaque explicit
  streams are implemented in the opt-in CUDA lane. Device-count discovery
  remains owned by `terlan-cuda`, and non-blocking transfers remain future
  work.
- richer exception classification and thread-local text: the CXX bridge now
  translates every direct compatibility call through `Result<T>` and preserves
  a bounded native error payload at the Terlan protocol boundary.

The ordered milestones and gates are defined in [ROADMAP.md](ROADMAP.md). They
are generic binder milestones plus package-owned CXX adapter milestones; no
PyTorch special case lives in the compiler.

This CUDA surface is specifically LibTorch tensor placement. Direct Rust CUDA
bindings remain available through a separate external `terlan-cuda` package and
do not depend on the PyTorch operator roadmap.
