# Terlan LibTorch roadmap

This roadmap owns the PyTorch/LibTorch package surface. Generic C ABI and
NativeBoundary generation remain in the Terlan compiler; PyTorch declarations,
operator schemas, distribution pins, and execution tests remain in this
repository.

The package targets LibTorch's released stable C/AOTI ABI and versioned
`StableIValue` dispatcher. It does not bind the public C++ frontend and it does
not require PyTorch to serve as the proof package for Terlan's generic C++
generator.

## Invariants

- CPU execution is the mandatory baseline and must remain usable without CUDA.
- CUDA through LibTorch is an opt-in capability of this package, never a Terlan
  compiler dependency. Direct CUDA access belongs in a separate external
  `terlan-cuda` package and is not coupled to this roadmap's CPU prerequisites.
- Terlan sees typed values and opaque NativeBoundary handles, never tensor or
  GPU memory pointers.
- Every dispatcher operation declares its exact operator, overload, stack
  layout, extension ABI version, and ownership transfers as metadata.
- A package gate must validate observable operation semantics, not merely
  successful linking or a non-error return.
- Released LibTorch distributions are the compatibility target; the PyTorch
  source checkout is reference material, not a package build dependency.

## Completed baseline

### PT0: scalar tensor lifecycle

- [x] Link a pinned released LibTorch distribution.
- [x] Construct and read an `int64` scalar tensor.
- [x] Own and dispose `AtenTensorHandle` exactly once.
- [x] Expose rank, element count, dtype, device type/index, and ABI version.

### PT1: copied shape arrays

- [x] Describe owner, length source, and immediate-copy policy in metadata.
- [x] Copy borrowed LibTorch dimensions into an owned Terlan `List[Int]`.
- [x] Reject invalid lengths and non-empty null array pointers.

### PT2: owned tensor dispatcher result

- [x] Describe `aten::clone` and its default overload as metadata.
- [x] Duplicate the input tensor handle before placing it on the owning stack.
- [x] Steal the returned stack slot into a new Terlan-owned tensor handle.
- [x] Execute readback and independent disposal through Terlan.

### PT3: signed integer StableIValue

- [x] Encode a metadata-linked Terlan `Int` as a 64-bit `StableIValue`.
- [x] Execute `aten::unsqueeze(tensor, -1)` against released LibTorch.
- [x] Prove rank growth, copied shape correctness, clone interoperability, and
  independent cleanup in a PyTorch-owned Terlan test.
- [x] Reject unknown, missing, repeated, or unsupported dispatcher arguments.

## CPU closure before CUDA

### PT4: multiple tensor inputs and CPU matmul

- [x] Add a dispatcher stack value for a second owned tensor-handle copy.
- [x] Extend the NativeBoundary helper to validate and borrow two live tensor
  resources without exposing raw handles or permitting mutable aliasing.
- [x] Form deterministic `[1, 1]` tensors from the existing scalar and
  `unsqueeze` surface, then execute `aten::matmul` on CPU.
- [x] Assert exact output shape and value, and dispose both inputs and output.
- [x] Reject stale secondary handles, wrong handle types, missing duplication
  metadata, and unsupported output ownership with stable diagnostics.

PT4 is the minimum operation-model prerequisite for CUDA. It separates
multi-resource and dispatcher correctness from device/runtime failures.

### PT5: external package gate and CPU adversarial baseline

- [x] Add the compiler-repository `make terlan-pytorch-package-check` target.
- [ ] Make it locate or check out this sibling repository through normal package
  metadata rather than compiler-local source paths.
- [x] Run the pinned CPU release check, generated consumer, and semantic
  dispatcher tests from that compiler-level gate.
- [x] Add a fresh external consumer import test independent of generated package
  tests.
- [x] Add stable negative cases for missing LibTorch, incompatible version,
  scalar `matmul` shape mismatch, dtype mismatch, and stale handles.
- [x] Assert the successful CPU result's dtype and device observations.
- [x] Produce a machine-readable pass/skip/fail report with the LibTorch build
  hash and binding-manifest hash.

The external-consumer gate commits the generated package at an immutable Git
revision, fetches it through normal package metadata, deletes the original
package tree, and executes from the verified package cache. Closing this path
also fixed the generic C ABI generator to emit the declared Terlan namespace
and `[native.rust]` helper contract in generated package manifests.

The repository root is now itself the generated `terlan-pytorch` version
`0.0.7` package. Public package identity is independent from the native Rust
crate identity, C++20 is declared explicitly for the pinned LibTorch 2.13
headers, and warnings are errors for package-owned C++ compilation. The full
CPU gate passes 55 package tests, the generated NativeBoundary test, and the
immutable Git consumer directly from the materialized root. The remaining PT5
item is only the compiler-level remote checkout fallback when no sibling or
explicit override exists.

Invalid device placement remains attached to the later device surface: the
current package cannot request a device, so inventing a rejection-only API
would not validate LibTorch interop. It becomes mandatory when a public
placement operation is added. PT9 closes the dtype mismatch case through a
real float-alpha operation and two deliberately incompatible int64 tensors.

CUDA execution through LibTorch opens only after PT4 and PT5 are green. Broader
CPU operator coverage is not a prerequisite. This ordering does not block a
direct Rust CUDA package.

### PT6: mixed dispatcher stack and named shape overload

- [x] Generate a three-slot stack containing one owned tensor copy and two
  signed integer arguments.
- [x] Select the exact `aten::transpose.int` named overload.
- [x] Transpose a rank-2 tensor through package-owned and fresh external Terlan
  consumers while preserving exact value and rank.
- [x] Preserve independent ownership and disposal of input and result.

### PT7: named tensor arithmetic overload

- [x] Select exact `aten::mul.Tensor` rather than attempting to encode a
  `Scalar` as a StableIValue integer.
- [x] Execute `multiply(4, 5) = 20` in package-owned and fresh external Terlan
  consumers with independent cleanup.
- [x] Extend and test the generic direct C wrapper's mapping of multiple
  borrowed opaque handles, ready for generated stable C shims.
- [x] Publish scalar arithmetic shims only after a compatible float
  constructor/dtype surface exists; LibTorch stably rejects their C `double`
  alpha for the current int64-only tensors.

### PT8: float64 scalar lifecycle

- [x] Add generic C ABI `Float` argument/result protocol support.
- [x] Permit multiple typed constructors for one opaque NativeBoundary resource.
- [x] Bind stable float64 tensor construction and item readback.
- [x] Execute `multiply(2.5, 4.0) = 10.0`, assert dtype `7`, and dispose every
  handle through package-owned and fresh external Terlan consumers.

### PT9: stable C scalar arithmetic and dtype rejection

- [x] Bind `aoti_torch_aten_subtract_Tensor` with two borrowed tensor handles,
  one Terlan `Float`/C `double` alpha, and an owned result.
- [x] Execute `subtract(4.0, 1.5, 2.0) = 1.0` through package-owned and fresh
  external consumers.
- [x] Reject int64 inputs to the float-alpha shim with a stable package-level
  dtype-mismatch diagnostic.

### PT10: bool scalar lifecycle

- [x] Add generic C ABI `Bool` argument/result protocol support.
- [x] Bind LibTorch's stable bool scalar constructor and item readback.
- [x] Round-trip both `true` and `false`, assert bool dtype `11`, and dispose
  every handle through package-owned and fresh external Terlan consumers.

### PT11: nonlinear activation

- [x] Select the exact default `aten::relu` dispatcher operation with one
  independently duplicated tensor handle and one owned result.
- [x] Execute `relu(-2.5) = 0.0` for float64 through package-owned and fresh
  external consumers.
- [x] Prove the immutable public operation preserves the source value and lets
  source and result be disposed independently.

### PT12: bounded indexing and owned views

- [x] Bind the generated stable C `aoti_torch_aten_narrow` operation with one
  borrowed tensor, three signed indices, and one owned result.
- [x] Narrow a rank-1 tensor to shape `[1]`, preserve its exact value, and read
  the source after the operation through package-owned and fresh consumers.
- [x] Reject an out-of-range start with a stable C ABI diagnostic.
- [x] Keep the generic generated smoke consumer from inventing scalar values
  for direct C operations whose valid domain is package-specific.

### PT13: dimension-range flattening

- [x] Select the exact `aten::flatten.using_ints` overload with one owned
  StableIValue tensor copy and two signed dimension arguments.
- [x] Flatten a rank-2 float64 tensor to rank 1 while preserving its value and
  leaving the source independently readable and disposable.
- [x] Execute the same shape assertions through the fresh Git consumer.
- [x] Reject an out-of-range dimension interval with a stable diagnostic.

### PT14: dimension-list reduction

- [x] Add generic structured C metadata for borrowed `List[Int]` input arrays
  linked to an exact pointer-length parameter pair.
- [x] Generate checked Rust slice/length conversion and native helper `li:`
  transport without exposing the C pointer to Terlan.
- [x] Bind `aoti_torch_aten_amax` and reduce a rank-2 float64 tensor over
  dimension `[0]` to rank 1 through package-owned and fresh consumers.
- [x] Preserve source/result ownership and reject an out-of-range reduction
  dimension with a stable diagnostic.

### PT15: owned shaped tensor construction

- [x] Add generic reviewed fixed-input metadata for nullable C pointers,
  pointer-backed `int32_t` options, and scalar `int32_t` constants.
- [x] Bind `aoti_torch_aten_full` as `full_float(shape, value)` with fixed
  float64 dtype and CPU placement hidden behind the generated adapter.
- [x] Construct shape `[2, 3]`, assert rank `2`, element count `6`, dtype `7`,
  copied shape, and repeated value through reductions.
- [x] Execute construction in a dedicated fresh Git consumer and reject a
  negative dimension with a stable diagnostic.

### PT16: owned non-constant float64 data construction

- [x] Add generic structured `List[Float]` input arrays with checked
  `const double*`/length lowering and `lf:` native transport.
- [x] Permit package-owned C adapter sources to compile beside an externally
  linked stable C distribution without moving package logic into the compiler.
- [x] Publish `from_floats(values, shape)` for row-major CPU float64 data. The
  adapter creates a call-scoped non-owning view, clones it through LibTorch,
  destroys the temporary handle, and returns only the independently owned copy.
- [x] Prove non-constant `[2, 3]` data through package-owned and fresh Git
  consumers, and reject a value-count/shape mismatch as
  `c_abi_status_7003`.
- [x] Resolve the VM's untyped empty-list wire form against the declared
  `List[Float]` argument and construct an owned `[0, 3]` tensor.

### PT17: affine forward composition

- [x] Bind the pinned CPU generated stable-C
  `aoti_torch_cpu_add_Tensor` operation as `add(left, right, alpha)` with two
  borrowed inputs and one independently owned result.
- [x] Compose non-constant `[2, 3]` input data, `[3, 1]` weights, matrix
  multiplication, and broadcast bias addition into an affine forward pass.
- [x] Require output shape `[2, 1]`, maximum `7.0` by reduction readback,
  float64 dtype, and unchanged readable input/bias tensors through
  package-owned and fresh Git consumers.
- [x] Reject incompatible `[2, 3] + [4]` broadcasting with a stable C ABI
  status diagnostic.
- [x] Keep the public operation device-neutral; the later CUDA profile must
  bind its matching generated backend shim without changing Terlan call sites.

### PT18: classifier forward and probability normalization

- [x] Bind the exact `aten::softmax.int` dispatcher overload with one owned
  tensor copy, a signed dimension, and null inherited dtype.
- [x] Compose non-constant input, zero classifier weights, broadcast bias,
  logits, and dimension-1 softmax into an owned `[2, 2]` probability tensor.
- [x] Require float64 dtype, four elements, unchanged zero logits, and exact
  uniform probability maximum `0.5` through package-owned and fresh Git
  consumers.
- [x] Reject dimension `2` for rank-2 logits through a stable C ABI status.
- [x] Add generic `generated_smoke: package_owned` metadata so operations with
  package-specific valid shapes remain public and mandatory in package gates
  without being invoked on the generic scalar smoke fixture.

### PT19: classifier predictions and optional dispatcher values

- [x] Extend the generic C ABI generator with reviewed `bool_argument` and
  `owned_optional_int_argument` StableIValue stack entries.
- [x] Validate allocator/destructor symbol signatures and guard heap-backed
  optional values until ownership transfers into `torch_call_dispatcher`.
- [x] Bind exact `aten::argmax(Tensor, int?, bool)` as
  `argmax(tensor, dimension, keep_dimension)` without a PyTorch-specific
  compiler branch.
- [x] Require int64 prediction dtype and both `[2]` reduced and `[2, 1]`
  keep-dimension shapes through package-owned and fresh Git consumers.
- [x] Compose classifier probabilities into deterministic class indices and
  reject an out-of-range prediction dimension with a stable C ABI status.

### PT20: model evaluation and scalar loss

- [x] Bind exact `aten::mse_loss(Tensor, Tensor, int)` as
  `mse_loss(predicted, target, reduction)` through two independently owned
  dispatcher inputs and a signed reduction value.
- [x] Prove mean reduction (`1`) returns an owned scalar float64 tensor with
  exact loss `1.0` while both equal-shape source tensors remain readable.
- [x] Extend classifier evaluation with an equal-shape zero target and require
  exact mean loss `0.25` alongside probability and class-index observations.
- [x] Reject incompatible `[2, 2]` and `[3]` loss operands through the stable C
  ABI in the fresh immutable Git consumer, and reject reduction values outside
  the documented `0`/`1`/`2` domain.

### PT21: batched labels and classification loss

- [x] Add generic `float_argument` StableIValue metadata, validating every
  declared Float exactly once and encoding its IEEE-754 bits with
  `f64::to_bits()`.
- [x] Add package-owned `from_ints(values, shape)` construction through a
  call-scoped int64 LibTorch view followed by an owned clone.
- [x] Bind exact `aten::cross_entropy_loss` with owned logits/target inputs,
  null class weights, reduction, ignore index, and float label smoothing.
- [x] Compose `[2, 2]` zero logits and `[0, 1]` batched int64 targets with
  smoothing `0.1`, requiring scalar loss between `0.69` and `0.70` while both
  source tensors remain readable.
- [x] Reject float class targets and label smoothing above `1.0` through stable
  C ABI diagnostics in the fresh immutable Git consumer.

### PT22: explicit gradient and functional parameter update

- [x] Audit released stable autograd entry points: `torch_set_requires_grad`
  exists, but no stable gradient getter closes observable graph autograd;
  `aten::_backward` additionally requires Tensor-list and unit-output stacks.
- [x] Bind exact `aten::mse_loss_backward` as
  `mse_loss_gradient(grad_output, predicted, target, reduction)` using three
  independently owned dispatcher inputs.
- [x] Require mean-loss gradient `0.5` for uniform `[2, 2]` parameter `2.0`
  against target `1.0`, preserving the public parameter and target handles.
- [x] Scale the gradient by learning rate `0.1`, functionally update the
  parameter to approximately `1.95`, and require MSE to fall from `1.0` to
  approximately `0.9025`.
- [x] Reject incompatible gradient operands through a stable C ABI status in
  the fresh immutable Git consumer.
- [ ] Close graph autograd only after generic owned Tensor-list inputs,
  unit-output dispatcher calls, and stable gradient retrieval are all proven.

### PT23: rank-4 NCHW convolution

- [x] Audit the pinned distribution's convolution exports and reject the
  deprecated device-generic `aoti_torch_convolution` shim as the package target.
- [x] Add a package-owned stable-C adapter over generated
  `aoti_torch_cpu_convolution`, containing its `AtenTensorHandle*` optional-bias
  representation and fixing non-transposed 2D convolution with zero output
  padding.
- [x] Publish `conv2d(input, weight, bias, stride, padding, dilation, groups)`
  using three borrowed opaque handles, three copied integer lists, and a signed
  group count without any PyTorch-specific compiler logic.
- [x] Widen the ABI fixture to rank 4 and require exact NCHW
  `[1,1,3,3] × [1,1,2,2] -> [1,1,2,2]` shape, float64 dtype, four output
  elements, maximum `4.0`, exact MSE `0.0` against a tensor filled with `4.0`,
  and readable source tensors.
- [x] Execute the same semantics from an immutable fresh Git consumer against
  pinned LibTorch and reject incompatible input/weight channels through a stable
  C ABI status.

### PT24: adaptive average pooling

- [x] Select the pinned generated CPU
  `aoti_torch_cpu__adaptive_avg_pool2d` single-result shim, avoiding the
  multiple-output ownership required by adaptive max pooling.
- [x] Publish `adaptive_avg_pool2d(input, output_size)` directly from structured
  metadata using the existing borrowed-handle and copied-`List[Int]` machinery;
  add no compiler or package-owned C adapter logic.
- [x] Reduce a non-constant NCHW `[1,1,4,4]` quadrant input to exact
  `[1,1,2,2]` values `[1.0,2.0,3.0,4.0]`, with maximum `4.0`, float64 dtype,
  source readability, and MSE `0.0` against the complete expected tensor.
- [x] Execute the same operation from an immutable fresh Git consumer and reject
  a one-element output-size list through a stable C ABI status.

### PT25: composed vision classifier inference

- [x] Compose existing public operations into one Terlan inference graph:
  `conv2d -> relu -> adaptive_avg_pool2d -> flatten -> matmul -> add ->
  softmax -> argmax`.
- [x] Require global pooled features at shape `[1,1]`, logits and probabilities
  at `[1,2]`, exact uniform probabilities `[0.5,0.5]` with MSE `0.0`, and int64
  prediction `[0]`.
- [x] Prove the graph through package-owned tests and an immutable fresh Git
  consumer against the ABI fixture and pinned LibTorch distribution.
- [x] Dispose input, parameters, every intermediate, expected tensor, reduction,
  probability, and prediction independently; add no fused native helper and no
  compiler special case.

### PT26: embedding lookup

- [x] Select exact default `aten::embedding` through the StableIValue dispatcher
  because the pinned distribution exposes no generated CPU embedding symbol.
- [x] Publish `embedding(weight, indices, padding_index,
  scale_gradient_by_frequency, sparse)` through two copied opaque handles, one
  signed integer, and two boolean stack values without new compiler machinery.
- [x] Look up batched int64 indices `[0,2]` in a float64 `[3,2]` table, requiring
  owned `[2,2]` output, exact MSE `0.0`, and readable weight/index sources.
- [x] Execute package-owned and immutable Git consumers against the fixture and
  pinned LibTorch, and stably reject index `3` for a three-row table.

### PT27: composed sequence classifier

- [x] Audit transformer normalization and pooled-embedding targets: exact layer
  norm needs generic dispatcher integer lists plus optional tensors, while native
  layer norm returns three tensors and embedding bag returns four; PT29 closes
  the single-result exact layer-norm route.
- [x] Compose `embedding -> amax -> matmul -> add -> softmax -> argmax` in
  ordinary Terlan as the broadest sequence classifier supported by the current
  generic boundary.
- [x] Require batched `[1,2]` indices, `[1,2,2]` embeddings, `[1,2]` pooled
  features/probabilities, exact probability MSE `0.0`, and int64 prediction
  `[0]` through fixture and pinned immutable consumers.
- [x] Dispose every source, parameter, intermediate, expected tensor, and result;
  stably reject a `[3,2]` projection matrix applied to `[1,2]` features.

### PT28: inference batch normalization

- [x] Bind exact default `aten::batch_norm` with required public weight, bias,
  running mean, and running variance tensors; encode its four present `Tensor?`
  schema positions as owned allocator-backed nested StableIValues.
- [x] Add generic `owned_optional_handle_copy` dispatcher lowering with validated
  allocator/destructor symbols and failure-safe ownership transfer; add no
  generated CPU shim, package adapter, or operator-specific compiler branch.
- [x] Normalize uniform NCHW `[1,2,2,2]` input `3.0` with weight `2.0`, bias
  `1.0`, mean `1.0`, variance `4.0`, epsilon `0.0`, requiring exact output
  `3.0`, unchanged shape/dtype, MSE `0.0`, and readable sources.
- [x] Execute through package-owned and immutable consumers against fixture and
  pinned LibTorch; stably reject a three-element mean for two input channels.
- [ ] Training-mode statistic updates remain deferred until their mutation and
  observable-running-stat ownership contract is specified independently.

### PT29: layer normalization and transformer residual

- [x] Add generic `owned_int_list_argument` dispatcher lowering backed by
  validated stable-list allocate/push/delete symbols and cleanup on partial
  construction failure.
- [x] Bind exact default `aten::layer_norm` with a public normalized-shape list,
  allocator-backed optional weight/bias copies, epsilon, and cuDNN flag; add no
  CPU shim, package adapter, or operator-specific compiler branch.
- [x] Normalize `[1.0,3.0,2.0,4.0]` over trailing shape `[2]` to exact
  `[-1.0,1.0,-1.0,1.0]`, preserving shape, float64 dtype, parameters, source,
  and complete-content MSE `0.0`.
- [x] Compose `layer_norm -> matmul -> add` as a transformer-style residual
  block with exact `[0.0,4.0,1.0,5.0]` output and independent disposal of every
  source, parameter, intermediate, expected tensor, reduction, and result.
- [x] Execute direct and composed paths through fixture and immutable consumers;
  stably reject normalized shape `[3]` for trailing input width `2`.

### PT30: normalized self-attention block

- [x] Compose existing generated functions only:
  `layer_norm -> Q/K/V matmul -> transpose -> score matmul -> scale -> softmax
  -> context matmul -> residual add`.
- [x] Require exact uniform `[0.5,0.5;0.5,0.5]` attention, unit context,
  residual `[2.0,4.0;5.0,3.0]`, and MSE `0.0` for all three complete tensors.
- [x] Preserve the input and independently dispose normalization parameters,
  projection weights, every Q/K/V and attention intermediate, expectations,
  reductions, losses, and results.
- [x] Execute through fixture and pinned immutable consumers; stably reject a
  `[3,2]` query projection applied to normalized `[2,2]` features.
- [x] Add no fused attention helper, package adapter, public function, or
  compiler special case.

### PT31: GELU and normalized feed-forward block

- [x] Add generic `owned_string_literal` dispatcher lowering with validated
  stable-string allocate/delete symbols and cleanup before ownership transfer.
- [x] Bind exact default `aten::gelu` with fixed schema value
  `approximate="none"`; preserve the input and return an independently owned
  float64 tensor.
- [x] Compose `layer_norm -> expansion matmul -> bias -> gelu -> projection
  matmul -> bias -> residual add` with `[2,2] -> [2,4] -> [2,2]` shapes.
- [x] Require residual values approximating
  `[0.8413447461,3.8413447461,4.8413447461,1.8413447461]` within `1e-7` MSE,
  dispose every handle, and execute through fixture and immutable consumers.
- [x] Stably reject a `[3,4]` expansion matrix for normalized width `2`; add no
  fused feed-forward helper, package adapter, or operator-specific compiler
  branch.

### PT32: complete pre-normalized transformer encoder

- [x] Compose PT30 attention and PT31 feed-forward into one ordinary Terlan
  graph with two layer normalizations and two residual joins.
- [x] Require exact attention residual `[2.0,4.0;5.0,3.0]`, `[2,4]` hidden
  expansion, and final encoder output approximating
  `[1.8413447461,4.8413447461;5.8413447461,2.8413447461]` within `1e-7` MSE.
- [x] Preserve the original input and independently dispose every parameter,
  attention intermediate, feed-forward intermediate, reduction, and result.
- [x] Execute through package-owned and immutable consumers; stably reject a
  `[2,3]` output projection joined to a width-`2` residual stream.
- [x] Add no public operation, fused encoder helper, package adapter, native
  symbol, or compiler change.

### PT33: transformer encoder classifier and evaluation

- [x] Extend the complete encoder with token maximum pooling, explicit batch
  insertion, affine two-class logits, softmax probabilities, argmax prediction,
  and cross-entropy evaluation.
- [x] Preserve `[2,2]` encoder output, pool to `[2]`, classify at `[1,2]`, and
  require exact `[0.5,0.5]` probabilities with prediction `[0]` and float64
  cross-entropy between `0.69` and `0.70`.
- [x] Correct the package-owned fixture's generic `unsqueeze` dimension
  insertion so dimension `0` produces `[1,2]` from a two-element vector.
- [x] Structure the immutable model as reusable Terlan attention and
  feed-forward functions, keeping VM artifact nesting bounded and cleanup local
  to each returned tensor boundary.
- [x] Execute the complete graph through package-owned and immutable consumers;
  stably reject class index `2` for two-class logits.
- [x] Preserve the source, dispose every encoder/classifier/evaluation handle,
  and add no public operation, native symbol, package adapter, or compiler
  change.

### PT34: token IDs through transformer classification

- [x] Start from int64 token IDs `[0,1]`, select an exact `[2,2]` embedding,
  and feed those owned features through the complete PT32 encoder and PT33
  classifier/evaluation graph.
- [x] Require exact embedded values `[1.0,3.0;4.0,2.0]` at MSE `0.0`, exact
  two-class probabilities `[0.5,0.5]`, prediction `[0]`, and cross-entropy
  between `0.69` and `0.70`.
- [x] Express attention and feed-forward as ordinary reusable Terlan functions
  so the full immutable consumer stays within the VM artifact nesting bound.
- [x] Execute through package-owned and immutable consumers against both the ABI
  fixture and pinned LibTorch; stably reject token ID `2` for a two-row
  embedding table.
- [x] Dispose the embedding table, token IDs, every model intermediate, and all
  results; add no public operation, native symbol, package adapter, or compiler
  change.

### PT35: batched token-transformer inference

- [x] Promote token IDs from `[T]` to `[B,T]` and embeddings/encoder state from
  `[T,D]` to `[B,T,D]`, using batch size `2`, token count `2`, and width `2`.
- [x] Exercise batched Q/K/V projections and score/context multiplication,
  transpose key dimensions `1` and `2`, and normalize attention over dimension
  `2` without adding a fused attention binding.
- [x] Pool dimension `1` to `[B,D]`, classify both samples together, and require
  exact `[2,2]` uniform probabilities, predictions `[0,0]`, and mean
  cross-entropy between `0.69` and `0.70`.
- [x] Widen only the ABI fixture's `matmul` shape model and real transpose
  dimension swapping; keep the generated public surface at 39 functions.
- [x] Execute package-owned and immutable consumers against the fixture and
  pinned LibTorch, dispose every batched handle, and stably reject incompatible
  batch dimensions `[2,2,2] × [3,2,2]`.

### PT36: batched vision-classifier inference

- [x] Run two distinct NCHW images together at `[2,1,3,3]`, preserving batch
  size through convolution, ReLU, adaptive average pooling, and feature
  flattening.
- [x] Require exact convolution values `4.0` for the unit image and `8.0` for
  the doubled image, exact pooled features `[4.0,8.0]`, and MSE `0.0` for both
  complete tensors.
- [x] Classify `[2,1]` features together, requiring exact uniform `[2,2]`
  probabilities, int64 predictions `[0,0]`, and mean cross-entropy between
  `0.69` and `0.70`.
- [x] Execute package-owned and immutable consumers against the fixture and
  pinned LibTorch; stably reject a target batch of length `1` for two image
  logits.
- [x] Dispose both image paths and all shared parameters/intermediates; add no
  public function, fixture behavior, native symbol, package adapter, or
  compiler change.

### PT37: grouped depthwise-pointwise CNN

- [x] Execute two-channel, two-image input `[2,2,3,3]` through a depthwise
  `[2,1,2,2]` convolution with `groups=2`, preserving independent channel
  topology at `[2,2,2,2]`.
- [x] Mix the depthwise channels through a pointwise `[2,2,1,1]` convolution
  with `groups=1`, proving a reusable depthwise-separable CNN backbone.
- [x] Require exact depthwise values `4.0`/`8.0`, exact pointwise values
  `8.0`/`16.0`, exact pooled features `[8.0,8.0;16.0,16.0]`, and MSE `0.0`
  for every complete intermediate.
- [x] Classify both samples with exact uniform probabilities, predictions
  `[0,0]`, and mean cross-entropy between `0.69` and `0.70` through fixture and
  pinned immutable consumers.
- [x] Stably reject `groups=2` with three output channels; dispose every handle
  and add no public function, fixture behavior, native symbol, package adapter,
  or compiler change.

### PT38: normalized residual CNN block

- [x] Compose depthwise `groups=2` convolution, inference batch normalization,
  ReLU, pointwise channel mixing, a second batch normalization, residual
  addition, and ReLU at stable `[2,2,2,2]` shape.
- [x] Reuse immutable normalization parameters across both branch stages while
  retaining independent NativeBoundary ownership and cleanup.
- [x] Require exact normalized depthwise values `1.0`/`2.0`, normalized
  pointwise values `2.0`/`4.0`, residual values `3.0`/`6.0`, pooled features
  `[3.0,3.0;6.0,6.0]`, and MSE `0.0` throughout.
- [x] Classify both samples with exact uniform probabilities, predictions
  `[0,0]`, and mean cross-entropy between `0.69` and `0.70` through fixture and
  pinned immutable consumers.
- [x] Stably reject a three-channel branch joined to a two-channel residual;
  use bounded ordinary Terlan stage helpers and add no public function, fixture
  behavior, native symbol, package adapter, or compiler change.

### PT39: stacked batched transformer encoders

- [x] Reuse the ordinary Terlan attention and feed-forward functions to execute
  two consecutive pre-normalized encoder layers over `[2,2,2]` batched token
  state.
- [x] Reuse immutable layer-normalization parameters across both layers while
  preserving independent intermediate handles and cleanup.
- [x] Require the complete second-layer tensor to approximate
  `[2.6826894921,6.6826894921;7.6826894921,3.6826894921]` and its reversed
  companion batch within `1e-7` MSE.
- [x] Require exact pooled shape `[2,2]`, pooled rows
  `[7.6826894921,6.6826894921]`, uniform probabilities, predictions `[0,0]`,
  and mean cross-entropy between `0.69` and `0.70` through fixture and pinned
  immutable consumers.
- [x] Stably reject a width-`3` projected state joined between width-`2`
  encoder layers; add no public function, fixture behavior, native symbol,
  package adapter, or compiler change.

### PT40: additive causal attention mask

- [x] Broadcast a `[2,2]` additive causal mask over `[2,2,2]` batched attention
  scores before the existing final-dimension softmax.
- [x] Require exact attention rows `[1.0,0.0]` and `[0.5,0.5]`, proving the
  first token cannot observe the future token while the second observes both.
- [x] Require exact context and attention-residual tensors, then complete the
  feed-forward, pooling, batched classification, prediction, and cross-entropy
  path.
- [x] Extend the ABI fixture with one reviewed causal-mask marker derived from
  copied tensor contents so its representative softmax distinguishes masked
  from uniform attention; retain real full-tensor validation against LibTorch.
- [x] Execute package-owned and immutable consumers against fixture and pinned
  LibTorch; stably reject a `[3,3]` mask applied to `[2,2,2]` scores and add no
  public function, native symbol, package adapter, or compiler change.

### PT41: batched encoder-decoder cross-attention

- [x] Attend a caller-owned `[2,2,2]` decoder state to a separately owned
  `[2,3,2]` memory tensor, proving query and memory sequence lengths need not
  match.
- [x] Normalize decoder and memory independently, project Q from the decoder
  and K/V from memory, transpose the memory token dimension, and require exact
  uniform attention at `[2,2,3]`.
- [x] Require the complete asymmetric context, attention residual, feed-forward
  output, pooled features, probabilities, predictions, and cross-entropy path
  through package-owned and immutable consumers.
- [x] Stably reject decoder and memory tensors with incompatible leading batch
  dimensions before a context handle is returned.
- [x] Dispose both source tensors and every intermediate independently; add no
  public function, fixture behavior, native symbol, package adapter, or
  compiler change.

### PT42: complete causal cross-attention decoder layer

- [x] Compose token embedding, additive causal self-attention, separately owned
  encoder-memory cross-attention, and the existing normalized feed-forward path
  as one batched decoder layer.
- [x] Require the exact `[2,2,2]` causal self-attention residual and subsequent
  cross-attention residual against `[2,3,2]` memory before feed-forward.
- [x] Require the complete decoder output and pooled features within `1e-7`
  MSE, then produce uniform probabilities, predictions `[0,0]`, and mean
  cross-entropy in `0.69–0.70`.
- [x] Stably reject a width-`3` memory context joined to width-`2` decoder state
  before a new residual handle is returned.
- [x] Execute package-owned and immutable consumers against fixture and pinned
  LibTorch; dispose every source and intermediate and add no public function,
  fixture behavior, native symbol, package adapter, or compiler change.

### PT43: tokenized source-target sequence model

- [x] Build `[2,3,2]` encoder memory from source token IDs and a source-owned
  embedding table rather than caller-supplied float memory.
- [x] Build `[2,2,2]` decoder state from independent target token IDs and a
  distinct target-owned embedding table, requiring exact values for both token
  streams.
- [x] Execute the complete causal self-attention, memory cross-attention,
  feed-forward, pooling, classification, prediction, and cross-entropy graph.
- [x] Stably reject source and target streams with incompatible batch sizes
  before cross-attention scores are returned.
- [x] Execute fixture and pinned immutable consumers, dispose all source/target
  IDs, tables, tensors, and intermediates, and retain 39 public functions with
  no fixture, adapter, or compiler change.

PT43 closes the planned broad CPU composition series. Subsequent work targets
the concrete deferred value, interchange, inference, diagnostics, and optional
CUDA capabilities below rather than adding more synthetic model graphs.

## Serialized model inference track

### MI0: stateless TorchScript load and forward

- [x] Extend the generic C ABI package generator to compile declared C and C++
  adapter sources separately and map borrowed Terlan `String` arguments to
  call-scoped `const char *` values without a PyTorch compiler special case.
- [x] Add package-owned `run_torchscript(input, path)` for the constrained
  single-tensor-input, single-tensor-output `forward` surface.
- [x] Generate a real serialized TorchScript model from the pinned LibTorch C++
  API, load it through `torch::jit::load`, and require exact `3.0 -> 6.0`
  inference while the borrowed input remains unchanged.
- [x] Contain every C++ exception inside the adapter, return a separately owned
  tensor handle, and stably distinguish missing model status `7041` from corrupt
  archive status `7042`.
- [x] Execute package-owned and fresh immutable Git consumers against both the
  deterministic fixture and pinned LibTorch; retain all 43 broad-composition
  tests and increase the generated public surface from 39 to 40 functions.

MI0 deliberately reloads the archive for each convenience call. MI1 below adds
the reusable form after closing generic multiple-resource ownership.

### MI1: reusable TorchScript model resource

- [x] Extend the generic C ABI generator from one assumed opaque resource to
  any manifest-declared set, with one Rust owner and destructor per type.
- [x] Store native resources in a generated tagged handle enum and validate
  both wire identity and stored type before every borrow, disposal, or result
  insertion.
- [x] Add the thread-confined, package-owned `TorchScriptModel` resource with
  `load_torchscript(path)`, `torchscript_forward(model, input)`, and
  `dispose_model(model)` while retaining the stateless MI0 convenience API.
- [x] Load the pinned archive once, execute independent `3.0 -> 6.0` and
  `4.0 -> 8.0` forwards, preserve both inputs, independently dispose both
  outputs, then dispose the model exactly once.
- [x] Reject a disposed model through the generic `stale_handle` diagnostic;
  continue to contain load and forward exceptions behind stable C statuses.
- [x] Exercise the 43-function surface with 45 package tests and extend the
  immutable Git consumer and execution report with reusable-model gates.

MI1 establishes multiple independently owned native resource classes and a
cross-resource call (`TorchScriptModel` borrowing `Tensor`) without adding a
PyTorch rule to the compiler. Structured multi-input/output model values remain
a later value-model slice.

### MI2: two positional tensor inputs

- [x] Add `torchscript_forward2(model, first, second)` as an explicit
  two-positional-tensor forward without introducing a compiler special case.
- [x] Borrow the model and both independently owned tensor inputs for one C++
  call, preserve both inputs, and return one separately owned tensor.
- [x] Generate a pinned order-sensitive TorchScript model implementing
  `left - right * 2.0` and require exact `3.0, 4.0 -> -5.0` semantics.
- [x] Reject invoking a single-input model through the two-input surface with
  stable forward status `7043`, containing the TorchScript schema exception.
- [x] Execute the 44-function surface with 46 package tests and add immutable
  Git-consumer and package-report gates for ordered inputs and arity rejection.

MI2 proves positional multi-input forwarding with the existing generic
multi-borrow machinery. It intentionally does not represent `List[Tensor]` as
an ordinary Terlan list: list ownership for opaque NativeBoundary handles must
be designed generically before variable-arity model calls are exposed.

### MI3: owned two-tensor model output

- [x] Add a third opaque resource, `TensorPair`, that owns exactly two tensor
  results returned inside one TorchScript tuple.
- [x] Add `torchscript_forward_pair(model, input)`, validate tuple shape and
  tensor element types inside C++, and contain all exceptions behind C status.
- [x] Extract each element through `tensor_pair_first` and
  `tensor_pair_second` as a new independently owned `Tensor` handle.
- [x] Dispose the `TensorPair` before reading either extracted tensor, proving
  refcounted result ownership no longer depends on the tuple container.
- [x] Run a pinned `(input * 2.0, input * 3.0)` model at input `2.0`, requiring
  exact outputs `4.0` and `6.0` with the input unchanged.
- [x] Reject a plain tensor result as stable status `7046`; reserve separate
  statuses for wrong tuple arity (`7047`) and non-tensor elements (`7048`).
- [x] Exercise the 48-function surface with 47 package tests and extend the
  immutable consumer and report with tensor-pair execution/rejection gates.

`TensorPair` is a deliberate ownership bridge, not a language tuple. Generic
NativeBoundary tuple/record lowering must eventually replace fixed pair
containers before arbitrary structured model outputs become public.

### MI4: copied TorchScript float output

- [x] Add `torchscript_forward_float(model, input): Float` using the generic
  C ABI `double` result path rather than allocating another native handle.
- [x] Require the C++ adapter to accept only an `IValue` double and copy it
  before the scoped forward result is destroyed.
- [x] Run a pinned `float(input.item()) * 2.0` model at input `3.0`, requiring
  exact Terlan result `6.0` and unchanged readable input.
- [x] Reject tensor and other non-float results as stable status `7049`.
- [x] Restore and test the generic generated-Cargo `[workspace]` opt-out so
  package adapters remain independent of enclosing workspaces.
- [x] Exercise the 49-function surface with 48 package tests and add immutable
  consumer/report gates for float execution and output-type rejection.

MI4 is the first non-handle model result and follows ordinary Python scalar
return behavior. Integer, boolean, string, optional, and nested mixed outputs
remain separate reviewed slices.

### MI5: copied TorchScript integer output

- [x] Add `torchscript_forward_int(model, input): Int` through the existing
  generic fallible C ABI `int64_t` output path.
- [x] Require the C++ adapter to accept only an `IValue` integer and copy it
  before the scoped forward result is destroyed.
- [x] Run a pinned `int(input.item()) + 3` model at integer input `4`, requiring
  exact Terlan result `7` and unchanged readable input.
- [x] Reject tensor and other non-integer results as stable status `7050`.
- [x] Exercise the 50-function surface with 49 package tests and add immutable
  consumer/report gates for integer execution and output-type rejection.

MI5 extends copied scalar model results without another opaque resource or any
PyTorch-specific compiler logic. TorchScript booleans remain a distinct slice
because C/C++ `bool`, Rust `bool`, and Terlan `Bool` require their own reviewed
ABI proof; strings and optional values additionally require ownership design.

### MI6: copied TorchScript boolean output

- [x] Add `torchscript_forward_bool(model, input): Bool` through the existing
  generic fallible C ABI `bool` output path.
- [x] Require the C++ adapter to accept only an `IValue` boolean and copy it
  before the scoped forward result is destroyed.
- [x] Run a pinned `input.item() > 0` model at integer input `1`, requiring an
  exact true Terlan result and unchanged readable input.
- [x] Reject tensor and other non-boolean results as stable status `7051`.
- [x] Exercise the 51-function surface with 50 package tests and add immutable
  consumer/report gates for boolean execution and output-type rejection.

MI6 completes direct copied primitive model outputs for Terlan `Float`, `Int`,
and `Bool`. String results require a separately owned copied-buffer design;
optional results require an explicit generic presence/value representation.

### MI7: owned copied TorchScript string output

- [x] Extend the generic Terlan C ABI metadata with an owned UTF-8 output
  contract naming its byte-length parameter, immediate-copy policy, and
  consuming destructor.
- [x] Add `torchscript_forward_string(model, input): String` without exposing a
  native pointer or lifetime through the public Terlan surface.
- [x] Require the C++ adapter to allocate exact result bytes, require the
  generated Rust wrapper to copy them before calling the named destructor, and
  reject invalid UTF-8 deterministically.
- [x] Run a pinned conditional-string model at integer input `1`, requiring
  exact output `"positive"` and unchanged readable input.
- [x] Reject tensor and other non-string results as stable status `7052`.
- [x] Exercise the 52-function surface with 51 package tests and add immutable
  consumer/report gates for string execution and output-type rejection.

MI7 closes the first non-scalar copied output across the generic C ABI. Empty
strings use a null pointer with zero length; non-empty values are copied before
destruction, and neither the VM nor generated package retains native storage.
Optional, list, dictionary, and nested mixed outputs remain distinct value-model
slices.

### MI8: owned copied TorchScript integer-list output

- [x] Extend generic C ABI metadata with an owned integer-array output contract
  naming its element-count parameter, immediate-copy policy, and consuming
  destructor.
- [x] Add `torchscript_forward_int_list(model, input): List[Int]` without
  exposing native array storage through Terlan.
- [x] Copy all `int64_t` elements before invoking the named destructor; represent
  an empty result as an ordinary empty Terlan list.
- [x] Run a pinned `[value, value + 1, value + 2]` model at input `4`, requiring
  exact output `[4,5,6]` and unchanged readable input.
- [x] Reject tensor and other non-integer-list results as stable status `7053`.
- [x] Exercise the 53-function surface with 52 package tests and immutable
  consumer/report gates for list execution and output-type rejection.

MI8 reuses the existing `List[Int]` VM value and `ok_ints` wire protocol. The
new work is entirely at the generic owned C-array boundary and in the external
PyTorch package. Optional outputs remain deferred until Terlan has an explicit
generic presence/value representation.

### MI9: owned copied TorchScript floating-point-list output

- [x] Generalize the generic owned-array contract with an explicit element
  kind while preserving `int64` as the backward-compatible default.
- [x] Add `torchscript_forward_float_list(model, input): List[Float]` without
  exposing native array storage through Terlan.
- [x] Require matching `double **` producer and `double *` destructor shapes,
  copy every element before destruction, and use the existing `ok_floats`
  NativeBoundary protocol.
- [x] Run a pinned `[value, value + 0.5, value + 1.0]` model at input `1.5`,
  requiring exact output `[1.5,2.0,2.5]` and unchanged readable input.
- [x] Reject tensor and other non-floating-point-list results as stable status
  `7054`.
- [x] Exercise the 54-function surface with 53 package tests and immutable
  consumer/report gates for floating-point-list execution and type rejection.

MI9 proves that owned arrays are one generic element-typed compiler mechanism,
not an integer-only or PyTorch-specific branch. Optional, dictionary, and
nested mixed outputs remain separate value-model slices.

### MI10: validated copied TorchScript boolean-list output

- [x] Add a generic `bool8` owned-array element using portable `uint8_t`
  storage instead of relying on C++ `bool` layout or `vector<bool>` storage.
- [x] Validate every copied byte as exactly `0` or `1` before constructing an
  ordinary Terlan `List[Bool]`.
- [x] Add the generic `ok_bools` NativeBoundary reply with populated, empty,
  and malformed-reply VM coverage.
- [x] Add `torchscript_forward_bool_list(model, input): List[Bool]` and copy a
  typed LibTorch BoolList through the reviewed byte encoding.
- [x] Run a pinned sign-classification model at input `2`, requiring exact
  output `[true,false,false]` and unchanged readable input.
- [x] Reject tensor and other non-boolean-list results as stable status `7055`.
- [x] Exercise the 55-function surface with 54 package tests and immutable
  consumer/report gates for boolean-list execution and type rejection.

MI10 closes homogeneous scalar-list outputs for `Int`, `Float`, and `Bool`.
Optional, string-list, dictionary, and nested mixed outputs remain separate
value-model slices.

### MI11: copied TorchScript string-list output

- [x] Add a generic owned string-array contract with `char ***` values, owned
  `size_t **` byte lengths, a `size_t *` count, immediate UTF-8 copying, and one
  named destructor for all native allocations.
- [x] Generate `Vec<String>` and the existing `ok_strings` NativeBoundary reply
  without using NUL termination; preserve empty string elements exactly.
- [x] Add `torchscript_forward_string_list(model, input): List[String]` over a
  validated generic LibTorch IValue list whose every element is a string.
- [x] Run a pinned conditional model at input `1`, requiring exact output
  `["positive","","ready"]` and unchanged readable input.
- [x] Reject tensor and other non-string-list results as stable status `7056`.
- [x] Exercise the 56-function surface with 55 package tests and immutable
  consumer/report gates for string-list execution and type rejection.

MI11 closes homogeneous scalar-list outputs for `Int`, `Float`, `Bool`, and
`String`. Optional, dictionary, and nested mixed outputs remain separate
value-model slices.

## LibTorch CUDA track

This track covers tensors placed on CUDA through a CUDA-enabled LibTorch
distribution. It is distinct from `terlan-cuda`, which may target the CUDA
driver/runtime through maintained Rust bindings such as `cudarc` or `cust`.
Neither package may place a CUDA dependency in the Terlan compiler or exchange
raw device pointers through Terlan values.

### CUDA0: capability model

- [ ] Replace the single CUDA availability bit with independent observations:
  NVIDIA driver/device, CUDA toolkit/compiler, and CUDA-enabled LibTorch.
- [ ] Do not require `nvcc` for a prebuilt LibTorch-only execution path; require
  it only when compiling package-owned CUDA source.
- [ ] Report stable states such as `driver-unavailable`, `device-unavailable`,
  `toolkit-unavailable`, `libtorch-cuda-unavailable`, and `available`.
- [ ] Keep the default CPU gate green for every unavailable state.

### CUDA1: synchronous device surface

- [ ] Pin and verify a CUDA-enabled LibTorch distribution separately from the
  CPU archive.
- [ ] Expose device availability/count and validated device selection through
  package-owned Terlan APIs.
- [ ] Add the StableIValue forms required for device placement and synchronous
  copy flags without exposing CUDA pointers.
- [ ] Move a tensor CPU → CUDA → CPU and verify value, dtype, shape, device type,
  and device index at every observable boundary.
- [ ] Synchronize explicitly before host readback.

### CUDA2: CPU/CUDA operation parity

- [ ] Run the PT4 `matmul` fixture on CUDA using the same public Tensor API.
- [ ] Compare exact CPU and CUDA output values and shapes.
- [ ] Prove deterministic destruction of source, device copy, result, and host
  readback handles.
- [ ] Run this path from opt-in `make cuda-package-check`; never add it to the
  default CPU release requirement. The final gate name should distinguish this
  LibTorch path, for example `terlan-pytorch-cuda-check`, from the direct
  `terlan-cuda` package gate.

### CUDA3: adversarial runtime and cleanup

- [ ] Cover invalid device indices, missing driver/runtime libraries,
  unsupported capability, incompatible CUDA LibTorch, allocation failure,
  stale handles, and device mismatch.
- [ ] Preserve stable public errors without exposing thread-local or upstream
  exception payloads.
- [ ] Produce a machine-readable CUDA execution report containing detection,
  distribution, device, operation, cleanup, and skip/fail evidence.

### CUDA4: asynchronous streams

- [ ] Introduce opaque package-owned stream resources only after synchronous
  placement and operation parity are green.
- [ ] Define stream ownership, current-stream interaction, synchronization,
  cancellation, and process cleanup before enabling non-blocking copies.
- [ ] Reject cross-owner streams, stale streams, and unsynchronized host
  readback deterministically.

## Later value and tensor surfaces

- [ ] Boolean, floating-point, optional, list, tuple, and multiple-result
  `StableIValue` forms.
- [x] General owned float64 tensor construction from shaped Terlan data without
  leaking caller buffers or deleter callbacks.
- [ ] Consume `terlan-ndarray` arrays through the reviewed versioned DLPack
  handoff as `Tensor.from_array`, preserving supported CPU dtype and shape.
  This package must not depend on Polars; `DataFrame.to_torch` remains a
  Polars-owned convenience surface.
- [ ] Broader arithmetic, reductions, dtype conversion, structured
  multi-input/output inference, and additional serialization formats.
- [ ] Copied native error detail under a reviewed lifetime and privacy policy.

These later surfaces must reuse the same metadata and NativeBoundary machinery;
neither CUDA nor PyTorch may introduce handwritten compiler special cases.
