# Terlan LibTorch roadmap

This roadmap owns the PyTorch/LibTorch package surface. Generic C++/CXX and
NativeBoundary generation remain in the Terlan compiler; PyTorch declarations,
operator schemas, distribution pins, and execution tests remain in this
repository.

The target package binds released LibTorch C++ binaries through generated
`cxx` bridges. The former stable C/AOTI wrapper sources are removed; a frozen
manifest and isolated fixture remain only as migration and compatibility-test
inputs, not as the package architecture.
PyTorch is also the real-package proof for Terlan's generic C++ generator.

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

## Active behavioral parity program

The completed PT, MI, CUDA, and interoperability tracks prove that the package
architecture works. They do not establish PyTorch API parity. The generated
package currently contains 2,829 CXX-generated package function declarations
across 121 modules. The frozen ATen inventory records 2,127 implemented
schemas, twelve explicit equivalents, 122 non-applicable schemas, and zero
missing schemas. The public callable inventory records 1,391 implemented, one
equivalent, 2,033 non-applicable, and zero missing. The latest completion adds retained tensor-list output
compositions for dequantize, quantize, split, unbind, and unsafe-split
families, plus a CPU-validated depthwise-3D convolution composition over the
generated CXX convolution surface. It also adds seeded CPU `Generator` ownership
and a generated CXX `record_stream` wrapper that preserves LibTorch's external
CUDA stream identity for allocator lifetime tracking, with a native integration
fixture on the CUDA gate. The public alias slice now also covers top-level
`torch.clamp_`, `torch.rsub`, and sparse `torch.saddmm` with executable CPU
parity evidence. The Tensor in-place comparison aliases (`eq_`, `ne_`, `lt_`,
`le_`, `gt_`, `ge_`, and their named variants) now reuse the generated scalar
comparison mutation kernels with dedicated CPU parity evidence.
The CXX generation path now preserves generated owned/mutation adapter leaves;
the Terlan post-generation patch only applies the small sparse compatibility
substitutions required by the pinned LibTorch headers.
Package execution reports now hash the active CXX manifest and normalized
generated manifest, with the verifier rejecting stale provenance.
The CXX check runner now executes the generated test tree directly instead of
compiling one monolithic aggregate source, keeping fixture coverage while
reducing compiler load on the full surface.
The optimized CPU tree has now executed all 84 generated CXX/runtime fixtures
successfully against the regenerated helper and pinned LibTorch distribution.
The deterministic-algorithm configuration family is now exposed through exact
LibTorch global-context CXX wrappers with executable CPU state-transition
coverage, and the float32 matmul precision query/setter is available through
the same bridge. Autograd gradient-mode and anomaly-mode global state is
likewise exposed through generated CXX wrappers with executable CPU
state-transition coverage. The warning-mode global query/setter is also bound
through the same bridge with executable CPU evidence. CPU autocast enablement,
cache state, and dtype query/setter operations now use the same exact
global-context CXX bridge, with executable CPU state-transition coverage and
restoration of the caller's original state. The process-wide default dtype is
also exposed through the exported `c10` default-dtype CXX state with an
executable CPU round-trip fixture. Inference-mode state is now queried through
the exact CXX TLS state, with an RAII transition probe and restoration check;
the existing Vulkan and CUDNN availability predicates are also classified as
implemented public callables.
The global CPU random-generator seed query and manual-seed operation now use
the existing owned `Generator` CXX resource with executable CPU evidence. The
implementation-generated `torch.seed()` operation now shares this path. CPU
RNG-state serialization and restoration are also available through the Tensor
CXX ownership path with a native round-trip fixture. CPU thread and inter-op
thread initialization, queries, and setters are covered by the same bridge.
The public flush-denormal toggle now has an ownership-neutral generated CXX
wrapper over the CPU MXCSR controls (the bundled LibTorch header declares this
API but its binary does not export it), with an executable CPU enable/restore
check. Because the operation has an exact supported CPU contract, the generated
wrapper is counted as an implemented public equivalent rather than excluded.
The same rule now covers `hsplit`, `vsplit`, and `dsplit`: each is generated as
a dimensioned composition over the existing `tensor_split` CXX leaves and is
validated by the aggregate behavior gate.
and generator-backed `rand`, `randn`, `randperm`, `rand_like`, `randn_like`,
`randint`, and scalar/low-range `randint_like` execution (including `.out`
variants), plus LU-unpacking default and `.out`
variants, six `randint_like` high/low/tensor/output forms, `to.other`, and
`to.device` device/dtype conversion, full `to.dtype_layout` optional-property
conversion, nested tensor construction, and functional `to_padded_tensor`
conversion, file-backed `from_file` construction and output, quantized
`qscheme` metadata, and the
all `gradient` tensor-list, scalar-spacing, optional-dimension, and
tensor-spacing forms, with executable tuple, mutable-output, dtype, shape,
CPU-device, quantization-scheme, and gradient-value validation against
LibTorch.
Only exact executable classifications count toward parity.

The packed-sequence recurrent slice now has direct generated CXX wrappers for
`gru.data`, `lstm.data`, `rnn_tanh.data`, and `rnn_relu.data`. A minimal packed
CPU case (`batch_sizes = [2, 1]`) validates the frozen LibTorch dispatchers;
the wrappers preserve tensor-list ownership through the generated `cxx`
boundary.

The quantized CPU slice now has direct generated CXX wrappers for
`quantized_max_pool1d/2d/3d` functional and retained-output forms plus
`quantized_batch_norm` functional and retained-output forms. The generated
Rust/CXX integration test validates the quantized tensor shapes and
mutable-output contracts against the pinned LibTorch CPU dispatchers.

The latest indexed CXX slices add exact covariance, CTC-loss, and default
`rand`/`randn`/`randint`, like-variants, bounded and in-place `random` forms,
RReLU-with-noise mutation, STFT/ISTFT spectral forms, segment reduction,
vector upsampling overloads, sparse matrix multiplication, scaled dot-product
attention, legacy SVD forms, quantized allocation,
legacy `Tensor.is_shared` storage metadata and `Tensor.share_memory_` shared-memory mutation/return-self behavior, fixed-size `nonzero_static` coordinate and retained-output forms, nested `to_padded_tensor.out` retained-output composition over the executable functional kernel, `Tensor.module_load` copy/detach compatibility, `Tensor.unique` flatten/dimension compatibility plus unique/consecutive-unique tuple/output forms, fake-quantization cachemask
outputs, histogram tuple/output forms with range/weight forwarding,
tensor-source `set_` mutations,
`torch.nn.functional.local_response_norm`, weighted/legacy-reduction
`torch.nn.functional.multilabel_soft_margin_loss` through generated CXX
surfaces, `torch.nn.functional.lp_pool1d` including ±infinity norm semantics,
and `torch.nn.functional.max_unpool1d` with explicit stride, padding, and
target-output-size support, plus `torch.nn.functional.lp_pool2d` and
`torch.nn.functional.lp_pool3d` across norm, stride, and ceil-mode forms.
It also adds `torch.nn.functional.gumbel_softmax`, preserving the legacy
`eps` call parameter while delegating soft and straight-through hard behavior
to LibTorch's C++ functional. It also adds Gaussian negative log-likelihood
loss with tensor/scalar variance, broadcasting, stability epsilon, full
constant, and reduction semantics.
It also adds `torch.nn.functional.interpolate` with generated CXX option
forwarding for size/scale-factor forms, interpolation modes, corner alignment,
recomputed scales, and antialiasing, with CPU and CUDA shape validation.
The legacy `upsample`, `upsample_bilinear`, and `upsample_nearest` public
aliases now route through the same generated interpolation surface.
The top-level and functional `conv1d`/`conv3d` aliases now point to the
existing generated convolution leaves with CPU/CUDA parity evidence.
Squeeze and view-copy public aliases now point to the generated view and
mutation leaves, preserving the tested alias/copy contracts.
Tensor partition aliases (`hsplit`, `vsplit`, `dsplit`, and `tensor_split`)
and `transpose_copy` now use their generated shape/partition leaves.
Arithmetic, comparison, logical, extrema, and integer-shift public aliases
now point at the tested generated Binary/Division/Tensor leaves.
Top-level aliases for logit, msort, quantization zero-point, masked select,
put/index-put, and sort now reuse their tested generated counterparts.
Top-level nan-statistics and mean aliases now reuse the generated reduction
leaves with dedicated parity evidence.
Product, standard-deviation, median, and NaN-median aliases now reuse the
tested reduction and special-function leaves.
CPU evidence validates weighted/ranged histogram tuple and output forms; the
pinned CUDA build has no histogram backend registrations, so those tests are
reported as CPU-only skips. CPU and CUDA runtime evidence validates local
response normalization across rank-3/rank-4 inputs, defaults, and empty-input
behavior, multilabel soft-margin loss across weight and reduction modes, and
one-dimensional Lp pooling across default/explicit stride, ceil mode, and
positive/negative infinite norm types, and max-unpool scatter behavior across
default stride, explicit padding, and target output sizes,
alongside covariance, both CTC-length forms, random-family
shapes/dtypes, RReLU execution, and LU-unpacking tuple/output forms.

The strict `pytorch-feature-parity-check` Make target now composes the frozen
ATen/public zero-missing inventory checks, Terlan format/grouped-binding/
reference/script policy checks, and the regenerated CPU and CUDA CXX behavior
gates. This is the authoritative closure command for the supported inventory.

The triangular-index slice now covers `tril_indices` and `triu_indices`,
including retained `.out` storage and configured dtype paths, through generated
CXX leaves with a CPU runtime parity test.

The `Tensor.is_pinned` slice now covers default and explicit-device CPU queries
through generated CXX wrappers with a passing LibTorch runtime parity test.

The resize slice now covers functional, retained-output, and in-place forms,
including optional memory-format handling, through generated CXX wrappers.

The explicit-stride allocation slice now covers functional and retained-output
`empty_strided` forms through generated CXX wrappers with passing CPU runtime
parity evidence.

The physical-permutation allocation slice adds functional and retained-output
`empty_permuted` forms through the same generated CXX bridge.

The multilabel-loss slice adds functional and retained-output
`multilabel_margin_loss` through a dedicated generated CXX completion module,
with passing CPU runtime parity evidence.

The quantization slice adds generated CXX completion wrappers for
`fake_quantize_per_tensor_affine`, `fake_quantize_per_channel_affine`, and
`quantize_per_tensor`, `dequantize`, and dynamic per-tensor quantization, with
independent CPU value, shape, dtype, and quantization-state evidence for the
supported forms.

The hashing slice adds functional and retained-output `hash_tensor` wrappers
through generated CXX leaves with exact CPU value evidence.

The resize-as slice adds functional, retained-output, and in-place
`resize_as` wrappers through generated CXX leaves with passing CPU value and
shape evidence.

The slice-inverse slice adds generated CXX leaves for full, start-only,
end-only, and bounded `slice_inverse` forms, preserving LibTorch mutation and
aliasing behavior through typed optional-bound composition.

The new-empty-strided slice adds a retained-output CXX leaf that preserves
the input options while materializing requested sizes and strides.

The CUDA stream slice now classifies `torch.cuda.Stream.synchronize` against
the owned `pytorch.Cuda.synchronize` handle operation, with package-level CUDA
runtime coverage in `CudaTest.terl`.

The sparse reduction slice now classifies `torch.sparse.sum` through a
generated-package composition that preserves optional dimensions and dtype,
with an isolated COO runtime fixture covering all-axis, selected-axis, and
typed reductions. The public sparse matrix slice also classifies
`torch.sparse.addmm` against the generated fused-`addmm` path, with a sparse
operand runtime fixture covering the dense result values. The legacy public
`torch.spmm` and `torch.dsmm` aliases are classified against the generated
`mm` sparse-matrix path and share sparse-product value coverage in that fixture.
Top-level `torch.xlogy_`, `torch.take_along_dim`, and `torch.index_put_` are
also now classified against their existing generated mutation/gather paths,
with the corresponding Xlogy, alias, and index runtime fixtures.
The top-level `torch.round_` and `torch.t_copy` aliases are likewise mapped to
the generated unary-storage and view-copy targets with existing runtime tests.
`torch.linalg.householder_product` is now classified against the generated
dense linalg target covered by the dense parity fixture.
The top-level `torch.alias_copy`, `torch.detach_copy`, and `torch.resize_as_`
aliases now use their generated view and resize mutation targets with existing
runtime fixtures.
The scalar-bound `torch.clamp_min_` and `torch.clamp_max_` aliases now use the
generated in-place clamp leaves covered by `ClampParityTest.terl`.
Tensor in-place random-distribution methods (`bernoulli_`, `uniform_`,
`exponential_`, `cauchy_`, `geometric_`, `log_normal_`, and `normal_`) are now
classified against their generated distribution targets with dedicated CPU
fixtures.
The Tensor-level `index_put_` mutation alias now shares the generated indexing
target and optional-index runtime fixture.
The deprecated `torch.Tensor.ndimension` synonym now shares the generated
`Tensor.dim` target and boundary fixture.
The Tensor-method `take_along_dim` alias now shares the generated gather target
and alias parity fixture.
The package-owned `MutationCompletion` module is now part of the generated
manifest, and `torch.Tensor.renorm_` is classified against its retained-handle
composition with mutation parity coverage. The deprecated
`torch.Tensor.solve(self, other)` method now has an explicit receiver-ordering
composition over `linalg.solve`, with CPU parity evidence against both the
generated reference and an independent numerical solution oracle.
The deprecated Tensor `symeig` export now has a fixed-arity wrapper preserving
its upper-triangle default and empty-vector branch, with dedicated CPU parity
evidence. The remaining deprecated Tensor `eig`, `lu`, and `lstsq` exports stay
explicitly non-applicable until wrappers can preserve their legacy return
contracts without narrowing behavior.

The RMS-normalization slice adds generated CXX leaves for default, weighted,
explicit-epsilon, and weighted-epsilon `rms_norm` forms.

The NLL-loss slice adds generated CXX functional and retained-output leaves
with absent and present class weights, validated against scalar CPU oracles.

The 2-D NLL-loss slice adds generated CXX functional and retained-output
leaves with absent and present class weights, validated against exact 4-D CPU
oracles.

The dtype view-copy slice adds generated functional and retained-output
`view_copy.dtype` leaves, preserving LibTorch's storage reinterpretation and
pre-typed output contract with passing CPU runtime evidence.

The random-permutation slice adds generated functional and retained-output
`randperm` leaves with pinned Long defaults and passing CPU structural runtime
evidence.

The instance-normalization slice adds a generated CXX functional
`instance_norm` leaf with the optional parameter set composed as nullopt,
plus CPU shape/dtype runtime evidence.

The multi-margin-loss slice adds generated CXX functional and retained-output
leaves with absent and present class weights, validated against scalar CPU
oracles.

The bilinear-upsampling slice adds generated functional and retained-output
`upsample_bilinear2d` leaves with default scale factors.

The bicubic-upsampling slice adds generated functional and retained-output
`upsample_bicubic2d` leaves with default scale factors and passing CPU
shape/dtype runtime evidence.

The linear-upsampling slice adds generated functional and retained-output
`upsample_linear1d` leaves with default scale factors.

The nearest-upsampling slice adds generated functional and retained-output
`upsample_nearest1d` leaves with default scale factors and passing CPU
shape/dtype runtime evidence.

The 2-D/3-D nearest-upsampling slice adds generated functional and
retained-output `upsample_nearest2d` and `upsample_nearest3d` leaves with
passing combined CPU shape/dtype runtime evidence.

The tensor-data mutation slice adds a generated CXX `set_data` leaf with
passing CPU equality evidence for replacement semantics.

The upsampling-backward slice adds generated functional and retained-output
leaves for nearest 1-D/2-D/3-D and linear 1-D gradients, with combined CPU
shape/dtype runtime evidence.

The trilinear 3-D slice adds generated functional and retained-output forward
and backward leaves with passing combined CPU shape/dtype runtime evidence.

The 2-D interpolation-backward slice adds generated bilinear and bicubic
functional and retained-output backward leaves with passing combined CPU
shape/dtype runtime evidence.

The dense dtype slice now covers `bmm.dtype*` and `mm.dtype*` through typed
output-conversion wrappers, and the statistical slice adds retained-output
`norm.Scalar_out` dispatch with integer/float Scalar coverage. Recurrent
dispatcher-only schemas are explicitly classified as non-applicable. Stable
sort and unsafe partition aliases now route through the generated Selection
and ShapePartition CXX leaves with dedicated Terlan-facing wrappers. Scalar
threshold and renorm in-place forms now preserve retained-handle mutation
through typed copy-back wrappers.
The indexed-selection slice also covers `take_along_dim*` through the typed
gather leaves, including retained output storage.
Scalar-tensor construction now preserves integer versus floating dtype
inference through a typed wrapper as well.
Inherited `Tensor.new_empty/new_full/new_ones/new_zeros` functional and
retained-output forms now use the typed creation leaves.
Copy-preserving `split_copy`, `split_with_sizes_copy`, and `unbind_copy`
wrappers now clone each partition result before returning the tensor list.
The legacy `vander` schema now composes the generated linalg Vandermonde
leaves and supports explicit width plus increasing-column output.
The scalar-tensor slice also covers retained `scalar_tensor.out` storage.
The metadata slice now covers `is_contiguous.memory_format` through a public
Terlan adapter over the generated CXX memory-format leaf, plus dimensioned
`norm` dtype and retained-output overloads through generated CXX compositions.
The tensor-list slice now also exposes `pad_sequence` through an indexed
LibTorch CXX leaf and a passing CPU runtime parity test.
The NaN-statistics slice adds generated CXX compositions for `nanmean` and
`nansum`, including dimensioned, dtype-controlled, and retained-output paths,
with passing CPU runtime evidence.

The broader pinned-runtime inventory contains 3,425 exported callables across
`torch`, `Tensor`, neural-network, autograd, optimizer, numerical, probability,
sparse, quantization, AMP, CUDA, distributed, export, serialization, and data
namespaces. Its policy currently classifies 1,374 callables from executable
evidence plus one equivalent storage-handle contract, 2,050
unsupported Python/frontend exports as non-applicable, and leaves zero
callable signatures missing. The latest completion adds compositional
`softmin`, `softsign`, `tanhshrink`, `grid_sample`, and `normalize` operators
to the generated CXX surface, with real LibTorch CPU runtime evidence. The
CUDA surface now also includes a generated `synchronize_device` composition
that owns and releases its stream handle. The typed CXX verifier aggregates
the generated behavior fixtures into one test module and enables incremental
compilation, while retaining a one-hour child-process bound for the pinned
compiler's full-surface native build.
the composite `pytorch-parity-check` requires both this inventory and ATen to
reach zero missing entries.

The primary package now uses the generated Rust/CXX helper. The generated CXX
contract snapshot is the sole migration baseline while PP1-PP7 expansion
continues; the historical C ABI manifest has been retired from the package and
build path.

Parity is measured against the public API of pinned PyTorch/LibTorch 2.13.0.
Every referenced callable must be classified exactly once as implemented,
provided by an equivalent Terlan API, non-applicable with a stable reason, or
missing. A callable is implemented only when executable tests prove its return
values, shape, dtype, device, mutation, ownership, error, and CPU/CUDA behavior
where the upstream operation supports both devices. Missing count zero is a
release gate; successful symbol lookup or linking is not implementation
evidence.

Python runtime mechanisms are not copied literally. Dynamic monkey patching,
Python descriptors, pickle execution, Python object identity, and CPython
extension hooks may be classified as non-applicable. Tensor, numerical,
autograd, module, optimizer, serialization, sparse, quantization, distributed
training, and device behavior may not be excluded merely because it is large or
difficult. Terlan-native APIs may replace Python syntax only when they preserve
the observable behavior and the mapping is recorded explicitly.

### PP0: frozen API inventory and executable classification

- [x] Freeze 2,261 exact non-internal ATen overload schemas from the pinned
  LibTorch headers, including the build identity and deterministic header-tree
  hash. The initial honest baseline is 17 implemented and 2,244 missing.
- [x] Freeze 3,425 public callables for `torch`, `Tensor`, `torch.nn`,
  `torch.nn.functional`, `torch.autograd`, `torch.optim`, `torch.linalg`,
  `torch.fft`, `torch.special`, `torch.distributions`, `torch.sparse`,
  `torch.ao`, `torch.amp`, `torch.cuda`, `torch.distributed`, `torch.export`,
  serialization, and data utilities at the pinned upstream revision. The
  reproducible upstream oracle is `scripts/freeze-pytorch-public-api.py`; its
  committed output is `tests/fixtures/pytorch-2.13-public-api.json`.
- [ ] Record source revision, LibTorch build hash, namespace, callable kind,
  overload family, device support, mutability, and reference documentation for
  every frozen entry.
- [x] Add a Terlan-owned ATen and broader public-callable inventory checker and
  machine-readable policies/reports, plus composite report and strict gates.
  Python may generate frozen oracle input; active package lifecycle, generated
  source ownership, and exception-boundary checks are driven by typed Terlan
  scripts. The migration-contract comparison remains a bounded differential
  verifier until its JSON contract reader is moved into the standard library.
- [x] Reject duplicate classification, stale aliases, unavailable Terlan
  targets, unknown exclusions, version drift, and unclassified callables.
- [x] Reconcile existing executable dispatcher surfaces with the frozen policy;
  exact clone, GELU, softmax, batch-normalization, layer-normalization, and
  cross-entropy schemas are classified against `DispatcherTest.terl` evidence.
- [x] Seed the policy from the existing 105-function surface without claiming
  unsupported overloads or composed-model examples as distinct PyTorch APIs.
- [x] Add `make pytorch-api-inventory-check`; strict mode must fail while any
  callable remains missing and report mode must remain usable during closure.
- [ ] Inventory this roadmap section after execution and update it if the
  frozen upstream surface exposes an unmodeled capability family.

### PP1: tensor value, construction, and metadata parity

- [x] Add nominal `DType` and `Device` values for typed CPU/CUDA construction;
  cover all ten admitted scalar dtypes without exposing native pointers.
- [x] Expand canonical `DType` to all 46 public scalar-type tags in pinned
  LibTorch and implement exact `allclose`, `equal`, `is_same_size`, `can_cast`,
  `promote_types`, and four `result_type` schemas. CPU and mandatory CUDA native
  evidence covers tolerance/NaN policy, exact versus shape-only comparison,
  safe-casting categories, promotion, every tensor/scalar operand order,
  invalid tags, null outputs, and caller-owned result initialization in
  `tests/native/type_relations.rs`; public overload evidence is in
  `tests/pytorch/TypeRelationsParityTest.terl` and `tests/pytorch/CudaTest.terl`.
  Individual construction and kernel families continue to reject scalar tags
  unsupported by their selected backend rather than narrowing the shared type.
- [x] Implement all 26 exact Bartlett, Blackman, Hamming, Hann, and Kaiser
  window-construction schemas, including periodic/default forms, custom
  coefficients, explicit dtype/device/pinned-memory policy, and retained
  outputs. Native CPU and mandatory CUDA evidence covers exact values, edge
  lengths, output reuse, malformed requests, and allocation failure in
  `tests/native/windows.rs`; public behavior is exercised by
  `tests/pytorch/WindowParityTest.terl` and `tests/pytorch/CudaTest.terl`.
- [x] Classify exact tensor rank, element-count, storage-offset, and default
  contiguity metadata after extending the mandatory CUDA round-trip evidence;
  CPU, zero-extent, non-contiguous, and CUDA behavior is covered by
  `ConstructionTest.terl`, `NonContiguousTest.terl`, and `CudaTest.terl`.
- [x] Execute the generated focused runtime fixtures with the native helper on
  both pinned backends: CPU `TensorTest.terl` and CUDA `CudaTest.terl` each
  report one passing test with zero failures.
- [x] Validate the generated package-level `NewEmptyStridedParityTest.terl`;
  both allocating and retained-output tests pass against the CPU CXX helper.
- [x] Implement typed `full`, `zeros`, `ones`, and uninitialized `empty`
  construction with shared shape, contiguous-stride, overflow, and device
  validation.
- [x] Expose shape strides, storage offset, and contiguity through immediate
  copies or scalar values from LibTorch's stable ABI.
- [x] Prove scalar, matrix, zero-extent, all-dtype, source-ownership,
  non-contiguous transpose, negative-dimension, shape-overflow, and invalid
  CUDA-ordinal behavior against the ABI fixture and pinned LibTorch.
- [x] Implement typed `arange`, `linspace`, `logspace`, and rectangular `eye`
  construction through package-owned LibTorch operators, including explicit
  dtype, device, and pinned-memory policy.
- [x] Prove ascending and descending exclusive ranges, inclusive linear
  endpoints, arbitrary logarithmic bases, every rectangular identity element,
  metadata ownership, zero-step rejection, negative-step-count rejection, and
  invalid identity dimensions.
- [x] Close dtype-inference, tensor-endpoint, `out`, layout, and remaining
  overload behavior before classifying the exact range and identity ATen
  schemas as implemented.
- [ ] Complete tensor construction from scalars, nested values, ranges,
  identities, uninitialized/zero/one/full buffers, and random distributions.
- [ ] Model all admitted scalar dtypes, complex values, layouts, strides,
  storage offsets, memory formats, devices, and gradient flags with typed
  Terlan values rather than integer constants in the public API.
- [ ] Complete clone/copy/detach/contiguous/pin/share conversions and exact
  CPU/CUDA transfer semantics without exposing native pointers.
- [ ] Prove empty, scalar, zero-extent, non-contiguous, overlapping, large,
  non-finite, complex, and unsupported-device behavior.
- [ ] Inventory this roadmap section after execution and update it if parity
  evidence reveals another foundational tensor representation.

### PP2: indexing, shape, and mutation parity

- [x] Implement all five exact optional-bound slice view, independent-copy,
  retained-copy, scatter, and retained-scatter schemas in `pytorch.Slice`.
  Native CPU and required-CUDA evidence exhausts absent/from/until/bounded,
  negative, clipped, empty, non-contiguous, alias-lifetime, error, and gradient
  behavior in `tests/native/slicing.rs`; direct public evidence executes every
  schema and optional-bound form on both devices in
  `tests/pytorch/SliceParityTest.terl`.
- [x] Implement all 21 exact block-diagonal, Cartesian-product,
  concatenation-alias, and row/column/depth stacking schemas in
  `pytorch.TensorJoining`. Native CPU and required-CUDA evidence covers every
  retained-output form, independent output storage, non-contiguous and empty
  inputs, invalid lists/dimensions/devices, and autograd in
  `tests/native/tensor_joining.rs`; every public operation executes on both
  devices in `tests/pytorch/TensorJoiningParityTest.terl`.
- [x] Implement 18 exact retained-output, narrowing, unfolding, trace-gradient,
  zeroing, image-column, and tensor-contraction schemas in
  `pytorch.ShapeCompletion`. Native CPU and required-CUDA tests prove values,
  non-contiguous inputs, view lifetime, output independence, malformed
  requests, gradients, and the pinned CPU-only behavior of `narrow_copy.out`
  in `tests/native/shape_completion.rs`; the shared public test executes on
  both devices in `tests/pytorch/ShapeCompletionParityTest.terl`.
- [x] Implement ten exact diagonal-view, diagonal-scatter output, and
  diagonal/select/slice/unfold backward schemas in `pytorch.ShapeGradient`.
  Explicit gradients, output independence, view lifetime, negative dimensions,
  and invalid-dimension requests execute on CPU and mandatory CUDA in
  `tests/native/shape_gradient.rs`; every public entry point executes in
  `tests/pytorch/ShapeGradientParityTest.terl`.
- [x] Implement eight exact affine-grid, affine-grid backward, and generic,
  planar, and volumetric sampling schemas in `pytorch.Spatial`. Native CPU and
  mandatory CUDA tests prove interpolation/padding modes, corner alignment,
  out-of-range coordinates, output reuse, invalid requests, and gradients in
  `tests/native/spatial.rs`; public values and output behavior execute in
  `tests/pytorch/SpatialParityTest.terl`.
- [x] Implement six exact strided view, copy, scatter, retained-output, and
  in-place schemas with both inherited and explicit storage offsets. Native
  CPU and required-CUDA tests prove alias lifetime, independent copy storage,
  metadata mutation, overlapping-read gradients, empty and scalar shapes, and
  invalid requests in `tests/native/strided.rs`; all twelve public entry points
  execute in `tests/pytorch/StridedParityTest.terl`.
- [x] Bind and execute 46 exact integer-axis and integer-list schemas in
  `pytorch.Shape`, including view/copy pairs, axis movement, selection,
  diagonals, channel and pixel transforms, matrix powers, padding, pooling,
  and target-shape reduction. Real LibTorch evidence is in
  `tests/pytorch/ShapeParityTest.terl`.
- [x] Bind and execute 31 exact dense view, alias, detach, transpose,
  conjugation, and real/complex storage schemas in `pytorch.View`. Public
  mixed-case PyTorch names remain exact while Rust adapters use explicit
  snake-case identifiers. Real LibTorch lifecycle evidence is in
  `tests/pytorch/ViewParityTest.terl`.
- [x] Extend `pytorch.Shape` with eight exact expand/copy, list-axis movement,
  narrow-copy, roll, rotation, and unflatten schemas. Generated declarations
  preserve PyTorch defaults, while executable evidence distinguishes aliased
  strides from owned copies in `tests/pytorch/ShapeTransformParityTest.terl`.
- [x] Complete 18 exact diagonal, matrix-power, pixel-layout, permutation,
  repetition, rolling, rotation, squeeze, triangular, unsqueeze, and view-copy
  retained-output schemas through six shared shape dispatcher families. CPU,
  mandatory CUDA, complete packet equality, source immutability, output
  resizing, and public Terlan evidence is in `tests/native/shape_output.rs`
  and `tests/pytorch/ShapeTransformParityTest.terl`.
- [x] Complete 11 exact detach, squeeze, axis-swap, transpose, triangular,
  unsqueeze, and zero in-place schemas through shared unary and shape
  dispatcher families. CPU, mandatory CUDA, functional equivalence, source
  independence, detached-autograd behavior, and invalid-dimension failure
  atomicity are proven in `tests/native/shape_mutation.rs` and
  `tests/pytorch/ShapeMutationParityTest.terl`.
- [x] Bind and execute 16 exact immutable ordering, bucketization, diagonal,
  index, mask, scatter, membership, conditional-selection, and copied-window
  schemas in `pytorch.Indexing`. Pinned LibTorch evidence in
  `tests/pytorch/IndexingParityTest.terl` proves values, shapes, stable duplicate
  ordering, accumulation, source ownership, and independent result storage.
- [x] Complete seven exact copy, masked-scatter, and indexed-put functional,
  retained-output, and in-place schemas. CPU and mandatory CUDA evidence proves
  output resizing, source ownership, indexed mutation, and LibTorch's specified
  partial mutation before a short masked-scatter source reports an error in
  `tests/native/mutation_completion.rs` and
  `tests/pytorch/MutationCompletionParityTest.terl`.
- [x] Complete all 13 exact `chunk`, scalar/list `split`, scalar/list/tensor
  `tensor_split`, `dsplit`, `hsplit`, `vsplit`, and `unbind` view schemas through
  one owned tensor-list boundary. Each result element has an independent handle
  while retaining upstream shared-storage aliasing. CPU and mandatory CUDA
  evidence covers exact values and shapes, uneven partitions, explicit indices,
  non-contiguous inputs, mutation aliasing, source-lifetime survival, autograd,
  invalid requests, and the pinned requirement that tensor-valued split controls
  remain on CPU for CUDA data in `tests/native/shape_partition.rs` and
  `tests/pytorch/ShapePartitionParityTest.terl`.
- [x] Complete all four exact tensor-index, retained-output, tensor-count, and
  scalar-count `repeat_interleave` schemas through 12 explicit paths covering
  every absent/present dimension and output-size state. CPU and mandatory CUDA
  evidence covers exact values, shapes, dtype/device preservation, retained
  output resizing, non-contiguous input, source independence, empty results,
  gradients, invalid counts/dimensions/types, and output-size mismatches. The
  CUDA mismatch is isolated because pinned ATen reports it asynchronously as a
  device assertion on synchronization; evidence is in
  `tests/native/repeat_interleave.rs` and
  `tests/pytorch/RepeatInterleaveParityTest.terl`.
- [x] Complete the exact functional and retained-output `constant_pad_nd`
  schemas plus generic `pad`, and publish both `torch.constant_pad_nd` and
  `torch.nn.functional.pad`. Native and public CPU and mandatory-CUDA evidence
  covers exact constant integer/float values, reflect/replicate/circular modes,
  negative cropping, non-contiguous inputs, retained-output resizing and
  failure atomicity, ownership, and autograd in `tests/native/padding.rs` and
  `tests/pytorch/PaddingParityTest.terl`.
- [x] Complete the three exact `atleast_1d.Sequence`, `atleast_2d.Sequence`,
  and `atleast_3d.Sequence` list-in/list-out schemas and publish the variadic
  `torch.atleast_*` family. Native and public CPU and mandatory-CUDA evidence
  covers scalar/vector/matrix rank rules, exact values, non-contiguous inputs,
  view aliasing, independent handle lifetimes, and autograd in
  `tests/native/atleast_sequence.rs` and
  `tests/pytorch/AtleastSequenceParityTest.terl`.
- [x] Complete the exact `broadcast_tensors`, default/explicit-indexing
  `meshgrid`, `flatten_dense_tensors`, and `unflatten_dense_tensors` schemas,
  and publish `torch.broadcast_tensors` and `torch.meshgrid`. Native and public
  CPU and mandatory-CUDA evidence covers exact broadcast/grid values, `ij` and
  `xy` indexing, mixed-device view preservation, flatten/unflatten storage
  aliasing, non-contiguous inputs, failures, and autograd in
  `tests/native/tensor_list_transforms.rs` and
  `tests/pytorch/TensorListTransformsParityTest.terl`.
- [ ] Complete basic and advanced indexing, masks, gather/scatter, take/put,
  slicing, selection, assignment, and index reduction behavior.
- [ ] Complete remaining public reshape/view/resize, squeeze/unsqueeze, flatten,
  transpose, permutation, expand, repeat, concatenate, and stack variants.
- [ ] Preserve upstream aliasing and mutation behavior explicitly; operations
  must not silently copy when PyTorch promises a view or mutate shared storage
  without a typed mutation contract.
- [ ] Inventory this roadmap section after execution and update it if storage
  aliasing exposes another required ownership primitive.

### PP3: numerical and reduction parity

- [x] Implement eight exact stable argsort-output, pairwise-distance,
  matrix-chain, retained clone, contiguity, and cross-product schemas in
  `pytorch.NumericalCompletion`. Native CPU and required-CUDA evidence covers
  dtype-retaining output, inferred/explicit options, one/empty matrix chains,
  non-contiguous copy versus contiguous alias behavior, malformed requests,
  and autograd in `tests/native/numerical_completion.rs`; every public form
  executes on both devices in
  `tests/pytorch/NumericalCompletionParityTest.terl`.
- [x] Bind and execute the exact no-argument `abs`, `ceil`, `floor`, `frac`,
  `reciprocal`, `round`, `sign`, `square`, and `trunc` schemas through the
  generated dispatcher ABI. Real LibTorch evidence covers signed fractions,
  directed rounding, ties-to-even rounding, every finite sign class, and
  source-handle ownership in `tests/pytorch/UnaryParityTest.terl`.
- [x] Bind and execute 48 additional exact immutable one-input schemas across
  inverse and direct trigonometry, hyperbolic functions, exponentials,
  logarithms, activations, special-value predicates, logical and bitwise
  operations, scalar reductions, trace, nonzero coordinates, and paired
  floating-point mantissa/exponent decomposition. Every schema
  executes against pinned LibTorch in
  `tests/pytorch/UnaryFamilyParityTest.terl`.
- [x] Bind and execute 56 exact immutable two-tensor schemas in the dedicated
  `pytorch.Binary` module. Real LibTorch evidence covers arithmetic, bitwise,
  comparisons and aliases, logical operations, extrema, contractions,
  special functions, masking, and explicit index selection in
  `tests/pytorch/BinaryParityTest.terl`, plus tolerance-aware elementwise
  comparison in `tests/pytorch/FunctionalParityTest.terl`.
- [x] Bind and execute 56 exact immutable numerical and special schemas in
  `pytorch.Special`. Real LibTorch evidence covers aliases, matrix functions,
  real-to-complex promotion, ordering, reversal, Bessel functions, probability
  transforms, and the remaining one-input `torch.special` family in
  `tests/pytorch/SpecialParityTest.terl`.
- [x] Bind and execute ten exact tuple-returning indexed-selection schemas in
  `pytorch.Selection`, covering cumulative extrema, kth and top-k ranking,
  dimension extrema, median/nanmedian, mode, and sorting. Both value and index
  tensors execute against pinned LibTorch in
  `tests/pytorch/SelectionParityTest.terl`.
- [x] Bind and execute exact cosine-similarity and pairwise-distance schemas in
  `pytorch.Distance`. Pinned LibTorch evidence covers values, reduced shapes,
  numerical epsilon behavior, and source ownership in
  `tests/pytorch/FunctionalParityTest.terl`.
- [x] Complete 117 numeric, bitwise, logical, activation, conversion, special,
  inverse, circular, hyperbolic, reduction, predicate, backward, decimal-round,
  and decomposition unary storage schemas through the exact StableIValue
  dispatcher. Typed wrappers preserve retained output and mutation contracts,
  including two-output storage, without another native bridge. CPU, mandatory
  CUDA, source-ownership, and failure-atomicity evidence is in
  `tests/native/unary_storage.rs` and
  `tests/pytorch/UnaryStorageParityTest.terl` plus
  `tests/pytorch/UnaryResidualParityTest.terl`.
- [x] Complete the 15 remaining `fmax`, `fmin`, `maximum`, `minimum`, `fmod`,
  and logical tensor storage schemas. Exact tensor forms use the StableIValue
  dispatcher while scalar `fmod` reuses the typed scalar bridge. CPU,
  mandatory CUDA, source-ownership, and failure-atomicity evidence is in
  `tests/native/binary_storage.rs` and
  `tests/pytorch/BinaryStorageParityTest.terl`.
- [x] Complete 23 exact residual two-tensor and dimension-extrema output
  schemas through two explicit dispatcher families. The compiler expands each
  declared family into ordinary validated bindings without inferring operator
  identities. CPU, mandatory CUDA, full-tensor equality, output ownership, and
  Terlan worker evidence is in `tests/native/binary_output.rs` and
  `tests/pytorch/BinaryOutputParityTest.terl`.
- [x] Complete the three scalar `copysign` functional, retained-output, and
  in-place schemas through the shared typed scalar bridge. Both integer and
  floating-point scalar paths have CPU, mandatory CUDA, source-ownership, and
  public Terlan wrapper evidence in `tests/native/binary_storage.rs` and
  `tests/pytorch/BinaryStorageParityTest.terl`.
- [x] Complete ten exact arctangent, greatest-common-divisor, Heaviside,
  hypotenuse, incomplete-gamma, least-common-multiple, exponent-scaling, and
  next-representable-value in-place schemas through one shared binary
  dispatcher family. CPU, mandatory CUDA, functional equivalence, both-input
  ownership, mixed float/integer operands, and invalid-broadcast failure
  atomicity are proven in `tests/native/binary_mutation.rs` and
  `tests/pytorch/BinaryMutationParityTest.terl`.
- [x] Complete 17 residual alias, angle, physical-conjugate, detach, exponential,
  truncation, Bessel, log-gamma, lift, and logarithm storage schemas through two
  explicit unary dispatcher families. CPU, mandatory CUDA, source-ownership,
  public-wrapper, and functional-equivalence evidence is in
  `tests/native/unary_storage.rs` and
  `tests/pytorch/UnaryResidualParityTest.terl`.
- [x] Complete 12 exact digamma, inverse, log-sigmoid, sorting, negation,
  activation, sign, square-root, transpose-copy, and real/complex view-copy
  retained-output schemas through the shared unary output dispatcher family.
  CPU, mandatory CUDA, complete result equality, source ownership, and
  failure-atomicity evidence is in `tests/native/unary_storage.rs` and
  `tests/pytorch/UnaryOutputCompletionTest.terl`.
- [x] Complete seven exact digamma, negation, activation, sign, square-root,
  and transpose in-place schemas through the shared unary mutation dispatcher
  family. CPU, mandatory CUDA, functional equivalence, and failure atomicity
  are proven in `tests/native/unary_storage.rs` and
  `tests/pytorch/UnaryOutputCompletionTest.terl`.
- [x] Complete seven `bmm`, `ger`, `matmul`, `mm`, `mv`, `orgqr`, and `outer`
  retained-output schemas through one explicit dense dispatcher family. CPU,
  mandatory CUDA, full-tensor equality, output resizing, and public Terlan
  evidence is in `tests/native/binary_output.rs` and
  `tests/pytorch/DenseParityTest.terl`.
- [x] Complete eight cumulative-extrema, kth-value, median, mode, NaN-median,
  sort, and top-k retained tuple-output schemas through five explicit selection
  dispatcher families. CPU, mandatory CUDA, complete values-and-indices
  equality, and public Terlan evidence is in `tests/native/binary_output.rs`
  and `tests/pytorch/SelectionParityTest.terl`.
- [x] Complete 12 exact extrema, index, truth, nonzero-count, and NaN-median
  retained-output schemas through 16 explicit reduction wrappers. Separate
  wrappers preserve provided and omitted optional dimensions without exposing
  nullable values in Terlan. CPU, mandatory CUDA, full-tensor equality, source
  ownership, and public Terlan evidence is in
  `tests/native/statistics_output.rs` and
  `tests/pytorch/StatisticsParityTest.terl`.
- [x] Complete nine exact log-sum-exp, mean, product, standard-deviation, sum,
  and variance retained-output schemas through 13 explicit reduction wrappers.
  Global overloads and omitted optional dimensions remain distinct executable
  paths while inherited dtype is represented without dynamic scalar-type
  values. CPU, mandatory CUDA, full-tensor equality, source ownership, and
  public Terlan evidence is in `tests/native/statistics_output.rs` and
  `tests/pytorch/StatisticsParityTest.terl`.
- [x] Complete all eight exact correction-based standard-deviation, variance,
  paired-mean functional, and retained-output schemas through 48 typed
  operations covering omitted and explicit dimensions, omitted, integer, and
  floating correction, and keep-dimension behavior. Direct paired results use
  an owned ABI tuple container whose elements are extracted as independently
  owned tensors before the container is released. CPU, mandatory CUDA,
  non-contiguous input, output resizing, ownership, and invalid dimensions are
  proven in `tests/native/correction_statistics_completion.rs` and
  `tests/pytorch/CorrectionStatisticsCompletionParityTest.terl`.
- [x] Complete four exact tensor copysign, multivariate log-gamma retained-output
  and in-place, and tensor-weight lerp in-place schemas. CPU and mandatory CUDA
  evidence proves functional equivalence, output ownership, and storage mutation
  through `tests/native/mutation_completion.rs` and
  `tests/pytorch/MutationCompletionParityTest.terl`.
- [x] Complete 13 exact cumulative-reduction, GELU, polygamma, and logit
  functional, retained-output, in-place, and backward schemas through 24 public
  operations. Distinct omitted and explicit dtype, approximation, and epsilon
  paths preserve the complete optional behavior rather than hardcoding defaults.
  CPU, mandatory CUDA, output ownership, mutation, and invalid-option atomicity
  are proven in `tests/native/optional_numerics.rs` and
  `tests/pytorch/OptionalNumericsParityTest.terl`.
- [x] Complete all three exact functional, retained-output, and in-place
  `nan_to_num` schemas through 24 explicit paths covering every independent
  absent/present NaN, positive-infinity, and negative-infinity replacement
  state. CPU and mandatory CUDA evidence proves exact dtype-limit defaults,
  custom replacements, complex component behavior, integer no-op behavior,
  non-contiguous inputs, source ownership, output resizing, error atomicity,
  gradients, and invalid boundary handles in `tests/native/nan_to_num.rs` and
  `tests/pytorch/NanToNumParityTest.terl`.
- [x] Complete both exact optional-weight `bincount` schemas, both equal-width
  `histc` schemas, and all three functional, retained-output, and in-place
  `index_reduce` schemas. Typed public operations cover unweighted and weighted
  counts, minimum output lengths, automatic and explicit integer/floating
  histogram bounds, and product, mean, maximum, and minimum indexed reductions
  with both `include_self` states. CPU and mandatory CUDA evidence proves exact
  values and dtypes, non-contiguous inputs, output resizing, source ownership,
  transactional errors, leaf-mutation rejection, supported gradients, pinned
  non-differentiable histogram and weighted-count behavior, and invalid
  boundary handles in `tests/native/aggregation.rs` and
  `tests/pytorch/AggregationParityTest.terl`.
- [ ] Complete unary, binary, comparison, logical, bitwise, rounding,
  transcendental, special-value, clamp, and elementwise selection families.
- [ ] Complete reductions, cumulative operations, sorting, top-k, uniqueness,
  histograms, statistics, and dtype/axis/output variants.
- [ ] Complete broadcasting, promotion, scalar overload, `out`, in-place, and
  autograd behavior for every applicable family.
- [ ] Inventory this roadmap section after execution and update it if operator
  schema coverage reveals another numerical family.

### PP4: linear algebra, FFT, sparse, and special domains

- [x] Bind and execute 40 exact dense two-tensor schemas in `pytorch.Dense`,
  including batched and ordinary products, matrix-vector and outer products,
  complex construction, shape/type matching, reflector products, ORGQR,
  explicit activation backward kernels, and tensor-order special functions.
  Real LibTorch evidence is in `tests/pytorch/DenseParityTest.terl`.
- [x] Bind and execute 73 exact statistical and decomposition schemas across
  `pytorch.Statistics` and canonical `pytorch.Tensor` reductions, including
  pinned compatibility behavior for deprecated
  Cholesky and nuclear-norm names, rank and pseudoinverse, pairwise distance,
  variance, paired sample/population moments, extrema and index reductions,
  truth and count reductions, cumulative and complete or dimension mean/sum,
  dimension-wise paired moments, scalar/dimension/excluded-dimension norms,
  bounded renormalization, distance, stable log-sum-exp, products, sorting,
  logical dimension reductions, combinations, retained cumulative reductions,
  Cholesky factorization/inversion/solve, and Frobenius and nuclear norms.
  CPU, mandatory CUDA, complete packet equality, source immutability, output
  resizing, and public Terlan evidence is in
  `tests/native/statistics_output.rs` and
  `tests/pytorch/StatisticsParityTest.terl`.
- [x] Bind and execute 16 exact tuple-returning factorization, decomposition,
  inversion, solve, determinant, and triangular-system schemas in
  `pytorch.Linalg`. CPU evidence covers two- and three-tensor return ownership,
  information tensors, caller-selected string modes, decomposition shapes, and
  numerical results in `tests/pytorch/LinalgTupleParityTest.terl`.
- [x] Bind and execute 14 exact single-result factorization, determinant,
  inverse, solve, rank, pseudoinverse, vector-product, and Householder schemas
  in `pytorch.Linalg`. CPU evidence covers factor reuse, tensor-valued
  tolerances, legacy and current solve paths, dimensions, finite outputs, and
  numerical results in `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Bind and execute 20 exact non-`out` complex, real, Hermitian, 1D, 2D,
  N-dimensional, and frequency-shift schemas in `pytorch.Fft`. CPU evidence
  covers backward, forward, and orthonormal normalization, selected axes,
  all-axis shifts, shape contracts, and round-trip numerical results in
  `tests/pytorch/FftParityTest.terl`.
- [x] Bind and execute seven additional exact non-`out` special-function
  schemas in `pytorch.Special`, covering Airy Ai, scaled modified Bessel K,
  spherical Bessel J0, integer-order polygamma, decimal rounding, and
  dimension-list log-sum-exp. CPU evidence covers reference values, retained
  dimensions, caller-owned inputs, and generated dispatcher ownership in
  `tests/pytorch/SpecialParityTest.terl`.
- [x] Bind and execute exact `special_logit`, `special_softmax`, and
  `special_log_softmax` schemas with both omitted and explicit optional
  arguments. The compiler now owns a reusable optional-float StableIValue
  path, while CPU evidence covers epsilon clamping, dtype selection,
  normalization, source ownership, and optional cleanup in
  `tests/pytorch/SpecialParityTest.terl`.
- [x] Bind and execute eight additional exact non-`out` linalg schemas for
  string-order condition numbers, Hermitian eigenvalues, least squares,
  float-tolerance rank and pseudoinverse, SVD and singular values, and
  Vandermonde matrices. CPU evidence covers default and explicit optionals,
  four-result ownership, rank and decomposition values, output shapes, and
  caller-owned inputs in `tests/pytorch/LinalgTupleParityTest.terl` and
  `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Bind and execute exact named matrix norm, named general norm, and tensor
  solve schemas with omitted and explicit dimension/dtype optionals. CPU
  evidence covers scalar results, retained dtype, optional list ownership,
  tensor-equation dimension permutation, and source preservation in
  `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Bind and execute exact `linalg_multi_dot` through a reusable compiler
  dispatcher constructor for `List[Tensor]`. The generated adapter duplicates
  each resource, transfers it only after a successful stable-list push, and
  cleans every partial failure path. CPU evidence covers a three-matrix chain,
  optimized result values, source ownership, and Terlan worker list transport
  in `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Bind and execute tensor-optional matrix-rank and pseudoinverse schemas
  across every absent/present tolerance combination, plus the exact aliasing
  `linalg_diagonal` schema. CPU evidence covers numerical cutoff behavior,
  caller-owned tolerance resources, and a diagonal view that remains usable
  after its source handle is disposed in
  `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Bind and execute exact scalar-optional `linalg_cond` with separate
  default, integer-order, and floating-order entry points. CPU evidence proves
  the reviewed direct ATen scalar boundary, numerical behavior, and source
  ownership after LibTorch's stable dispatcher rejected `Scalar` in
  `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Bind and execute every remaining non-`out` scalar linalg schema:
  matrix norm, general norm, vector norm, and internal power-sum. Separate
  integer and floating-order entry points cover absent/default and explicit
  dimension/dtype behavior through one shared direct ATen executor, with CPU
  evidence for all 17 generated APIs in
  `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Establish mutable dispatcher output ownership with exact
  `linalg_det.out`. The compiler duplicates caller-owned output storage for
  dispatch, discards the returned owned alias, and retains the original public
  handle; CPU evidence proves in-place population and source preservation in
  `tests/pytorch/LinalgSingleParityTest.terl`.
- [x] Generalize mutable dispatcher ownership to disjoint additional output
  resources and execute `linalg_cholesky_ex.L`, `linalg_eigh.eigvals`,
  `linalg_inv_ex.inverse`, and `linalg_svd.U`. CPU evidence proves two- and
  three-output mutation, exact output dtypes, retained caller handles, and
  immutable source tensors in `tests/pytorch/LinalgTupleParityTest.terl`.
- [x] Execute the complete LU output-overload family: `linalg_lu.out`,
  `linalg_lu_factor.out`, and `linalg_lu_factor_ex.out`, including mixed
  floating-point factor and Int32 pivot/status output storage.
- [x] Execute fixed-arity tuple outputs for LDL factorization, QR, signed
  log-determinant, and checked linear solve with retained caller-owned storage.
- [x] Execute nine core single-output linalg overloads spanning factorization,
  inverse, LDL/LU solve, contraction, triangular solve, cross, and vector dot.
- [x] Execute nine extended single-output linalg overloads spanning complex
  eigenvalues, spectral operations, matrix functions, tensor equations, and
  resource-list matrix-chain multiplication.
- [x] Execute complex `linalg_eig.out` and both absent/default and explicit
  optional paths for four-output `linalg_lstsq.out`, retaining all caller-owned
  output handles across complex and mixed-dtype decompositions.
- [x] Execute all matrix-rank and pseudoinverse output tolerance schemas across
  absent, absolute-only, relative-only, combined, scalar, and tensor cutoff
  forms with retained caller-owned Int64 and floating output storage.
- [x] Execute all named-order condition, matrix-norm, and general-norm output
  schemas with omitted/default and explicit dimension/dtype behavior.
- [x] Execute all Scalar-based condition, matrix-norm, general-norm, and
  vector-norm output schemas through one shared direct-ATen executor, covering
  absent, integer, floating, dimension, keep-dimension, and dtype forms.
- [x] Complete all 99 frozen `torch.linalg` schemas, including matrix/tensor
  contraction, decomposition, solving, rank, norm, spectral behavior, and all
  caller-owned output overloads.
- [x] Complete all 42 frozen `torch.fft` schemas, including real, complex, and
  Hermitian transforms across one, two, and N dimensions; retained output
  storage; shifts; normalization; and full/real frequency grids.
- [x] Execute all 54 tensor-input `torch.special` output schemas through
  retained caller-owned storage, including unary and binary functions,
  optional and reduction parameters, and every tensor-order polynomial
  family. CPU evidence is in `tests/pytorch/SpecialOutParityTest.terl` and
  `tests/pytorch/SpecialTensorOutParityTest.terl`.
- [x] Establish the direct ATen Scalar contract for special functions with
  integer and floating values in both operand positions and both immutable
  and retained-output forms. Chebyshev T provides the executable reference
  family in `tests/pytorch/SpecialScalarParityTest.terl`.
- [x] Complete all 170 frozen `torch.special` schemas. The final Scalar slice
  covers 15 families, both operand positions, integer and floating values,
  immutable results, and retained output storage through eight shared native
  primitives and 120 typed Terlan entry points.
- [x] Bind and execute 60 foundational sparse schemas in `pytorch.Sparse`:
  dense-to-COO and compressed-layout conversion, coalescing, sparse/dense rank
  metadata, COO and compressed index/value alias/copy/output accessors, sparse
  masks, sparse-to-sparse copy, functional/output/in-place resize families,
  generic layout selection, configured sparse-to-dense conversion, and every
  inferred/explicit-size COO, CSR, CSC, BSR, BSC, and generic compressed
  constructor schema. Sampled sparse matrix multiplication covers integer and
  floating scalar controls plus retained output storage; sparse-to-dense
  backward covers absent, enabled, and disabled gradient masking.
  `tests/pytorch/SparseParityTest.terl` and `tests/native/sparse.rs` cover COO,
  CSR, CSC, BSR, and BSC values, optional dense-rank arguments, ownership,
  mutation, malformed ranks and blocks, and direct fallbacks for LibTorch
  sparse copy/output dispatcher gaps.
- [x] Close the nine remaining sparse-adjacent embedding and gather schemas in
  their existing `pytorch.Tensor` and `pytorch.Gradients` owners. Evidence
  covers retained embedding/gather/dense-gradient outputs, four-result
  embedding bags with absent/present sample weights and padding, functional,
  output, and in-place renormalization, and sparse gradient materialization.
  The frozen ATen inventory now has no unclassified schema containing
  `sparse`.
- [x] Complete all 56 remaining bitwise AND/OR/XOR, left/right shift,
  remainder, power, and floating-power overload schemas. Twelve typed native
  primitives preserve exact integer values and cover scalar-left,
  scalar-right, retained-output, and in-place forms; operation-specific
  Terlan entry points keep native operation IDs private. Evidence in
  `tests/pytorch/BinaryScalarParityTest.terl` and
  `tests/native/binary_scalar.rs` covers every operation/shape combination,
  operand order, source retention, mutation, invalid operation IDs, invalid
  bitwise dtypes, and failure atomicity.
- [x] Complete all 24 remaining `div`, `divide`, and `true_divide` schemas.
  `pytorch.Division` exposes typed integer and floating scalar overloads while
  retaining exact tensor, rounding-mode, output, and in-place behavior. Direct
  ATen scalar bindings avoid the stable dispatcher ABI's missing `Scalar`
  conversion; tensor overloads continue through exact dispatcher contracts.
  `tests/pytorch/DivisionParityTest.terl` and `tests/native/division.rs` prove
  all 36 typed entry points, aliases, source ownership, invalid rounding modes,
  and mutation failure atomicity.
- [x] Complete all nine remaining `mul` and `multiply` schemas by extending
  the shared binary scalar engine with one reviewed operation ID. Existing
  immutable tensor multiplication remains canonical; typed scalar,
  retained-output, and in-place forms reuse the same native storage contract.
  `tests/pytorch/MultiplicationParityTest.terl` and
  `tests/native/binary_scalar.rs` prove both aliases, integer and floating
  scalars, output retention, mutation, and unchanged sources.
- [x] Complete all five remaining `floor_divide` schemas through the shared
  typed binary engine. `tests/pytorch/FloorDivisionParityTest.terl` and the
  native binary suite prove tensor and scalar functional, retained-output, and
  in-place behavior for positive and negative operands without changing the
  canonical immutable tensor API.
- [x] Complete all seven remaining `xlogy` schemas through operation ID 10 in
  the shared typed binary engine. Scalar-left and scalar-right, tensor,
  retained-output, and in-place paths preserve operand order and source
  ownership. `tests/pytorch/XlogyParityTest.terl` and the native binary suite
  prove stable zero behavior and nontrivial logarithmic results.
- [x] Complete all 21 remaining `add`, `sub`, `subtract`, and `rsub` schemas
  through nine shared scaled-arithmetic primitives. Sixty-two documented
  Terlan wrappers cover integer and floating operands crossed with integer and
  floating alpha values without exposing operation identifiers. Native and
  Terlan evidence proves functional, retained-output, in-place, reverse-order,
  source-ownership, invalid-operation, and failure-atomicity behavior in
  `tests/native/scaled_arithmetic.rs` and
  `tests/pytorch/ScaledArithmeticParityTest.terl`. The native suite also
  executes scaled addition, division, multiplication, floor division, and
  `xlogy` on CUDA with synchronous host readback under the required CUDA lane.
- [x] Complete all 55 remaining comparison scalar, retained-output, and
  in-place schemas for `eq`, `ne`, `not_equal`, `lt`, `less`, `le`,
  `less_equal`, `gt`, `greater`, `ge`, and `greater_equal`. Six canonical
  native operation IDs drive 88 documented typed Terlan wrappers. Native
  evidence covers integer and floating scalars plus every storage contract;
  alias-level Terlan tests distinguish strict and inclusive comparisons at
  equality while preserving both sources. The required CUDA lane executes all
  six canonical operations through functional, retained-output, and in-place
  paths before synchronous host assertions.
- [x] Complete all 22 remaining `clamp`, `clamp_min`, `clamp_max`, and `clip`
  schemas through six shared scalar/tensor native primitives. Typed Terlan
  wrappers cover absent minimum or maximum bounds, integer, floating, mixed,
  tensor, retained-output, and in-place forms. CPU and mandatory CUDA evidence
  is in `tests/native/clamp.rs` and `tests/pytorch/ClampParityTest.terl`.
- [x] Complete all four `lerp` scalar/tensor functional and retained-output
  schemas through four shared native primitives. Typed Terlan wrappers retain
  integer and floating scalar distinctions; CPU and mandatory CUDA evidence is
  in `tests/native/lerp.rs` and `tests/pytorch/LerpParityTest.terl`.
- [x] Complete all five remaining `where` schemas through shared scalar,
  tensor-output, and owned tensor-list primitives. Typed wrappers cover all
  integer/floating scalar pairings, while CPU and mandatory CUDA evidence in
  `tests/native/where.rs` and `tests/pytorch/WhereParityTest.terl` proves index
  ownership, output retention, and invalid-condition failure atomicity.
- [x] Complete all six `index_fill` scalar/tensor functional, retained-output,
  and in-place schemas through six direct ATen primitives. Typed wrappers
  preserve integer and floating scalar distinctions. CPU and mandatory CUDA
  evidence in `tests/native/index_fill.rs` and
  `tests/pytorch/IndexFillParityTest.terl` proves values, source retention,
  mutation, output ownership, and invalid-index failure atomicity.
- [x] Complete the seven missing `index_add`, `index_copy`, and `index_select`
  output, in-place, and backward schemas through shared indexed-mutation
  primitives. Typed wrappers preserve integer and floating alpha values. CPU
  and mandatory CUDA evidence in `tests/native/index_mutation.rs` and
  `tests/pytorch/IndexMutationParityTest.terl` proves every storage contract,
  selection-gradient scattering, and prevalidation that prevents invalid
  indices from partially mutating caller-owned tensors.
- [x] Complete all 11 remaining `fill`, `masked_fill`, and `fill_diagonal_`
  schemas through shared scalar/tensor native primitives. Typed wrappers cover
  integer and floating values, diagonal wrapping, and every output/in-place
  contract. CPU and mandatory CUDA evidence in `tests/native/fill.rs` and
  `tests/pytorch/FillParityTest.terl` proves values, source ownership, and
  prevalidated failure atomicity for malformed masks and non-scalar values.
- [x] Complete all 16 remaining plain, scalar, additive, legacy-reduced, and
  modern-reduced `scatter` schemas through six shared native primitives.
  Public Terlan APIs replace reduction strings with typed legacy and modern
  reduction values. CPU and mandatory CUDA evidence in
  `tests/native/scatter.rs` and `tests/pytorch/ScatterParityTest.terl` proves
  all five modern reductions, every storage contract, source ownership, and
  invalid-mode, reduction, and index failure atomicity.
- [x] Complete all 12 remaining scalar/tensor `isin`, `searchsorted`, and
  `bucketize` schemas through 10 shared native primitives. Public Terlan APIs
  provide typed insertion sides, integer and floating scalar forms, optional
  sorter forms, output-width selection, and retained output storage. CPU and
  mandatory CUDA evidence in `tests/native/lookup.rs` and
  `tests/pytorch/LookupParityTest.terl` proves values, dtype, source ownership,
  and invalid-mode, side, and sorter failure atomicity.
- [x] Complete all 25 exact `addbmm`, `addcdiv`, `addcmul`, `addmm`, `addmv`,
  `addr`, and `baddbmm` functional, retained-output, and in-place schemas
  through three shared fused-arithmetic primitives. Public Terlan APIs preserve
  exact integer and floating scalar values through the canonical `Scalar`
  type. CPU and mandatory CUDA evidence in
  `tests/native/fused_arithmetic.rs` and
  `tests/pytorch/FusedArithmeticParityTest.terl` covers values, dtype-selecting
  overloads, source ownership, output retention, mutation, and failure
  atomicity.
- [ ] Complete sparse layouts and operations and their supported
  CUDA combinations when CUDA LibTorch evidence is available; CPU behavior is
  classified for every sparse-named schema in the frozen inventory.
- [ ] Inventory this roadmap section after execution and update it if a domain
  requires a package-neutral interchange primitive.

### PP5: autograd, modules, and optimizer parity

- [x] Complete the five remaining exact in-place and feature-wise dropout
  schemas in `pytorch.DropoutCompletion`. Native and public CPU/required-CUDA
  evidence proves inference identity, deterministic training boundaries at
  probabilities zero and one, functional source ownership, mutation, and
  invalid-probability failure atomicity in
  `tests/native/dropout_completion.rs` and
  `tests/pytorch/DropoutCompletionParityTest.terl`.
- [x] Implement all nine exact one-, two-, and three-dimensional convolution
  and transposed-convolution schemas in `pytorch.Convolution`, including
  numeric and named padding plus independently present or absent bias. Native
  CPU and required-CUDA evidence covers exact values, grouped convolution,
  source ownership, malformed dimensions/groups/devices, and autograd in
  `tests/native/convolution.rs`; all public defaults and explicit forms execute
  on both devices in `tests/pytorch/ConvolutionParityTest.terl`.
- [x] Bind and execute nine exact immutable loss schemas in `pytorch.Loss`,
  covering L1, smooth-L1, Huber, soft-margin, Poisson NLL, hinge embedding,
  cosine embedding, margin ranking, and triplet margin behavior. Pinned
  LibTorch evidence in `tests/pytorch/LossParityTest.terl` covers numerical
  results, reduction defaults, mixed tensor/primitive stacks, and source
  ownership.
- [x] Bind and execute exact Huber and smooth-L1 backward kernels in
  `pytorch.Loss`. Pinned LibTorch evidence covers unreduced gradients across
  quadratic and linear regions while preserving all three caller-owned inputs.
- [x] Bind and execute exact functional element, alpha, and feature dropout
  schemas in `pytorch.Regularization`. Deterministic training and inference
  branches execute against pinned LibTorch without mutating caller-owned
  tensors in `tests/pytorch/FunctionalParityTest.terl`.
- [x] Bind and execute exact native dropout backward behavior with a mixed
  boolean mask and explicit scale in `tests/pytorch/FunctionalParityTest.terl`.
- [x] Bind and execute ten exact immutable gradient kernels in
  `pytorch.Gradients`, covering cumulative product, dense embedding, gather,
  GELU/GLU, masked selection, soft-margin, and fake-quantization cache masks.
  Pinned LibTorch evidence in `tests/pytorch/GradientKernelsParityTest.terl`
  proves numerical output, shapes, frequency scaling, scatter behavior, and
  caller-owned source lifetimes.
- [x] Complete exact `requires_grad_` leaf mutation through the stable autograd
  boundary. CPU and mandatory CUDA execution plus public Terlan backward and
  gradient retrieval prove observable graph participation in
  `tests/native/mutation_completion.rs` and
  `tests/pytorch/MutationCompletionParityTest.terl`.

- [x] Bind and execute five exact adaptive-average, average, and maximum
  pooling schemas in `pytorch.Pooling`, preserving all trailing spatial
  defaults. Numerical evidence spans one, two, and three dimensions in
  `tests/pytorch/PoolingParityTest.terl`. The bare
  `aten::max_pool2d_backward` schema is classified non-applicable because the
  pinned distribution provides no executable CPU or CUDA backend kernel for
  it; executable autograd uses indexed backward schemas.
- [x] Bind and execute eight tuple-returning adaptive, indexed, and fractional
  maximum pooling schemas in one, two, and three dimensions. The package uses
  the compiler-owned opaque-resource tuple ABI, transactional helper storage,
  and tuple protocol decoding; deterministic random-sample, numerical, and
  ownership evidence is in
  `tests/pytorch/PoolingParityTest.terl`.
- [x] Bind and execute eight exact adaptive, fractional, and indexed
  maximum-pool backward or unpool schemas in two and three dimensions.
  Full-cycle evidence produces indices through the matching forward operator,
  routes gradients back to source shape, and reconstructs sparse unpooled
  outputs in `tests/pytorch/PoolingParityTest.terl`.
- [x] Bind and execute four exact two- and three-dimensional average-pooling
  forward/backward schemas with inherited divisors, explicit backward spatial
  parameters, source-shape restoration, and gradient conservation in
  `tests/pytorch/PoolingParityTest.terl`.
- [x] Complete 23 exact adaptive-average, adaptive-maximum, average,
  fractional-maximum, indexed-maximum, and unpool retained-output or gradient
  schemas. Four additional explicit-divisor paths preserve optional
  `avg_pool2d` and `avg_pool3d` behavior without nullable public parameters.
  CPU, mandatory CUDA, tuple-output ownership, source preservation, malformed
  kernel rejection, and public Terlan execution are proven by
  `tests/native/pooling_output.rs` and
  `tests/pytorch/PoolingOutputParityTest.terl`. The non-executable bare
  `max_pool2d_backward.out` schema is classified with its base schema.
- [x] Extend the reviewed CXX-backed Tensor backward surface to explicit
  vector-Jacobian seeds for non-scalar outputs, including retain/create-graph
  controls. The generated CXX wrapper calls LibTorch's native Tensor backward
  contract directly, with CPU and CUDA evidence in
  `AutogradGradientParityTest.terl` and `CudaTest.terl`; the graph-preserving
  gradient accessor proves second derivatives on both backends, and CUDA
  execution is recorded as `autograd-explicit-gradient` in the generated
  report.
- [x] Add the functional `torch::autograd::grad` contract as a generated CXX
  Tensor operation with explicit retain/create-graph controls. CPU and CUDA
  tests prove non-accumulating vector-Jacobian products and second derivatives;
  CUDA execution is recorded as `autograd-functional-grad`.
- [x] Extend functional autograd to multiple roots and inputs through a
  generated list-valued CXX wrapper. CPU and CUDA tests prove independent
  vector-Jacobian products for two roots, with CUDA execution recorded as
  `autograd-multiple-roots`.
- [x] Add multiple-root `torch::autograd::backward` through the generated CXX
  list-input wrapper. The native call preserves LibTorch accumulation and
  retain/create-graph controls; its generated Tensor return token is disposed
  by the public Terlan `Unit` alias. CPU (7/7) and CUDA (10/10) evidence covers
  the contract, with CUDA execution recorded as
  `autograd-multiple-roots-backward`.
- [x] Expose inference mode as an owned generated CXX RAII guard instead of
  only a process-state probe. Nested enter/disable/restore behavior passes on
  CPU and CUDA; disposal restores the prior LibTorch state and CUDA evidence is
  recorded as `autograd-inference-mode-guard`.
- [x] Expose CPU autocast as an owned generated CXX RAII guard that restores
  enabled state, dtype, and cache state across nested scopes. CPU and CUDA
  LibTorch packages pass the focused 3/3 state suite; CUDA evidence is
  recorded as `autograd-autocast-cpu-guard`.
- [x] Expose anomaly detection as an owned generated CXX RAII guard that
  restores both enablement and NaN-check flags across nested scopes. CPU and
  CUDA focused suites pass 3/3; CUDA evidence is recorded as
  `autograd-anomaly-mode-guard`.
- [x] Expose warning policy as an owned generated CXX RAII guard that restores
  `warnAlways` across nested scopes. CPU and CUDA focused suites pass 3/3;
  CUDA evidence is recorded as `autograd-warning-mode-guard`.
- [x] Expose deterministic-algorithm configuration as an owned generated CXX
  RAII guard that restores enablement and warn-only state across nested scopes.
  CPU and CUDA focused suites pass 3/3; CUDA evidence is recorded as
  `autograd-deterministic-algorithms-guard`.
- [x] Expose float32 matmul precision as an owned generated CXX RAII guard that
  restores the prior precision across nested scopes. CPU and CUDA focused
  suites pass 3/3; CUDA evidence is recorded as
  `autograd-float32-matmul-precision-guard`.
- [x] Expose default dtype as an owned generated CXX RAII guard that restores
  the prior scalar type across nested scopes. CPU and CUDA focused suites pass
  3/3; CUDA evidence is recorded as `autograd-default-dtype-guard`.
- [ ] Complete graph autograd, higher-order gradients, multiple roots,
  retained graphs, gradient modes, hooks with typed lifetimes, and deterministic
  failure/cancellation behavior.
- [ ] Implement parameter and module ownership, state dictionaries,
  train/eval mode, buffers, initialization, composition, and every public
  `torch.nn`/`torch.nn.functional` family.
- [ ] Complete public optimizer, scheduler, clipping, mixed-precision, and
  checkpoint semantics with device-portable state.
- [ ] Inventory this roadmap section after execution and update it if module
  state requires another compiler or VM ownership primitive.

### PP6: serialization, export, data, and distributed parity

- [ ] Complete safe tensor/model/state serialization without executing pickle,
  plus package-owned mappings for TorchScript, export, and admitted graph forms.
- [ ] Complete typed dataset, sampler, batching, collation, worker, and
  streaming behavior using Terlan process semantics.
- [ ] Complete applicable distributed tensor/training collectives through
  VM-owned supervision, cancellation, transport, and replication contracts.
- [ ] Inventory this roadmap section after execution and update it if external
  formats or distributed backends require another capability package.

### PP7: CPU/CUDA parity and closure

- [ ] Execute every CUDA-applicable classified operation on CPU and CUDA from
  the same public Terlan source and compare shape, dtype, values, errors, and
  ownership within documented numerical tolerances.
- [ ] Cover streams, events, synchronization, mixed precision, allocator
  pressure, cancellation, device mismatch, and multi-device behavior without
  making CUDA a compiler dependency.
- [ ] Reach zero missing callables, zero unclassified upstream tests, and zero
  undocumented public Terlan declarations in machine-readable reports.
- [ ] Add one strict `make pytorch-feature-parity-check` gate composing API,
  CPU, CUDA, autograd, model, interoperability, documentation, and source-policy
  validation.
- [ ] Inventory the complete roadmap and replace completed execution detail
  with durable package documentation before declaring feature parity.

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
- [x] Make it locate or check out this sibling repository through normal package
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
source-resolution item is closed by the compiler repository's checksum-pinned
Git bundle fixture: the gate creates a clean workspace with no sibling package,
resolves the exact revision through ordinary dependency metadata, verifies its
Git tree and clean checkout, and executes a fresh importing consumer without
network access. Publishing the package or pushing the pinned commit is not a
completion requirement.

The compiler gate also owns a machine-readable baseline-policy proof. CPU
execution is required when the native runtime is available; an unavailable
native runtime can produce only the explicit
`native_execution_unavailable` skip reason when skip policy is enabled. CUDA
remains an optional profile with `cuda_libtorch_unavailable` as its stable skip
reason. Requesting CUDA from the CPU profile is rejected before execution, and
placeholder C symbol identities are rejected by the generic binding generator.
This policy check validates dependency wiring and both source and normalized
manifest hashes even on its simulated skip path.

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
- [x] Add a package-owned LibTorch C++ adapter behind the generated C ABI for
  trainable leaf creation, scalar backward, copied gradient retrieval, and a
  functional SGD step. No PyTorch-specific compiler path is permitted.
- [x] Prove one leaf can execute real graph autograd, expose exact gradient
  `0.5`, lower MSE from `1.0` to approximately `0.9025`, and return a new leaf
  that can participate in a second backward pass.
- [x] Prove one affine graph propagates gradients into independent weight and
  bias leaves through `matmul`, autograd-aware `add`, and MSE before updating
  both leaves functionally.
- [x] Reject integer leaves, non-scalar backward roots, missing gradients,
  updates before backward, and invalid learning rates with stable C ABI status
  diagnostics.
- [x] Add an owned multi-parameter optimizer resource with explicit zeroing,
  Adam state, checkpointing, and process-owner/cancellation semantics after the
  functional training path is stable.
- [x] Extend the owned optimizer resource with stateful SGD, momentum,
  dampening, weight decay, Nesterov validation, step counting, and
  checkpoint/restore parity while retaining the existing Adam contract.

Gate: `make autograd-check LIBTORCH=vendor/libtorch-2.13.0+cpu/libtorch`.

### PT22A: PPO training prerequisites

Status: Complete.

- [x] Establish graph autograd and optimizer steps inside `terlan-pytorch`;
  `terlan-ndarray` remains the explicit dense-array package dependency while
  TNXP keeps runtime tensor exchange package-neutral.
- [x] Bind differentiable tensor division, exponentiation, logarithm, clamp,
  elementwise minimum, mean, sum, and negation through reviewed LibTorch APIs.
- [x] Bind indexed action selection through `gather` and deterministic seeded
  categorical sampling without transferring probability buffers through
  Terlan values.
- [x] Implement clipped policy-ratio loss, value loss, entropy bonus, advantage
  normalization, and combined actor/critic loss as ordinary Terlan functions
  composed from public tensor operations.
- [x] Add an owned Adam optimizer over multiple trainable tensors. Its native
  state must remain package-owned and must not expose C++ pointers or require
  ndarray conversion.
- [x] Execute one deterministic PPO minibatch update in Terlan and prove finite
  loss, bounded clipping, changed actor and critic parameters, gradient reset,
  deterministic seeded replay, and complete cleanup.
- [x] Execute the same `pytorch.Ppo.minibatch_update` source on CPU and CUDA
  through unchanged public Terlan call sites when CUDA LibTorch is explicitly
  selected.

Gate: `make ppo-check LIBTORCH=vendor/libtorch-2.13.0+cpu/libtorch`.

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
Training-mode statistic updates are closed by the functional
`pytorch.Training.batch_norm_training` contract. It clones both running
statistics, executes exact `aten::batch_norm` in training mode against those
clones, and returns the output plus updated mean and variance as independently
owned tensors. Caller-owned input, affine parameters, and statistics remain
unchanged.

- [x] Execute functional training-mode batch normalization with exact output,
  updated running mean/variance, source preservation, and complete disposal.

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

### PT44: owned dtype conversion

- [x] Reuse the synchronous stable-C allocation and copy path shared with
  device placement instead of adding a second conversion engine.
- [x] Add `to_dtype(tensor, dtype)` for admitted Bool, unsigned/signed integer,
  Float16/32/64, and BFloat16 identifiers while retaining the source device.
- [x] Prove Float64 -> Int64 -> Float64 scalar semantics, Float64 -> Bool,
  shaped Float64 -> Float32, unchanged source ownership, and independently
  disposed outputs.
- [x] Reject unknown dtype identifiers before LibTorch allocation with stable
  status `7064`.
- [x] Keep conversion in a focused package test and record its lifecycle and
  error evidence in the machine-readable package report.
- [x] Execute Float64 -> Float32 -> Float64 conversion on a real CUDA device,
  preserve CUDA placement through both conversions, and record the operations
  in the checked CUDA execution report.

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

### MI12: optional owned TorchScript tensor output

- [x] Extend the generic C ABI generator to recognize `Option[OpaqueType]`
  returns and map successful null output pointers to `None` without allocating
  a resource identity.
- [x] Add canonical package-helper `ok_none` and `ok_some_handle` replies that
  decode into ordinary Terlan `Option[T]` constructors.
- [x] Add `torchscript_forward_optional_tensor(model, input): Option[Tensor]`
  and require the loaded forward schema to declare an optional result.
- [x] Execute one reusable model through both branches, requiring `3.0` to
  produce a separately owned `Some(6.0)` and `0.0` to produce `None` while
  both inputs remain unchanged.
- [x] Reject an ordinary tensor-returning model with stable status `7057`.
- [x] Exercise fixture, released LibTorch, generated package, immutable Git
  consumer, and machine-readable report gates.

MI12 closes the first optional resource result without introducing a
PyTorch-specific compiler rule. Dictionaries and nested mixed outputs remain
separate value-model slices.

### MI13: owned TorchScript tensor-list output

- [x] Extend the generic C ABI generator with an owned handle-array contract
  that names the output length, temporary-array destructor, and opaque element
  type.
- [x] Generate `Vec<Tensor>` and `ok_handles` package-helper replies while
  assigning an independent VM resource identity to every returned tensor.
- [x] Add `torchscript_forward_tensor_list(model, input): List[Tensor]` and
  require the loaded forward schema to return exactly `List[Tensor]`.
- [x] Execute `[input * 2.0, input]` at input `3.0`, require values `[6.0,3.0]`,
  keep the input unchanged, and dispose both returned tensors independently.
- [x] Reject an ordinary tensor-returning model with stable status `7058`.
- [x] Exercise fixture, released LibTorch, generated package, immutable Git
  consumer, and machine-readable report gates.

MI13 establishes the generic owned-resource-list boundary required by future
dictionary and nested model results. The C array is temporary transport
storage; ownership of each non-null tensor handle transfers independently.

## LibTorch CUDA track

This track covers tensors placed on CUDA through a CUDA-enabled LibTorch
distribution. It is distinct from `terlan-cuda`, which may target the CUDA
driver/runtime through maintained Rust bindings such as `cudarc` or `cust`.
Neither package may place a CUDA dependency in the Terlan compiler or exchange
raw device pointers through Terlan values.

### CUDA0: capability model

- [x] Replace the single CUDA availability bit with independent observations:
  NVIDIA driver/device, CUDA toolkit/compiler, and CUDA-enabled LibTorch.
- [x] Do not require `nvcc` for a prebuilt LibTorch-only execution path; require
  it only when compiling package-owned CUDA source.
- [x] Report stable states such as `driver-unavailable`, `device-unavailable`,
  `toolkit-unavailable`, `libtorch-cuda-unavailable`, and `available`.
- [x] Keep the default CPU gate green for every unavailable state.

### CUDA1: synchronous device surface

- [x] Pin and verify the CUDA-enabled LibTorch `2.13.0+cu129` distribution
  separately from the CPU archive, including the required stream exports.
- [x] Validate device selection through package-owned tensor placement and
  stream construction APIs. Device count remains a `terlan-cuda` discovery
  concern until LibTorch exposes it through an admitted C contract.
- [x] Add synchronous `Tensor.to_device`, device type/index inspection, and
  blocking-copy behavior through LibTorch's stable C ABI without exposing CUDA
  pointers.
- [x] Move a tensor CPU → CUDA → CPU and verify value, dtype, shape, device type,
  and device index at every observable boundary.
- [x] Synchronize explicitly before host readback.

The CPU placement path and opt-in CUDA round trip are green on the validated
host. The CUDA lane requires a separately supplied CUDA LibTorch distribution
and driver-visible device; it does not make the CUDA archive part of the
default CPU release.

### CUDA2: CPU/CUDA operation parity

- [x] Run the PT4 `matmul` fixture on CUDA using the same public Tensor API.
- [x] Compare exact CPU and CUDA output values and shapes.
- [x] Prove deterministic destruction of source, device copy, result, and host
  readback handles.
- [x] Run this path from opt-in `make cuda-check`; never add it to the default
  CPU release requirement. Once selected, the gate fails stably when its CUDA
  LibTorch distribution or driver-visible device is unavailable.

### CUDA3: adversarial runtime and cleanup

- [x] Reject invalid device indices and stale stream handles through the
  executable CUDA adversarial gate.
- [x] Cover missing driver/device discovery, disabled CUDA capability, missing
  or incompatible CUDA LibTorch, and unresolved CUDA runtime dependencies with
  stable executable preflight and runtime gates.
Deterministic allocation-failure injection and true multi-device mismatch are
future CUDA fault-injection work. Activate them when a validated multi-GPU test
host and a non-production allocator fault hook both exist.
- [x] Preserve stable public errors for admitted stream failures without
  exposing thread-local or upstream exception payloads.
- [x] Produce a machine-readable CUDA capability report containing independent
  driver, device, toolkit, and LibTorch observations plus stable skip evidence.
- [x] Produce the passing CUDA execution report containing operation,
  synchronization, and cleanup evidence once the CUDA lane can execute.

### CUDA4: explicit streams before asynchronous copies

- [x] Introduce opaque package-owned stream resources only after synchronous
  placement and operation parity are green.
- [x] Define stream ownership, current-stream interaction, synchronization,
  and deterministic cleanup while all tensor copies remain blocking.
- [x] Reject stale streams and invalid stream devices deterministically.
Non-blocking copies remain out of scope. Cancellation, process ownership, and
unsynchronized host-readback rejection become mandatory before such an API is
admitted.

## ML package interoperability track

This track keeps package ownership directional and prevents a dependency mesh.
`terlan-ndarray` owns dense homogeneous arrays, `terlan-polars` owns columnar
tables, and `terlan-pytorch` owns tensors and model execution. Packages exchange
ordinary typed values or the versioned pointer-free tensor packet; PyTorch must
not depend on Polars.

Consumers use `Tensor.from_packet` for package-neutral TNXP values. The older
`Tensor.from_array(Bytes)` spelling remains a compatibility alias. The package
manifest explicitly depends on `terlan-ndarray`; immutable Git consumers prove
that the dependency resolves into `terlan.lock` and remains usable from cache
after both source repositories disappear. The PyTorch release closure also
builds ndarray from its current sources and executes an advanced ndarray
matrix multiplication before importing the result into PyTorch. This keeps the
runtime protocol package-neutral while preventing PyTorch releases from
silently drifting away from the canonical Terlan array implementation.

- [x] Declare ndarray in both the generated and repository package manifests,
  validate publish-safe path/Git/registry dependency metadata in the compiler,
  and prove immutable consumer resolution records ndarray in `terlan.lock`.

- [x] Execute `terlan-ndarray` Float64 export -> VM `Bytes` -> PyTorch tensor
  import across independent native helpers, preserving exact shape, dtype,
  values, and cleanup.
- [x] Execute non-constant ndarray Bool, Int64, and Float64 packets through
  PyTorch `narrow` and typed scalar extraction with exact dtype, shape, value,
  and cleanup assertions.
- [x] Execute ndarray rank-2 matrix multiplication, export its owned result,
  import it into PyTorch, and prove exact shape/value plus independent cleanup.
- [x] Reject duplicate claims, wrong consumers, stale ndarray handles,
  malformed layouts, unsupported devices, and unsupported dtypes through the
  executable ndarray interoperability gate; pair it with VM actor-exit and
  helper-failure cleanup tests.
- [x] Audit the private packet-manager deleter at exact cardinality one for
  success and rejection, and execute both ndarray-first and tensor-first
  destruction orders while proving the remaining owner stays usable.
- [x] Add Polars selected-column -> tensor-packet conversion for rectangular,
  homogeneous numeric data, with explicit null handling and copy evidence.
The ndarray rank-1/rank-2 -> Polars conversion is owned by the Polars package,
not PyTorch. Its package roadmap must preserve explicit column names, axis
mapping, dtype, and deterministic unsupported-rank rejection.
- [x] Add Polars -> ndarray conversion only when selected columns resolve to one
  compatible numeric dtype and rectangular dense shape; reject mixed, nested,
  categorical, or unresolved-null data rather than coercing silently.
- [x] Prove both direct Polars -> PyTorch packet consumption and transformed
  Polars -> ndarray -> PyTorch consumers without introducing package cycles.
  The direct Iris lane preserves `[150, 4]`; the transformed lane performs an
  ndarray transpose to `[4, 150]`. Both reduce to the exact maximum `7.9`.
- [x] Require owned pointer-free exchange for the baseline. Zero-copy remains a
  later benchmark-triggered optimization and cannot block package convergence.
- [x] Add `Tensor.to_packet` for contiguous CPU snapshots of Bool, UInt8,
  Int32, Int64, Float32, and Float64 tensors; materialize non-contiguous inputs
  in logical row-major order and copy CUDA tensors synchronously through CPU.
- [x] Prove bidirectional CUDA -> PyTorch -> CUDA packet exchange for all six
  common dtypes at shape `[2, 2]`, with explicit CUDA streams and independent
  producer, consumer, packet, and restored-buffer lifetimes.
- [x] Prove the imported DLPack manager and exported native byte temporary are
  each deleted exactly once while the resulting tensor or VM `Bytes` remains
  independently usable.
OpenCV image-matrix exchange is owned by the OpenCV package. It must reuse TNXP
before any direct OpenCV-to-PyTorch dependency is considered.

## Later value and tensor surfaces

Dictionary and nested mixed `StableIValue` results remain demand-triggered
future surfaces.
- [x] General owned float64 tensor construction from shaped Terlan data without
  leaking caller buffers or deleter callbacks.
- [x] Implement `Tensor.from_array` over the VM's pointer-free, versioned
  DLPack tensor packet and prove owned CPU float64 import, shape/value
  preservation, and truncated-payload rejection.
- [x] Prove complete `terlan-ndarray` export -> VM `Bytes` ->
  `Tensor.from_array` execution across independent package helpers, including
  exact shape, dtype, values, and cleanup. This package does not depend on
  Polars; consumers compose `DataFrame.to_array` with `Tensor.from_array`.
Do not introduce a VM one-shot tensor broker unless a measured future zero-copy
path requires move-only transfer. The current pointer-free packet is an owned
copied value and needs no broker lifetime semantics.
- [x] Add owned synchronous conversion across the admitted common scalar dtypes.
Dictionary/nested model outputs, additional serialization formats, and copied
native error detail remain future surfaces. Each requires a concrete consumer,
plus a reviewed lifetime and privacy contract for copied diagnostics.

These later surfaces must reuse the same metadata and NativeBoundary machinery;
neither CUDA nor PyTorch may introduce handwritten compiler special cases.

## Terlan-owned support automation

Repository support and validation orchestration is written in Terlan. External
programs such as `c++`, `git`, `curl`, `nm`, and `ldd` may only be launched
through bounded `std.system.Process` commands with explicit duration and output
limits.

- [x] Replace LibTorch admission, fetching, CUDA discovery, TorchScript fixture
  generation, package materialization, adversarial validation, ndarray
  interoperability, and external Git-consumer shell harnesses with Terlan
  scripts and permanent declarative fixtures.
- [x] Preserve all 86 external package-consumer scenarios as individual
  `.terl.in` programs with a Terlan-owned scenario manifest and runner.
- [x] Add `make terlan-script-check` as a hard gate that rejects repository-owned
  `.sh` support scripts while excluding vendored dependencies and build output.
- [x] Move Make-only grouped-binding, function-reference, formatting, and cleanup
  orchestration into typed `CheckTerlanLint.terls` and `Clean.terls` runners.
- [x] Replace the generated CXX surface and fixture postprocessors with
  `PatchCxxGeneratedSurface.terls` and `PatchCxxGeneratedTests.terls`; both
  active Make/CXX-check paths now run through `terlc`, with differential parity
  against the retired Python transforms on a fresh 2,829-function bind.
- [x] Add the generated `AddFlushDenormal.terls` manifest transform for
  ownership-neutral wrappers when a public PyTorch entry has no directly
  bindable equivalent.
- [x] Make generated crates consume the canonical CXX `build.rs` and Rust
  bridge, including CUDA library/header admission and relocatable runtime
  search paths; source-ownership checks now reject build-configuration drift.
- [x] Make the typed CXX runner perform LibTorch admission itself before
  regeneration, so direct `RunCxxChecks.terls` invocations cannot bypass the
  pinned version, build-hash, header, and runtime prerequisite checks.

## Generated CXX migration

The accepted native architecture is now `terlc bind cpp` plus `cxx`; manually
maintained per-operator C ABI wrappers are migration debt and are not the
target architecture. Generated ABI adaptation is allowed and is owned by the
compiler generator.

The first real LibTorch slice is executable: maintained Clang selection reads
`at::Tensor`, `at::native::ones`, and zero-argument Tensor transformations from
frozen upstream headers. The generator emits the `cxx::bridge`, `IntArrayRef`
lowering, trailing-default omission, by-value Tensor ownership conversion,
exact overload selection, exception containment, helper, Terlan module, and
lifecycle test. `make cxx-check LIBTORCH=...` compiles, links, creates a real
tensor, executes `abs`, `detach`, `neg`, `relu`, the exact Tensor and Scalar
`mul` overloads (with both integer and floating-point scalar lowering), and
`item`. It owns and independently disposes both `at::Tensor` and cross-namespace
`c10::Scalar` results. A generated specialization adapter then invokes
`c10::Scalar::to<long>()`; the behavioral gate proves `ones([1]) * 3` converts
back to exactly `3` before all resources are disposed. The generated enum path
now exposes all 46 non-sentinel `c10::ScalarType` values, constructs
`std::optional<c10::ScalarType>` for `ones_dtype`, and invokes the inherited
`TensorBase::scalar_type()` getter through the concrete Tensor resource. The
runtime gate proves a requested `Float64` tensor reports `Float64` exactly.
The same compiler-owned lowering now constructs all eight reviewed layouts,
string-constructs `c10::Device`, and supplies optional pinned-memory policy.
The complete `ones_options` signature executes without a package-authored
wrapper, while generated `layout`, `device`, `rank`, and `element_count`
projections prove `Strided`, `"cpu"`, rank `2`, and four elements exactly.
Borrowed `TensorBase::sizes()` and `TensorBase::strides()` views are now copied
through a generic generated collection adapter and must report `[2,2]` and
`[2,1]`. Generated primitive projections additionally prove defined, CPU,
CUDA, gradient, leaf, and default-format contiguity predicates. The contiguity
binding consumes the extracted C++ `MemoryFormat::Contiguous` default without
a package-authored wrapper or duplicated public argument. Exact generated
selection of `c10::Scalar::to<double>()` proves floating-point transport from a
22-member template overload family. Generic exact-symbol function and mapping
families now reduce repeated author policy for homogeneous declarations while
expanding back to independently validated normalized evidence. Six additional
Tensor property predicates prove storage, dtype-category, quantization, and
sparse-layout behavior. Generated owned-resource-list adaptation now selects
`Tensor::unbind(long)` directly, consumes its extracted `dim = 0` default or
an explicit `unbind_dim` parameter, contains exceptions, copies every returned
Tensor before atomically publishing handles, and leaves no package-authored
C++ shim. The same generic adapter now covers `chunk`, both `split` overloads,
and all three `tensor_split` overloads, including scalar, `IntArrayRef`, and
secondary-Tensor arguments with exact overload selection and extracted
defaults. Eight more exact unary Tensor methods (`ceil`, default-format
`clone`, `floor`, `reciprocal`, zero-decimal `round`, `sign`, `square`, and
`trunc`) are declared through the same homogeneous function/policy family and
executed as one ownership chain on CPU and CUDA. Generic resource-list input
lowering now binds `at::cat`, `at::stack`, their extracted default-dimension forms, and
the list-to-list free functions `at::broadcast_tensors` and both `at::meshgrid`
overloads without
package-authored C++: the generator creates the `cxx` collector, validates and
copies all handles, lowers to LibTorch's `IListRef<Tensor>` or
`ArrayRef<Tensor>`, contains failures,
and owns the result. The same generated path covers `block_diag`,
`cartesian_prod`, `column_stack`, `dstack`, `hstack`, and `vstack` through one
declarative function/policy family. Repeated handles work, and CPU/CUDA tests dispose inputs
before reading the result. Broadcast outputs are also copied completely before
any public handle is published. Five corresponding `*_out` functions now use a
compiler-generated mutable free-function adapter: list inputs are copied before
the unique mutable borrow, `Tensor&` results are discarded as aliases, and all
exceptions are contained behind stable package policy. That same generated path
now covers the joining aliases `concat`, `concatenate`, and `row_stack`, their
exact output functions, and extracted default-dimension forms for the applicable
APIs. Public default arguments now preserve the existing callable names and
signatures rather than creating separate default-suffixed aliases. Declarative
cross-module type imports let `pytorch.TensorJoining` reuse the one
`pytorch.Tensor` resource identity and the same generated helper/handle
table. The generated path now also owns the complete 13-function
`pytorch.ShapePartition` contract: `chunk`, both `split` forms, all three
`tensor_split` forms, `unbind`, and scalar/list forms of `dsplit`, `hsplit`,
and `vsplit`. Exact free functions whose upstream first argument is named
`self` are escaped by the compiler generator rather than by package wrappers.
The twenty-two-function `pytorch.Regularization` module now selects upstream
`alpha_dropout`, `dropout`, `feature_dropout`, `native_dropout`,
`native_dropout.out`, `native_dropout_backward`, `native_layer_norm`,
`native_layer_norm.out`, `native_group_norm`, `native_group_norm.out`,
`native_group_norm_backward`, `native_group_norm_backward.out`,
`native_layer_norm_backward`, `native_layer_norm_backward.out`,
`native_batch_norm`, `native_batch_norm.out`,
`native_batch_norm_backward`, `native_batch_norm_backward.out`,
`log_sigmoid_forward`, `log_sigmoid_forward.output`,
`log_sigmoid_backward`, and `log_sigmoid_backward.grad_input`
directly instead of routing them through the legacy
dispatcher. The
five-function `pytorch.DropoutCompletion` module also binds its functional and
in-place variants directly. The maintained extractor now derives `in_out` from
unannotated non-const C++ references, while the bridge renderer escapes the
reserved CXX `self` spelling for mutable free functions. The six-function
`pytorch.GluCompletion` module now proves generated copying of
secondary Tensor resources before retained-output mutation, including forward,
backward, JVP, and backward-JVP calls. The generated path now also owns all
eight `pytorch.Spatial` affine-grid and grid-sampler
contracts, exercising copied Tensor inputs and integer-list dimensions in both
functional and retained-output calls. The current real LibTorch migration
package also owns the complete 102-contract `pytorch.Binary` surface directly
from extracted C++ metadata, including aliases, retained outputs, in-place
forms, and dual-output dimension reductions. Representative arithmetic,
logical, remainder, mutation, reduction, shape, and CPU/CUDA device behavior
are covered by the generated CXX gates. The package contains 2760 generated
functions over 2351 exact metadata symbols and
classifications. The latest source-only promotion batch closes 36 public
low-level and internal callable contracts, including exact generated CXX
wrappers for `Tensor.chalf`, `unsafe_chunk`, and `scalar_tensor`.
It also materializes all twelve Tensor dtype convenience aliases
(`float`, `double`, `half`, `int`, `long`, `bool`, `bfloat16`, `byte`,
`cdouble`, `cfloat`, `char`, and `short`) over the validated `to_dtype` CXX
leaf, plus overloaded `Tensor.cpu` and `Tensor.cuda` placement aliases over
`to_device`, a Tensor-level `to_sparse_coo` alias over the generated sparse
leaf, and its matching `to_dense` alias. CUDA stream recording remains
exposed through the typed `pytorch.Cuda.record_stream` leaf so the opaque
`Stream` ownership boundary is preserved.
The same compatibility layer now materializes `ndimension` and `nelement`
over the generated rank and element-count leaves, plus `type_as` and the
default `is_pinned` form. The CUDA behavior test now exercises the public
placement and stream aliases directly. A dedicated Tensor view-alias test now
routes `adjoint`, `conj`, `detach`, `positive`, `ravel`, and `t` through the
generated View CXX leaves, while `resolve_conj`, `resolve_neg`, and
`get_device` are exposed over their exact view/type-relation leaves.
The reviewed CXX compatibility surface now also exposes the common
`Tensor.to` overloads for dtype conversion and device placement, while the
existing generated `to_other` leaf covers tensor-copy conversion;
`TensorToAliasParityTest.terl` exercises all three paths and the public
inventory promotes `torch.Tensor.to(?)` from non-applicable to implemented
with executable evidence.
The same generated leaves now back the common floating-point `Tensor.clip`
overload and tensor `add_`/`sub_`/`subtract_`/`multiply_` in-place aliases; a dtype-based
`Tensor.element_size` projection covers the scalar storage-width contract.
The parity and CXX behavior fixtures verify clamp bounds, mutation, restoration,
and element width without adding a handwritten ABI surface. The existing random
in-place and scalar clamp CXX leaves also cover the frozen `Tensor.random_`,
`clamp_`, `clamp_min_`, `clamp_max_`, and `clip_` exports. The public inventory
consequently reports 1373 implemented callables, 1 equivalent, 2051
non-applicable, and 0 missing.
`make cxx-cuda-check LIBTORCH=...` independently builds that
generated package against frozen cu129 LibTorch and executes CUDA tensor
creation, placement/property predicates, arithmetic, scalar extraction, and
resource disposal on device 0. Its ownership gate disposes the source tensor
before inspecting and disposing the returned CUDA slices; this is generated-CXX
CUDA evidence rather than evidence inherited from the legacy C ABI package.

The first partial migration inside the large `pytorch.Tensor` module now moves
its package-neutral interchange boundary as well: `from_packet`, the
`from_array` compatibility alias, and `to_packet` select two exact
`terlan::tensor_packet` declarations from maintained Clang metadata and cross a
compiler-generated CXX bridge. No package-authored C ABI is used for this
path. Its pointer-free TNXP codec owns imported tensor storage and exported
bytes, validates layout bounds, contains malformed-input failures before the
`noexcept` bridge, and synchronously snapshots CUDA tensors to CPU. Three
direct generated-crate tests exercise contiguous ownership round trips,
zero-stride import, and malformed-packet containment against both LibTorch
distributions. Two additional direct generated-crate tests validate dtype and
tagged-Scalar type relations, including invalid enum containment. The Terlan
execution gates now include 79 tests on CPU and 77 on CUDA, including actual
CUDA-to-owned-CPU packet exchange. These three partial Tensor contracts are not
counted in the 891 completed-module operations below.

All 21 existing `pytorch.TensorJoining`, both `pytorch.Distance`, all 13
`pytorch.ShapePartition`, all four `pytorch.Regularization`, and all five
`pytorch.DropoutCompletion`, plus all six `pytorch.GluCompletion` public
contracts, all eight `pytorch.Spatial` contracts, all ten
`pytorch.ShapeGradient` contracts, and all 12
`pytorch.PaddingBackwardCompletion` contracts, plus all nine
`pytorch.IntegrationCompletion` contracts and all three `pytorch.NanToNumNative`
contracts, all 12 `pytorch.Strided` contracts, and all six
`pytorch.IndexFillNative` contracts, all 11 `pytorch.FillNative` contracts,
all seven `pytorch.IndexMutationNative`
contracts, all five `pytorch.LerpNative` contracts, and all five
`pytorch.WhereNative` contracts now have exact CXX replacements.
All 32 `pytorch.ActivationCompletion` contracts now bind the 16 exact
hardshrink, softshrink, softplus, and threshold forward, backward, and output
overloads across integer and floating public Scalar forms.
All 36 `pytorch.ExtendedActivationCompletion` contracts now bind 18 exact
CELU, ELU, hardtanh, and leaky-ReLU functional, retained-output, in-place, and
backward declarations across the same two public Scalar forms.
All 28 `pytorch.Loss` contracts now bind exact embedding, ranking, regression,
likelihood, and triplet loss declarations, including retained-output and
explicit backward forms. The four internal `nll_loss_forward`/
`nll_loss_backward` forms (functional and retained-output) are validated against
the public autograd result in `tests/pytorch/NllLossInternalParityTest.terl`.
The corresponding four `nll_loss2d_forward`/`nll_loss2d_backward` forms are
validated against the 2-D public autograd result in
`tests/pytorch/NllLoss2dInternalParityTest.terl`.
The internal `nll_loss_nd` schema is likewise validated against the same
autograd result in `tests/pytorch/NllLossNdInternalParityTest.terl`.
The internal rrelu-with-noise backward pair and multi-margin-loss backward pair
are validated against deterministic CPU slopes and public autograd gradients in
`tests/pytorch/RreluWithNoiseBackwardParityTest.terl`.
The `batch_norm_update_stats` tuple and retained-output forms are also bound
through exact CXX wrappers and validated against CPU reference statistics.
The 2-D grid-sampler backward functional and retained-output forms are now
bound through exact CXX wrappers and validated against one another on CPU.
The corresponding 3-D functional and retained-output backward forms are also
bound and validated on CPU. The functional and retained-output
`multilabel_margin_loss_backward` forms are now bound through exact CXX
wrappers and validated against one another on CPU, including the required
Long-target and floating workspace contracts.
The functional and retained-output `convolution_backward` forms are now
bound through exact CXX wrappers, including optional bias-size lowering and
all three output-mask controls, and validated against one another on CPU.
The generic functional and retained-output `convolution` forms are now
bound through exact generated CXX wrappers with optional-bias lowering and
validated against one another on CPU.
The matching functional and retained-output `multilabel_margin_loss_forward`
tuple forms are now bound through exact generated CXX wrappers and validated
against one another on CPU.
The internal `nested_to_padded_tensor` form is now bound through an exact
generated CXX wrapper with optional output-size lowering and CPU validation.
The CPU-backed `lstm_cell` form is now bound through an exact generated CXX
recurrent leaf with tensor-list hidden-state and optional-bias lowering, and
validated on CPU for biased and bias-free forms.
The CPU-backed `rnn_tanh_cell` form is now bound through the same exact
generated CXX recurrent surface with optional-bias lowering, and validated on
CPU for biased and bias-free forms.
The CPU-backed `rnn_relu_cell` form is now bound through that recurrent
surface as well, with both optional-bias forms validated through generated CXX
on CPU.
The CPU-backed `gru_cell` form is now bound through the recurrent surface with
the same optional-bias lowering and validated through generated CXX on CPU.
The CPU-backed `rnn_tanh.input`, `rnn_relu.input`, `gru.input`, and `lstm.input`
sequence forms are now bound through the generated recurrent surface with
tensor-list parameter/hidden-state lowering and validated together on CPU.
The CPU-backed `rrelu_with_noise_functional` tuple form is now bound through
the generated random surface with optional-generator lowering and validated
against its deterministic evaluation behavior on CPU.
The CPU-backed `rrelu_with_noise` value and retained-output forms are now
bound through generator-aware generated CXX wrappers and validated together on
CPU, including the mutable noise contract.
The CPU-backed `nonzero_numpy` tensor-list form is now bound through the
generated `WhereNative` resource-list surface and validated for coordinate
tensor ownership, shape, and values on CPU.
The CPU-backed `slow_conv3d_forward` implementation-only alias is now
classified against the exact generated `slow_conv3d` leaf and its CPU evidence.
Its retained-output `slow_conv3d_forward.output` form is also bound through a
generated mutable CXX leaf and validated against the functional result.
The retained-output forms for `slow_conv3d`, `slow_conv_dilated2d/3d`, and
`slow_conv_transpose2d/3d` are now bound through the same generated mutable
CXX convolution surface and validated against their functional counterparts.
The CPU backend-availability predicates `is_vulkan_available` and
`cudnn_is_acceptable` are now exposed through a generated CXX module and
validated against the pinned CPU LibTorch result.
The CPU-backed `sym_is_contiguous` predicate is now bound through the same
generated backend-availability CXX module and validated for the contiguous CPU
memory format.
The retained-output `thnn_conv2d.out` schema is now bound through the generated
low-level convolution CXX module and validated against its functional result on
CPU.
The autograd-internal `linear_backward` tuple family was probed directly against
the pinned CPU LibTorch dispatcher and remains explicitly non-applicable because
it exposes no executable CPU result; high-level linear behavior remains covered.
The CPU-backed `thnn_conv2d` legacy schema is now bound through a generated
low-level CXX convolution module with optional-bias lowering and validated on
CPU.
The CPU-backed `slow_conv3d` and `slow_conv_dilated2d` legacy schemas are now
bound through the same generated low-level CXX convolution module with
optional-bias lowering and validated together on CPU.
The CPU-backed `slow_conv_dilated3d`, `slow_conv_transpose2d`, and
`slow_conv_transpose3d` legacy schemas are now bound through that module as
well, with optional-bias and output-padding lowering validated together on CPU.
The `value_selecting_reduction_backward` implementation-only schema is now
bound through an exact generated CXX leaf and validated against its expected
CPU scatter gradient.
All 18 `pytorch.ShapeCompletion` contracts now bind exact channel shuffle,
transpose/select/narrow, unfold, trace-gradient, patch transform, contraction,
and zero declarations. CPU executes every contract; CUDA executes all 17 with
kernels in the frozen distribution, while `narrow_copy.out` retains its pinned
no-CUDA-kernel rejection contract.
All 20 `pytorch.Indexing` contracts now bind exact ordering, bucketization,
diagonal, indexed copy/selection, membership, masking, scatter, conditional
selection, copied-window, retained-output, and in-place declarations. Every
contract executes through generated CXX on both CPU and CUDA.
All 17 `pytorch.Gradients` contracts now bind 15 exact cumulative-product,
embedding, fake-quantization, gather, GELU, logit, GLU, masked-selection, and
soft-margin gradient declarations. CPU and CUDA execute dense and sparse
embedding routes, omitted/present logit epsilon, string mode, and all retained
outputs through generated CXX.
All 47 `pytorch.Dense` contracts now bind exact dense arithmetic, linear
algebra, activation-gradient, complex construction, special-function,
shape/type matching, inherited Tensor-method, and retained-output declarations.
Every contract executes through generated CXX on both CPU and CUDA.
All 21 `pytorch.FunctionalCompletion` contracts now bind exact linear,
log-softmax, group-normalization, finite-difference, and masked-scatter
backward declarations. The compiler supplies optional Tensor values, inserts
omitted middle optionals, and converts compatibility integer dtypes through
the extracted enum; every contract executes through generated CXX on CPU and
CUDA.
All four `pytorch.PaddingNative` contracts now bind exact constant-pad and
mode-based pad declarations. Generated adapters preserve both historical
tagged-Scalar name layouts, optional fill values, and retained output; every
contract executes through generated CXX on CPU and CUDA.
All 16 `pytorch.BinaryCrossEntropyCompletion` contracts now bind the six exact
probability, backward, logits, and retained-output declarations. Generated
optional-resource alignment covers absent, element-weight-only,
positive-weight-only, and both-weight calls, including ATen's abbreviated
`pos_weight` spelling; every contract executes through generated CXX on CPU
and CUDA.
All 36 `pytorch.Division` contracts now bind exact `div`, `divide`, and
`true_divide` Scalar/Tensor declarations across functional, retained-output,
rounding-mode, and logically-const in-place forms. The compiler constructs
present optional string views for explicit rounding modes; every contract
executes through generated CXX on CPU and CUDA.
All 20 `pytorch.SliceNative` contracts now bind exact slice-view, copied-slice,
retained-output, and slice-scatter declarations. Generated optional-integer
alignment covers absent, start-only, end-only, and fully bounded calls; every
contract executes through generated CXX on CPU and CUDA.
All 44 `pytorch.Fft` contracts now bind 42 exact discrete, Hermitian, real,
frequency-grid, and frequency-shift declarations. Generated adapters lower
ordinary and optional dimension ArrayRefs, preserve public integer-list and
normalization defaults, and select overloaded factories without an author
shim; every contract executes through generated CXX on CPU and CUDA where its
public placement contract permits it.
All 56 `pytorch.Pooling` contracts now bind 48 exact adaptive, average,
maximum, fractional-maximum, backward, unpool, and retained-output declarations.
Generated owned tuple carriers, explicit required-optional absence, and dual
retained-output borrowing execute across the full contract set on CPU and CUDA.
All 18 `pytorch.Selection` contracts now bind exact cumulative extrema,
dimension reduction, ranked selection, median, mode, sorting, and top-k
declarations. Ten owned tuple and eight dual retained-output calls execute on
both devices through one shared workload.
All five `pytorch.TensorListTransforms` contracts now bind exact broadcasting,
mesh-grid, dense flatten, and dense unflatten declarations. Compiler-generated
`prepend_resource` lowering combines the stable public first Tensor plus
remainder list into one copied `TensorList`; all contracts execute on CPU and
CUDA without a package-authored wrapper.
All 12 `pytorch.NumericalCompletionNative` contracts now bind exact stable-sort
output, pairwise-distance, matrix-chain, clone-output, contiguity, and
cross-product declarations. Compiler-generated direct and optional integer-enum
lowering targets extracted `c10::MemoryFormat` metadata; every contract executes
on CPU and CUDA without a package-authored wrapper.
All 11 `pytorch.LinearStorageCompletion` contracts now bind exact bilinear,
temporal convolution, cumulative-extrema gradient, and storage-rebinding
declarations. Generated tuple ownership atomically publishes all three
`conv_tbc_backward` results; every contract executes on CPU and CUDA without a
package-authored wrapper.
All 16 `pytorch.OutputCompletion` contracts now bind exact complex, distance,
expand-copy, legacy LU/QR, dropout-gradient, polar, reflection/replication
padding, renormalization, determinant, and softmax retained-output
declarations. Exact upstream parameter spellings, inferred optional dtype,
and dual mutable outputs are compiler-generated; numeric, shape, dtype,
source-preservation, and placement checks execute on CPU and CUDA without a
package-authored wrapper.
All 34 `pytorch.View` contracts now bind exact alias, detachment, lift,
minimum-rank, transpose, conjugation, real/imaginary, and real/complex storage
view declarations. Generated family expansion selects both free functions and
Tensor methods, preserves acronym-bearing upstream names, and lowers the three
sequence overloads through compiler-owned Tensor-list collection; every
contract executes on CPU and CUDA without a package-authored wrapper.
All 18 `pytorch.ConvolutionNative` contracts now bind nine exact one-, two-,
and three-dimensional regular, string-padding, and transposed-convolution
declarations. Six generated function families project absent and present bias
forms through upstream `std::optional<Tensor>` parameters; shape, dtype,
numeric, source-preservation, and placement checks execute on CPU and CUDA
without a package-authored wrapper.
All 25 `pytorch.TypeRelationsNative` contracts now bind exact comparison,
property, storage-identity, geometry, device, dtype-promotion, four
`result_type`, and gradient-retention declarations. Compiler-generated
parameter permutation maps the public Tensor-first scalar overload to the
upstream Scalar-first declaration, while reviewed enum guards contain invalid
historical integer dtype values before C++ invocation. Every contract executes
on CPU and CUDA without a package-authored wrapper.
All 32 `pytorch.CorrectionStatisticsCompletion` contracts now bind exact
correction-based `std`, `var`, `std_mean`, and `var_mean` declarations.
Compiler-generated adapters lower omitted and present optional dimensions and
integer/float corrections, own paired Tensor returns atomically, and safely
borrow both retained outputs. Every functional, paired-return, and
retained-output contract executes numerically on CPU and CUDA without a
package-authored wrapper.
All 12 `pytorch.RepeatInterleaveNative` contracts now bind four exact upstream
index, retained-output, tensor-count, and scalar-count declarations. Generic
`cpp_ignore` lowering keeps compatibility-only positional values in the frozen
public contract while omitting them from generated CXX declarations and exact
LibTorch calls. Every optional dimension/output-size state executes on CPU and
CUDA without a package-authored wrapper.
Strided
operations prove generated optional offset lowering and logical-const metadata
mutation. Index fill proves compiler-generated tagged Scalar choice lowering,
exact Tensor overload selection, and compiler-generated logically-const method
mutation without a package-authored operator wrapper. Lerp extends the same
proof through functional, retained-output, and in-place Scalar/Tensor overloads
with a secondary Tensor resource.
WhereNative proves exact true-index list ownership, repeated tagged-Scalar
lowering in one call, mixed Scalar/Tensor selection, and retained-output
mutation on CPU and CUDA.
FusedArithmeticNative preserves its three public selector contracts as
generated Terlan composition over 25 private exact CXX leaves. All seven
functional, retained-output, and in-place schemas execute numerically on CPU
and CUDA, including CUDA-only half-input/float-accumulation dtype forms for
`addmm` and `baddbmm`, without a package-authored dispatch wrapper.
ScaledArithmeticNative likewise preserves nine public selector contracts as
generated Terlan composition over 16 private exact CXX leaves. Exact
Tensor/Scalar `add`, `sub`, and `rsub` functional and retained-output overloads,
plus supported `add_` and `sub_` mutations, execute on CPU and CUDA with both
integer and floating tagged Scalars and no package-authored dispatch wrapper.
ClampNative preserves six public selector contracts as generated Terlan
composition over 18 private exact CXX leaves. Scalar-bound presence and
integer/float choices plus all three tensor-bound modes execute across
functional, retained-output, and in-place forms on CPU and CUDA without a
package-authored C++ wrapper.
AggregationNative preserves seven public selector contracts as generated
Terlan composition over 11 private exact CXX leaves. Optional bincount weights,
integer/float histogram bounds, and the `prod`, `mean`, `amax`, and `amin`
indexed-reduction choices execute across functional, retained-output, and
in-place forms on CPU and CUDA, including both `include_self` states, without a
package-authored C++ wrapper.
ScatterNative now applies the same generated composition to six public selector
contracts over 18 exact CXX leaves, covering plain, additive, legacy-reduced,
and modern-reduced tensor/scalar scatter. The focused CPU contract executes
through the generated helper; CUDA package assembly is wired to the same façade.
RangeNative now binds its 14 packed constructor contracts through 29 exact CXX
leaves covering `arange`, `eye`, `linspace`, `logspace`, and legacy `range`.
The focused CPU contract executes packed optional dtype/layout/device lowering
through the generated helper.
SpecialScalarNative now binds eight scalar-left/right and retained-output
dispatchers through 112 exact CXX leaves covering all 14 special scalar
operations, including polynomial and logarithmic/zeta families. Its focused
contract validates integer and floating scalar forms, output mutation, shape,
dtype, and device placement through the generated helper.
CreationCompletionNative now binds four dense creation/like dispatchers over 16
exact CXX leaves for empty, full, ones, zeros, and all retained-output forms.
Its focused CPU contract validates configured dtype/device creation and output
mutation through the generated helper.
BinaryScalarNative now binds twelve tensor/scalar, retained-output, and in-place
dispatchers over 108 exact CXX leaves for bitwise, arithmetic, remainder, power,
floor-division, xlogy, fmod, and copysign operations. Its focused CPU contract
validates tensor/scalar direction, mutation, output storage, shape, dtype, and
device placement through the generated helper.
LookupNative now binds ten public membership, sorted-search, and bucketization
dispatchers over 12 exact CXX leaves. Its focused CPU contract exercises tensor
and scalar forms, retained outputs, optional side/sorter lowering, shape, dtype,
and device placement through the generated helper; CUDA package assembly is
wired to the same façade.
UnaryStorageNative now binds all 156 retained-output and in-place unary
contracts through 156 exact CXX leaves, including trigonometric, logical,
rounding, reduction, backward, alias, and tensor-copy families. Its focused CPU
contract validates representative output mutation, in-place storage, paired
`frexp` outputs, reductions, shape, dtype, and device placement; CUDA package
assembly is wired to the same façade.
Unary now binds all 57 immutable unary, predicate, reduction, coordinate, and
tuple-returning `frexp` contracts through exact ATen free functions, including
the previously absent `argwhere` and `isfinite` metadata declarations. The
generated CPU and CUDA CXX boundaries compile against the frozen LibTorch
distributions.
Shape now binds all 82 shape, view, permutation, padding, indexing, and
retained-output contracts through exact ATen functions and Tensor methods,
including compiler-selected integer and integer-list overloads.
Special now binds all 141 mathematical, matrix, polynomial, scalar-overload,
retained-output, and in-place contracts through exact ATen functions and Tensor
methods, including optional epsilon and dtype lowering. Representative CPU and
CUDA behavior is covered by the generated CXX gates.
Statistics now binds all 87 statistical, reduction, norm, cumulative, and
retained-output contracts through exact ATen functions, including the complete
24-family out-parameter surface. The compiler materializes omitted trailing
optional C++ parameters as explicit absent values for exact overload calls;
CPU and CUDA generated packages both compile successfully.
CorrectionStatisticsPairNative now binds all 16 legacy TensorPair contracts
through generated ownership around LibTorch's native `std::tuple<Tensor,
Tensor>` return carrier. The package-owned alias preserves the exact public
TensorPair contract without a handwritten operator ABI; CPU and CUDA generated
packages compile successfully.
Sparse now binds all 55 sparse-dimension, conversion, indexing, resize,
constructor, and sampled-addmm contracts through extracted ATen dispatchers
and package-owned C++ option adapters; generated CPU and CUDA packages both
compile successfully.
Linalg now binds all 153 factorization, decomposition, solve, inverse, rank,
norm, least-squares, and tensor-equation contracts through extracted ATen
callables and package-owned C++ option adapters; generated CPU and CUDA
packages both compile successfully.
IndexMutationNative extends the same generated adapters across indexed
addition, copying, selection output, selection backward, Scalar alpha, and
logically-const in-place methods with exact CPU and CUDA numeric evidence.
FillNative proves structural tagged-Scalar prefix matching plus direct
functional, retained-output, and in-place full, diagonal, and masked filling.
`bindings/cxx/migration.json` and `bindings/cxx/contract-snapshot.json` record
completed modules and cap the reviewed contract at 1,760 operations. That
denominator includes all 1,573 direct module declarations plus 187 expanded
function-family members; `scripts/check-cxx-migration.py` fails on contract
drift or snapshot growth. The current migration coverage is therefore
1,760/1,760 operations (100%), with zero remaining legacy operations; the
final four Tensor TorchScript collection/optional contracts are now compiled
through CXX alongside the other Tensor contracts. The primary generated
package now uses that single CXX handle world; the CXX contract snapshot is the
retained audit artifact.

The full `make cxx-check` orchestrator now completes successfully after
regenerating the CXX package and compiling/running the complete generated CPU
fixture tree. The CUDA gate remains dependent on a driver-visible GPU; the
pinned CUDA package report preserves the last successful RTX A4500 execution
evidence.
Package-level public fixtures that use `List.length()` now import the standard
Terlan `List` module explicitly, and `make package-check` runs its generated
test tree incrementally so every fixture shares the same compiler-owned cache.

The migration sequence is:

- [x] Add compiler-owned CXX generation for resources, lists, public defaults,
  mutable output functions, exceptions, and cross-module type imports.
- [x] Migrate the first complete modules (`pytorch.ActivationCompletion`,
  `pytorch.ExtendedActivationCompletion`,
  `pytorch.TensorJoining`,
  `pytorch.Distance`, `pytorch.ShapePartition`, `pytorch.Regularization`, and
  `pytorch.DropoutCompletion`, plus `pytorch.GluCompletion`) with exact CPU and
  CUDA behavioral evidence and a machine-readable non-regression gate. The
  same gate now includes `pytorch.Spatial`, `pytorch.ShapeGradient`, and
  `pytorch.PaddingBackwardCompletion`, plus `pytorch.IntegrationCompletion` and
  `pytorch.NanToNumNative`, `pytorch.Strided`, `pytorch.FillNative`,
  `pytorch.IndexFillNative`,
  `pytorch.IndexMutationNative`, `pytorch.LerpNative`, and
  `pytorch.WhereNative`, plus `pytorch.Loss`, `pytorch.ShapeCompletion`, and
  `pytorch.Indexing`, `pytorch.Gradients`, `pytorch.Dense`, and
  `pytorch.FunctionalCompletion`, plus `pytorch.PaddingNative` and
  `pytorch.BinaryCrossEntropyCompletion`, `pytorch.Division`, and
  `pytorch.SliceNative`, `pytorch.Fft`, `pytorch.Pooling`, and
  `pytorch.Selection`, `pytorch.TensorListTransforms`, and
  `pytorch.NumericalCompletionNative`, and
  `pytorch.LinearStorageCompletion`, `pytorch.OutputCompletion`, and
  `pytorch.View`, `pytorch.ConvolutionNative`, and
  `pytorch.TypeRelationsNative`, and
  `pytorch.CorrectionStatisticsCompletion`, and
  `pytorch.RepeatInterleaveNative`, `pytorch.FusedArithmeticNative`, and
  `pytorch.ScaledArithmeticNative`, `pytorch.ClampNative`, and
  `pytorch.AggregationNative`.
  The gate also now includes the complete `pytorch.Unary` and `pytorch.Shape`
  modules generated from exact ATen metadata.
- [x] Migrate the remaining existing modules, adding generic generator lowering
  shapes only when an existing public contract requires them.
- [x] Switch the primary package and all consumers to the generated-CXX helper.
- [x] Remove the handwritten C ABI wrapper sources and gate the active package
  against their reintroduction; retain only the generated CXX contract snapshot
  and isolated stable-C fixture needed for contract/negative compatibility tests.
- [x] Delete the historical C ABI manifest and retire its migration-only audit;
  the equivalent CXX contract snapshot is now archived and active.
- [x] Drive both frozen inventories to zero missing entries for the supported
  pinned LibTorch surface; the remaining PP1-PP7 work is semantic expansion,
  not inventory classification.
- [ ] Continue PP1-PP7 semantic expansion for additional frontend and backend
  behavior beyond the currently supported classification.
