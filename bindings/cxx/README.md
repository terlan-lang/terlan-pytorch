# Generated CXX binding input

This directory contains include-only inputs for Terlan's maintained C++
metadata extractor. It contains no ABI wrapper and no per-operator C++
implementation. `terlc bind cpp` owns the generated `cxx::bridge`, C++ value
ownership adapter, `ArrayRef` lowering, default-argument omission, exception
containment, Rust helper, Terlan module, and lifecycle test.

The initial executable slice selects `at::Tensor`, `at::native::ones`, and the
`abs`, `detach`, `neg`, and `relu` Tensor methods from the frozen LibTorch
headers. The extractor's repeatable `--qualified-symbol` option keeps the
metadata inventory limited to those exact declarations.

The executable slice also selects both `Tensor::mul` overloads and the
non-template `Tensor::item`. Generated adapters disambiguate Tensor versus
Scalar multiplication, lower Terlan `Int` and `Float` into `c10::Scalar`, and
own the cross-namespace `c10::Scalar` item result without an author shim.
The exact `Scalar::to<long>()` specialization is selected by stable Clang ID
and invoked through generated exception containment; the behavior test checks
the resulting integer value, not only handle allocation.
The generated `DType` surface covers all 46 non-sentinel `c10::ScalarType`
values. `ones_dtype` lowers the reviewed atom through generated
`std::optional<c10::ScalarType>` construction, while the inherited
`TensorBase::scalar_type()` getter is generated as `dtype`; runtime evidence
asserts that a requested `Float64` tensor reports `Float64` exactly.
The same generated option lowering covers all eight non-sentinel `c10::Layout`
values, string-constructed `c10::Device`, and optional `bool`. The full
`ones_options` call supplies dtype, layout, device, and pinned-memory policy.
Generated projections then observe `Float64`, `Strided`, and canonical device
spelling `"cpu"` through `TensorBase` getters and `Device::str()`.

The generated slice also uses generic contained primitive projections for the
inherited `TensorBase::dim()` and `TensorBase::numel()` methods. The behavior
gate requires a `[2, 2]` tensor to report rank `2` and element count `4`.
Generated collection adapters copy the borrowed results of
`TensorBase::sizes()` and `TensorBase::strides()`; the same tensor must report
shape `[2, 2]` and contiguous strides `[2, 1]`. Generated Boolean projections
prove that the tensor is defined, contiguous in the C++ default memory format,
on CPU, not on CUDA, does not require gradients, and is an autograd leaf. The
`is_contiguous` adapter omits the extracted trailing `MemoryFormat` default, so
the package manifest does not restate or hand-code that C++ default. Exact
selection of `c10::Scalar::to<double>()` from its 22-member template overload
family proves generated floating-point projection without a specialization
shim. Homogeneous exact-symbol and policy families now declare the unary
Tensor transforms and six additional Tensor property predicates without
repeating complete function or classification objects; generated normalized
evidence still expands every member independently.
The same compiler-owned collection machinery now handles owned resource
lists. `Tensor::unbind` is selected directly from Clang metadata, including its
trailing `dim = 0` default, and `unbind_dim` exposes the explicit parameter.
Generated C++ contains exceptions and owns the returned vector; generated Rust
copies every Tensor successfully before publishing any handles. CPU and CUDA
tests dispose the source first, then inspect and independently dispose every
returned Tensor.
Without changing that adapter, the slice also binds the exact `chunk`, both
`split`, and all three `tensor_split` overloads. Their generated paths cover
scalar counts, borrowed integer lists, overload selection, extracted defaults,
and a separately validated Tensor argument. The CPU gate executes all six;
the CUDA gate exercises chunking and index-based tensor splitting.
The homogeneous unary family now additionally selects `ceil`, default-format
`clone`, `floor`, `reciprocal`, zero-decimal `round`, `sign`, `square`, and
`trunc`. One family declaration supplies the public shape and one policy family
supplies containment; each exact Clang symbol remains independently visible in
normalized evidence. CPU and CUDA gates execute the complete chain.
Generic resource-list input lowering now also selects `at::cat` and `at::stack`
directly from their generated ATen headers. The package supplies only upstream includes, extracted
declaration, and public policy; Terlan generates the `cxx` collector, validates
and copies every handle (including repeated handles), lowers the collector to
LibTorch's `IListRef<Tensor>` or `ArrayRef<Tensor>`, contains exceptions, and owns the result. CPU
and CUDA gates dispose the inputs before inspecting the concatenated tensor.
The same collector and owned-list result machinery now compose for
`at::broadcast_tensors`: a free function consumes `List[Tensor]`, returns
`List[Tensor]`, and publishes no result handle until every returned Tensor has
been copied successfully.
Both `at::meshgrid` overloads use that list-to-list path. The explicit overload
also proves generic `String` to `c10::string_view` lowering through `rust::Str`,
with `xy` and default indexing behavior checked by exact output shapes.
The same declarative list-input path now binds `block_diag`, `cartesian_prod`,
`column_stack`, `dstack`, `hstack`, and `vstack` as one public function and
policy family. Terlan expands the family into six independently visible exact
symbols and generates every C++ and Rust adapter.
Five corresponding `*_out` declarations exercise generated mutable free-function
adaptation. Package metadata identifies the mutable `Tensor&`; Terlan generates
the pinned CXX parameter, stable exception envelope, alias-result disposal, and
handle-table mutation without an author-written output shim.
The same list-input and mutable-output machinery now covers the complete joining
alias/default slice selected here: `concat`, `concatenate`, and `row_stack`, plus
`cat_out`, `stack_out`, `concat_out`, `concatenate_out`, and `row_stack_out`.
`pytorch.TensorJoining` imports the single `pytorch.Tensor` handle type;
both modules therefore use one generated Rust/CXX helper and one typed handle
table. Public `dimension: Int = 0` defaults match the existing package contract,
while extracted C++ defaults remain the source of omitted native arguments.

The complete module migrations include all 21 existing
`pytorch.TensorJoining` functions, both `pytorch.Distance` functions, and all 13
`pytorch.ShapePartition` functions with exact generated-CXX replacements and no
package-authored adapter implementation. All four `pytorch.Regularization`
functions and all five `pytorch.DropoutCompletion` functions are likewise bound
directly to their upstream ATen declarations. The maintained extractor derives
`in_out` from the latter module's ordinary non-const C++ references, and the
compiler generates their mutable alias containment. All six
`pytorch.GluCompletion` contracts now use direct calls as well; generated CXX
copies their secondary Tensor inputs before borrowing retained outputs.
All eight `pytorch.Spatial` affine-grid and grid-sampler contracts reuse that
same generated secondary-resource and integer-list lowering.
All ten `pytorch.ShapeGradient` diagonal, select, slice, and unfold contracts
reuse those compiler-generated integer-list and retained-output adapters as
well.
All 12 `pytorch.PaddingBackwardCompletion` reflection and replication backward
contracts reuse the same adapters across 1D, 2D, and 3D inputs.
All nine `pytorch.IntegrationCompletion` cumulative trapezoid, trapezoid,
`trapz`, and KL-divergence contracts are direct too; the public integer and
floating spacing functions deliberately share extracted C++ `Scalar`
overloads.
All three `pytorch.NanToNumNative` contracts use compiler-generated primitive
optional presence/value lowering across functional, retained-output, and
in-place calls.
All 12 `pytorch.Strided` contracts use exact optional storage-offset lowering
across view, copy, metadata-mutation, retained-output, and scatter paths.
All six `pytorch.IndexFillNative` contracts use exact Scalar/Tensor overload
selection; the tagged integer-or-float Scalar choice and logically-const
`Tensor::index_fill_` mutations are lowered and contained by the compiler.
All five `pytorch.LerpNative` contracts likewise use exact Scalar/Tensor
overload selection across functional, retained-output, and in-place calls.
They prove that logically-const mutable methods can copy a secondary Tensor
resource while the compiler constructs the tagged integer-or-float `Scalar`.
All five `pytorch.WhereNative` contracts use exact overloads for true-index
extraction, Scalar/Scalar and mixed Scalar/Tensor selection, and retained
output. The two-Scalar call proves that tagged Scalar lowering is repeatable
within one generated signature, while index extraction reuses generated owned
Tensor-list results.
All seven `pytorch.IndexMutationNative` contracts bind indexed addition,
copying, selection output, and selection backward directly. They combine
tagged Scalar alpha, integer-list shape lowering, copied secondary resources,
retained output, and logically-const in-place Tensor methods without an
operator-specific wrapper.
All 11 `pytorch.FillNative` contracts bind full-tensor, diagonal, and masked
Scalar/Tensor filling directly across functional, retained-output, and
logically-const in-place calls. Structural tagged-Scalar recognition preserves
the public `value_*` prefix even where upstream names the parameter
`fill_value`.
All 32 `pytorch.ActivationCompletion` contracts bind the 16 exact hardshrink,
softshrink, softplus, and threshold forward, backward, and retained-output
overloads. Integer and floating public functions share their extracted C++
`Scalar` declarations while retaining distinct Terlan contracts.
All 36 `pytorch.ExtendedActivationCompletion` contracts likewise bind 18 exact
CELU, ELU, hardtanh, and leaky-ReLU functional, retained-output, in-place, and
backward declarations. Their integer and floating public Scalar forms share
the same compiler-generated calls.
All 19 `pytorch.Loss` contracts bind exact upstream declarations for embedding,
ranking, regression, likelihood, and triplet losses, including retained-output
and explicit backward calls.
All 18 `pytorch.ShapeCompletion` contracts bind exact channel shuffle,
transpose/select/narrow, unfold, trace-gradient, patch transform, contraction,
and zero declarations. The inherited `Tensor::unfold` method and both `narrow`
overloads come directly from maintained Clang metadata.
All 20 `pytorch.Indexing` contracts bind exact ordering, bucketization,
diagonal, indexed copy/selection, membership, masking, scatter, conditional
selection, and copied-window declarations. Functional, retained-output, and
logically-const in-place paths all use compiler-generated adapters.
All 17 `pytorch.Gradients` contracts bind 15 exact cumulative-product,
embedding, fake-quantization, gather, GELU, logit, GLU, masked-selection, and
soft-margin gradient declarations. Omitted and present logit epsilon forms
share extracted optional-double declarations, while four output paths retain
caller storage through generated mutable adapters.
All 47 `pytorch.Dense` contracts bind exact tensor/tensor dense arithmetic,
linear algebra, activation-gradient, complex construction, special-function,
shape/type matching, and retained-output declarations. The four inherited
`Tensor::*_as` methods and every overloaded namespace function are selected
from maintained Clang metadata without package-side adapters.
All 21 `pytorch.FunctionalCompletion` contracts bind exact linear,
log-softmax, group-normalization, finite-difference, and masked-scatter
backward declarations. Compiler-generated optional-resource lowering covers
present and omitted tensors, including `weight`/`bias` and
`prepend`/`append` optionals in the middle of exact upstream signatures;
historical integer dtype arguments are cast through extracted enum metadata.
All four `pytorch.PaddingNative` contracts bind exact constant-pad functional
and retained-output declarations plus mode-based pad with omitted or explicit
fill values. The compiler recognizes both maintained tagged-Scalar naming
conventions and emits every branch without a package-authored dispatcher.
All 16 `pytorch.BinaryCrossEntropyCompletion` contracts bind the six exact
probability, backward, logits, and retained-output declarations. Generated
optional-resource alignment preserves absent, element-weight-only,
positive-weight-only, and both-weight calls; the compiler recognizes ATen's
`pos_weight` abbreviation without changing the stable public
`positive_weight` spelling.
All 36 `pytorch.Division` contracts bind the exact `div`, `divide`, and
`true_divide` Scalar/Tensor overloads across functional, retained-output, and
logically-const in-place calls. Required rounding modes lower through a
compiler-generated present `std::optional<string_view>` value.
All 20 `pytorch.SliceNative` contracts bind five exact `slice`, `slice_copy`,
and `slice_scatter` declarations. Generated optional-integer alignment emits
all absent, start-only, end-only, and fully bounded variants across views,
owned copies, retained outputs, and scatter calls.
All 44 `pytorch.Fft` contracts bind 42 exact FFT, Hermitian FFT, real FFT,
frequency-grid, and frequency-shift declarations. Compiler-generated adapters
omit transform-size optionals, lower required dimensions into ordinary or
optional integer ArrayRefs, preserve normalization and integer-list defaults,
and normalize extracted braced defaults for exact overload selection.
All 56 `pytorch.Pooling` contracts bind 48 exact adaptive, average, maximum,
fractional-maximum, backward, unpool, and retained-output declarations.
Compiler-generated owned tuple carriers publish both returned Tensor handles
atomically, required omitted C++ optionals lower to `std::nullopt`, and six
dual-output calls borrow distinct retained Tensor handles without a
package-authored wrapper. The shared behavior workload executes every contract
against both CPU and CUDA LibTorch.
All 18 `pytorch.Selection` contracts bind exact cumulative extrema, ranked
selection, dimension reduction, mode, median, sorting, and top-k declarations.
Ten functional calls use generated owned Tensor tuple carriers and eight output
calls use generated dual retained-output borrowing. Every contract executes on
both CPU and CUDA through the same device-parameterized workload.
All five `pytorch.TensorListTransforms` contracts bind exact broadcasting,
mesh-grid, dense flatten, and dense unflatten declarations. A generated
`prepend_resource` mapping combines the public required first Tensor and the
remaining Tensor list into one compiler-owned collector for LibTorch's single
`TensorList` parameter. Every contract executes on both CPU and CUDA without a
package-authored forwarding wrapper.
All 12 `pytorch.NumericalCompletionNative` contracts bind exact stable-sort
output, pairwise distance, matrix-chain multiplication, clone output,
contiguity, and cross-product declarations. Direct and optional historical
integer parameters lower through extracted `c10::MemoryFormat` metadata; every
contract executes through generated CXX on CPU and CUDA.
All 11 `pytorch.LinearStorageCompletion` contracts bind exact bilinear,
temporal convolution, cumulative-extrema gradient, and storage-rebinding
declarations. The generated owned tuple carrier now publishes all three
`conv_tbc_backward` tensors atomically; every contract executes on CPU and
CUDA.
All 16 `pytorch.OutputCompletion` contracts bind exact complex, distance,
expand-copy, legacy LU/QR, dropout-gradient, polar, reflection/replication
padding, renormalization, determinant, and softmax retained-output
declarations. Every output—including both `slogdet_out` tensors—executes with
numeric, shape, dtype, source-preservation, and placement checks on CPU and
CUDA.
All 34 `pytorch.View` contracts bind exact alias, detachment, lift,
minimum-rank, transpose, conjugation, real/imaginary, and real/complex storage
view declarations. Generated free-function, immutable-method, and Tensor-list
families execute every contract on CPU and CUDA without a package-authored
forwarding wrapper.
All 18 `pytorch.ConvolutionNative` contracts bind nine exact regular,
string-padding, and transposed-convolution declarations across one, two, and
three dimensions. Generated absent/present optional-resource projection
executes all contracts on CPU and CUDA without a package-authored forwarding
wrapper.
All 25 `pytorch.TypeRelationsNative` contracts bind exact comparison, tensor
property, storage-identity, geometry, device, dtype-promotion, four
`result_type`, and gradient-retention declarations. Generated parameter
permutation and tagged-Scalar lowering select the exact upstream overloads,
while reviewed enum guards contain invalid historical integer dtype values.
Every contract executes on CPU and CUDA without a package-authored forwarding
wrapper.
All 32 `pytorch.CorrectionStatisticsCompletion` contracts bind exact
correction-based `std`, `var`, `std_mean`, and `var_mean` declarations.
Compiler-generated adapters lower omitted and present optional dimensions and
integer/float corrections, own paired Tensor returns, and safely borrow both
retained outputs. Every functional, paired-return, and retained-output contract
executes numerically on CPU and CUDA without a package-authored forwarding
wrapper.
All 12 `pytorch.RepeatInterleaveNative` contracts bind the four exact upstream
index, retained-output, tensor-count, and scalar-count declarations. The
compiler retains compatibility-only positional values in the public Terlan
contract while omitting them from generated CXX declarations and exact LibTorch
calls. Every optional dimension/output-size state executes on CPU and CUDA
without a package-authored forwarding wrapper.
All three `pytorch.FusedArithmeticNative` public contracts are generated Terlan
compositions over 25 private, exact CXX leaves. The stable operation selector
therefore remains outside C++, while functional, retained-output, and in-place
forms bind exact `addbmm`, `addcdiv`, `addcmul`, `addmm`, `addmv`, `addr`, and
`baddbmm` declarations. CPU and CUDA tests execute all 21 ordinary paths; CUDA
also proves the exact half-input/float-accumulation `addmm` and `baddbmm` dtype
functional and retained-output overloads.
All nine `pytorch.ScaledArithmeticNative` public contracts are generated Terlan
compositions over 16 private exact CXX leaves. Six homogeneous private families
select the Tensor/Scalar `add`, `sub`, and `rsub` functional and retained-output
overloads plus every supported `add_` and `sub_` mutation. CPU and CUDA tests
execute all 24 storage/operand paths with integer and floating Scalar lowering.
All six `pytorch.ClampNative` public contracts are generated Terlan
compositions over 18 private exact CXX leaves. Scalar-bound presence and
integer/float choices plus all three tensor-bound modes select exact `clamp`,
`clamp_min`, and `clamp_max` functional, retained-output, and mutation
overloads. CPU and CUDA tests execute all 18 selector/form combinations without
package-authored C++ glue.
All seven `pytorch.AggregationNative` public contracts are generated Terlan
compositions over 11 private exact CXX leaves. Optional bincount weights,
integer/float histogram bounds, and all four indexed-reduction names select
exact `bincount`, `histc`, and `index_reduce` functional, retained-output, and
mutation declarations. CPU and CUDA tests execute both weight states, both
bound types, every reduction, and both `include_self` states without
package-authored C++ glue.
The complete `pytorch.Binary` module is now generated directly from extracted
LibTorch metadata: all 102 functional, retained-output, and in-place public
contracts (including tensor aliases and `max_dim`/`min_dim` dual outputs) are
covered without a package-authored operator shim. CPU and CUDA behavior tests
exercise representative arithmetic, logical, remainder, integer mutation,
dual-output reduction, alias, shape, and device paths.
Distance additionally proves two-resource free calls with primitive defaults.
ShapePartition proves all
scalar, integer-list, and Tensor-index forms of `chunk`, `split`,
`tensor_split`, `unbind`, `dsplit`, `hsplit`, and `vsplit`. The complete
generated migration package currently contains 2207 generated functions over 1940 exact
metadata symbols. The Tensor TorchScript surface is complete as well: scalar,
tuple, integer/float lists, boolean/string lists, optional Tensor, and Tensor-list
outputs all lower through generated CXX adapters. The migration ledger reports
1,760 of 1,760 frozen legacy operations with no remaining C-ABI contracts.
The six public `pytorch.ScatterNative` selector contracts
use 18 generated CXX leaves for plain, additive, legacy-reduced, and
modern-reduced tensor/scalar scatter; the focused CPU contract executes
through the generated helper. `pytorch.RangeNative` now binds its 14 packed
constructor contracts through 29 exact CXX leaves for `arange`, `eye`,
`linspace`, `logspace`, and legacy `range`; focused CPU coverage exercises
optional dtype/layout/device lowering. `pytorch.SpecialScalarNative` adds eight
public dispatchers over 112 exact CXX leaves for the complete scalar-left/right
and retained-output special-function surface. `pytorch.CreationCompletionNative`
adds four public creation/like dispatchers over 16 exact CXX leaves. The
`pytorch.BinaryScalarNative` adds twelve tensor/scalar, retained-output, and
in-place dispatchers over 108 exact CXX leaves. `pytorch.LookupNative` adds ten
membership, sorted-search, and bucketization dispatchers over 12 exact CXX
leaves. `pytorch.UnaryStorageNative` adds all 156 retained-output and in-place
unary contracts over 156 exact CXX leaves. `pytorch.Unary` adds all 57 immutable
unary and predicate contracts over 57 exact CXX leaves. `pytorch.Shape` adds all
82 shape and view contracts over 82 exact CXX leaves. The machine-readable migration ledger and contract gate are
`migration.json` and `scripts/check-cxx-migration.py`; they reject public
signature drift and any increase in the legacy operation/source counts.
`pytorch.Special` adds all 141 mathematical, matrix, polynomial, scalar-overload,
retained-output, and in-place contracts over 130 exact CXX leaves, with
representative CPU/CUDA behavior coverage.
`pytorch.Statistics` adds all 87 reduction, norm, cumulative, tuple, and
retained-output contracts over exact extracted ATen symbols, including its 24
out-parameter families. Omitted trailing optional C++ parameters are now
materialized by the compiler as explicit absent values for selected overloads;
the generated Statistics package compiles against both CPU and CUDA LibTorch,
and its focused CPU behavior test passes.
`pytorch.CorrectionStatisticsPairNative` adds all 16 legacy TensorPair contracts
through generated ownership around LibTorch's native tuple return carrier. A
package-owned alias preserves the exact public TensorPair contract without an
operator shim; its generated package compiles against both CPU and CUDA
LibTorch.
`pytorch.Sparse` adds all 55 sparse-dimension, conversion, indexing, resize,
constructor, and sampled-addmm contracts through extracted ATen dispatchers
and package-owned C++ option adapters; its generated package compiles against
both CPU and CUDA LibTorch.
`pytorch.Linalg` adds all 153 factorization, decomposition, solve, inverse,
rank, norm, least-squares, and tensor-equation contracts through extracted
ATen callables and package-owned C++ option adapters; its generated package
compiles against both CPU and CUDA LibTorch.
`pytorch.Cuda` adds all five CUDA stream resource contracts through a generated
CXX-owned stream resource and package-owned implementation source; CPU and
CUDA packages both compile, with CPU calls reporting unavailable CUDA support.
`pytorch.Optimizer` adds all 11 Adam/SGD lifecycle, update, checkpoint, restore,
and cancellation contracts through a generated CXX-owned optimizer resource;
the real CPU package compiles successfully.

The same generated package now selects the package-owned
`terlan::tensor_packet` namespace directly from extracted C++ metadata.
`Tensor.from_packet`, the `from_array` compatibility alias, and
`Tensor.to_packet` therefore cross a compiler-generated `cxx::bridge`; the
package source declares no C ABI for this path. The pointer-free TNXP codec
copies six shared dtypes, validates shape/stride/storage bounds, snapshots CUDA
tensors through CPU, and catches every C++ failure before its `noexcept` CXX
entry point. Direct generated-crate tests prove contiguous ownership round
trips, zero-stride broadcast import, and malformed-packet containment. The
Terlan gates execute 79 behavior tests on CPU and 77 on CUDA, including actual
CUDA-to-owned-CPU packet export. Five direct generated-crate tests cover packet
ownership and dtype-relation containment on each LibTorch distribution. Because
the rest of `pytorch.Tensor` is still being migrated, these three functions are
deliberately excluded from the complete-module ledger's 1,186-operation
numerator.

Run the real compile, link, helper, and Terlan execution gate with:

```sh
make cxx-check LIBTORCH=vendor/libtorch-2.13.0+cpu/libtorch
make cxx-cuda-check LIBTORCH=vendor/libtorch-2.13.0+cu129/libtorch
```

The CUDA gate builds the same generated CXX package against frozen cu129
LibTorch and executes `ones_device("cuda:0")`, CUDA/CPU placement predicates,
all Lerp and WhereNative overload categories, all indexed mutation paths, and
all FillNative paths,
all 56 Pooling value, tuple, backward, unpool, retained-output, and dual-output
paths,
all 18 Selection tuple and dual retained-output paths,
all five TensorListTransforms list-input and list-output paths,
all 12 NumericalCompletionNative stable-sort, distance, matrix-chain, clone,
contiguous, and cross-product paths,
all 11 LinearStorageCompletion bilinear, temporal convolution, backward,
cumulative-extrema gradient, and storage-rebinding paths,
all ActivationCompletion and ExtendedActivationCompletion integer/floating
forward, backward, output, and in-place paths,
all Loss forward, backward, and retained-output paths,
all CUDA-supported ShapeCompletion paths (the frozen distribution has no
`narrow_copy.out` CUDA kernel),
all Indexing functional, retained-output, and in-place paths,
all Gradients dense/sparse, optional-epsilon, string-mode, functional, and
retained-output paths,
all Dense tensor/tensor, complex, special-function, linear-algebra, inherited
method, and retained-output paths,
all FunctionalCompletion linear, log-softmax, group-normalization,
finite-difference, optional-resource, and retained-output paths,
all PaddingNative integer/floating Scalar, optional-value, functional, and
retained-output paths,
dtype/layout/storage checks, scalar multiplication and extraction, and complete
resource disposal on a real CUDA device. It also executes `unbind` and proves
that both returned slices remain CUDA tensors after the source is disposed,
then repeats that ownership check for generated chunk and tensor-split paths.
Cosine similarity and pairwise distance execute on both CPU and CUDA with exact
result/device checks. The partition family likewise executes every migrated
form on both devices. Alpha, element, and feature dropout plus native dropout
backward execute through their direct generated calls on both devices.
The remaining functional and in-place dropout variants do as well.
GLU forward, backward, JVP, and backward-JVP functional/output paths have
numerical and ownership evidence on both devices.
Affine-grid forward/backward and generic, 2D, and 3D sampling likewise execute
on CPU and CUDA, including every retained-output form.
Diagonal/scatter and diagonal/select/slice/unfold backward calls execute on both
devices too, including all five retained-output forms and integer-list shapes.
Reflection and replication padding backward calls execute in 1D, 2D, and 3D on
both devices, including all six retained-output forms.
Cumulative and reduced integration plus KL divergence execute numerically on
both devices through Tensor-, integer-, and floating-spacing overloads.
Nan-to-number executes absent and present optional replacement combinations on
both devices, including retained-output resizing and in-place mutation.
Strided view/copy/mutation/scatter forms execute with inherited and explicit
storage offsets on both devices. Scalar and Tensor-valued index fill executes
all six functional, retained-output, and in-place forms on CPU and CUDA.

The generated CXX contract snapshot is now the sole migration baseline. The
historical C ABI manifest has been retired from the package and build path.
Further parity work must continue through generated CXX and must not add
another package-authored C or C++ operator shim.
