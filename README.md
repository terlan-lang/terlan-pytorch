# terlan-pytorch

PyTorch tensors and model inference for Terlan.

This package brings the core tensor and inference capabilities of
[PyTorch](https://github.com/pytorch/pytorch) to Terlan applications through
the released LibTorch distribution. The API uses owned Terlan values and
explicit tensor disposal.

## Available now

The current CPU implementation supports:

- scalar, shaped float64, and shaped int64 tensor construction;
- pointer-free DLPack packet import into owned CPU tensors;
- shape, rank, element count, dtype, and device inspection;
- cloning, indexing, reshaping, transposition, and reductions;
- matrix multiplication, arithmetic, ReLU, GELU, softmax, and argmax;
- convolution, adaptive average pooling, embedding, batch normalization, and
  layer normalization;
- functional training-mode batch normalization with independently owned
  updated running statistics;
- MSE, explicit MSE gradients, and cross-entropy loss;
- real graph autograd with trainable leaves, backward propagation, copied
  gradients, functional SGD, and owned Adam and momentum-SGD optimizer state;
- PPO loss, sampling, minibatch, and device-parity building blocks;
- composed CNN, classifier, attention, and transformer execution;
- stateless and reusable TorchScript model inference;
- Tensor, optional tensor, tensor-list, fixed tensor-pair, scalar, string, and
  homogeneous scalar-list TorchScript results;
- synchronous tensor placement through `Tensor.to_device`;
- owned conversion between admitted boolean, integer, Float16/32/64, and
  BFloat16 tensor dtypes through `Tensor.to_dtype`.

The complete generated API is documented in
[`docs/pytorch.Tensor.md`](docs/pytorch.Tensor.md).

## Example

```terlan
module example.Pytorch.

import pytorch.Tensor.{amax, dispose, from_floats, item_float, relu, shape}.

pub run(): Bool ->
    let input = from_floats([-2.0, 3.0], [2]);
    let output = relu(input);
    let maximum = amax(output, [0], false);
    let valid = shape(output) == [2] and item_float(maximum) == 3.0;
    dispose(maximum);
    dispose(output);
    dispose(input);
    valid.
```

Every operation that returns a `Tensor`, `TensorPair`, or `TorchScriptModel`
returns a separately owned resource. Dispose each resource when it is no longer
needed.

## Setup

The verified release host is Linux x86_64. macOS, Windows, Linux AArch64, and
embedded targets are not release-gated yet.

The package currently targets LibTorch `2.13.0+cpu`. From this repository, run:

```sh
make release-check
```

This downloads the pinned LibTorch archive, verifies its checksum and required
symbols, generates the package, and runs the native, Terlan, and immutable Git
consumer checks. The persistent CPU evidence is written to
`reports/package-execution-report.json`.

To use an existing compatible LibTorch installation:

```sh
make package-check LIBTORCH=/path/to/libtorch
```

`TERLC` can override the compiler executable:

```sh
make package-check \
  TERLC=/path/to/terlc \
  LIBTORCH=/path/to/libtorch
```

The package does not build PyTorch from source.

## CUDA

The package includes synchronous device placement and an opt-in CUDA execution
gate. CUDA execution requires a CUDA-enabled LibTorch distribution and a
driver-visible NVIDIA device. The matching CUDA toolkit runtime libraries must
also be discoverable through `LD_LIBRARY_PATH`:

```sh
make cuda-check LIBTORCH=/path/to/cuda/libtorch
```

The admitted CUDA distribution must include the developer headers
`include/cuda.h` and `include/cuda_runtime_api.h`; runtime-only CUDA bundles are
rejected before compilation. Generated executables use relocatable package
RPATHs for the bundled `cuda-runtime` layout, while development invocations may
provide equivalent toolkit directories through `LD_LIBRARY_PATH`.

LibTorch `2.13.0+cu129` CPU-to-CUDA-to-CPU execution with explicit stream
selection, synchronization, matrix multiplication, and on-device dtype
conversion is validated by the checked-in report at
`reports/cuda-execution-report.json`. Tensor transfers and dtype conversion
remain blocking. CUDA remains an opt-in lane and raw device pointers are not
exposed through the Terlan API.

The self-hosted CI lane reads the distribution from `LIBTORCH_CUDA_ROOT` and
its colon-separated runtime library directories from `CUDA_LIBRARY_PATH`.
The CUDA gate rejects missing devices, incompatible distributions, missing
stream exports, and unresolved runtime libraries before executing Terlan code.

## Development

Regenerate the package after changing binding metadata or package-owned native
adapters:

```sh
make generate
```

Run the lightweight fixture-backed checks:

```sh
make fixture-check
```

The primary package now uses generated CXX bindings. `make cxx-check` and
`make cxx-cuda-check` generate
Rust/C++ bindings through `cxx`, compare migrated public contracts against the
legacy package, and execute real LibTorch behavior. All 1,760 legacy
operations are migrated across `pytorch.ActivationCompletion` and
`pytorch.ExtendedActivationCompletion`,
`pytorch.TensorJoining`, `pytorch.Distance`,
`pytorch.ShapePartition`, `pytorch.Regularization`, and
`pytorch.DropoutCompletion`, plus `pytorch.GluCompletion` and
`pytorch.Spatial`, `pytorch.ShapeGradient`, and
`pytorch.PaddingBackwardCompletion`, `pytorch.IntegrationCompletion`, and
`pytorch.NanToNumNative`, `pytorch.Strided`, `pytorch.FillNative`,
`pytorch.IndexFillNative`,
`pytorch.IndexMutationNative`, `pytorch.LerpNative`, `pytorch.Loss`,
`pytorch.ShapeCompletion`, `pytorch.Indexing`, `pytorch.Gradients`,
`pytorch.Dense`, `pytorch.FunctionalCompletion`, `pytorch.PaddingNative`,
`pytorch.BinaryCrossEntropyCompletion`, `pytorch.Division`,
`pytorch.SliceNative`, `pytorch.Fft`, `pytorch.Pooling`, `pytorch.Selection`,
`pytorch.TensorListTransforms`, `pytorch.NumericalCompletionNative`,
`pytorch.LinearStorageCompletion`, `pytorch.OutputCompletion`, `pytorch.View`,
`pytorch.ConvolutionNative`, `pytorch.TypeRelationsNative`,
`pytorch.CorrectionStatisticsCompletion`, `pytorch.RepeatInterleaveNative`,
`pytorch.FusedArithmeticNative`, `pytorch.ScaledArithmeticNative`,
`pytorch.ClampNative`, `pytorch.AggregationNative`, `pytorch.WhereNative`,
`pytorch.Statistics`, `pytorch.CorrectionStatisticsPairNative`, and
`pytorch.Sparse`, `pytorch.Linalg`, `pytorch.Cuda`, and `pytorch.Optimizer`; the
hand-maintained C wrapper sources have been removed, with only a frozen
migration manifest retained for contract auditing; all Tensor
TorchScript collection/optional contracts now use the generated-CXX handle world.
The orchestration behind both CXX targets is typed in
`scripts/RunCxxChecks.terls`; Make targets only validate inputs and dispatch to
that script. Its generated behavior fixtures are aggregated into one native
test module and compiled incrementally, while preserving every fixture body
and the full public-test set.
The grouped-binding, function-reference, format, and cleanup targets likewise
dispatch to `scripts/CheckTerlanLint.terls` and `scripts/Clean.terls`, keeping
filesystem and compiler orchestration in typed Terlan code.

Tensor interchange has also crossed that boundary: `Tensor.from_packet`, its
`from_array` compatibility alias, and `Tensor.to_packet` now use exact
compiler-generated CXX bindings to a package-neutral, pointer-free TNXP codec.
The codec contains malformed-input failures before the generated bridge and
owns both imported tensor storage and exported bytes. The real distribution
gates execute 79 Terlan tests on CPU and 77 on CUDA, plus five direct CXX
boundary tests on each build. These three partial `pytorch.Tensor` contracts do
remain outside the completed-module migration count. The generated CXX contract
snapshot is the sole migration baseline; the historical C ABI manifest has
been retired from the package and build path.

Report exact progress against both frozen PyTorch 2.13 inventories:

```sh
make pytorch-parity-report
```

`pytorch-api-inventory-report` covers the 2,261 exact ATen overload schemas;
`pytorch-public-api-inventory-report` covers the broader 3,425-callable Python
surface. The corresponding `pytorch-parity-check` target is intentionally
strict and succeeds only after both policies have zero missing entries. To
re-freeze the public oracle, provide the Python executable from the pinned
reference environment explicitly:

```sh
make pytorch-public-api-inventory-freeze \
  PYTORCH_REFERENCE_PYTHON=/path/to/pytorch-2.13/bin/python
```

Validate pointer-free ndarray-to-PyTorch tensor exchange with both packages
checked out next to each other:

```sh
make ndarray-interop-check \
  LIBTORCH=/path/to/libtorch \
  NDARRAY=/path/to/terlan-ndarray
```

The package manifest declares `terlan-ndarray` as an ordinary Terlan
dependency, and the immutable-consumer gate resolves and locks that dependency
without either source checkout remaining available. The release gate also
builds the current ndarray helper and proves that an ndarray matrix
multiplication result can be imported and consumed as an independently owned
PyTorch tensor. PyTorch does not depend on Polars; its runtime ndarray boundary
remains the versioned, pointer-free TNXP value contract.

Run the released CPU distribution and immutable consumer checks:

```sh
make release-check
```

The compiler-owned full package gate is:

```sh
make -C ../terlan terlan-pytorch-package-check \
  TERLAN_PYTORCH_DIR="$(pwd)"
```

## Remaining expansion

- additional optimizer families;
- dictionary and nested mixed TorchScript result values;
- typed CUDA device discovery, additional model formats, and release targets.

Implementation sequencing and remaining gates are tracked in
[`docs/ROADMAP.md`](docs/ROADMAP.md). The reviewed native target is documented
in [`docs/TARGET_SURFACE.md`](docs/TARGET_SURFACE.md).
