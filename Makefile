TERLC ?= ../terlan/target/debug/terlc
TERLAN_VM_RUNNER ?= $(abspath $(dir $(TERLC))terlan-vm)
LIBTORCH ?=
NDARRAY ?= ../terlan-ndarray
NDARRAY_HELPER := $(abspath $(NDARRAY))/_build/rust-target/debug/native-boundary-helper
TERLAN_NDARRAY_NATIVE_BOUNDARY_HELPER_PATH ?= $(NDARRAY_HELPER)
export TERLAN_NDARRAY_NATIVE_BOUNDARY_HELPER_PATH
# The package build and migration audit are CXX-first.  The contract snapshot
# is the reviewed, adapter-neutral baseline for generated CXX bindings.
CXX_CONTRACT_SNAPSHOT := bindings/cxx/contract-snapshot.json
MANIFEST := libtorch-cxx.json
GENERATED := generated
CXX_MANIFEST := libtorch-cxx.json
CXX_GENERATED := _build/libtorch-cxx-generated
CXX_TARGET := _build/libtorch-cxx-target
CXX_CUDA_GENERATED := _build/libtorch-cxx-cuda-generated
CXX_CUDA_TARGET := _build/libtorch-cxx-cuda-target
PINNED_LIBTORCH := $(CURDIR)/vendor/libtorch-2.13.0+cpu/libtorch
CUDA_REPORT := $(CURDIR)/reports/cuda-execution-report.json
PACKAGE_REPORT := $(CURDIR)/reports/package-execution-report.json
CUDA_RUNTIME_ROOT := $(CURDIR)/vendor/cuda-runtime/usr/lib/x86_64-linux-gnu
CUDA_RUNTIME_PATH := $(CUDA_RUNTIME_ROOT)/libcusparseLt/12:$(CUDA_RUNTIME_ROOT)/nvshmem/12

.PHONY: generate cxx-check cxx-cuda-check pytorch-feature-parity-check fetch-libtorch fixture-check package-check ndarray-dependency-check terlan-format-check terlan-grouped-binding-check terlan-function-reference-check terlan-script-check pytorch-api-inventory-report pytorch-api-inventory-check pytorch-public-api-inventory-freeze pytorch-public-api-policy-seed pytorch-public-api-inventory-report pytorch-public-api-inventory-check pytorch-parity-report pytorch-parity-check autograd-check ppo-check cuda-check ndarray-interop-check release-check clean cxx-check-legacy cxx-cuda-check-legacy

generate:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	rm -rf "$(GENERATED)"
	"$(TERLC)" bind cpp --manifest "$(MANIFEST)" --out "$(GENERATED)"
	cp -a package-src/. "$(GENERATED)/src/"
	"$(TERLC)" run scripts/PatchCxxGeneratedSurface.terls -- "$(GENERATED)" "$(CURDIR)"
	"$(TERLC)" run scripts/CheckCxxExceptionSurface.terls -- "$(CURDIR)" "$(GENERATED)"
	"$(TERLC)" run scripts/CheckCxxSourceOwnership.terls -- "$(CURDIR)" "$(GENERATED)"
	"$(TERLC)" fmt --write "$(GENERATED)/src"
	mkdir -p "$(GENERATED)/tests/pytorch"
	cp tests/pytorch/NativeBoundaryTest.terl "$(GENERATED)/tests/pytorch/NativeBoundaryTest.terl"
	cp tests/pytorch/EmbeddingParityTest.terl "$(GENERATED)/tests/pytorch/EmbeddingParityTest.terl"
	cp tests/pytorch/DispatcherTest.terl "$(GENERATED)/tests/pytorch/DispatcherTest.terl"
	cp tests/pytorch/DevicePlacementTest.terl "$(GENERATED)/tests/pytorch/DevicePlacementTest.terl"
	cp tests/pytorch/DLPackTest.terl "$(GENERATED)/tests/pytorch/DLPackTest.terl"
	cp tests/pytorch/DtypeTest.terl "$(GENERATED)/tests/pytorch/DtypeTest.terl"
	cp tests/pytorch/DtypeAliasParityTest.terl "$(GENERATED)/tests/pytorch/DtypeAliasParityTest.terl"
	cp tests/pytorch/TensorMethodAliasParityTest.terl "$(GENERATED)/tests/pytorch/TensorMethodAliasParityTest.terl"
	cp tests/pytorch/TensorToAliasParityTest.terl "$(GENERATED)/tests/pytorch/TensorToAliasParityTest.terl"
	cp tests/pytorch/TensorSparseAliasParityTest.terl "$(GENERATED)/tests/pytorch/TensorSparseAliasParityTest.terl"
	cp tests/pytorch/TensorViewAliasParityTest.terl "$(GENERATED)/tests/pytorch/TensorViewAliasParityTest.terl"
	cp tests/pytorch/ConstructionTest.terl "$(GENERATED)/tests/pytorch/ConstructionTest.terl"
	cp tests/pytorch/TypeRelationsParityTest.terl "$(GENERATED)/tests/pytorch/TypeRelationsParityTest.terl"
	cp tests/pytorch/ChalfParityTest.terl "$(GENERATED)/tests/pytorch/ChalfParityTest.terl"
	cp tests/pytorch/GeqrfOutParityTest.terl "$(GENERATED)/tests/pytorch/GeqrfOutParityTest.terl"
	cp tests/pytorch/DenseDtypeParityTest.terl "$(GENERATED)/tests/pytorch/DenseDtypeParityTest.terl"
	cp tests/pytorch/NormOutputParityTest.terl "$(GENERATED)/tests/pytorch/NormOutputParityTest.terl"
	cp tests/pytorch/NormDimensionedCxxParityTest.terl "$(GENERATED)/tests/pytorch/NormDimensionedCxxParityTest.terl"
	cp tests/pytorch/NanStatisticsCxxParityTest.terl "$(GENERATED)/tests/pytorch/NanStatisticsCxxParityTest.terl"
	cp tests/pytorch/EinsumCxxParityTest.terl "$(GENERATED)/tests/pytorch/EinsumCxxParityTest.terl"
	cp tests/pytorch/AliasParityTest.terl "$(GENERATED)/tests/pytorch/AliasParityTest.terl"
	cp tests/pytorch/MutationAliasParityTest.terl "$(GENERATED)/tests/pytorch/MutationAliasParityTest.terl"
	cp tests/pytorch/LegacyAliasParityTest.terl "$(GENERATED)/tests/pytorch/LegacyAliasParityTest.terl"
	cp tests/pytorch/TriangularSolveOutParityTest.terl "$(GENERATED)/tests/pytorch/TriangularSolveOutParityTest.terl"
	cp tests/pytorch/ScalarTensorParityTest.terl "$(GENERATED)/tests/pytorch/ScalarTensorParityTest.terl"
	cp tests/pytorch/NewCreationParityTest.terl "$(GENERATED)/tests/pytorch/NewCreationParityTest.terl"
	cp tests/pytorch/VanderParityTest.terl "$(GENERATED)/tests/pytorch/VanderParityTest.terl"
	cp tests/pytorch/LerpParityTest.terl "$(GENERATED)/tests/pytorch/LerpParityTest.terl"
	cp tests/pytorch/WindowParityTest.terl "$(GENERATED)/tests/pytorch/WindowParityTest.terl"
	cp tests/pytorch/AttentionParityTest.terl "$(GENERATED)/tests/pytorch/AttentionParityTest.terl"
	cp tests/pytorch/SvdParityTest.terl "$(GENERATED)/tests/pytorch/SvdParityTest.terl"
	cp tests/pytorch/QuantizedAllocationParityTest.terl "$(GENERATED)/tests/pytorch/QuantizedAllocationParityTest.terl"
	cp tests/pytorch/UniqueParityTest.terl "$(GENERATED)/tests/pytorch/UniqueParityTest.terl"
	cp tests/pytorch/HistogramParityTest.terl "$(GENERATED)/tests/pytorch/HistogramParityTest.terl"
	cp tests/pytorch/StorageParityTest.terl "$(GENERATED)/tests/pytorch/StorageParityTest.terl"
	cp tests/pytorch/SpectralParityTest.terl "$(GENERATED)/tests/pytorch/SpectralParityTest.terl"
	cp tests/pytorch/SegmentReduceParityTest.terl "$(GENERATED)/tests/pytorch/SegmentReduceParityTest.terl"
	cp tests/pytorch/UpsampleVectorParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleVectorParityTest.terl"
	cp tests/pytorch/SparseMatmulParityTest.terl "$(GENERATED)/tests/pytorch/SparseMatmulParityTest.terl"
	cp tests/pytorch/SparsePublicAddmmParityTest.terl "$(GENERATED)/tests/pytorch/SparsePublicAddmmParityTest.terl"
	cp tests/pytorch/SparsePublicSumParityTest.terl "$(GENERATED)/tests/pytorch/SparsePublicSumParityTest.terl"
	cp tests/pytorch/LuUnpackParityTest.terl "$(GENERATED)/tests/pytorch/LuUnpackParityTest.terl"
	cp tests/pytorch/UnravelIndexParityTest.terl "$(GENERATED)/tests/pytorch/UnravelIndexParityTest.terl"
	cp tests/pytorch/ToOtherParityTest.terl "$(GENERATED)/tests/pytorch/ToOtherParityTest.terl"
	cp tests/pytorch/ToPaddedTensorParityTest.terl "$(GENERATED)/tests/pytorch/ToPaddedTensorParityTest.terl"
	cp tests/pytorch/NativeDropoutParityTest.terl "$(GENERATED)/tests/pytorch/NativeDropoutParityTest.terl"
	cp tests/pytorch/NativeLayerNormParityTest.terl "$(GENERATED)/tests/pytorch/NativeLayerNormParityTest.terl"
	cp tests/pytorch/NativeGroupNormParityTest.terl "$(GENERATED)/tests/pytorch/NativeGroupNormParityTest.terl"
	cp tests/pytorch/NativeGroupNormBackwardParityTest.terl "$(GENERATED)/tests/pytorch/NativeGroupNormBackwardParityTest.terl"
	cp tests/pytorch/NativeLayerNormBackwardParityTest.terl "$(GENERATED)/tests/pytorch/NativeLayerNormBackwardParityTest.terl"
	cp tests/pytorch/NativeBatchNormParityTest.terl "$(GENERATED)/tests/pytorch/NativeBatchNormParityTest.terl"
	cp tests/pytorch/NativeBatchNormBackwardParityTest.terl "$(GENERATED)/tests/pytorch/NativeBatchNormBackwardParityTest.terl"
	cp tests/pytorch/LstmCellParityTest.terl "$(GENERATED)/tests/pytorch/LstmCellParityTest.terl"
	cp tests/pytorch/RnnTanhCellParityTest.terl "$(GENERATED)/tests/pytorch/RnnTanhCellParityTest.terl"
	cp tests/pytorch/RnnReluCellParityTest.terl "$(GENERATED)/tests/pytorch/RnnReluCellParityTest.terl"
	cp tests/pytorch/GruCellParityTest.terl "$(GENERATED)/tests/pytorch/GruCellParityTest.terl"
	cp tests/pytorch/RecurrentSequenceParityTest.terl "$(GENERATED)/tests/pytorch/RecurrentSequenceParityTest.terl"
	cp tests/pytorch/RecurrentDataParityTest.terl "$(GENERATED)/tests/pytorch/RecurrentDataParityTest.terl"
	cp tests/pytorch/ThnnConv2dParityTest.terl "$(GENERATED)/tests/pytorch/ThnnConv2dParityTest.terl"
	cp tests/pytorch/SlowConvolutionParityTest.terl "$(GENERATED)/tests/pytorch/SlowConvolutionParityTest.terl"
	cp tests/pytorch/BackendAvailabilityParityTest.terl "$(GENERATED)/tests/pytorch/BackendAvailabilityParityTest.terl"
	cp tests/pytorch/FlushDenormalParityTest.terl "$(GENERATED)/tests/pytorch/FlushDenormalParityTest.terl"
	cp tests/pytorch/DeterminismParityTest.terl "$(GENERATED)/tests/pytorch/DeterminismParityTest.terl"
	cp tests/pytorch/MatmulPrecisionParityTest.terl "$(GENERATED)/tests/pytorch/MatmulPrecisionParityTest.terl"
	cp tests/pytorch/AutogradStateParityTest.terl "$(GENERATED)/tests/pytorch/AutogradStateParityTest.terl"
	cp tests/pytorch/AutogradGradientParityTest.terl "$(GENERATED)/tests/pytorch/AutogradGradientParityTest.terl"
	cp tests/pytorch/WarningStateParityTest.terl "$(GENERATED)/tests/pytorch/WarningStateParityTest.terl"
	cp tests/pytorch/GlobalRandomStateParityTest.terl "$(GENERATED)/tests/pytorch/GlobalRandomStateParityTest.terl"
	cp tests/pytorch/RngStateParityTest.terl "$(GENERATED)/tests/pytorch/RngStateParityTest.terl"
	cp tests/pytorch/SeedParityTest.terl "$(GENERATED)/tests/pytorch/SeedParityTest.terl"
	cp tests/pytorch/ThreadingParityTest.terl "$(GENERATED)/tests/pytorch/ThreadingParityTest.terl"
	cp tests/pytorch/AutocastStateParityTest.terl "$(GENERATED)/tests/pytorch/AutocastStateParityTest.terl"
	cp tests/pytorch/DefaultDtypeParityTest.terl "$(GENERATED)/tests/pytorch/DefaultDtypeParityTest.terl"
	cp tests/pytorch/InferenceModeParityTest.terl "$(GENERATED)/tests/pytorch/InferenceModeParityTest.terl"
	cp tests/pytorch/SymbolicConstraintsParityTest.terl "$(GENERATED)/tests/pytorch/SymbolicConstraintsParityTest.terl"
	cp tests/pytorch/QuantizationInternalsParityTest.terl "$(GENERATED)/tests/pytorch/QuantizationInternalsParityTest.terl"
	cp tests/pytorch/QuantizedInternalsParityTest.terl "$(GENERATED)/tests/pytorch/QuantizedInternalsParityTest.terl"
	cp tests/pytorch/NonzeroNumpyParityTest.terl "$(GENERATED)/tests/pytorch/NonzeroNumpyParityTest.terl"
	cp tests/pytorch/LogSigmoidParityTest.terl "$(GENERATED)/tests/pytorch/LogSigmoidParityTest.terl"
	cp tests/pytorch/NllLossInternalParityTest.terl "$(GENERATED)/tests/pytorch/NllLossInternalParityTest.terl"
	cp tests/pytorch/FromFileParityTest.terl "$(GENERATED)/tests/pytorch/FromFileParityTest.terl"
	cp tests/pytorch/IndexTensorOptionalParityTest.terl "$(GENERATED)/tests/pytorch/IndexTensorOptionalParityTest.terl"
	cp tests/pytorch/GeneratorParityTest.terl "$(GENERATED)/tests/pytorch/GeneratorParityTest.terl"
	cp tests/pytorch/RangeParityTest.terl "$(GENERATED)/tests/pytorch/RangeParityTest.terl"
	cp tests/pytorch/TriangularIndicesParityTest.terl "$(GENERATED)/tests/pytorch/TriangularIndicesParityTest.terl"
	cp tests/pytorch/IsPinnedParityTest.terl "$(GENERATED)/tests/pytorch/IsPinnedParityTest.terl"
	cp tests/pytorch/ResizeParityTest.terl "$(GENERATED)/tests/pytorch/ResizeParityTest.terl"
	cp tests/pytorch/EmptyPermutedParityTest.terl "$(GENERATED)/tests/pytorch/EmptyPermutedParityTest.terl"
	cp tests/pytorch/MultilabelMarginLossParityTest.terl "$(GENERATED)/tests/pytorch/MultilabelMarginLossParityTest.terl"
	cp tests/pytorch/FakeQuantizeParityTest.terl "$(GENERATED)/tests/pytorch/FakeQuantizeParityTest.terl"
	cp tests/pytorch/DistributionParityTest.terl "$(GENERATED)/tests/pytorch/DistributionParityTest.terl"
	cp tests/pytorch/DistributionInplaceParityTest.terl "$(GENERATED)/tests/pytorch/DistributionInplaceParityTest.terl"
	cp tests/pytorch/NormalParityTest.terl "$(GENERATED)/tests/pytorch/NormalParityTest.terl"
	cp tests/pytorch/CovParityTest.terl "$(GENERATED)/tests/pytorch/CovParityTest.terl"
	cp tests/pytorch/CtcLossParityTest.terl "$(GENERATED)/tests/pytorch/CtcLossParityTest.terl"
	cp tests/pytorch/RandomParityTest.terl "$(GENERATED)/tests/pytorch/RandomParityTest.terl"
	cp tests/pytorch/BernoulliParityTest.terl "$(GENERATED)/tests/pytorch/BernoulliParityTest.terl"
	cp tests/pytorch/MultinomialParityTest.terl "$(GENERATED)/tests/pytorch/MultinomialParityTest.terl"
	cp tests/pytorch/QuantizationMetadataParityTest.terl "$(GENERATED)/tests/pytorch/QuantizationMetadataParityTest.terl"
	cp tests/pytorch/ViewDtypeParityTest.terl "$(GENERATED)/tests/pytorch/ViewDtypeParityTest.terl"
	cp tests/pytorch/QuantileParityTest.terl "$(GENERATED)/tests/pytorch/QuantileParityTest.terl"
	cp tests/pytorch/NanquantileParityTest.terl "$(GENERATED)/tests/pytorch/NanquantileParityTest.terl"
	cp tests/pytorch/HashTensorParityTest.terl "$(GENERATED)/tests/pytorch/HashTensorParityTest.terl"
	cp tests/pytorch/ResizeAsParityTest.terl "$(GENERATED)/tests/pytorch/ResizeAsParityTest.terl"
	cp tests/pytorch/StridedParityTest.terl "$(GENERATED)/tests/pytorch/StridedParityTest.terl"
	cp tests/pytorch/SpatialParityTest.terl "$(GENERATED)/tests/pytorch/SpatialParityTest.terl"
	cp tests/pytorch/ShapeGradientParityTest.terl "$(GENERATED)/tests/pytorch/ShapeGradientParityTest.terl"
	cp tests/pytorch/ShapeCompletionParityTest.terl "$(GENERATED)/tests/pytorch/ShapeCompletionParityTest.terl"
	cp tests/pytorch/SliceParityTest.terl "$(GENERATED)/tests/pytorch/SliceParityTest.terl"
	cp tests/pytorch/SliceInverseParityTest.terl "$(GENERATED)/tests/pytorch/SliceInverseParityTest.terl"
	cp tests/pytorch/NewEmptyStridedParityTest.terl "$(GENERATED)/tests/pytorch/NewEmptyStridedParityTest.terl"
	cp tests/pytorch/RmsNormParityTest.terl "$(GENERATED)/tests/pytorch/RmsNormParityTest.terl"
	cp tests/pytorch/NllLossParityTest.terl "$(GENERATED)/tests/pytorch/NllLossParityTest.terl"
	cp tests/pytorch/NllLoss2dParityTest.terl "$(GENERATED)/tests/pytorch/NllLoss2dParityTest.terl"
	cp tests/pytorch/NllLoss2dInternalParityTest.terl "$(GENERATED)/tests/pytorch/NllLoss2dInternalParityTest.terl"
	cp tests/pytorch/NllLossNdInternalParityTest.terl "$(GENERATED)/tests/pytorch/NllLossNdInternalParityTest.terl"
	cp tests/pytorch/RreluWithNoiseBackwardParityTest.terl "$(GENERATED)/tests/pytorch/RreluWithNoiseBackwardParityTest.terl"
	cp tests/pytorch/RreluWithNoiseFunctionalParityTest.terl "$(GENERATED)/tests/pytorch/RreluWithNoiseFunctionalParityTest.terl"
	cp tests/pytorch/RreluWithNoiseValueParityTest.terl "$(GENERATED)/tests/pytorch/RreluWithNoiseValueParityTest.terl"
	cp tests/pytorch/ViewCopyDtypeParityTest.terl "$(GENERATED)/tests/pytorch/ViewCopyDtypeParityTest.terl"
	cp tests/pytorch/RandpermParityTest.terl "$(GENERATED)/tests/pytorch/RandpermParityTest.terl"
	cp tests/pytorch/InstanceNormParityTest.terl "$(GENERATED)/tests/pytorch/InstanceNormParityTest.terl"
	cp tests/pytorch/MultiMarginLossParityTest.terl "$(GENERATED)/tests/pytorch/MultiMarginLossParityTest.terl"
	cp tests/pytorch/UpsampleBilinear2dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleBilinear2dParityTest.terl"
	cp tests/pytorch/UpsampleBicubic2dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleBicubic2dParityTest.terl"
	cp tests/pytorch/UpsampleLinear1dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleLinear1dParityTest.terl"
	cp tests/pytorch/UpsampleNearest1dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleNearest1dParityTest.terl"
	cp tests/pytorch/UpsampleNearest2dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleNearest2dParityTest.terl"
	cp tests/pytorch/UpsampleNearest3dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleNearest3dParityTest.terl"
	cp tests/pytorch/UpsampleNearest23dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleNearest23dParityTest.terl"
	cp tests/pytorch/SetDataParityTest.terl "$(GENERATED)/tests/pytorch/SetDataParityTest.terl"
	cp tests/pytorch/UpsampleBackwardParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleBackwardParityTest.terl"
	cp tests/pytorch/UpsampleTrilinear3dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleTrilinear3dParityTest.terl"
	cp tests/pytorch/UpsampleBackward2dParityTest.terl "$(GENERATED)/tests/pytorch/UpsampleBackward2dParityTest.terl"
	cp tests/pytorch/QuantizePerTensorParityTest.terl "$(GENERATED)/tests/pytorch/QuantizePerTensorParityTest.terl"
	cp tests/pytorch/TensorJoiningParityTest.terl "$(GENERATED)/tests/pytorch/TensorJoiningParityTest.terl"
	cp tests/pytorch/ConvolutionParityTest.terl "$(GENERATED)/tests/pytorch/ConvolutionParityTest.terl"
	cp tests/pytorch/NumericalCompletionParityTest.terl "$(GENERATED)/tests/pytorch/NumericalCompletionParityTest.terl"
	cp tests/pytorch/DropoutCompletionParityTest.terl "$(GENERATED)/tests/pytorch/DropoutCompletionParityTest.terl"
	cp tests/pytorch/LinearStorageCompletionParityTest.terl "$(GENERATED)/tests/pytorch/LinearStorageCompletionParityTest.terl"
	cp tests/pytorch/LossOutputCompletionParityTest.terl "$(GENERATED)/tests/pytorch/LossOutputCompletionParityTest.terl"
	cp tests/pytorch/IntegrationCompletionParityTest.terl "$(GENERATED)/tests/pytorch/IntegrationCompletionParityTest.terl"
	cp tests/pytorch/GluCompletionParityTest.terl "$(GENERATED)/tests/pytorch/GluCompletionParityTest.terl"
	cp tests/pytorch/OutputCompletionParityTest.terl "$(GENERATED)/tests/pytorch/OutputCompletionParityTest.terl"
	cp tests/pytorch/FunctionalCompletionParityTest.terl "$(GENERATED)/tests/pytorch/FunctionalCompletionParityTest.terl"
	cp tests/pytorch/BinaryCrossEntropyCompletionParityTest.terl "$(GENERATED)/tests/pytorch/BinaryCrossEntropyCompletionParityTest.terl"
	cp tests/pytorch/ActivationCompletionParityTest.terl "$(GENERATED)/tests/pytorch/ActivationCompletionParityTest.terl"
	cp tests/pytorch/PaddingBackwardCompletionParityTest.terl "$(GENERATED)/tests/pytorch/PaddingBackwardCompletionParityTest.terl"
	cp tests/pytorch/ExtendedActivationCompletionParityTest.terl "$(GENERATED)/tests/pytorch/ExtendedActivationCompletionParityTest.terl"
	cp tests/pytorch/CorrectionStatisticsCompletionParityTest.terl "$(GENERATED)/tests/pytorch/CorrectionStatisticsCompletionParityTest.terl"
	cp tests/pytorch/ShapePartitionParityTest.terl "$(GENERATED)/tests/pytorch/ShapePartitionParityTest.terl"
	cp tests/pytorch/RepeatInterleaveParityTest.terl "$(GENERATED)/tests/pytorch/RepeatInterleaveParityTest.terl"
	cp tests/pytorch/NanToNumParityTest.terl "$(GENERATED)/tests/pytorch/NanToNumParityTest.terl"
	cp tests/pytorch/AggregationParityTest.terl "$(GENERATED)/tests/pytorch/AggregationParityTest.terl"
	cp tests/pytorch/PaddingParityTest.terl "$(GENERATED)/tests/pytorch/PaddingParityTest.terl"
	cp tests/pytorch/AtleastSequenceParityTest.terl "$(GENERATED)/tests/pytorch/AtleastSequenceParityTest.terl"
	cp tests/pytorch/TensorListTransformsParityTest.terl "$(GENERATED)/tests/pytorch/TensorListTransformsParityTest.terl"
	cp tests/pytorch/CreationCompletionParityTest.terl "$(GENERATED)/tests/pytorch/CreationCompletionParityTest.terl"
	cp tests/pytorch/RangeMetadataTest.terl "$(GENERATED)/tests/pytorch/RangeMetadataTest.terl"
	cp tests/pytorch/OptionalOutputTest.terl "$(GENERATED)/tests/pytorch/OptionalOutputTest.terl"
	cp tests/pytorch/TensorListOutputTest.terl "$(GENERATED)/tests/pytorch/TensorListOutputTest.terl"
	cp tests/pytorch/TrainingTest.terl "$(GENERATED)/tests/pytorch/TrainingTest.terl"
	cp tests/pytorch/SgdTest.terl "$(GENERATED)/tests/pytorch/SgdTest.terl"
	cp tests/pytorch/ReadmeExampleTest.terl "$(GENERATED)/tests/pytorch/ReadmeExampleTest.terl"
	"$(TERLC)" fmt --write "$(GENERATED)/tests"
	"$(TERLC)" run scripts/PatchCxxGeneratedTests.terls -- "$(GENERATED)"
	mkdir -p "$(GENERATED)/native/rust/tests"
	# The legacy Rust tests target the removed C ABI and are not part of the
	# CXX-first generated package. CXX-specific integration tests follow.
	cp tests/native/histogramdd_tensor_bins_cxx.rs "$(GENERATED)/native/rust/tests/histogramdd_tensor_bins_cxx.rs"
	cp tests/native/histogram_options_cxx.rs "$(GENERATED)/native/rust/tests/histogram_options_cxx.rs"
	cp tests/native/quantized_internals_cxx.rs "$(GENERATED)/native/rust/tests/quantized_internals_cxx.rs"
	cp tests/native/cuda_record_stream_cxx.rs "$(GENERATED)/native/rust/tests/cuda_record_stream_cxx.rs"
	cp tests/native/determinism_cxx.rs "$(GENERATED)/native/rust/tests/determinism_cxx.rs"
	cp tests/native/matmul_precision_cxx.rs "$(GENERATED)/native/rust/tests/matmul_precision_cxx.rs"
	cp tests/native/autograd_state_cxx.rs "$(GENERATED)/native/rust/tests/autograd_state_cxx.rs"
	cp tests/native/warning_state_cxx.rs "$(GENERATED)/native/rust/tests/warning_state_cxx.rs"
	cp tests/native/global_random_state_cxx.rs "$(GENERATED)/native/rust/tests/global_random_state_cxx.rs"
	cp tests/native/rng_state_cxx.rs "$(GENERATED)/native/rust/tests/rng_state_cxx.rs"
	cp tests/native/seed_cxx.rs "$(GENERATED)/native/rust/tests/seed_cxx.rs"
	cp tests/native/threading_cxx.rs "$(GENERATED)/native/rust/tests/threading_cxx.rs"
	cp tests/native/autocast_state_cxx.rs "$(GENERATED)/native/rust/tests/autocast_state_cxx.rs"
	cp tests/native/default_dtype_cxx.rs "$(GENERATED)/native/rust/tests/default_dtype_cxx.rs"
	cp tests/native/inference_mode_cxx.rs "$(GENERATED)/native/rust/tests/inference_mode_cxx.rs"
	# The generated CXX crate intentionally shares the checked-in native dependency graph.
	# Copying the lockfile keeps regeneration offline and reproducible.
	cp native/rust/Cargo.lock "$(GENERATED)/native/rust/Cargo.lock"
	"$(TERLC)" run scripts/MaterializePackage.terls -- "$(GENERATED)" "$(CURDIR)"

cxx-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_libtorch_env_missing]: LIBTORCH must point at a released LibTorch distribution" >&2; exit 1; }
	"$(TERLC)" run scripts/RunCxxChecks.terls -- cpu "$(CURDIR)" "$(abspath $(TERLC))" "$(LIBTORCH)"

# Compatibility alias: the typed orchestrator is the single CXX check path.
cxx-check-legacy: cxx-check

cxx-cuda-check: export LD_LIBRARY_PATH := $(CUDA_RUNTIME_PATH):$(LIBTORCH)/lib:$(LD_LIBRARY_PATH)
cxx-cuda-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_cuda_libtorch_env_missing]: LIBTORCH must point at a CUDA-enabled LibTorch distribution" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckLibTorch.terls -- "$(LIBTORCH)" cuda
	"$(TERLC)" run scripts/CheckCudaDevice.terls -- "$(CURDIR)"
	"$(TERLC)" run scripts/RunCxxChecks.terls -- cuda "$(CURDIR)" "$(abspath $(TERLC))" "$(LIBTORCH)"

# Strict CXX-first closure gate for the frozen PyTorch inventories.  The
# CPU/CUDA checks regenerate the package from the same CXX manifest and execute
# the complete generated behavior corpus; the inventory checks reject any new
# unclassified schema or public callable.
pytorch-feature-parity-check: pytorch-parity-check terlan-format-check terlan-grouped-binding-check terlan-function-reference-check terlan-script-check cxx-check
	$(MAKE) cxx-cuda-check TERLC="$(TERLC)" LIBTORCH="$(LIBTORCH)"

# Compatibility alias: CUDA uses the same typed orchestrator and environment setup.
cxx-cuda-check-legacy: cxx-cuda-check

package-check: export LD_LIBRARY_PATH := $(CUDA_RUNTIME_PATH):$(LIBTORCH)/lib:$(LD_LIBRARY_PATH)
package-check: ndarray-dependency-check terlan-format-check terlan-grouped-binding-check terlan-function-reference-check terlan-script-check
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_libtorch_env_missing]: LIBTORCH must point at a released LibTorch distribution" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckLibTorchNegative.terls -- "$(TERLC)" "$(LIBTORCH)"
	"$(TERLC)" run scripts/CheckCudaDeviceNegative.terls -- "$(TERLC)"
	"$(TERLC)" run scripts/CheckLibTorch.terls -- "$(LIBTORCH)"
	"$(TERLC)" run scripts/BuildTorchScriptFixture.terls -- "$(LIBTORCH)" \
		"$(CURDIR)/fixtures/models/double.pt" \
		"$(CURDIR)/fixtures/models/weighted_difference.pt" \
		"$(CURDIR)/fixtures/models/tensor_pair.pt" \
		"$(CURDIR)/fixtures/models/scalar_float.pt" \
		"$(CURDIR)/fixtures/models/scalar_int.pt" \
		"$(CURDIR)/fixtures/models/scalar_bool.pt" \
		"$(CURDIR)/fixtures/models/scalar_string.pt" \
		"$(CURDIR)/fixtures/models/int_list.pt" \
		"$(CURDIR)/fixtures/models/float_list.pt" \
		"$(CURDIR)/fixtures/models/bool_list.pt" \
		"$(CURDIR)/fixtures/models/string_list.pt" \
		"$(CURDIR)/fixtures/models/optional_tensor.pt" \
		"$(CURDIR)/fixtures/models/tensor_list.pt"
	$(MAKE) generate
	LIBTORCH="$(LIBTORCH)" \
	LD_LIBRARY_PATH="$(LIBTORCH)/lib:$${LD_LIBRARY_PATH:-}" \
		cargo test --manifest-path "$(GENERATED)/native/rust/Cargo.toml"
	LIBTORCH="$(LIBTORCH)" \
	LD_LIBRARY_PATH="$(LIBTORCH)/lib:$${LD_LIBRARY_PATH:-}" \
		cargo build --manifest-path "$(GENERATED)/native/rust/Cargo.toml" --bin native-boundary-helper
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- cuda "$(TERLC)" "$(CURDIR)" unavailable
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- dtype "$(TERLC)" "$(CURDIR)"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- construction "$(TERLC)" "$(CURDIR)"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- range "$(TERLC)" "$(CURDIR)"
	"$(TERLC)" run scripts/MaterializePackage.terls -- "$(GENERATED)" "$(CURDIR)"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" --incremental test "$(GENERATED)/tests"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/AutogradTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/NonContiguousTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/RangeBehaviorTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/UnaryParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/UnaryFamilyParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/UnaryStorageParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/UnaryResidualParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/UnaryOutputCompletionTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/BinaryParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/BinaryStorageParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/BinaryMutationParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/MutationCompletionParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/OptionalNumericsParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/BinaryOutputParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/BinaryScalarParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/DivisionParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/MultiplicationParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/FloorDivisionParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/XlogyParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ScaledArithmeticParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ComparisonScalarParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ClampParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LerpParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/WhereParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/IndexFillParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/IndexMutationParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/FillParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ScatterParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LookupParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SpecialParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SpecialOutParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SpecialTensorOutParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SpecialScalarParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SpecialScalarFamilyParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SparseParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ShapeParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ShapeTransformParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ShapeMutationParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PoolingParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PoolingOutputParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/FusedArithmeticParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/SelectionParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LinalgTupleParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LinalgSingleParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LinalgOutToleranceParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LinalgScalarOutParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/FftParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/FftOutParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/DenseParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/StatisticsParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/LossParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/FunctionalParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/GradientKernelsParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/IndexingParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/ViewParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/AtleastSequenceParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/TensorListTransformsParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/AdamTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoTensorOpsTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoSamplingTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoLossTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoMinibatchTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoDeviceParityTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- autograd "$(TERLC)" "$(CURDIR)"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- adam "$(TERLC)" "$(CURDIR)"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- sgd "$(TERLC)" "$(CURDIR)"
	@test -x "$(TERLAN_VM_RUNNER)" || { echo "error[terlan_pytorch_vm_runner_missing]: missing terlan-vm: $(TERLAN_VM_RUNNER)" >&2; exit 1; }
	TERLAN_VM_RUNNER="$(TERLAN_VM_RUNNER)" \
		"$(TERLC)" run scripts/CheckConsumer.terls -- "$(TERLC)" "$(CURDIR)" "$(LIBTORCH)" "$(NDARRAY)"
	python3 ./scripts/write-package-report.py \
		--repository "$(CURDIR)" \
		--generated "$(GENERATED)" \
		--libtorch "$(LIBTORCH)" \
		--output "$(PACKAGE_REPORT)"
	python3 ./scripts/check-package-report.py "$(PACKAGE_REPORT)"

terlan-grouped-binding-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckTerlanLint.terls -- grouped "$(TERLC)" "$(CURDIR)"

terlan-format-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckTerlanLint.terls -- format "$(TERLC)" "$(CURDIR)"

ndarray-dependency-check:
	@test -f "$(NDARRAY)/terlan.toml" || { echo "error[terlan_pytorch_ndarray_package_missing]: NDARRAY must point at terlan-ndarray" >&2; exit 1; }
	$(MAKE) -C "$(NDARRAY)" abi-check TERLC="$(abspath $(TERLC))"
	@test -x "$(NDARRAY_HELPER)" || { echo "error[terlan_pytorch_ndarray_helper_missing]: $(NDARRAY_HELPER)" >&2; exit 1; }

terlan-function-reference-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckTerlanLint.terls -- references "$(TERLC)" "$(CURDIR)"

terlan-script-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckTerlanScripts.terls -- "$(CURDIR)"

pytorch-api-inventory-report:
	"$(TERLC)" run scripts/CheckPytorchApi.terls -- --allow-missing

pytorch-api-inventory-check:
	"$(TERLC)" run scripts/CheckPytorchApi.terls

pytorch-public-api-inventory-freeze:
	@test -n "$(PYTORCH_REFERENCE_PYTHON)" || { echo "error[pytorch_reference_python_missing]: PYTORCH_REFERENCE_PYTHON must name a Python with torch 2.13.0+cpu" >&2; exit 1; }
	"$(PYTORCH_REFERENCE_PYTHON)" scripts/freeze-pytorch-public-api.py \
		--output tests/fixtures/pytorch-2.13-public-api.json

pytorch-public-api-policy-seed:
	python3 scripts/seed-pytorch-public-api-policy.py \
		--aten-reference tests/fixtures/pytorch-2.13-aten-api.json \
		--aten-policy tests/pytorch-api-policy.json \
		--public-reference tests/fixtures/pytorch-2.13-public-api.json \
		--output tests/pytorch-public-api-policy.json
	python3 scripts/classify-pytorch-public-unavailable.py \
		--reference tests/fixtures/pytorch-2.13-public-api.json \
		--policy tests/pytorch-public-api-policy.json

pytorch-public-api-inventory-report:
	"$(TERLC)" run scripts/CheckPytorchApi.terls -- --public --allow-missing

pytorch-public-api-inventory-check:
	"$(TERLC)" run scripts/CheckPytorchApi.terls -- --public

pytorch-parity-report: pytorch-api-inventory-report pytorch-public-api-inventory-report

pytorch-parity-check: pytorch-api-inventory-check pytorch-public-api-inventory-check

autograd-check: ndarray-dependency-check
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_libtorch_env_missing]: LIBTORCH must point at a released LibTorch distribution" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckLibTorch.terls -- "$(LIBTORCH)"
	$(MAKE) generate
	LIBTORCH="$(LIBTORCH)" cargo build --manifest-path "$(GENERATED)/native/rust/Cargo.toml" --bin native-boundary-helper
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/AutogradTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/AdamTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoTensorOpsTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- autograd "$(TERLC)" "$(CURDIR)"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- adam "$(TERLC)" "$(CURDIR)"

ppo-check: autograd-check
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoSamplingTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoLossTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoMinibatchTest.terl"
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/PpoDeviceParityTest.terl"

cuda-check: export LD_LIBRARY_PATH := $(CUDA_RUNTIME_PATH):$(LIBTORCH)/lib:$(LD_LIBRARY_PATH)
cuda-check: export TERLAN_PYTORCH_REQUIRE_CUDA := 1
cuda-check: ndarray-dependency-check
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_cuda_libtorch_env_missing]: LIBTORCH must point at a CUDA-enabled LibTorch distribution" >&2; exit 1; }
	@test -f "$(LIBTORCH)/lib/libtorch_cuda.so" -o \
		-f "$(LIBTORCH)/lib/libtorch_cuda.dylib" -o \
		-f "$(LIBTORCH)/lib/torch_cuda.lib" -o \
		-f "$(LIBTORCH)/bin/torch_cuda.dll" || { echo "error[terlan_pytorch_cuda_distribution]: CUDA LibTorch library is missing" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckCudaDevice.terls
	"$(TERLC)" run scripts/CheckLibTorch.terls -- "$(LIBTORCH)" cuda
	$(MAKE) generate
	LIBTORCH="$(LIBTORCH)" cargo test --manifest-path "$(GENERATED)/native/rust/Cargo.toml"
	LIBTORCH="$(LIBTORCH)" cargo build --manifest-path "$(GENERATED)/native/rust/Cargo.toml" --bin native-boundary-helper
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test tests/pytorch/CudaTest.terl
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test tests/pytorch/TypeRelationsParityTest.terl tests/pytorch/ShapeCompletionParityTest.terl tests/pytorch/SliceParityTest.terl tests/pytorch/TensorJoiningParityTest.terl tests/pytorch/ConvolutionParityTest.terl tests/pytorch/NumericalCompletionParityTest.terl tests/pytorch/DropoutCompletionParityTest.terl tests/pytorch/LinearStorageCompletionParityTest.terl tests/pytorch/LossOutputCompletionParityTest.terl tests/pytorch/IntegrationCompletionParityTest.terl tests/pytorch/GluCompletionParityTest.terl tests/pytorch/OutputCompletionParityTest.terl tests/pytorch/FunctionalCompletionParityTest.terl tests/pytorch/BinaryCrossEntropyCompletionParityTest.terl tests/pytorch/ActivationCompletionParityTest.terl tests/pytorch/PaddingBackwardCompletionParityTest.terl tests/pytorch/ExtendedActivationCompletionParityTest.terl tests/pytorch/CorrectionStatisticsCompletionParityTest.terl tests/pytorch/ShapePartitionParityTest.terl tests/pytorch/RepeatInterleaveParityTest.terl tests/pytorch/NanToNumParityTest.terl tests/pytorch/AggregationParityTest.terl tests/pytorch/PaddingParityTest.terl tests/pytorch/AtleastSequenceParityTest.terl tests/pytorch/TensorListTransformsParityTest.terl
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- cuda "$(TERLC)" "$(CURDIR)" available
	python3 ./scripts/write-cuda-report.py \
		--libtorch "$(LIBTORCH)" \
		--output "$(CUDA_REPORT)"
	python3 ./scripts/check-cuda-report.py "$(CUDA_REPORT)"

ndarray-interop-check: ndarray-dependency-check
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_libtorch_env_missing]: LIBTORCH must point at a released LibTorch distribution" >&2; exit 1; }
	@test -f "$(NDARRAY)/terlan.toml" || { echo "error[terlan_pytorch_ndarray_package_missing]: NDARRAY must point at terlan-ndarray" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckLibTorch.terls -- "$(LIBTORCH)"
	$(MAKE) generate
	LIBTORCH="$(LIBTORCH)" cargo build --manifest-path "$(GENERATED)/native/rust/Cargo.toml" --bin native-boundary-helper
	TERLAN_NDARRAY_NATIVE_BOUNDARY_HELPER_PATH="$(NDARRAY_HELPER)" \
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(CURDIR)/tests/pytorch/NdarrayInteropTest.terl"
	TERLAN_NDARRAY_NATIVE_BOUNDARY_HELPER_PATH="$(NDARRAY_HELPER)" \
	TERLAN_PYTORCH_NATIVE_BOUNDARY_HELPER_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" run scripts/RunAdversarial.terls -- ndarray "$(TERLC)" "$(CURDIR)" "$(NDARRAY)"

fetch-libtorch:
	"$(TERLC)" run scripts/FetchLibTorch.terls

release-check: fetch-libtorch
	$(MAKE) cxx-check LIBTORCH="$(PINNED_LIBTORCH)"
	$(MAKE) package-check LIBTORCH="$(PINNED_LIBTORCH)"
	$(MAKE) ndarray-interop-check \
		LIBTORCH="$(PINNED_LIBTORCH)" NDARRAY="$(NDARRAY)"

fixture-check: ndarray-dependency-check terlan-script-check
	rm -rf fixtures/dist
	mkdir -p fixtures/dist/lib
	cc -shared -fPIC fixtures/libtorch_stable_c_stub.c -o fixtures/dist/lib/libtorch_cpu.so -lm
	cc -shared -fPIC -x c /dev/null -o fixtures/dist/lib/libtorch.so
	cc -shared -fPIC -x c /dev/null -o fixtures/dist/lib/libc10.so
	"$(TERLC)" run scripts/BuildTorchScriptFixture.terls -- "$(CURDIR)/fixtures/dist" \
		"$(CURDIR)/fixtures/models/double.pt" \
		"$(CURDIR)/fixtures/models/weighted_difference.pt" \
		"$(CURDIR)/fixtures/models/tensor_pair.pt" \
		"$(CURDIR)/fixtures/models/scalar_float.pt" \
		"$(CURDIR)/fixtures/models/scalar_int.pt" \
		"$(CURDIR)/fixtures/models/scalar_bool.pt" \
		"$(CURDIR)/fixtures/models/scalar_string.pt" \
		"$(CURDIR)/fixtures/models/int_list.pt" \
		"$(CURDIR)/fixtures/models/float_list.pt" \
		"$(CURDIR)/fixtures/models/bool_list.pt" \
		"$(CURDIR)/fixtures/models/string_list.pt" \
		"$(CURDIR)/fixtures/models/optional_tensor.pt" \
		"$(CURDIR)/fixtures/models/tensor_list.pt"
	@echo "[terlan-pytorch-fixture-check] stable-C compatibility fixture validated; CXX gates use real LibTorch headers."

clean:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/Clean.terls -- "$(CURDIR)"
