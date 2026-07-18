TERLC ?= ../terlan/target/debug/terlc
LIBTORCH ?=
MANIFEST := bindings/libtorch-c.json
GENERATED := generated
PINNED_LIBTORCH := $(CURDIR)/vendor/libtorch-2.13.0+cpu/libtorch

.PHONY: generate fetch-libtorch fixture-check package-check release-check clean

generate:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	rm -rf "$(GENERATED)"
	"$(TERLC)" bind c --manifest "$(MANIFEST)" --out "$(GENERATED)"
	mkdir -p "$(GENERATED)/tests/pytorch"
	cp tests/pytorch/DispatcherTest.terl "$(GENERATED)/tests/pytorch/DispatcherTest.terl"
	./scripts/materialize-package.sh "$(GENERATED)" "$(CURDIR)"

package-check:
	@test -n "$(LIBTORCH)" || { echo "error[terlan_pytorch_libtorch_env_missing]: LIBTORCH must point at a released LibTorch distribution" >&2; exit 1; }
	./scripts/check-libtorch-negative.sh
	./scripts/check-libtorch.sh "$(LIBTORCH)"
	./scripts/build-torchscript-fixture.sh "$(LIBTORCH)" \
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
		"$(CURDIR)/fixtures/models/string_list.pt"
	$(MAKE) generate
	LIBTORCH="$(LIBTORCH)" cargo test --manifest-path "$(GENERATED)/native/rust/Cargo.toml"
	LIBTORCH="$(LIBTORCH)" cargo build --manifest-path "$(GENERATED)/native/rust/Cargo.toml" --bin native-boundary-helper
	./scripts/materialize-package.sh "$(GENERATED)" "$(CURDIR)"
	TERLAN_SAFE_NATIVE_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(GENERATED)/tests"
	./scripts/check-consumer.sh "$(TERLC)" "$(CURDIR)" "$(LIBTORCH)"
	python3 ./scripts/write-package-report.py \
		--repository "$(CURDIR)" \
		--generated "$(GENERATED)" \
		--libtorch "$(LIBTORCH)" \
		--output "$(GENERATED)/package-execution-report.json"
	python3 ./scripts/check-package-report.py "$(GENERATED)/package-execution-report.json"

fetch-libtorch:
	./scripts/fetch-libtorch.sh

release-check: fetch-libtorch
	$(MAKE) package-check LIBTORCH="$(PINNED_LIBTORCH)"

fixture-check:
	rm -rf fixtures/dist
	mkdir -p fixtures/dist/lib
	cc -shared -fPIC fixtures/libtorch_stable_c_stub.c -o fixtures/dist/lib/libtorch_cpu.so -lm
	cc -shared -fPIC -x c /dev/null -o fixtures/dist/lib/libtorch.so
	cc -shared -fPIC -x c /dev/null -o fixtures/dist/lib/libc10.so
	./scripts/build-torchscript-fixture.sh "$(CURDIR)/fixtures/dist" \
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
		"$(CURDIR)/fixtures/models/string_list.pt"
	$(MAKE) generate
	LIBTORCH="$(CURDIR)/fixtures/dist" cargo test --manifest-path "$(GENERATED)/native/rust/Cargo.toml"
	LIBTORCH="$(CURDIR)/fixtures/dist" cargo build --manifest-path "$(GENERATED)/native/rust/Cargo.toml" --bin native-boundary-helper
	./scripts/materialize-package.sh "$(GENERATED)" "$(CURDIR)"
	TERLAN_SAFE_NATIVE_PATH="$(CURDIR)/$(GENERATED)/native/rust/target/debug/native-boundary-helper" \
		"$(TERLC)" test "$(GENERATED)/tests"
	./scripts/check-consumer.sh "$(TERLC)" "$(CURDIR)" "$(CURDIR)/fixtures/dist"

clean:
	rm -rf "$(GENERATED)" fixtures/dist fixtures/models/double.pt \
		fixtures/models/weighted_difference.pt fixtures/models/tensor_pair.pt \
		fixtures/models/scalar_float.pt fixtures/models/scalar_int.pt \
		fixtures/models/scalar_bool.pt fixtures/models/scalar_string.pt \
		fixtures/models/int_list.pt fixtures/models/float_list.pt \
		fixtures/models/bool_list.pt fixtures/models/string_list.pt
