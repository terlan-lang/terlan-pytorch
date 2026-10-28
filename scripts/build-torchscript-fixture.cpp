#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <torch/script.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 14) {
    std::cerr << "usage: build-torchscript-fixture <double-output.pt> "
                 "<weighted-difference-output.pt> <tensor-pair-output.pt> "
                 "<scalar-float-output.pt> <scalar-int-output.pt> "
                 "<scalar-bool-output.pt> <scalar-string-output.pt> "
                 "<int-list-output.pt> <float-list-output.pt> "
                 "<bool-list-output.pt> <string-list-output.pt> "
                 "<optional-tensor-output.pt> <tensor-list-output.pt>\n";
    return 2;
  }

  try {
    torch::jit::Module module("TerlanDoubleModel");
    module.define(R"JIT(
      def forward(self, input: Tensor) -> Tensor:
        return input * 2.0
    )JIT");
    module.save(argv[1]);

    torch::jit::Module weighted_difference("TerlanWeightedDifferenceModel");
    weighted_difference.define(R"JIT(
      def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return left - right * 2.0
    )JIT");
    weighted_difference.save(argv[2]);

    torch::jit::Module tensor_pair("TerlanTensorPairModel");
    tensor_pair.define(R"JIT(
      def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        return input * 2.0, input * 3.0
    )JIT");
    tensor_pair.save(argv[3]);

    torch::jit::Module scalar_float("TerlanScalarFloatModel");
    scalar_float.define(R"JIT(
      def forward(self, input: Tensor) -> float:
        return float(input.item()) * 2.0
    )JIT");
    scalar_float.save(argv[4]);

    torch::jit::Module scalar_int("TerlanScalarIntModel");
    scalar_int.define(R"JIT(
      def forward(self, input: Tensor) -> int:
        return int(input.item()) + 3
    )JIT");
    scalar_int.save(argv[5]);

    torch::jit::Module scalar_bool("TerlanScalarBoolModel");
    scalar_bool.define(R"JIT(
      def forward(self, input: Tensor) -> bool:
        return input.item() > 0
    )JIT");
    scalar_bool.save(argv[6]);

    torch::jit::Module scalar_string("TerlanScalarStringModel");
    scalar_string.define(R"JIT(
      def forward(self, input: Tensor) -> str:
        if input.item() > 0:
          return "positive"
        return "non_positive"
    )JIT");
    scalar_string.save(argv[7]);

    torch::jit::Module int_list("TerlanIntListModel");
    int_list.define(R"JIT(
      def forward(self, input: Tensor) -> List[int]:
        value = int(input.item())
        return [value, value + 1, value + 2]
    )JIT");
    int_list.save(argv[8]);

    torch::jit::Module float_list("TerlanFloatListModel");
    float_list.define(R"JIT(
      def forward(self, input: Tensor) -> List[float]:
        value = float(input.item())
        return [value, value + 0.5, value + 1.0]
    )JIT");
    float_list.save(argv[9]);

    torch::jit::Module bool_list("TerlanBoolListModel");
    bool_list.define(R"JIT(
      def forward(self, input: Tensor) -> List[bool]:
        value = int(input.item())
        return [value > 0, value == 0, value < 0]
    )JIT");
    bool_list.save(argv[10]);

    torch::jit::Module string_list("TerlanStringListModel");
    string_list.define(R"JIT(
      def forward(self, input: Tensor) -> List[str]:
        if input.item() > 0:
          return ["positive", "", "ready"]
        return ["non_positive", "", "waiting"]
    )JIT");
    string_list.save(argv[11]);

    torch::jit::Module optional_tensor("TerlanOptionalTensorModel");
    optional_tensor.define(R"JIT(
      def forward(self, input: Tensor) -> Optional[Tensor]:
        if input.item() > 0:
          return input * 2.0
        return None
    )JIT");
    optional_tensor.save(argv[12]);

    torch::jit::Module tensor_list("TerlanTensorListModel");
    tensor_list.define(R"JIT(
      def forward(self, input: Tensor) -> List[Tensor]:
        return [input * 2.0, input]
    )JIT");
    tensor_list.save(argv[13]);
  } catch (const c10::Error& error) {
    std::cerr << "failed to create TorchScript fixture: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
