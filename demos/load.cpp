// Uses the weights saved in the xor demo to test xor-ing
#include "module/linear.hpp"
#include "module/relu.hpp"
#include "module/sequential.hpp"
#include "serial/checkpoint.hpp"

#include <filesystem>
#include <print>

using namespace NN;

int main() {
  Sequential model(
      // clang-format off
    Linear(2, 4, 0),
    ReLU(),
    Linear(4, 1, 0)
      // clang-format on
  );

  Serial::load(model, "saves/xor.nnst");
  model.eval();
  Tensor test(1, 2);
  for (const auto &l : std::vector<std::vector<int>>{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }) {
    test.set_data(l);
    auto out = model(test);
    std::println("prediction for {}: {}", test.p->data, out.p->data);
  }
  return 0;
}
