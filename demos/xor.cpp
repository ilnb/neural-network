#include "nn/loss.hpp"
#include "nn/module/linear.hpp"
#include "nn/module/relu.hpp"
#include "nn/module/sequential.hpp"
#include "optim/adam.hpp"
#include "tensor/tensor.hpp"
#include <print>

int main() {
  NN::Sequential model(
      // clang-format off
    NN::Linear(2, 4),
    NN::ReLU(),
    NN::Linear(4, 1)
      // clang-format on
  );

  NN::Tensor x(4, 2);
  x.set_data(0, 0, 0, 1, 1, 0, 1, 1);
  NN::Tensor y(4, 1);
  y.set_data(0, 1, 1, 0);
  NN::Optim::Adam opt(model.params());

  for (size_t e = 0; e < 1500; e++) {
    auto pred = model.forward(x);
    auto loss = NN::Loss::mse(pred, y);

    loss.backward();
    opt.step();
    opt.zero_grad();

    if (e % 100 == 0)
      std::println("epoch {:>4}: loss = {}", e, loss.item());
  }

  NN::Tensor test(1, 2);
  test.set_data(0, 1);

  auto out = model.forward(test);
  std::println("prediction for {}: {}", test.p->data, out.p->data);
}
