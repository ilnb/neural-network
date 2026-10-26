#include "nn/activations.hpp"
#include "nn/linear.hpp"
#include "nn/loss.hpp"
#include "nn/sequential.hpp"
#include "optim/sgd.hpp"
#include "tensor/tensor.hpp"
#include <print>

int main() {
  NN::Sequential model(
      // clang-format off
    NN::Linear(2, 4),
    NN::Act::ReLU(),
    NN::Linear(4, 1)
      // clang-format on
  );

  NN::Tensor x(4, 2);
  x.set_data(0, 0, 0, 1, 1, 0, 1, 1);
  NN::Tensor y(4, 1);
  y.set_data(0, 1, 1, 0);
  NN::Optim::SGD opt(model.params(), 0.01);

  for (size_t e = 0; e < 5000; e++) {
    auto pred = model.forward(x);
    auto loss = NN::Loss::mse(pred, y);

    loss.backward();
    opt.step();
    opt.zero_grad();

    if (e % 1000 == 0)
      std::println("epoch {:>4}: loss = {}", e, loss.item());
  }

  NN::Tensor test(1, 2);
  test.set_data(0, 1);

  auto out = model.forward(test);
  std::println("prediction for {}: {}", test.p->data, out.p->data);
}
