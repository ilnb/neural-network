#include "data/dataloader.hpp"
#include "data/tensor_dataset.hpp"
#include "loss/loss.hpp"
#include "module/linear.hpp"
#include "module/relu.hpp"
#include "module/sequential.hpp"
#include "optim/adam.hpp"
#include "tensor/tensor.hpp"
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

  Tensor x(4, 2);
  x.set_data(0, 0, 0, 1, 1, 0, 1, 1);
  Tensor y(4, 1);
  y.set_data(0, 1, 1, 0);
  Optim::Adam opt(model.params(), { .lr = 0.02 });

  Data::DataLoader dl(Data::TensorDataset(x, y), { .batch_size = 2, .shuf = true, .seed = 0 });

  for (size_t e = 0; e < 500; e++) {
    model.train();
    Tensor loss;
    for (auto [x_b, y_b] : dl.epoch()) {
      auto pred = model(x_b);
      loss = Loss::mse(pred, y_b);

      loss.backward();
      opt.step();
      opt.zero_grad();
    }

    if (e % 100 == 0) {
      model.eval();
      std::println("epoch {:>4}: loss = {}", e, loss.item());
    }
  }

  model.eval();
  Tensor test(1, 2);
  for (const auto &l : std::vector<std::vector<int>>{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }) {
    test.set_data(l);
    auto out = model(test);
    std::println("prediction for {}: {}", test.p->data, out.p->data);
  }
}
