// XOR as a 2 -> 4 -> 1 network, trained with Adam on mean squared error.
//
// The targets are 0 and 1 and the output is a single unit with no activation,
// so this is a regression problem rather than a classification one: the values
// converge to 0/1 instead of becoming probabilities.
#include "data/dataloader.hpp"
#include "loss/loss.hpp"
#include "module/linear.hpp"
#include "module/relu.hpp"
#include "module/sequential.hpp"
#include "optim/adam.hpp"

#include <print>

using namespace NN;

int main() {
  // Sequentially runs the layers, transferring the outputs to the next layer.
  Sequential model(
      // clang-format off
    Linear(2, 4, 0),
    ReLU(),
    Linear(4, 1, 0)
      // clang-format on
  );

  // Truth table: (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0.
  Tensor x(4, 2);
  x.set_data(0, 0, 0, 1, 1, 0, 1, 1);
  Tensor y(4, 1);
  y.set_data(0, 1, 1, 0);
  Optim::Adam opt(model.params(), { .lr = 0.02 });

  // Batch size 2 means two optimizer steps per epoch, so the epoch count below
  // is really a step count.
  Data::DataLoader dl(Data::TensorDataset(x, y), { .batch_size = 2, .shuf = true, .seed = 0 });

  for (size_t e = 0; e < 1500; e++) {
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
