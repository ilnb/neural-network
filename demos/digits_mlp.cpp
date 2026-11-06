#include "autograd/grad_mode.hpp"
#include "data/dataloader.hpp"
#include "data/dataset.hpp"
#include "data/split.hpp"
#include "data/tensor_dataset.hpp"
#include "loss/loss.hpp"
#include "module/linear.hpp"
#include "module/relu.hpp"
#include "module/sequential.hpp"
#include "module/softmax.hpp"
#include "optim/adam.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace NN;

struct Metrics {
  double loss = 0, acc = 0;
};

// Mean loss and accuracy of `model` over a whole dataset.
//
// Batches are averaged by sample count, since the last one may be smaller.
Metrics evaluate(Sequential &model, const Data::Dataset &ds, const size_t batch_size) {
  Autograd::NoGradGuard g;

  Metrics m;
  size_t seen = 0;

  for (size_t i = 0; i < ds.size(); i += batch_size) {
    const size_t n = std::min(batch_size, ds.size() - i);

    std::vector<size_t> idx(n);
    std::ranges::iota(idx, i);

    auto [xb, yb] = ds.batch(idx);
    auto pred = model(xb);
    const size_t C = pred.p->shape.back();

    m.loss += static_cast<double>(Loss::ce(pred, yb).item()) * static_cast<double>(n);
    for (size_t r = 0; r < n; r++) {
      size_t best = 0, truth = 0;
      for (size_t c = 1; c < C; c++) {
        if (pred.p->data[r * C + c] > pred.p->data[r * C + best])
          best = c;
        if (yb.p->data[r * C + c] > yb.p->data[r * C + truth])
          truth = c;
      }
      m.acc += static_cast<double>(best == truth);
    }
    seen += n;
  }

  m.loss /= static_cast<double>(seen);
  m.acc /= static_cast<double>(seen);
  return m;
}

int main() {
  Tensor X, Y;
  {
    std::ifstream f("assets/digits.csv");
    if (!f) {
      std::println(stderr, "Not able to open digits.csv");
      exit(1);
    }
    std::vector<dtype> feats, labels;
    std::string line;
    size_t lines = 0;
    while (std::getline(f, line)) {
      if (line.empty())
        continue;
      size_t col = 0;
      for (auto tok : std::views::split(line, ',')) {
        dtype v{};
        auto [ptr, err_c] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
        if (err_c != std::errc())
          throw std::runtime_error("bad number on line " + std::to_string(lines + 1));
        if (col < 64)
          feats.push_back(v / 16);
        else if (col == 64)
          labels.push_back(v);
        col++;
      }
      if (col != 65)
        throw std::runtime_error("expected 65 columns on line " + std::to_string(lines + 1));
      lines++;
    }

    std::vector<dtype> onehot(lines * 10);
    for (size_t i = 0; i < lines; i++)
      onehot[i * 10 + static_cast<size_t>(labels[i])] = 1;

    X.set_dims(lines, 64).set_data(std::move(feats));
    Y.set_dims(lines, 10).set_data(std::move(onehot));
  }

  Sequential model(
      // clang-format off
    Linear(64, 16, 0),
    ReLU(),
    Linear(16, 16, 0),
    ReLU(),
    Linear(16, 10, 0),
    Softmax(1)
      // clang-format on
  );

  Optim::Adam opt(model.params(), { .lr = 0.001 });

  Data::TensorDataset ds(X, Y);

  auto [train_set, test_set] = Data::train_test_split(ds, { .test_size = 0.2, .seed = 0 });

  Data::DataLoader loader(train_set, { .batch_size = 32, .shuf = true, .seed = 0 });

  for (size_t e = 0; e < 500; e++) {
    model.train();
    for (auto [x_b, y_b] : loader.epoch()) {
      auto pred = model(x_b);
      auto loss = Loss::ce(pred, y_b);

      loss.backward();
      opt.step();
      opt.zero_grad();
    }

    if (e % 100 == 0) {
      model.eval();
      const auto tr = evaluate(model, *train_set, 256);
      const auto te = evaluate(model, *test_set, 256);
      std::println("epoch {:>4}: train loss = {:.4e}, acc = {:.2f}% | test loss = {:.4e}, acc = {:.2f}%", e, tr.loss, tr.acc * 100, te.loss, te.acc * 100);
    }
  }

  auto [loss, acc] = evaluate(model, *test_set, 256);
  std::println("Test loss: {:.4e}, acc = {:.2f}%", loss, acc * 100);
  return 0;
}
