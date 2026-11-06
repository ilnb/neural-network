#pragma once

#include "module/base.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace NN {
  struct Linear : Module {
    Tensor w, b;
    std::mt19937 rng;

    Linear(size_t in, size_t out, uint32_t seed = std::random_device{}()) : w({ in, out }), b({ out }), rng(seed) {
      std::normal_distribution<dtype> dist(0, std::sqrt(2 / static_cast<dtype>(in))); // He init
      std::ranges::generate(w.p->data, [&]() { return dist(rng); });

      w.set_requires_grad(true);
      b.set_requires_grad(true);
    }

    Tensor forward(const Tensor &x) override { return x.matmul(w) + b; }
    std::vector<Tensor> params() const override { return { w, b }; }
  };
} // namespace NN
