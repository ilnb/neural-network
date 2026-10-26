#pragma once

#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace NN {
  struct Linear {
    Tensor w, b;

    Linear(size_t in, size_t out, uint32_t seed = 0) : w({in, out}), b({out}) {
      std::mt19937 m(seed);
      std::normal_distribution<dtype> dist(0, std::sqrt(2 / static_cast<dtype>(in)));
      std::ranges::generate(w.p->data, [&]() { return dist(m); });

      w.set_requires_grad(true);
      b.set_requires_grad(true);
    }

    inline Tensor forward(const Tensor &x) const { return x.matmul(w) + b; }
    inline std::vector<Tensor> params() const { return {w, b}; }
  };
} // namespace NN
