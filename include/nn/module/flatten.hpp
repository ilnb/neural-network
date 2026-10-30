#pragma once

#include "nn/module/base.hpp"
#include "tensor/tensor.hpp"
#include <stdexcept>
#include <vector>

namespace NN {
  struct Flatten : Module {
    Tensor forward(const Tensor &x) const override {
      if (x.empty() || x.p->shape.empty())
        throw std::invalid_argument("Non-empty batched tensor expected.");

      const size_t batch = x.p->shape[0];
      return x.reshape(batch, x.p->num_el / batch);
    }

    std::vector<Tensor> params() const override { return {}; }
  };
} // namespace NN
