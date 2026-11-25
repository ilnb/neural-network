#pragma once

#include "activations/activations.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"

namespace NN {
  struct Softmax : Stateless {
    size_t dim;
    Softmax(size_t dim) : dim(dim) {}
    [[nodiscard]] Tensor forward(const Tensor &x) override { return Act::softmax(x, dim); }
  };
} // namespace NN
