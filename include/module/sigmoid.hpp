#pragma once

#include "activations/activations.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"

namespace NN {
  struct Sigmoid : Stateless {
    Tensor forward(const Tensor &x) override { return Act::sigmoid(x); }
  };
} // namespace NN
