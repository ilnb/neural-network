#pragma once

#include "activations/activations.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"

namespace NN {
  struct ReLU : Stateless {
    Tensor forward(const Tensor &x) override { return Act::relu(x); }
  };
} // namespace NN
