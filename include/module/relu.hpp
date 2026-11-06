#pragma once

#include "activations/activations.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"

namespace NN {
  struct ReLU : Module {
    Tensor forward(const Tensor &x) override { return NN::Act::relu(x); }

    std::vector<Tensor> params() const override { return {}; }
  };
} // namespace NN
