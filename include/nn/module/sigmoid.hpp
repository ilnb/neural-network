#pragma once

#include "nn/activations.hpp"
#include "nn/module/base.hpp"
#include "tensor/tensor.hpp"

namespace NN {
  struct Sigmoid : Module {
    Tensor forward(const Tensor &x) const override { return NN::Act::sigmoid(x); }

    std::vector<Tensor> params() const override { return {}; }
  };
} // namespace NN
