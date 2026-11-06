#pragma once

#include "activations/activations.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"

namespace NN {
  struct Softmax : Module {
    size_t dim;
    Softmax(size_t dim) : dim(dim) {}
    Tensor forward(const Tensor &x) override { return NN::Act::softmax(x, dim); }
    std::vector<Tensor> params() const override { return {}; }
  };
} // namespace NN
