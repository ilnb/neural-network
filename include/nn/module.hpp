#pragma once

#include "tensor/tensor.hpp"
#include <vector>

namespace NN {
  struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor &x) const = 0;
    virtual std::vector<Tensor> params() const = 0;
  };
} // namespace NN
