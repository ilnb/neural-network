#pragma once

#include "tensor/tensor.hpp"
#include <vector>

namespace NN {
  struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor &x) = 0;
    virtual std::vector<Tensor> params() const = 0;
    virtual void train(bool mode = true) { training = mode; }
    void eval() { train(false); }

  protected:
    bool training = true;
  };
} // namespace NN
