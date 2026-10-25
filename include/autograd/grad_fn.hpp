#pragma once

#include "tensor/tensor.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace NN::Autograd {
  struct GradFn {
    std::vector<std::shared_ptr<Impl>> inputs;
    std::function<void(const Tensor &)> backward;
  };
} // namespace NN::Autograd
