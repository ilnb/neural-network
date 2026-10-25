#pragma once

#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Autograd {
  // Undos broadcasting to target `shape`.
  Tensor sum_to_shape(const Tensor &g, const std::vector<size_t> &shape);

  // Adds in the gradient.
  void accumulate(Impl &p, const Tensor &g);

  // Backprop
  void backward(const Tensor &root);
} // namespace NN::Autograd
