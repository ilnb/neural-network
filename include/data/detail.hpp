#pragma once

#include "tensor/tensor.hpp"

#include <vector>

// Raw data helpers.
//
// Everything here performs forward computation only: no `GradFn` is attached
// and `requires_grad` is never consulted.
namespace NN::Data::Detail {
  // Stacks tensors of identical shape along a new leading axis.
  //
  // The result has shape `{n} + shape`, where `n` is `ts.size()`.
  //
  // Throws `std::invalid_argument` if `ts` is empty, or if the tensors do not
  // all share a shape.
  Tensor stack(const std::vector<Tensor> &ts);
} // namespace NN::Data::Detail
