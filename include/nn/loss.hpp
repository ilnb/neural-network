#pragma once

#include "tensor/tensor.hpp"

namespace NN::Loss {
  // Mean Squared Error
  dtype mse(const Tensor &pred, const Tensor &target);

  // Binary Cross Entropy loss
  dtype bce(const Tensor &pred, const Tensor &target);

  // Cross Entropy loss, with dim as the classification axis
  dtype ce(const Tensor &pred, const Tensor &target, size_t dim);

  // CE loss over the last axis
  inline dtype ce(const Tensor &pred, const Tensor &target) {
    Tensor::validate(pred, target);
    return ce(pred, target, pred.p->shape.size() - 1);
  }
} // namespace NN::Loss
