#pragma once

#include "tensor/tensor.hpp"

namespace NN::Loss {
  // Mean Squared Error
  Tensor mse(const Tensor &pred, const Tensor &target);

  // Binary Cross Entropy loss
  Tensor bce(const Tensor &pred, const Tensor &target);

  // Cross Entropy loss, with dim as the classification axis
  Tensor ce(const Tensor &pred, const Tensor &target, size_t dim);

  // CE loss over the last axis
  inline Tensor ce(const Tensor &pred, const Tensor &target) {
    Tensor::validate(pred, target);
    return ce(pred, target, pred.p->shape.size() - 1);
  }

  namespace Detail {
    Tensor mse(const Tensor &pred, const Tensor &target);

    Tensor bce(const Tensor &pred, const Tensor &target);

    Tensor ce(const Tensor &pred, const Tensor &target, size_t dim);
  } // namespace Detail
} // namespace NN::Loss
