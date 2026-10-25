#pragma once

#include "tensor/tensor.hpp"

namespace NN::Act {
  dtype sigmoid(dtype v);

  Tensor relu(const Tensor &x);

  Tensor sigmoid(const Tensor &x);

  Tensor softmax(const Tensor &x, size_t dim);

  Tensor log_softmax(const Tensor &x, size_t dim);

  namespace Detail {
    Tensor relu(const Tensor &x);

    Tensor sigmoid(const Tensor &x);

    Tensor softmax(const Tensor &x, size_t dim);

    Tensor log_softmax(const Tensor &x, size_t dim);
  } // namespace Detail
} // namespace NN::Act
