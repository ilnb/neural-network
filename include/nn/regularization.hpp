#pragma once

#include "tensor/tensor.hpp"

namespace NN::Reg {
  Tensor l1(const Tensor &weight);

  Tensor l2(const Tensor &weight);

  // `alpha` is L1-coefficient
  Tensor elastic_net(const Tensor &weight, const dtype alpha);

  namespace Detail {
    Tensor l1(const Tensor &weight);

    Tensor l2(const Tensor &weight);

    Tensor elastic_net(const Tensor &weight, const dtype alpha);
  } // namespace Detail
} // namespace NN::Reg
