#pragma once

#include "tensor/tensor.hpp"

namespace NN::Reg {
  // L1 norm of `weight`: `sum_i |w_i|`.
  //
  // The gradient at `w_i == 0` is taken as `0`.
  //
  // Throws `std::invalid_argument` if `weight` is empty.
  Tensor l1(const Tensor &weight);

  // Half the squared L2 norm: `0.5 * sum_i w_i^2`.
  //
  // Throws `std::invalid_argument` if `weight` is empty.
  Tensor l2(const Tensor &weight);

  // Elastic net: `alpha * sum_i |w_i| + (1 - alpha) * 0.5 * sum_i w_i^2`.
  //
  // `alpha` is the L1 coefficient: 0 is pure L2, 1 is pure L1.
  //
  // Throws `std::invalid_argument` if `weight` is empty,
  //  or `alpha` not in `[0, 1]`.
  Tensor elastic_net(const Tensor &weight, const dtype alpha);

  // Raw regularization kernels.
  //
  // These compute the scalar penalty only and never attach a `GradFn`; the
  // public wrappers above add the backward rules. Validation matches the
  // public versions.
  namespace Detail {
    Tensor l1(const Tensor &weight);

    Tensor l2(const Tensor &weight);

    Tensor elastic_net(const Tensor &weight, const dtype alpha);
  } // namespace Detail
} // namespace NN::Reg
