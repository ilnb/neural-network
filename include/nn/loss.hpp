#pragma once

#include "tensor/tensor.hpp"

namespace NN::Loss {
  // Mean squared error: `(1/n) * sum_i (pred_i - target_i)^2`.
  //
  // Throws `std::invalid_argument` if either tensor is empty,
  // malformed, or the shapes differ.
  Tensor mse(const Tensor &pred, const Tensor &target);

  // Binary cross entropy, averaged over all elements:
  // `-(1/n) * sum_i [t_i log p_i + (1 - t_i) log(1 - p_i)]`.
  //
  // Targets must be binary and predictions must lie in `[0, 1]`.
  // A term is 0 when `p_i == t_i` at either extreme,
  // and the loss is `inf` when `p_i == 0, t_i == 1` or
  // `p_i == 1, t_i == 0`. NaN predictions propagate to the result.
  //
  // Throws `std::invalid_argument` if either tensor is empty or malformed, the
  // shapes differ, a target is not binary, or a prediction lies outside
  // `[0, 1]`.
  Tensor bce(const Tensor &pred, const Tensor &target);

  // Cross entropy along the classification axis `dim`:
  // `-(1/slices) * sum_i t_i log p_i`, with `slices = num_el / shape[dim]`,
  // so the loss is averaged over the slices.
  //
  // Targets are soft labels in `[0, 1]`; a zero target contributes nothing.
  // Probabilities must be non-negative, and `p_i == 0` with `t_i > 0` gives
  // `inf`.
  //
  // Throws `std::out_of_range` if `dim >= rank`, and `std::invalid_argument`
  // if either tensor is empty or malformed, the shapes differ, a target lies
  // outside `[0, 1]`, or a probability is negative.
  Tensor ce(const Tensor &pred, const Tensor &target, size_t dim);

  // CE loss over the last axis. `pred` must have at least one axis.
  inline Tensor ce(const Tensor &pred, const Tensor &target) {
    Tensor::validate(pred, target);
    return ce(pred, target, pred.p->shape.size() - 1);
  }

  // Raw loss kernels.
  //
  // These compute the scalar loss only and never attach a `GradFn`; the
  // public wrappers above add the backward rules. Validation matches the
  // public versions.
  namespace Detail {
    Tensor mse(const Tensor &pred, const Tensor &target);

    Tensor bce(const Tensor &pred, const Tensor &target);

    Tensor ce(const Tensor &pred, const Tensor &target, size_t dim);
  } // namespace Detail
} // namespace NN::Loss
