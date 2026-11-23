#pragma once

#include "tensor/tensor.hpp"

namespace NN::Loss {
  // Mean squared error: `(1/n) * sum_i (pred_i - target_i)^2`.
  //
  // Throws `std::invalid_argument` if either tensor is empty,
  // malformed, or the shapes differ.
  [[nodiscard]] Tensor mse(const Tensor &pred, const Tensor &target);

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
  [[nodiscard]] Tensor bce(const Tensor &pred, const Tensor &target);

  // Cross entropy along the classification axis `dim`, taking logits:
  // `-(1/slices) * sum_i t_i log softmax(x)_i`, with
  // `slices = num_el / shape[dim]`, so the loss is averaged over the slices.
  //
  // Softmax is applied internally, subtracting the maximum logit before
  // exponentiating, so logits of any magnitude are accepted and the backward
  // pass never divides by a probability. Targets are soft labels in `[0, 1]`;
  // a zero target contributes nothing. A `-inf` logit with target mass gives
  // `inf`, and NaN logits propagate to the result.
  //
  // Throws `std::out_of_range` if `dim >= rank`, and `std::invalid_argument`
  // if either tensor is empty or malformed, the shapes differ, or a target lies
  // outside `[0, 1]`.
  [[nodiscard]] Tensor ce(const Tensor &pred, const Tensor &target, size_t dim);

  // CE loss over the last axis. `pred` must have at least one axis.
  [[nodiscard]] inline Tensor ce(const Tensor &pred, const Tensor &target) {
    Tensor::validate(pred, target);
    return ce(pred, target, pred.p->shape.size() - 1);
  }

  // Raw loss kernels.
  //
  // These compute the scalar loss only and never attach a `GradFn`; the
  // public wrappers above add the backward rules. Validation matches the
  // public versions.
  namespace Detail {
    [[nodiscard]] Tensor mse(const Tensor &pred, const Tensor &target);

    [[nodiscard]] Tensor bce(const Tensor &pred, const Tensor &target);

    [[nodiscard]] Tensor ce(const Tensor &pred, const Tensor &target, size_t dim);
  } // namespace Detail
} // namespace NN::Loss
