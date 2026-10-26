#pragma once

#include "nn/module.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <concepts>
#include <vector>

namespace NN::Act {
  // Standard sigmoid.
  dtype sigmoid(dtype v);

  // Element-wise `max(x, 0)`; NaN is propagated unchanged.
  Tensor relu(const Tensor &x);

  // Element-wise logistic sigmoid.
  Tensor sigmoid(const Tensor &x);

  // Normalizes `x` along `dim`: `exp(x_k) / sum_j exp(x_j)`, with the slice
  // maximum subtracted before exponentiating for numerical stability.
  //
  // Every slice along `dim` sums to 1. A rank-0 tensor is treated as a
  // one-element slice and yields 1. NaN and infinite entries poison their
  // slice.
  //
  // Throws `std::invalid_argument` if `x` is empty, and `std::out_of_range`
  // if `dim` is out of bounds (on a rank-0 tensor only `dim == 0` is valid).
  Tensor softmax(const Tensor &x, size_t dim);

  // Logarithm of `softmax(x, dim)`; A rank-0 tensor yields 0.
  // NaN and infinite entries poison their slice.
  //
  // Throws the same exceptions as `softmax`.
  Tensor log_softmax(const Tensor &x, size_t dim);

  // Relu module for pipeline
  struct ReLU : Module {
    Tensor forward(const Tensor &x) const override { return relu(x); }

    std::vector<Tensor> params() const override { return {}; }
  };

  // Sigmoid module for pipeline
  struct Sigmoid : Module {
    Tensor forward(const Tensor &x) const override { return sigmoid(x); }

    std::vector<Tensor> params() const override { return {}; }
  };

  // Raw activation kernels.
  //
  // These perform the forward computation only and never attach a `GradFn`.
  // The public wrappers above add the backward rules when `x` requires grad;
  // internal code that must not build a graph calls these directly.
  namespace Detail {
    Tensor relu(const Tensor &x);

    Tensor sigmoid(const Tensor &x);

    Tensor softmax(const Tensor &x, size_t dim);

    Tensor log_softmax(const Tensor &x, size_t dim);

    // Applies `f` to every element of `x` and returns a copy.
    // The result is a leaf.
    template <typename F>
      requires requires(F f, dtype x) {
        { f(x) } -> std::convertible_to<dtype>;
      }
    Tensor map(const Tensor &x, F f) {
      Tensor r = x.clone();
      std::ranges::for_each(r.p->data, [f](dtype &v) { v = f(v); });
      return r;
    }
  } // namespace Detail
} // namespace NN::Act
