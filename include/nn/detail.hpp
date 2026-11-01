#pragma once

#include "tensor/tensor.hpp"
#include <algorithm>
#include <concepts>

namespace NN::Detail {
  template <typename F>
    requires requires(F f, dtype x) {
      { f(x) } -> std::convertible_to<dtype>;
    }
  Tensor map_(Tensor &x, F f) {
    std::ranges::for_each(x.p->data, [f](dtype &v) { v = f(v); });
    return x;
  }

  // Applies `f` to every element of `x` and returns a copy.
  // `f` takes `dtype` or `dtype &` as input.
  // The result is a leaf.
  template <typename F>
    requires requires(F f, dtype x) {
      { f(x) } -> std::convertible_to<dtype>;
    }
  Tensor map(const Tensor &x, F f) {
    Tensor r = x.clone();
    return map_(r, f);
  }

  template <typename F>
    requires requires(F f, size_t idx, dtype x) {
      { f(idx, x) } -> std::convertible_to<dtype>;
    }
  Tensor map_(Tensor &x, F f) {
    dtype *d = x.p->data.data();
    for (size_t i = 0, n = x.p->num_el; i < n; i++)
      d[i] = f(i, d[i]);
    return x;
  }

  // Applies `f` to every element of `x` and returns a copy.
  // `f` takes index and value as inputs.
  // The result is a leaf.
  template <typename F>
    requires requires(F f, size_t idx, dtype x) {
      { f(idx, x) } -> std::convertible_to<dtype>;
    }
  Tensor map(const Tensor &x, F f) {
    Tensor r = x.clone();
    return map_(r, f);
  }
} // namespace NN::Detail
