#pragma once

#include "tensor/tensor.hpp"
#include <algorithm>
#include <concepts>

namespace NN::Detail {
  // Applies `f` to every element of `x` and returns a copy.
  // `f` takes `dtype` or `dtype &` as input.
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

  // Applies `f` to every element of `x` and returns a copy.
  // `f` takes index and value as inputs.
  // The result is a leaf.
  template <typename F>
    requires requires(F f, size_t idx, dtype x) {
      { f(idx, x) } -> std::convertible_to<dtype>;
    }
  Tensor map(const Tensor &x, F f) {
    Tensor r = x.clone();
    std::ranges::for_each(r.p->data, [f](dtype &v) { v = f(v); });
    return r;
  }
} // namespace NN::Detail
