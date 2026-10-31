#pragma once

#include "tensor/tensor.hpp"
#include <algorithm>

namespace NN::Detail {
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
} // namespace NN::Detail
