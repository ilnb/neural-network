#pragma once

#include "tensor/tensor.hpp"

namespace NN::Detail {
  // Applies `f` to every element of `x` and returns a copy.
  template <typename F>
  Tensor map(const Tensor &x, F f) {
    Tensor r = x;
    std::ranges::for_each(r.data, [f](dtype &v) { v = f(v); });
    return r;
  }
} // namespace NN::Detail
