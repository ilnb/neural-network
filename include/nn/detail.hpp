#pragma once

#include "tensor/tensor.hpp"

namespace NN::Detail {
  // Applies `f` to every element of `x` and returns a copy.
  template <typename F>
  Tensor map(const Tensor &x, F f) {
    Tensor r = x.clone();
    std::ranges::for_each(r.p->data, [f](dtype &v) { v = f(v); });
    return r;
  }
} // namespace NN::Detail
