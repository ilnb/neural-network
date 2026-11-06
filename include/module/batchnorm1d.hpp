#pragma once

#include "module/batchnorm.hpp"

namespace NN {
  // BatchNorm over the channel axis of `{N, C}` or `{N, C, L}`.
  //
  // In `{N, C, L}` every channel is normalized over both `N` and `L`, which is
  // the PyTorch convention for `nn.BatchNorm1d` on 3D input.
  struct BatchNorm1d : BatchNorm {
    BatchNorm1d(size_t n, const Config &c = {}) : BatchNorm(n, c) {}

  protected:
    bool accepts(size_t rank) const override { return rank == 2 or rank == 3; }
    const char *expected() const override { return "BatchNorm1d expects a rank 2 {N, C} or rank 3 {N, C, L} input."; }
  };
} // namespace NN
