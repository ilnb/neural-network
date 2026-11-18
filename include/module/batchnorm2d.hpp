#pragma once

#include "module/batchnorm.hpp"

namespace NN {
  // BatchNorm over the channel axis of `{N, C, H, W}`.
  //
  // Every channel is normalized over all of `N`, `H` and `W`.
  struct BatchNorm2d : BatchNorm {
    BatchNorm2d(size_t n, const Config &c = {}) : BatchNorm(n, c) {}

  protected:
    bool accepts(size_t rank) const override { return rank == 4; }
    const char *expected() const override { return "BatchNorm2d expects a rank 4 {N, C, H, W} input."; }
  };
} // namespace NN
