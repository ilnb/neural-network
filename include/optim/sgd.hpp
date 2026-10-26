#pragma once

#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Optim {
  struct SGD {
    std::vector<Tensor> params;
    dtype lr;

    void zero_grad() {
      for (auto &p : params)
        p.p->grad.reset();
    }

    void step() {
      Autograd::NoGradGuard g;
      for (auto &p : params) {
        if (p.p->grad)
          p.sub_(*p.p->grad * lr);
      }
    }
  };
} // namespace NN::Optim
