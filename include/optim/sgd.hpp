#pragma once

#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Optim {
  struct SGD {
    std::vector<Tensor> params, velocity;
    dtype lr, beta;

    explicit SGD(const std::vector<Tensor> &v, const dtype lr, const dtype beta = 0) : params(v), lr(lr), beta(beta) {
      velocity.reserve(v.size());
      for (const auto &p : params)
        velocity.emplace_back(Tensor(p.p->shape));
    }

    void zero_grad() {
      for (auto &p : params)
        p.p->grad.reset();
    }

    void step() {
      Autograd::NoGradGuard g;
      for (size_t i = 0; i < params.size(); i++) {
        auto &p = params[i], &v = velocity[i];
        if (!p.p->grad)
          continue;
        p -= *p.p->grad * lr;
        v *= beta;
        v += *p.p->grad;
        p -= v * lr;
      }
    }
  };
} // namespace NN::Optim
