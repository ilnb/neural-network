#pragma once

#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Optim {
  struct SGD {
    struct HParams {
      dtype lr = 0.01, beta = 0.9;
    };

    std::vector<Tensor> params, vel;
    HParams hp;

    explicit SGD(const std::vector<Tensor> &v) : SGD(v, HParams{}) {}

    SGD(const std::vector<Tensor> &v, const HParams &c) : params(v), hp(c) {
      vel.reserve(v.size());
      for (const auto &p : params)
        vel.emplace_back(Tensor(p.p->shape));
    }

    void zero_grad() {
      for (auto &p : params)
        p.p->grad.reset();
    }

    void step() {
      Autograd::NoGradGuard g;
      for (size_t i = 0, n = params.size(); i < n; i++) {
        auto &p = params[i], &v = vel[i];
        if (!p.p->grad)
          continue;

        auto &pd = p.p->data, &vd = v.p->data;
        const auto &g = *p.p->grad;
        for (size_t j = 0, itr = p.p->num_el; j < itr; j++) {
          pd[j] -= g.p->data[j] * hp.lr;
          vd[j] *= hp.beta;
          vd[j] += g.p->data[j];
          pd[j] -= vd[j] * hp.lr;
        }
      }
    }
  };
} // namespace NN::Optim
