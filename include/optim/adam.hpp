#pragma once

#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <vector>

namespace NN::Optim {
  struct Adam {
    struct HParams {
      dtype lr = 0.01, eps = 1e-8;
      dtype beta1 = 0.9, beta2 = 0.999;
    };

    std::vector<Tensor> params, mom, vel;
    HParams hp;
    size_t t = 0;

    explicit Adam(const std::vector<Tensor> &v) : Adam(v, HParams{}) {}

    Adam(const std::vector<Tensor> &v, const HParams &c) : params(v), hp(c) {
      mom.reserve(params.size());
      vel.reserve(params.size());
      for (const auto &p : params) {
        mom.emplace_back(Tensor(p.p->shape));
        vel.emplace_back(Tensor(p.p->shape));
      }
    }

    void zero_grad() {
      for (auto &p : params)
        p.p->grad.reset();
    }

    void step() {
      Autograd::NoGradGuard g;
      t++;

      const dtype b1_curr = 1 - std::pow(hp.beta1, t);
      const dtype b2_curr = 1 - std::pow(hp.beta2, t);

      for (size_t i = 0, n = params.size(); i < n; i++) {
        auto &p = params[i], &v = vel[i], &m = mom[i];

        if (!p.p->grad)
          continue;

        const auto &g = *p.p->grad;
        auto &md = m.p->data, &pd = p.p->data, &vd = v.p->data;

        for (size_t j = 0, itr = p.p->num_el; j < itr; j++) {
          const dtype grad = g.p->data[j];

          md[j] = hp.beta1 * md[j] + (1 - hp.beta1) * grad;
          vd[j] = hp.beta2 * vd[j] + (1 - hp.beta2) * grad * grad;

          const dtype m_hat = md[j] / b1_curr;
          const dtype v_hat = vd[j] / b2_curr;

          pd[j] -= hp.lr * m_hat / (std::sqrt(v_hat) + hp.eps);
        }
      }
    }
  };
} // namespace NN::Optim
