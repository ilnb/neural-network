#include "autograd/grad_mode.hpp"
#include "optim/adam.hpp"
#include "optim/sgd.hpp"
#include <cmath>

namespace NN::Optim {
  void Adam::zero_grad() {
    for (auto &p : params)
      p.p->grad.reset();
  }

  void Adam::step() {
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

  void SGD::zero_grad() {
    for (auto &p : params)
      p.p->grad.reset();
  }

  void SGD::step() {
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
} // namespace NN::Optim
