#include "module/batchnorm.hpp"
#include "autograd/grad_mode.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace NN {
  Tensor BatchNorm::forward(const Tensor &x) {
    if (x.empty())
      throw std::invalid_argument("Empty tensor.");
    if (!accepts(x.p->shape.size()))
      throw std::invalid_argument(expected());
    if (x.p->shape[1] != channels)
      throw std::invalid_argument("Wrong number of channels on axis 1.");

    auto &xd = x.p->data;
    const size_t xn = x.p->num_el;
    const auto &shape = x.p->shape;
    const size_t C = channels;
    // Elements per (sample, channel): 1 for {N, C}, L for {N, C, L}, H * W
    // for {N, C, H, W}.
    const size_t inner = xn / (shape[0] * C);
    // Elements per channel, across the whole batch.
    const size_t n = xn / C;

    std::vector<dtype> mu(C), var(C), inv_std(C);

    if (training) {
      for (size_t i = 0; i < xn; i++)
        mu[(i / inner) % C] += xd[i];
      for (size_t c = 0; c < C; c++)
        mu[c] /= static_cast<dtype>(n);

      for (size_t i = 0; i < xn; i++) {
        const size_t c = (i / inner) % C;
        const dtype d = xd[i] - mu[c];
        var[c] += d * d;
      }
      for (size_t c = 0; c < C; c++)
        var[c] /= static_cast<dtype>(n);
    } else {
      std::ranges::copy(running_mean.p->data, mu.begin());
      std::ranges::copy(running_var.p->data, var.begin());
    }

    for (size_t c = 0; c < C; c++)
      inv_std[c] = 1 / std::sqrt(var[c] + cfg.eps);

    // The running buffers are not part of the graph, so they are updated
    // through their raw storage.
    if (training) {
      const dtype corr = n > 1 ? static_cast<dtype>(n) / static_cast<dtype>(n - 1) : dtype(1);
      for (size_t c = 0; c < C; c++) {
        running_mean.p->data[c] = (1 - cfg.mom) * running_mean.p->data[c] + cfg.mom * mu[c];
        running_var.p->data[c] = (1 - cfg.mom) * running_var.p->data[c] + cfg.mom * var[c] * corr;
      }
    }

    Tensor r(shape);

    std::vector<dtype> xhat(xn);
    auto &yd = r.p->data;
    auto &gd = gamma.p->data, &bd = beta.p->data;

    for (size_t i = 0; i < xn; i++) {
      const size_t c = (i / inner) % C;
      xhat[i] = (xd[i] - mu[c]) * inv_std[c];
      yd[i] = xhat[i] * gd[c] + bd[c];
    }

    const bool requires_grad = x.p->requires_grad or gamma.p->requires_grad or beta.p->requires_grad;
    if (Autograd::grad_enabled and requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { x.p, gamma.p, beta.p };
      const bool batch_stats = training;
      fn->backward = [ap = x.p, ag = gamma.p, ab = beta.p, xhat = std::move(xhat), invstd = std::move(inv_std), C, inner, n, batch_stats](const Tensor &dz) {
        std::vector<dtype> dgamma(C), dbeta(C);

        const size_t zn = dz.p->num_el, an = ap->num_el;
        auto &ad = ap->data, &gd = ag->data;
        auto &zd = dz.p->data;

        for (size_t i = 0; i < zn; i++) {
          const size_t c = (i / inner) % C;
          dgamma[c] += zd[i] * xhat[i];
          dbeta[c] += zd[i];
        }

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          auto &xd = dx.p->data;
          for (size_t i = 0; i < an; i++) {
            const size_t c = (i / inner) % C;
            if (batch_stats)
              xd[i] = gd[c] * invstd[c] * (zd[i] - dbeta[c] / static_cast<dtype>(n) - xhat[i] * dgamma[c] / static_cast<dtype>(n));
            else
              xd[i] = gd[c] * invstd[c] * zd[i];
          }
          accumulate(*ap, dx);
        }
        if (ag->requires_grad) {
          Tensor t(ag->shape);
          t.p->data = std::move(dgamma);
          accumulate(*ag, t);
        }
        if (ab->requires_grad) {
          Tensor t(ab->shape);
          t.p->data = std::move(dbeta);
          accumulate(*ab, t);
        }
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }
} // namespace NN
