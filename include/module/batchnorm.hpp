#pragma once

#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace NN {
  // Shared base for BatchNorm1d and BatchNorm2d.
  //
  // Normalizes every axis except the channel axis (axis 1). A rank `r` input
  // `{N, C, ...}` is reduced to per-channel statistics; the remaining axes are
  // pooled, so `BatchNorm1d` on `{N, C, L}` and `BatchNorm2d` on
  // `{N, C, H, W}` share the same math.
  //
  // In training mode the layer uses batch statistics and updates the running
  // buffers. In eval mode it uses the running buffers. The forward pass is
  // differentiable in both modes; `gamma` and `beta` are the only parameters.
  //
  // The forward and backward are fused because the tensor API has no
  // tensor-tensor division or reciprocal. With `mu` the batch mean and `var`
  // the biased batch variance, `xhat = (x - mu) / sqrt(var + eps)`,
  // `y = gamma * xhat + beta`, and
  //
  // ```
  // dbeta[c]  = sum dz
  // dgamma[c] = sum dz * xhat
  // dx        = gamma * inv_std * (dz - dbeta / n - xhat * dgamma / n)
  // ```
  //
  // where `n` is the number of elements per channel. `running_var` is updated
  // with the unbiased variance `n / (n - 1) * var`, matching PyTorch; the
  // correction is skipped when `n == 1`.
  //
  // A graph supports one backward pass, like every other module.
  struct BatchNorm : Module {
    struct Config {
      dtype mom = 0.1, eps = 1e-5;
    };

    size_t channels;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
    Config cfg;

    // Creates a layer for `n` channels.
    //
    // `gamma` starts at 1, `beta` at 0, `running_mean` at 0 and
    // `running_var` at 1.
    //
    // Throws `std::invalid_argument` if `n == 0`, `mom` is not finite or not
    // in `[0, 1]`, or `eps` is not finite or not positive.
    BatchNorm(size_t n, const Config &c) : channels(n), gamma({ n }), beta({ n }), running_mean({ n }), running_var({ n }), cfg(c) {
      if (channels == 0)
        throw std::invalid_argument("n should be > 0.");
      if (!std::isfinite(cfg.mom) or cfg.mom < 0 or 1 < cfg.mom)
        throw std::invalid_argument("momentum should be in [0, 1].");
      if (!std::isfinite(cfg.eps) or cfg.eps <= 0)
        throw std::invalid_argument("eps should be a positive real number.");

      std::ranges::fill(gamma.p->data, dtype(1));
      std::ranges::fill(running_var.p->data, dtype(1));

      gamma.set_requires_grad(true);
      beta.set_requires_grad(true);
    }

    std::vector<Tensor> params() const override { return { gamma, beta }; }

    Tensor forward(const Tensor &x) override { return normalize(x); }

  protected:
    // True if the derived layer accepts an input of this rank.
    virtual bool accepts(size_t rank) const = 0;

    // Message used when `accepts` rejects the rank.
    virtual const char *expected() const = 0;

    Tensor normalize(const Tensor &x) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      if (!accepts(x.p->shape.size()))
        throw std::invalid_argument(expected());
      if (x.p->shape[1] != channels)
        throw std::invalid_argument("Wrong number of channels on axis 1.");

      const auto &shape = x.p->shape;
      const size_t C = channels;
      // Elements per (sample, channel): 1 for {N, C}, L for {N, C, L}, H * W
      // for {N, C, H, W}.
      const size_t inner = x.p->num_el / (shape[0] * C);
      // Elements per channel, across the whole batch.
      const size_t n = x.p->num_el / C;

      std::vector<dtype> mu(C), var(C), inv_std(C);

      if (training) {
        std::ranges::fill(mu, dtype(0));
        for (size_t i = 0; i < x.p->num_el; i++)
          mu[(i / inner) % C] += x.p->data[i];
        for (size_t c = 0; c < C; c++)
          mu[c] /= static_cast<dtype>(n);

        std::ranges::fill(var, dtype(0));
        for (size_t i = 0; i < x.p->num_el; i++) {
          const size_t c = (i / inner) % C;
          const dtype d = x.p->data[i] - mu[c];
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

      Tensor y(shape);
      std::vector<dtype> xhat(x.p->num_el);
      for (size_t i = 0; i < x.p->num_el; i++) {
        const size_t c = (i / inner) % C;
        xhat[i] = (x.p->data[i] - mu[c]) * inv_std[c];
        y.p->data[i] = xhat[i] * gamma.p->data[c] + beta.p->data[c];
      }

      const bool needs_grad = x.p->requires_grad or gamma.p->requires_grad or beta.p->requires_grad;
      if (Autograd::grad_enabled and needs_grad) {
        auto fn = std::make_shared<Autograd::GradFn>();
        fn->inputs = { x.p, gamma.p, beta.p };
        const bool batch_stats = training;
        fn->backward = [ap = x.p, ag = gamma.p, ab = beta.p, xhat = std::move(xhat), invstd = std::move(inv_std), C, inner, n, batch_stats](const Tensor &dz) {
          std::vector<dtype> dgamma(C, 0), dbeta(C, 0);
          for (size_t i = 0; i < dz.p->num_el; i++) {
            const size_t c = (i / inner) % C;
            dgamma[c] += dz.p->data[i] * xhat[i];
            dbeta[c] += dz.p->data[i];
          }
          if (ap->requires_grad) {
            Tensor dx(ap->shape);
            for (size_t i = 0; i < ap->num_el; i++) {
              const size_t c = (i / inner) % C;
              if (batch_stats)
                dx.p->data[i] = ag->data[c] * invstd[c] * (dz.p->data[i] - dbeta[c] / static_cast<dtype>(n) - xhat[i] * dgamma[c] / static_cast<dtype>(n));
              else
                dx.p->data[i] = ag->data[c] * invstd[c] * dz.p->data[i];
            }
            Autograd::accumulate(*ap, dx);
          }
          if (ag->requires_grad) {
            Tensor t(ag->shape);
            t.p->data = std::move(dgamma);
            Autograd::accumulate(*ag, t);
          }
          if (ab->requires_grad) {
            Tensor t(ab->shape);
            t.p->data = std::move(dbeta);
            Autograd::accumulate(*ab, t);
          }
        };
        y.p->grad_fn = std::move(fn);
        y.p->requires_grad = true;
      }

      return y;
    }
  };
} // namespace NN
