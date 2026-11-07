#pragma once

#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using NN::Autograd::GradFn, NN::Autograd::accumulate;

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

    Tensor forward(const Tensor &x) final;

  protected:
    // True if the derived layer accepts an input of this rank.
    virtual bool accepts(size_t rank) const = 0;

    // Message used when `accepts` rejects the rank.
    virtual const char *expected() const = 0;
  };
} // namespace NN
