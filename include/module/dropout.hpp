#pragma once

#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "module/base.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace NN {
  struct Dropout : Module {
    dtype p;
    std::vector<bool> mask;
    std::mt19937 rng;

    Dropout(const dtype p, const uint32_t seed = 0) : p(p), rng(seed) {
      if (!std::isfinite(p) || p < dtype(0) || p >= dtype(1))
        throw std::invalid_argument("dropout probability must be in [0, 1).");
    }

    Tensor forward(const Tensor &x) override {
      if (!training or !p)
        return x;

      Tensor r = x.clone();
      mask.resize(x.p->num_el);
      std::bernoulli_distribution dist(p);

      const dtype scale = dtype(1) / (dtype(1) - p);
      auto &rd = r.p->data, &xd = x.p->data;
      for (size_t i = 0, n = x.p->num_el; i < n; i++) {
        const bool kept = !dist(rng);
        mask[i] = kept;
        rd[i] = kept ? xd[i] * scale : dtype(0);
      }

      if (x.p->requires_grad) {
        auto fn = std::make_shared<NN::Autograd::GradFn>();
        fn->inputs = { r.p };
        fn->backward = [ap = r.p, mask = std::move(mask), scale](const Tensor &dz) {
          if (!ap->requires_grad)
            return;
          Tensor dx(ap->shape);
          auto &xd = dx.p->data, &zd = dz.p->data;
          for (size_t i = 0, n = ap->num_el; i < n; i++)
            xd[i] = zd[i] * mask[i] * scale;
          Autograd::accumulate(*ap, dx);
        };
        r.p->grad_fn = std::move(fn);
        r.p->requires_grad = true;
      }

      return r;
    }
  };
} // namespace NN
