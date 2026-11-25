#pragma once

#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "module/base.hpp"

#include "tensor/tensor.hpp"
#include <cmath>
#include <format>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using NN::Autograd::GradFn, NN::Autograd::accumulate;

namespace NN {
  // Inverted dropout.
  //
  // During training each element is zeroed with probability `p`, and the
  // survivors are scaled by `1 / (1 - p)` so the expected value is unchanged.
  // The mask is drawn per element, not per sample. With `p == 0` or outside
  // training mode the input is returned unchanged. Backward applies the same
  // mask and scale.
  //
  // Throws `std::invalid_argument` if `p` is not finite or lies outside
  // `[0, 1)`.
  struct Dropout : Stateless {
    dtype p;
    // Keep mask of the last forward; emptied once the backward node takes it.
    std::vector<bool> mask;
    std::mt19937 rng;

    // `seed` seeds the mask RNG; the default is nondeterministic.
    Dropout(const dtype p, const u32 seed = std::random_device{}()) : p(p), rng(seed) {
      if (!std::isfinite(p) || p < SC<dtype>(0) || p >= SC<dtype>(1))
        throw std::invalid_argument(std::format("Dropout: probability {} must be in [0, 1)).", p));
    }

    [[nodiscard]] Tensor forward(const Tensor &x) override {
      if (!training or !p)
        return x;

      Tensor r = x.clone();
      mask.resize(x.p->num_el);
      std::bernoulli_distribution dist(p);

      const dtype scale = SC<dtype>(1) / (SC<dtype>(1) - p);
      auto &rd = r.p->data, &xd = x.p->data;
      for (size_t i = 0, n = x.p->num_el; i < n; i++) {
        const bool kept = !dist(rng);
        mask[i] = kept;
        rd[i] = kept ? xd[i] * scale : SC<dtype>(0);
      }

      if (x.p->requires_grad) {
        auto fn = std::make_shared<GradFn>();
        fn->inputs = { x.p };
        fn->backward = [ap = x.p, mask = std::move(mask), scale](const Tensor &dz) {
          if (!ap->requires_grad)
            return;
          Tensor dx(ap->shape);
          auto &xd = dx.p->data, &zd = dz.p->data;
          for (size_t i = 0, n = ap->num_el; i < n; i++)
            xd[i] = zd[i] * mask[i] * scale;
          accumulate(*ap, dx);
        };
        r.p->grad_fn = std::move(fn);
        r.p->requires_grad = true;
      }

      return r;
    }
  };
} // namespace NN
