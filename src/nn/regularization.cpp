#include "nn/regularization.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace NN::Reg {
  Tensor l1(const Tensor &weight) {
    auto r = Detail::l1(weight);

    if (Autograd::grad_enabled and weight.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { weight.p };
      fn->backward = [ap = weight.p](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);
        const auto &w = ap->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++) {
          const dtype sign = (w[i] > 0) - (w[i] < 0);
          dx.p->data[i] = dz.p->data[0] * sign;
        }
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }

  Tensor l2(const Tensor &weight) {
    auto r = Detail::l2(weight);

    if (Autograd::grad_enabled and weight.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { weight.p };
      fn->backward = [ap = weight.p](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);
        const auto &w = ap->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++)
          dx.p->data[i] = dz.p->data[0] * w[i];
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }

  Tensor elastic_net(const Tensor &weight, const dtype alpha) {
    auto r = Detail::elastic_net(weight, alpha);

    if (Autograd::grad_enabled and weight.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { weight.p };
      fn->backward = [ap = weight.p, alpha](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);
        const auto &w = ap->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++) {
          const dtype sign = (w[i] > 0) - (w[i] < 0);
          dx.p->data[i] = dz.p->data[0] * (alpha * sign + (dtype(1) - alpha) * w[i]);
        }
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }

  namespace Detail {
    Tensor l1(const Tensor &weight) {
      if (weight.empty())
        throw std::invalid_argument("Empty tensor.");
      return Tensor(std::ranges::fold_left(weight.p->data, dtype(0), [](dtype x, dtype w) { return x + std::abs(w); }));
    }

    Tensor l2(const Tensor &weight) {
      if (weight.empty())
        throw std::invalid_argument("Empty tensor.");
      return Tensor(dtype(0.5) * std::ranges::fold_left(weight.p->data, dtype(0), [](dtype x, dtype w) { return x + w * w; }));
    }

    Tensor elastic_net(const Tensor &weight, const dtype alpha) {
      if (weight.empty())
        throw std::invalid_argument("Empty tensor.");
      if (alpha < 0 or 1 < alpha)
        throw std::invalid_argument("`alpha` should be in [0, 1].");
      return Tensor(std::ranges::fold_left(weight.p->data, dtype(0), [alpha](dtype x, dtype w) { return x + alpha * std::abs(w) + (dtype(1) - alpha) * dtype(0.5) * w * w; }));
    }
  } // namespace Detail
} // namespace NN::Reg
