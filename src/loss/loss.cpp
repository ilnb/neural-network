#include "loss/loss.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace NN::Loss {
  Tensor mse(const Tensor &pred, const Tensor &target) {
    auto r = Detail::mse(pred, target);

    if (Autograd::grad_enabled and (pred.p->requires_grad or target.p->requires_grad)) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { pred.p, target.p };

      fn->backward = [ap = pred.p, at = target.p](const Tensor &dz) {
        const dtype f = dz.p->data[0] * 2 / ap->num_el;
        const auto &pdata = ap->data, &tdata = at->data;

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          for (size_t i = 0, n = ap->num_el; i < n; i++)
            dx.p->data[i] = f * (pdata[i] - tdata[i]);
          Autograd::accumulate(*ap, dx);
        }
        if (at->requires_grad) {
          Tensor dt(at->shape);
          for (size_t i = 0, n = at->num_el; i < n; i++)
            dt.p->data[i] = -f * (pdata[i] - tdata[i]);
          Autograd::accumulate(*at, dt);
        }
      };

      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor bce(const Tensor &pred, const Tensor &target) {
    auto r = Detail::bce(pred, target);
    if (Autograd::grad_enabled and (pred.p->requires_grad or target.p->requires_grad)) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { pred.p, target.p };

      fn->backward = [ap = pred.p, at = target.p](const Tensor &dz) {
        const dtype f = dz.p->data[0] / ap->num_el;
        const auto &pdata = ap->data, &tdata = at->data;

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          for (size_t i = 0, n = ap->num_el; i < n; i++) {
            const dtype p = pdata[i], t = tdata[i];
            if (p != t)
              dx.p->data[i] = f * (p - t) / (p * (1 - p));
          }
          Autograd::accumulate(*ap, dx);
        }
        if (at->requires_grad) {
          Tensor dt(at->shape);
          for (size_t i = 0, n = at->num_el; i < n; i++) {
            const dtype p = pdata[i];
            if (p != 0 and p != 1)
              dt.p->data[i] = f * (std::log(1 - p) - std::log(p));
          }
          Autograd::accumulate(*at, dt);
        }
      };

      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor ce(const Tensor &pred, const Tensor &target, size_t dim) {
    auto r = Detail::ce(pred, target, dim);

    if (Autograd::grad_enabled and (pred.p->requires_grad or target.p->requires_grad)) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { pred.p, target.p };
      fn->backward = [ap = pred.p, at = target.p, dim](const Tensor &dz) {
        const dtype f = -dz.p->data[0] * static_cast<dtype>(ap->shape[dim]) / ap->num_el;
        const auto &pdata = ap->data;

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          for (size_t i = 0, n = ap->num_el; i < n; i++)
            dx.p->data[i] = f * at->data[i] / pdata[i];
          Autograd::accumulate(*ap, dx);
        }
        if (at->requires_grad) {
          Tensor dt(at->shape);
          for (size_t i = 0, n = at->num_el; i < n; i++)
            dt.p->data[i] = f * std::log(pdata[i]);
          Autograd::accumulate(*at, dt);
        }
      };

      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  namespace Detail {
    Tensor mse(const Tensor &pred, const Tensor &target) {
      Tensor::validate(pred, target);
      if (pred.p->shape != target.p->shape)
        throw std::invalid_argument("Incompatible shape.");

      double sum = 0;
      auto n = pred.p->num_el;
      auto &pdata = pred.p->data, &tdata = target.p->data;
      for (size_t i = 0; i < n; i++) {
        const double d = static_cast<double>(pdata[i]) - static_cast<double>(tdata[i]);
        sum += d * d;
      }
      return Tensor(static_cast<dtype>(sum / static_cast<double>(n)));
    }

    Tensor bce(const Tensor &pred, const Tensor &target) {
      Tensor::validate(pred, target);
      if (pred.p->shape != target.p->shape)
        throw std::invalid_argument("Incompatible shape.");

      auto n = pred.p->num_el;
      auto &pdata = pred.p->data, &tdata = target.p->data;

      for (size_t i = 0; i < n; i++) {
        const dtype y = tdata[i], p = pdata[i];
        if (p < 0 or p > 1)
          throw std::invalid_argument("Malformed tensor data. Probabilities must lie in [0, 1].");
        if (y != 0 and y != 1)
          throw std::invalid_argument("Malformed tensor data. Entries should be binary.");
      }

      double sum = 0;
      for (size_t i = 0; i < n; i++) {
        double y = tdata[i], p = pdata[i];
        if (p == 0) {
          if (y == 1)
            return Tensor(std::numeric_limits<dtype>::infinity());
          else
            continue;
        } else if (p == 1) {
          if (y == 0)
            return Tensor(std::numeric_limits<dtype>::infinity());
          else
            continue;
        }
        sum -= y * std::log(p);
        sum -= (1 - y) * std::log(1 - p);
      }
      return Tensor(static_cast<dtype>(sum / static_cast<double>(n)));
    }

    Tensor ce(const Tensor &pred, const Tensor &target, size_t dim) {
      Tensor::validate(pred, target);
      if (pred.p->shape != target.p->shape)
        throw std::invalid_argument("Incompatible shape.");
      if (dim >= pred.p->shape.size())
        throw std::out_of_range("Class axis out of range.");

      auto n = pred.p->num_el;
      auto &pdata = pred.p->data, &tdata = target.p->data;

      for (size_t i = 0; i < n; i++) {
        const dtype y = tdata[i], p = pdata[i];
        if (y < 0 or y > 1)
          throw std::invalid_argument("Malformed tensor data. Targets must lie in [0, 1].");
        if (p < 0)
          throw std::invalid_argument("Malformed tensor data. Probabilities must be non-negative.");
      }

      const auto slices = static_cast<double>(n) / pred.p->shape[dim];
      double sum = 0;

      for (size_t i = 0; i < n; i++) {
        const dtype y = tdata[i], p = pdata[i];
        if (y == 0)
          continue;
        if (p == 0)
          return Tensor(std::numeric_limits<dtype>::infinity());
        sum -= y * std::log(p);
      }

      return Tensor(static_cast<dtype>(sum / slices));
    }
  } // namespace Detail
} // namespace NN::Loss
