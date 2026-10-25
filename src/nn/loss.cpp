#include "nn/loss.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace NN::Loss {
  Tensor mse(const Tensor &pred, const Tensor &target) {
    auto r = Detail::mse(pred, target);

    if (pred.p->requires_grad or target.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {pred.p, target.p};

      fn->backward = [ap = pred.p, at = target.p](const Tensor &dz) {
        if (!ap->requires_grad)
          return;

        Tensor dx(ap->shape);
        const size_t n = ap->num_el;
        const dtype f = dz.p->data[0] / n;
        auto &data = dx.p->data, &pdata = ap->data, &tdata = at->data;

        for (size_t i = 0; i < n; i++)
          data[i] = f * (pdata[i] - tdata[i]);
        Autograd::accumulate(*ap, dx);
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor bce(const Tensor &pred, const Tensor &target) {
    auto r = Detail::bce(pred, target);
    if (pred.p->requires_grad or target.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {pred.p, target.p};

      fn->backward = [ap = pred.p, at = target.p](const Tensor &dz) {
        if (!ap->requires_grad)
          return;

        Tensor dx(ap->shape);
        const dtype f = dz.p->data[0] / ap->num_el;
        auto &xdata = dx.p->data, &pdata = ap->data, &tdata = at->data;

        for (size_t i = 0, n = ap->num_el; i < n; i++) {
          const dtype p = pdata[i];
          const dtype t = tdata[i];
          if (p != t)
            xdata[i] = f * (p - t) / (p * (1 - p));
        }
        Autograd::accumulate(*ap, dx);
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor ce(const Tensor &pred, const Tensor &target, size_t dim) {
    auto r = Detail::mse(pred, target);

    if (pred.p->requires_grad or target.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {pred.p, target.p};
      fn->backward = [ap = pred.p, at = target.p, dim](const Tensor &dz) {
        if (!ap->requires_grad)
          return;

        Tensor dx(ap->shape);
        const dtype f = -dz.p->data[0] / ap->num_el / ap->shape[dim];
        auto &xdata = dx.p->data, &pdata = ap->data, &tdata = at->data;

        for (size_t i = 0, n = ap->num_el; i < n; i++)
          xdata[i] = f * tdata[i] / pdata[i];
        Autograd::accumulate(*ap, dx);
      };

      r.p->grad_fn = fn;
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
      return Tensor(static_cast<dtype>(sum / static_cast<double>(2 * n)));
    }

    Tensor bce(const Tensor &pred, const Tensor &target) {
      Tensor::validate(pred, target);
      if (pred.p->shape != target.p->shape)
        throw std::invalid_argument("Incompatible shape.");

      double sum = 0;
      auto n = pred.p->num_el;
      auto &pdata = pred.p->data, &tdata = target.p->data;
      for (size_t i = 0; i < n; i++) {
        double y = tdata[i], p = pdata[i];
        if (y != 0 and y != 1)
          throw std::invalid_argument("Malformed tensor data. Entries should be binary.");
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
      const auto slices = static_cast<double>(n) / pred.p->shape[dim];
      double sum = 0;

      for (size_t i = 0; i < n; i++) {
        const double y = tdata[i], p = pdata[i];
        if (y < 0 or y > 1)
          throw std::invalid_argument("Malformed tensor data. Targets must lie in [0, 1].");
        if (y == 0)
          continue;
        if (p < 0)
          throw std::invalid_argument("Malformed tensor data. Probabilities must be non-negative.");
        if (p == 0)
          return Tensor(std::numeric_limits<dtype>::infinity());
        sum -= y * std::log(p);
      }

      return Tensor(static_cast<dtype>(sum / slices));
    }
  } // namespace Detail
} // namespace NN::Loss
