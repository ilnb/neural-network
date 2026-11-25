#include "loss/loss.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>

using NN::Autograd::GradFn, NN::Autograd::accumulate;

namespace NN::Loss {
  namespace {
    // Layout of the classification axis `dim`: `outer` slices before it,
    // `len` entries along it, and `inner` slices after it.
    struct Axis {
      size_t len, inner, outer;
    };

    Axis axis_info(const std::vector<size_t> &shape, size_t dim) {
      Axis a{ shape[dim], 1, 1 };
      for (size_t i = dim + 1; i < shape.size(); i++)
        a.inner *= shape[i];
      for (size_t i = 0; i < dim; i++)
        a.outer *= shape[i];
      return a;
    }

    // `log(sum(exp(x)))` for one slice, subtracting the maximum logit first.
    double slice_lse(const dtype *x, size_t len, size_t stride) {
      dtype mx = x[0];
      for (size_t j = 1; j < len; j++)
        mx = std::max(mx, x[j * stride]);

      double ex = 0;
      for (size_t j = 0; j < len; j++)
        ex += std::exp(SC<double>(x[j * stride]) - SC<double>(mx));

      return SC<double>(mx) + std::log(ex);
    }
  } // namespace

  Tensor mse(const Tensor &pred, const Tensor &target) {
    auto r = Detail::mse(pred, target);

    const bool requires_grad = pred.p->requires_grad or target.p->requires_grad;
    if (Autograd::grad_enabled and requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { pred.p, target.p };

      fn->backward = [ap = pred.p, at = target.p](const Tensor &dz) {
        const dtype f = dz.p->data[0] * 2 / ap->num_el;
        const auto &pdata = ap->data, &tdata = at->data;

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          auto &xd = dx.p->data;
          for (size_t i = 0, n = ap->num_el; i < n; i++)
            xd[i] = f * (pdata[i] - tdata[i]);
          accumulate(*ap, dx);
        }
        if (at->requires_grad) {
          Tensor dt(at->shape);
          auto &tdd = dt.p->data;
          for (size_t i = 0, n = at->num_el; i < n; i++)
            tdd[i] = -f * (pdata[i] - tdata[i]);
          accumulate(*at, dt);
        }
      };

      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor bce(const Tensor &pred, const Tensor &target) {
    auto r = Detail::bce(pred, target);

    const bool requires_grad = pred.p->requires_grad or target.p->requires_grad;
    if (Autograd::grad_enabled and requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { pred.p, target.p };

      fn->backward = [ap = pred.p, at = target.p](const Tensor &dz) {
        const dtype f = dz.p->data[0] / ap->num_el;
        const auto &pdata = ap->data, &tdata = at->data;

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          auto &xd = dx.p->data;
          for (size_t i = 0, n = ap->num_el; i < n; i++) {
            const dtype p = pdata[i], t = tdata[i];
            if (p != t)
              xd[i] = f * (p - t) / (p * (1 - p));
          }
          accumulate(*ap, dx);
        }
        if (at->requires_grad) {
          Tensor dt(at->shape);
          auto &td = dt.p->data;
          for (size_t i = 0, n = at->num_el; i < n; i++) {
            const dtype p = pdata[i];
            if (p != 0 and p != 1)
              td[i] = f * (std::log(1 - p) - std::log(p));
          }
          accumulate(*at, dt);
        }
      };

      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor ce(const Tensor &pred, const Tensor &target, size_t dim) {
    auto r = Detail::ce(pred, target, dim);

    const bool requires_grad = pred.p->requires_grad or target.p->requires_grad;
    if (Autograd::grad_enabled and requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { pred.p, target.p };
      fn->backward = [ap = pred.p, at = target.p, dim](const Tensor &dz) {
        const auto &xdata = ap->data, &tdata = at->data;
        const auto ax = axis_info(ap->shape, dim);
        const dtype scale = dz.p->data[0] / SC<dtype>(ax.outer * ax.inner);

        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          auto &xd = dx.p->data;

          for (size_t o = 0; o < ax.outer; o++) {
            for (size_t q = 0; q < ax.inner; q++) {
              const size_t base = o * ax.len * ax.inner + q;
              const double lse = slice_lse(xdata.data() + base, ax.len, ax.inner);

              double tsum = 0;
              for (size_t j = 0; j < ax.len; j++)
                tsum += tdata[base + j * ax.inner];

              for (size_t j = 0; j < ax.len; j++) {
                const size_t i = base + j * ax.inner;
                const double p = std::exp(SC<double>(xdata[i]) - lse);
                xd[i] = SC<dtype>(scale * (tsum * p - SC<double>(tdata[i])));
              }
            }
          }
          accumulate(*ap, dx);
        }

        if (at->requires_grad) {
          Tensor dt(at->shape);
          auto &td = dt.p->data;

          for (size_t o = 0; o < ax.outer; o++) {
            for (size_t q = 0; q < ax.inner; q++) {
              const size_t base = o * ax.len * ax.inner + q;
              const double lse = slice_lse(xdata.data() + base, ax.len, ax.inner);

              for (size_t j = 0; j < ax.len; j++) {
                const size_t i = base + j * ax.inner;
                td[i] = SC<dtype>(-SC<double>(scale) * (SC<double>(xdata[i]) - lse));
              }
            }
          }
          accumulate(*at, dt);
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
        throw std::invalid_argument(std::format("mse: shapes {} and {} are incompatible.", pred.p->shape, target.p->shape));

      double sum = 0;
      auto n = pred.p->num_el;
      auto &pdata = pred.p->data, &tdata = target.p->data;
      for (size_t i = 0; i < n; i++) {
        const auto d = SC<double>(pdata[i]) - SC<double>(tdata[i]);
        sum += d * d;
      }
      return Tensor(SC<dtype>(sum / SC<double>(n)));
    }

    Tensor bce(const Tensor &pred, const Tensor &target) {
      Tensor::validate(pred, target);
      if (pred.p->shape != target.p->shape)
        throw std::invalid_argument(std::format("bce: shapes {} and {} are incompatible.", pred.p->shape, target.p->shape));

      auto n = pred.p->num_el;
      auto &pdata = pred.p->data, &tdata = target.p->data;

      for (size_t i = 0; i < n; i++) {
        const dtype y = tdata[i], p = pdata[i];
        if (p < 0 or p > 1)
          throw std::invalid_argument("bce: malformed tensor data. probabilities must lie in [0, 1].");
        if (y != 0 and y != 1)
          throw std::invalid_argument("bce: malformed tensor data. entries must be binary.");
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
      return Tensor(SC<dtype>(sum / SC<double>(n)));
    }

    Tensor ce(const Tensor &pred, const Tensor &target, size_t dim) {
      Tensor::validate(pred, target);
      if (pred.p->shape != target.p->shape)
        throw std::invalid_argument(std::format("ce: shapes {} and {} are incompatible.", pred.p->shape, target.p->shape));
      if (dim >= pred.p->shape.size())
        throw std::out_of_range(std::format("ce: dim is {} out of bounds {}.", dim, pred.p->shape.size()));

      auto &xdata = pred.p->data, &tdata = target.p->data;
      const size_t n = pred.p->num_el;
      const auto ax = axis_info(pred.p->shape, dim);

      for (size_t i = 0; i < n; i++) {
        const dtype y = tdata[i];
        if (y < 0 or 1 < y)
          throw std::invalid_argument("ce: malformed tensor data. targets must lie in [0, 1].");
      }

      double sum = 0;
      for (size_t o = 0; o < ax.outer; o++) {
        for (size_t q = 0; q < ax.inner; q++) {
          const size_t base = o * ax.len * ax.inner + q;

          double tsum = 0, tx = 0;
          for (size_t j = 0; j < ax.len; j++) {
            const size_t i = base + j * ax.inner;
            const double t = tdata[i];
            tsum += t;
            if (t != 0)
              tx += t * SC<double>(xdata[i]);
          }
          if (tsum == 0)
            continue;

          sum += tsum * slice_lse(xdata.data() + base, ax.len, ax.inner) - tx;
        }
      }

      return Tensor(SC<dtype>(sum / SC<double>(ax.outer * ax.inner)));
    }
  } // namespace Detail
} // namespace NN::Loss
