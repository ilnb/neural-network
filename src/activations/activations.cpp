#include "activations/activations.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "nn/detail.hpp"
#include "tensor/detail.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace NN::Act {
  dtype sigmoid(dtype v) {
    if (v >= 0)
      return 1 / (1 + std::exp(-v));
    else
      return std::exp(v) / (1 + std::exp(v));
  }

  Tensor relu(const Tensor &x) {
    auto r = Detail::relu(x);

    if (Autograd::grad_enabled and x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { x.p };
      fn->backward = [ap = x.p](const Tensor &dz) {
        if (ap->requires_grad) {
          Tensor dx(ap->shape);
          auto &xdata = dx.p->data, &zdata = dz.p->data;
          auto &adata = ap->data;
          for (size_t i = 0, n = ap->num_el; i < n; i++)
            if (adata[i] > 0)
              xdata[i] = zdata[i];
          Autograd::accumulate(*ap, dx);
        }
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor sigmoid(const Tensor &x) {
    auto r = Detail::sigmoid(x);

    if (Autograd::grad_enabled and x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { x.p };
      std::weak_ptr<Impl> wr = r.p;
      fn->backward = [ap = x.p, wr](const Tensor &dz) {
        auto ar = wr.lock();
        if (!ap->requires_grad or !ar)
          return;
        Tensor dx(ap->shape);
        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &ydata = ar->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++) {
          const dtype y = ydata[i];
          xdata[i] = zdata[i] * y * (1 - y);
        }
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor softmax(const Tensor &x, size_t dim) {
    auto r = Detail::softmax(x, dim);

    if (Autograd::grad_enabled and x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { x.p };
      std::weak_ptr<Impl> wr = r.p;
      fn->backward = [ap = x.p, wr, dim](const Tensor &dz) {
        auto ar = wr.lock();
        if (!ap->requires_grad or !ar)
          return;
        if (ap->shape.empty()) {
          Autograd::accumulate(*ap, Tensor(0));
          return;
        }
        Tensor y(ar);
        auto s = NN::Detail::sum(NN::Detail::binary_broadcast(dz, y, std::multiplies<>()), dim);
        auto ex = NN::Detail::expand(s, dim, ap->shape[dim]);
        Tensor dx(ap->shape);
        auto &xdata = dx.p->data, &zdata = dz.p->data;
        const auto &ydata = ar->data, &sdata = ex.p->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++)
          xdata[i] = ydata[i] * (zdata[i] - sdata[i]);
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor log_softmax(const Tensor &x, size_t dim) {
    auto r = Detail::log_softmax(x, dim);

    if (Autograd::grad_enabled and x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = { x.p };

      std::weak_ptr<Impl> wr = r.p;
      fn->backward = [ap = x.p, wr, dim](const Tensor &dz) {
        auto ar = wr.lock();
        if (!ap->requires_grad or !ar)
          return;
        if (ap->shape.empty()) {
          Autograd::accumulate(*ap, Tensor(0));
          return;
        }
        Tensor y(ar);
        Tensor s = NN::Detail::expand(NN::Detail::sum(dz, dim), dim, ap->shape[dim]);
        Tensor dx(ap->shape);
        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &ydata = ar->data, &sdata = s.p->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++)
          xdata[i] = zdata[i] - std::exp(ydata[i]) * sdata[i];
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }

  namespace Detail {
    Tensor relu(const Tensor &x) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      return Tensor(NN::Detail::map(x, [](dtype v) { return v > 0 or std::isnan(v) ? v : 0; }));
    }

    Tensor sigmoid(const Tensor &x) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      return Tensor(NN::Detail::map(x, [](dtype v) { return NN::Act::sigmoid(v); }));
    }

    Tensor softmax(const Tensor &x, size_t dim) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      if (x.p->shape.empty()) {
        if (dim != 0)
          throw std::out_of_range("dim is out of bounds.");
        const dtype v = x.p->data[0];
        const dtype e = std::exp(v - v); // to preserve NaN and inf poisons
        return Tensor(e / e);
      }
      if (dim >= x.p->shape.size())
        throw std::out_of_range("dim is out of bounds.");

      Tensor r;
      r.set_dims(x.p->shape, false);
      const auto &sh = r.p->shape, &st = r.p->strides;
      auto &dt = r.p->data;
      const auto &xdt = x.p->data;
      const size_t n = sh.size(), d = sh[dim], sd = st[dim];
      const size_t outer = r.p->num_el / d;

      for (size_t idx = 0; idx < outer; idx++) {
        size_t off = 0;
        for (size_t i = n, tmp = idx; i-- > 0;) {
          const size_t sz = i == dim ? 1 : sh[i];
          const size_t c = tmp % sz;
          tmp /= sz;
          off += c * st[i];
        }

        dtype mx = xdt[off];
        for (size_t k = 1; k < d; k++)
          mx = std::max(mx, xdt[off + k * sd]);
        dtype s = 0;
        for (size_t k = 0; k < d; k++) {
          const dtype v = std::exp(xdt[off + k * sd] - mx);
          dt[off + k * sd] = v;
          s += v;
        }
        for (size_t k = 0; k < d; k++)
          dt[off + k * sd] /= s;
      }
      return r;
    }

    Tensor log_softmax(const Tensor &x, size_t dim) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      if (x.p->shape.empty()) {
        if (dim != 0)
          throw std::out_of_range("dim is out of bounds.");
        const dtype v = x.p->data[0];
        return Tensor(v - v);
      }
      if (dim >= x.p->shape.size())
        throw std::out_of_range("dim out of bounds.");

      Tensor r;
      r.set_dims(x.p->shape, false);
      const auto &sh = r.p->shape, &st = r.p->strides;
      auto &dt = r.p->data;
      const auto &xdt = x.p->data;
      const size_t n = sh.size(), d = sh[dim], sd = st[dim];
      const size_t outer = r.p->num_el / d;

      for (size_t idx = 0; idx < outer; idx++) {
        size_t off = 0;
        for (size_t i = n, tmp = idx; i-- > 0;) {
          const size_t sz = i == dim ? 1 : sh[i];
          const size_t c = tmp % sz;
          tmp /= sz;
          off += c * st[i];
        }

        dtype mx = xdt[off];
        for (size_t k = 1; k < d; k++)
          mx = std::max(mx, xdt[off + k * sd]);
        dtype s = 0;
        for (size_t k = 0; k < d; k++)
          s += std::exp(xdt[off + k * sd] - mx);
        const dtype l = std::log(s);
        for (size_t k = 0; k < d; k++)
          dt[off + k * sd] = (xdt[off + k * sd] - mx) - l;
      }
      return r;
    }
  } // namespace Detail
} // namespace NN::Act
