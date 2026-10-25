#include "nn/activations.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "nn/detail.hpp"
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
    if (x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {x.p};
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
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor sigmoid(const Tensor &x) {
    auto r = Detail::sigmoid(x);
    if (x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {x.p};
      fn->backward = [ap = x.p, ar = r.p](const Tensor &dz) {
        if (!ap->requires_grad)
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
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor softmax(const Tensor &x, size_t dim) {
    auto r = Detail::softmax(x, dim);
    if (x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {x.p};
      fn->backward = [ap = x.p, ar = r.p, dim](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor y(ar);
        auto tmp = dz * y;
        auto dx = y * (dz - tmp.sum(dim).expand(dim, ap->shape[dim]));
        Autograd::accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor log_softmax(const Tensor &x, size_t dim) {
    auto r = Detail::log_softmax(x, dim);

    if (x.p->requires_grad) {
      auto fn = std::make_shared<NN::Autograd::GradFn>();
      fn->inputs = {x.p};

      fn->backward = [ap = x.p, ar = r.p, dim](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor y(ar);
        Tensor s = dz.h_sum(dim).expand(dim, ap->shape[dim]);
        Tensor dx(ap->shape);
        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &ydata = ar->data, &sdata = s.p->data;
        for (size_t i = 0, n = ap->num_el; i < n; i++)
          xdata[i] = zdata[i] - std::exp(ydata[i]) * sdata[i];
        Autograd::accumulate(*ap, dx);
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }

    return r;
  }

  namespace Detail {
    Tensor relu(const Tensor &x) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      return Tensor(NN::Detail::map(x, [](dtype v) { return v > 0 ? v : 0; }));
    }

    Tensor sigmoid(const Tensor &x) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
      return Tensor(NN::Detail::map(x, [](dtype v) { return NN::Act::sigmoid(v); }));
    }

    Tensor softmax(const Tensor &x, size_t dim) {
      if (x.empty())
        throw std::invalid_argument("Empty tensor.");
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
