#include "nn/activations.hpp"
#include "nn/detail.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace NN::Act {
  Tensor relu(const Tensor &x) {
    return Detail::map(x, [](dtype v) { return v > 0 ? v : 0; });
  }

  dtype sigmoid(dtype v) {
    if (v >= 0)
      return 1 / (1 + std::exp(-v));
    else
      return std::exp(v) / (1 + std::exp(v));
  }

  Tensor sigmoid(const Tensor &x) {
    return Detail::map(x, [](dtype v) { return sigmoid(v); });
  }

  Tensor softmax(const Tensor &x, size_t dim) {
    if (x.empty())
      throw std::invalid_argument("Empty tensor.");
    if (dim >= x.p->shape.size())
      throw std::out_of_range("dim is out of bounds.");

    Tensor r;
    const auto &sh = r.p->shape, &st = r.p->strides;
    auto &dt = r.p->data;
    const auto &xdt = x.p->data;
    r.set_dims(x.p->shape, false);
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
    const auto &sh = r.p->shape, &st = r.p->strides;
    auto &dt = r.p->data;
    const auto &xdt = x.p->data;
    r.set_dims(x.p->shape, false);
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
} // namespace NN::Act
