#include "tensor/tensor.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "tensor/detail.hpp"
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace NN {
  using Autograd::GradFn, Autograd::accumulate;
  void Tensor::assign_shape(std::vector<size_t> &&dims, const bool clear) {
    if (!p)
      p = std::make_shared<Impl>();
    const size_t n = Detail::checked_num_el(dims);
    p->shape = std::move(dims);
    finalize_shape(n, clear);
  }

  void Tensor::finalize_shape(const size_t n, const bool clear) {
    p->num_el = n;
    p->strides.resize(p->shape.size());
    for (size_t i = 0, tmp = n; i < p->shape.size(); i++) {
      tmp /= p->shape[i];
      p->strides[i] = tmp;
    }
    if (clear)
      p->data.assign(n, 0);
    else
      p->data.resize(n);
  }

  Tensor &Tensor::set_dims(std::vector<size_t> &&v, const bool clear) {
    if (not empty())
      check_mutable();
    assign_shape(std::move(v), clear);
    return *this;
  }

  Tensor Tensor::reshape(std::vector<size_t> &&v) const {
    auto r = Detail::reshape(*this, std::move(v));
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::reshape(dz, std::vector(ap->shape)));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::transpose() const {
    auto r = Detail::transpose(*this);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::transpose(dz));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::cat(const Tensor &b, size_t dim) const {
    auto r = Detail::cat(*this, b, dim);

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};

      fn->backward = [ap = p, bp = b.p, dim](const Tensor &dz) {
        auto &ashape = ap->shape, &bshape = bp->shape;
        auto &zdata = dz.p->data;
        size_t inner = 1;
        for (size_t i = dim + 1, n = ashape.size(); i < n; i++)
          inner *= ashape[i];
        const size_t a_block = ashape[dim] * inner;
        const size_t b_block = bshape[dim] * inner;
        const size_t outer = ap->num_el / a_block;

        auto split_grad = [&](const std::vector<size_t> &shape, size_t blk, size_t off) {
          Tensor d(shape);
          auto &data = d.p->data;

          for (size_t i = 0; i < outer; i++) {
            const size_t dz_off = i * (a_block + b_block) + off;
            const size_t d_off = i * blk;
            std::copy_n(zdata.begin() + dz_off, blk, data.begin() + d_off);
          }

          return d;
        };
        if (ap->requires_grad)
          accumulate(*ap, split_grad(ap->shape, a_block, 0));
        if (bp->requires_grad)
          accumulate(*bp, split_grad(bp->shape, b_block, a_block));
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::stack(const Tensor &b, size_t dim) const {
    auto r = Detail::stack(*this, b, dim);

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};

      fn->backward = [ap = p, bp = b.p, dim](const Tensor &dz) {
        const auto &shape = ap->shape;
        const size_t n = shape.size();
        size_t inner = 1;
        for (size_t i = dim; i < n; i++)
          inner *= shape[i];
        const size_t block = inner;
        const size_t outer = ap->num_el / inner;

        auto split_grad = [&](const std::shared_ptr<Impl> &p, size_t which) {
          Tensor d(p->shape);
          for (size_t i = 0; i < outer; i++) {
            const size_t src = i * 2 * block + which * block;
            const size_t dst = i * block;
            std::copy_n(dz.p->data.begin() + src, block, d.p->data.begin() + dst);
          }
          return d;
        };

        if (ap->requires_grad)
          accumulate(*ap, split_grad(ap, 0));
        if (bp->requires_grad)
          accumulate(*bp, split_grad(bp, 1));
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }

    return r;
  }

  Tensor Tensor::permute(std::vector<size_t> &&v) const {
    auto r = Detail::permute(*this, v);

    if (p->requires_grad) {
      // inverse permutation
      const size_t n = p->shape.size();
      for (size_t i = 0; i < n; i++) {
        if (v[i] >= n)
          continue;
        size_t cur = i;
        size_t next = v[cur];
        while (next != i) {
          size_t tmp = v[next];
          v[next] = cur + n;
          cur = next;
          next = tmp;
        }
        v[i] = cur + n;
      }
      for (auto &x : v)
        x -= n;

      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, inv = std::move(v)](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::permute(dz, inv));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }

    return r;
  }

  bool Tensor::empty() const { return !p; }

  void Tensor::backward() const { Autograd::backward(*this); }

  void Tensor::set_requires_grad(const bool x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (p->grad_fn)
      throw std::logic_error("requires_grad can only be changed on a leaf tensor.");
    p->requires_grad = x;
  }

  void Tensor::check_mutable() const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if ((p->requires_grad or p->grad_fn))
      throw std::logic_error("In-place op involving autograd tensor.");
  }

  void Tensor::validate() const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");

    if (not Detail::valid_strides(p))
      throw std::invalid_argument("Malformed strides.");

    if (p->data.size() != p->num_el or Detail::checked_num_el(p->shape) != p->num_el)
      throw std::invalid_argument("Malformed num_el or data.");
  }

  Tensor Tensor::clone() const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    auto r = Tensor(p->shape);
    r.set_data(p->data);
    return r;
  }

  dtype Tensor::item() const {
    if (empty() or p->num_el != 1)
      throw std::domain_error("Tensor must contain exactly one element.");
    return p->data[0];
  }

  Tensor Tensor::sum() const {
    auto r = Detail::sum(*this);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);
        std::ranges::fill(dx.p->data, dz.p->data[0]);
        accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::mean() const {
    if (empty())
      throw std::domain_error("Empty tensor.");
    return sum() / p->num_el;
  }

  Tensor Tensor::min() const {
    if (empty())
      throw std::domain_error("Empty tensor.");

    const auto idx = std::ranges::min_element(p->data) - p->data.begin();
    Tensor r(p->data[idx]);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, idx](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);
        dx.p->data[idx] = dz.p->data[0];
        accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::max() const {
    if (empty())
      throw std::domain_error("Empty tensor.");

    const auto idx = std::ranges::max_element(p->data) - p->data.begin();
    Tensor r(p->data[idx]);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, idx](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);
        dx.p->data[idx] = dz.p->data[0];
        accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::expand(size_t dim, size_t size) const {
    auto r = Detail::expand(*this, dim, size);

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, dim](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::sum(dz, dim));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }

    return r;
  }

  Tensor Tensor::sum(size_t dim) const {
    auto r = Detail::sum(*this, dim);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, dim](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::expand(dz, dim, ap->shape[dim]));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::mean(size_t dim) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (dim >= p->shape.size())
      throw std::out_of_range("Out of bounds axis.");
    const dtype f = p->shape[dim];
    auto r = Detail::reduce(*this, dim, dtype{0}, std::plus<>());
    for (auto &x : r.p->data)
      x /= f;

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, dim](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        auto dx = Detail::expand(dz, dim, ap->shape[dim]);
        const dtype f = ap->shape[dim];
        for (auto &x : dx.p->data)
          x /= f;
        accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::max(size_t dim) const {
    auto [r, idx] = Detail::cmp(*this, dim, std::greater<>());

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, idx = std::move(idx), dim](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);

        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &zshape = dz.p->shape, &shape = ap->shape;
        auto &strides = ap->strides;

        for (size_t out = 0, n = dz.p->num_el; out < n; out++) {
          size_t off = 0, tmp = out;
          for (size_t i = shape.size(), j = zshape.size(); i-- > 0;) {
            if (i == dim)
              continue;
            j--;
            const size_t c = tmp % zshape[j];
            tmp /= zshape[j];
            off += c * strides[i];
          }
          xdata[off + idx[out] * strides[dim]] = zdata[out];
        }

        accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::min(size_t dim) const {
    auto [r, idx] = Detail::cmp(*this, dim, std::less<>());

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, idx = std::move(idx), dim](const Tensor &dz) {
        if (!ap->requires_grad)
          return;
        Tensor dx(ap->shape);

        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &zshape = dz.p->shape, &shape = ap->shape;
        auto &strides = ap->strides;

        for (size_t out = 0, n = dz.p->num_el; out < n; out++) {
          size_t off = 0, tmp = out;
          for (size_t i = shape.size(), j = zshape.size(); i-- > 0;) {
            if (i == dim)
              continue;
            j--;
            const size_t c = tmp % zshape[j];
            tmp /= zshape[j];
            off += c * strides[i];
          }
          xdata[off + idx[out] * strides[dim]] = zdata[out];
        }

        accumulate(*ap, dx);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::add(const dtype x) const {
    auto r = Detail::add(*this, x);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, dz);
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::add(const Tensor &b) const {
    validate(*this, b);
    auto r = Detail::binary_broadcast(*this, b, std::plus<>());

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};
      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Autograd::sum_to_shape(dz, ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, Autograd::sum_to_shape(dz, bp->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor &Tensor::add_(const dtype x) {
    check_mutable();
    return Detail::binary_inplace(*this, x, std::plus<>());
  }

  Tensor &Tensor::add_(const Tensor &b) {
    check_mutable();
    validate(*this, b);
    if (b.p->requires_grad or b.p->grad_fn)
      throw std::logic_error("In-place op involving autograd tensor.");
    if (not Detail::broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return Detail::binary_inplace(*this, b, std::plus<>());
  }

  Tensor Tensor::sub(const dtype x) const { return add(-x); }

  Tensor Tensor::sub(const Tensor &b) const {
    validate(*this, b);
    auto r = Detail::binary_broadcast(*this, b, std::minus<>());

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};
      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Autograd::sum_to_shape(dz, ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, Autograd::sum_to_shape(Detail::neg(dz), bp->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor &Tensor::sub_(const dtype x) { return add_(-x); }

  Tensor &Tensor::sub_(const Tensor &b) {
    check_mutable();
    validate(*this, b);
    if (b.p->requires_grad or b.p->grad_fn)
      throw std::logic_error("In-place op involving autograd tensor.");
    if (not Detail::broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return Detail::binary_inplace(*this, b, std::minus<>());
  }

  Tensor Tensor::mul(const Tensor &b) const {
    validate(*this, b);
    auto r = Detail::binary_broadcast(*this, b, std::multiplies<>());

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};
      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Autograd::sum_to_shape(Detail::binary_broadcast(dz, Tensor(bp), std::multiplies<>()), ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, Autograd::sum_to_shape(Detail::binary_broadcast(dz, Tensor(ap), std::multiplies<>()), bp->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::mul(const dtype x) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    auto r = Detail::mul(*this, x);

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, x](const Tensor &dz) { accumulate(*ap, dz * x); };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor &Tensor::mul_(const dtype x) {
    check_mutable();
    return Detail::binary_inplace(*this, x, std::multiplies<>());
  }

  Tensor &Tensor::mul_(const Tensor &b) {
    check_mutable();
    validate(*this, b);
    if (b.p->requires_grad or b.p->grad_fn)
      throw std::logic_error("In-place op involving autograd tensor.");
    if (not Detail::broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return Detail::binary_inplace(*this, b, std::multiplies<>());
  }

  Tensor Tensor::matmul(const Tensor &b) const {
    auto r = Detail::matmul(*this, b);

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};

      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        Tensor a(ap), b(bp);
        auto a_t = Detail::transpose(a), b_t = Detail::transpose(b);
        if (ap->requires_grad)
          accumulate(*ap, Autograd::sum_to_shape(Detail::matmul(dz, b_t), ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, Autograd::sum_to_shape(Detail::matmul(a_t, dz), bp->shape));
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::operator-() const {
    auto r = clone();
    std::ranges::for_each(r.p->data, [](dtype &x) { x = -x; });
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::neg(dz));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::operator+(const dtype x) const { return add(x); }

  Tensor operator+(const dtype x, const Tensor &a) { return a.add(x); }

  Tensor &Tensor::operator+=(const dtype x) { return add_(x); }

  Tensor Tensor::operator+(const Tensor &b) const { return add(b); }

  Tensor &Tensor::operator+=(const Tensor &b) { return add_(b); }

  Tensor Tensor::operator-(const dtype x) const { return sub(x); }

  Tensor operator-(const dtype x, const Tensor &a) {
    if (a.empty())
      throw std::invalid_argument("Empty tensor.");
    Tensor r(a.p->shape);
    auto &rdata = r.p->data, &data = a.p->data;
    for (size_t i = 0, n = a.p->num_el; i < n; i++)
      rdata[i] = x - data[i];
    if (a.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {a.p};
      fn->backward = [ap = a.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, Detail::neg(dz));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor &Tensor::operator-=(const dtype x) { return sub_(x); }

  Tensor &Tensor::operator-=(const Tensor &b) { return sub_(b); }

  Tensor Tensor::operator-(const Tensor &b) const { return sub(b); }

  Tensor Tensor::operator*(const dtype x) const { return mul(x); }

  Tensor operator*(const dtype x, const Tensor &a) { return a.mul(x); }

  Tensor &Tensor::operator*=(const dtype x) { return mul_(x); }

  Tensor Tensor::operator*(const Tensor &b) const { return mul(b); }

  Tensor &Tensor::operator*=(const Tensor &b) { return mul_(b); }

  Tensor Tensor::operator/(const dtype x) const {
    if (x == 0)
      throw std::domain_error("Division by zero.");
    return mul(1 / x);
  }

  Tensor &Tensor::operator/=(const dtype x) {
    if (x == 0)
      throw std::domain_error("Division by zero.");
    return mul_(1 / x);
  }

  namespace Raw {
    Tensor add(const Tensor &a, const Tensor &b) {
      Tensor::validate(a, b);
      return Detail::binary_broadcast(a, b, std::plus<>());
    }

    Tensor sub(const Tensor &a, const Tensor &b) {
      Tensor::validate(a, b);
      return Detail::binary_broadcast(a, b, std::minus<>());
    }

    Tensor mul(const Tensor &a, const Tensor &b) {
      Tensor::validate(a, b);
      return Detail::binary_broadcast(a, b, std::multiplies<>());
    }

    Tensor matmul(const Tensor &a, const Tensor &b) { return Detail::matmul(a, b); }
  } // namespace Raw
} // namespace NN
