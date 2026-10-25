#include "tensor/tensor.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include <algorithm>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace NN {
  using namespace Autograd;

  bool Tensor::valid_strides() const {
    const auto &strides = p->strides;
    const auto &shape = p->shape;
    // NOTE: contiguous row-major; relax once non-contiguous views exist.
    if (strides.size() != shape.size())
      return false;
    for (size_t i = shape.size(), acc = 1; i-- > 0;) {
      if (strides[i] != acc)
        return false;
      acc *= shape[i];
    }
    return true;
  }

  bool Tensor::broadcasts_into(const std::vector<size_t> &s, const std::vector<size_t> &b) {
    if (b.size() > s.size())
      return false;
    const size_t off = s.size() - b.size();
    for (size_t i = 0; i < b.size(); i++)
      if (b[i] != 1 and b[i] != s[off + i])
        return false;
    return true;
  }

  std::vector<size_t> Tensor::compute_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides(shape.size());
    size_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      strides[i] = acc;
      acc *= shape[i];
    }
    return strides;
  }

  void Tensor::assign_shape(std::vector<size_t> &&dims, const bool clear) {
    if (!p)
      p = std::make_shared<Impl>();
    const size_t n = checked_num_el(dims);
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

  std::vector<size_t> Tensor::aligned_strides(const Tensor &t, const std::vector<size_t> &out) {
    if (out.size() < t.p->shape.size())
      throw std::logic_error("Cant align strides to a lower rank shape.");
    std::vector<size_t> s(out.size());
    const size_t off = out.size() - t.p->shape.size();
    for (size_t i = 0; i < t.p->shape.size(); i++)
      if (t.p->shape[i] != 1)
        s[off + i] = t.p->strides[i];
    return s;
  }

  Tensor &Tensor::set_data(std::vector<dtype> &&v) {
    if (not empty())
      check_mutable();
    if (v.size() != checked_num_el(p->shape))
      throw std::invalid_argument("Data vectors are of different size.");
    p->data = std::move(v);
    p->num_el = p->data.size();
    return *this;
  }

  Tensor &Tensor::set_dims(std::vector<size_t> &&v, const bool clear) {
    if (not empty())
      check_mutable();
    assign_shape(std::move(v), clear);
    return *this;
  }

  Tensor Tensor::h_reshape(std::vector<size_t> &&v) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (p->num_el != checked_num_el(v))
      throw std::invalid_argument("Incompatible target shape.");

    Tensor x(std::move(v));
    x.p->data = p->data;
    return x;
  }

  Tensor Tensor::reshape(std::vector<size_t> &&v) const {
    auto r = h_reshape(std::move(v));
    if (!p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, dz.reshape(ap->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::h_transpose() const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    const size_t n = p->shape.size();
    if (n < 2)
      throw std::domain_error("Tensor needs to be rank >= 2 for transpose.");

    auto perm = std::views::iota(0zu, n) | std::ranges::to<std::vector>();
    std::swap(perm[n - 2], perm[n - 1]);
    return h_permute(std::move(perm));
  }

  Tensor Tensor::transpose() const {
    auto r = h_transpose();
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, dz.h_transpose());
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::h_cat(const Tensor &b, size_t dim) const {
    if (empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");
    if (dim >= p->shape.size())
      throw std::out_of_range("dim is out of bounds.");
    if (p->shape.size() != b.p->shape.size())
      throw std::invalid_argument("b is not of the same rank.");
    for (size_t i = 0; i < p->shape.size(); i++)
      if (i != dim and p->shape[i] != b.p->shape[i])
        throw std::invalid_argument("Dimensions other than dim should be equal.");

    size_t inner = 1;
    for (size_t i = dim + 1; i < p->shape.size(); i++)
      inner *= p->shape[i];
    const size_t outer = p->num_el / inner / p->shape[dim];
    const size_t a_sz = p->shape[dim] * inner;
    const size_t b_sz = b.p->shape[dim] * inner;

    auto new_shape = p->shape;
    new_shape[dim] += b.p->shape[dim];
    Tensor ret(std::move(new_shape));

    auto &shape = p->shape, &strides = p->strides;
    auto &data = p->data;
    auto &bstrides = b.p->strides;
    auto &rdata = ret.p->data, &bdata = b.p->data;

    for (size_t idx = 0; idx < outer; idx++) {
      size_t a_off = 0, b_off = 0;
      for (size_t i = dim, tmp = idx; i-- > 0;) {
        const size_t c = tmp % shape[i];
        tmp /= shape[i];
        a_off += c * strides[i];
        b_off += c * bstrides[i];
      }

      const size_t ret_off = idx * (a_sz + b_sz);
      std::copy_n(data.begin() + a_off, a_sz, rdata.begin() + ret_off);
      std::copy_n(bdata.begin() + b_off, b_sz, rdata.begin() + ret_off + a_sz);
    }

    return ret;
  }

  Tensor Tensor::cat(const Tensor &b, size_t dim) const {
    auto r = h_cat(b, dim);

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};

      fn->backward = [ap = p, bp = p, dim](const Tensor &dz) {
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

  Tensor Tensor::h_stack(const Tensor &b, size_t dim) const {
    if (empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &shape = p->shape;
    const size_t n = shape.size();
    if (dim > n)
      throw std::out_of_range("dim is out of bounds.");
    if (shape != b.p->shape)
      throw std::invalid_argument("b is not of the same shape.");

    const size_t axis = dim < n ? shape[dim] : 1;
    size_t inner = 1;
    for (size_t i = dim + 1; i < n; i++)
      inner *= shape[i];
    const size_t outer = p->num_el / inner / axis;
    const size_t block = axis * inner;

    std::vector<size_t> new_shape;
    new_shape.reserve(n + 1);
    for (size_t i = 0; i < dim; i++)
      new_shape.push_back(shape[i]);
    new_shape.push_back(2);
    for (size_t i = dim; i < n; i++)
      new_shape.push_back(shape[i]);
    Tensor ret(std::move(new_shape));

    auto &strides = p->strides, &bstrides = b.p->strides;
    auto &data = p->data;
    auto &rdata = ret.p->data, &bdata = b.p->data;
    for (size_t idx = 0; idx < outer; idx++) {
      size_t a_off = 0, b_off = 0;
      for (size_t i = dim, tmp = idx; i-- > 0;) {
        const size_t c = tmp % shape[i];
        tmp /= shape[i];
        a_off += c * strides[i];
        b_off += c * bstrides[i];
      }

      const size_t ret_off = idx * 2 * block;
      std::copy_n(data.begin() + a_off, block, rdata.begin() + ret_off);
      std::copy_n(bdata.begin() + b_off, block, rdata.begin() + ret_off + block);
    }

    return ret;
  }

  Tensor Tensor::stack(const Tensor &b, size_t dim) const {
    auto r = h_stack(b, dim);

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

  Tensor Tensor::h_permute(std::vector<size_t> &&v) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    const size_t n = p->shape.size();
    if (v.size() != n)
      throw std::invalid_argument("`perm` needs to be of the same size.");
    if (not std::ranges::is_permutation(v, std::views::iota(0uz, n)))
      throw std::invalid_argument("`perm` is not a permutation.");

    std::vector<size_t> new_shape(n);
    for (size_t i = 0; i < n; i++)
      new_shape[i] = p->shape[v[i]];

    Tensor r(std::move(new_shape));
    for (size_t idx = 0; idx < r.p->num_el; idx++) {
      size_t off = 0;
      for (size_t i = n, tmp = idx; i-- > 0;) {
        const size_t c = tmp % r.p->shape[i];
        tmp /= r.p->shape[i];
        off += c * p->strides[v[i]];
      }
      r.p->data[idx] = p->data[off];
    }

    return r;
  }

  Tensor Tensor::permute(std::vector<size_t> &&v) const {
    auto r = h_permute(std::move(v));

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
          accumulate(*ap, dz.permute(inv));
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

    if (not valid_strides())
      throw std::invalid_argument("Malformed strides.");

    if (p->data.size() != p->num_el)
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

  Tensor Tensor::h_sum() const {
    if (empty())
      throw std::domain_error("Empty tensor.");
    return Tensor(std::ranges::fold_left(p->data, 0, std::plus<>()));
  }

  Tensor Tensor::sum() const {
    auto r = h_sum();
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
        if (ap->requires_grad)
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
        if (ap->requires_grad)
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
    if (empty())
      throw std::domain_error("Empty tensor.");
    const auto &shape = p->shape;
    const size_t n = shape.size();
    if (dim > n)
      throw std::out_of_range("Out of bounds axis.");

    size_t inner = 1;
    for (size_t i = dim; i < n; i++)
      inner *= shape[i];
    const size_t outer = p->num_el / inner;

    std::vector<size_t> new_shape;
    new_shape.reserve(n + 1);
    for (size_t i = 0; i < dim; i++)
      new_shape.push_back(shape[i]);
    new_shape.push_back(size);
    for (size_t i = dim; i < n; i++)
      new_shape.push_back(shape[i]);

    Tensor r(std::move(new_shape));
    auto &data = p->data, &rdata = r.p->data;
    for (size_t i = 0; i < outer; i++) {
      for (size_t j = 0; j < size; j++) {
        std::copy_n(data.begin() + i * inner, inner, rdata.begin() + i * size * inner + j * inner);
      }
    }

    return r;
  }

  Tensor Tensor::h_sum(size_t dim) const { return h_reduce(dim, 0, std::plus<>()); }

  Tensor Tensor::sum(size_t dim) const {
    auto r = h_sum(dim);
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, dim](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, dz.expand(dim, ap->shape[dim]));
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
    auto r = h_reduce(dim, dtype{0}, std::plus<>());
    for (auto &x : r.p->data)
      x /= f;

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, dim](const Tensor &dz) {
        if (ap->requires_grad)
          return;
        auto dx = dz.expand(dim, ap->shape[dim]);
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
    auto [r, idx] = h_cmp(dim, std::greater<>());

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, idx = std::move(idx), dim](const Tensor &dz) {
        if (ap->requires_grad)
          return;
        Tensor dx(ap->shape);

        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &zshape = dz.p->shape, &shape = ap->shape;
        auto &strides = ap->strides;

        for (size_t out = 0, n = dx.p->num_el; out < n; out++) {
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
    auto [r, idx] = h_cmp(dim, std::less<>());

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, idx = std::move(idx), dim](const Tensor &dz) {
        if (ap->requires_grad)
          return;
        Tensor dx(ap->shape);

        auto &xdata = dx.p->data, &zdata = dz.p->data;
        auto &zshape = dz.p->shape, &shape = ap->shape;
        auto &strides = ap->strides;

        for (size_t out = 0, n = dx.p->num_el; out < n; out++) {
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

  Tensor Tensor::h_add(const dtype x) const {
    auto r = clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v += x; });
    return r;
  }

  Tensor Tensor::add(const dtype x) const {
    auto r = h_add(x);
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
    auto r = binary_broadcast(*this, b, std::plus<>());

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};
      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, sum_to_shape(dz, ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, sum_to_shape(dz, bp->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    validate(a, b);
    return binary_broadcast(a, b, std::plus<>());
  }

  Tensor &Tensor::add_(const dtype x) {
    check_mutable();
    std::ranges::for_each(p->data, [x](dtype &v) { v += x; });
    return *this;
  }

  Tensor &Tensor::add_(const Tensor &b) {
    check_mutable();
    validate(*this, b);
    if (b.p->requires_grad or b.p->grad_fn)
      throw std::logic_error("In-place op involving autograd tensor.");
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::plus<>());
  }

  Tensor Tensor::sub(const dtype x) const { return h_add(-x); }

  Tensor Tensor::sub(const Tensor &b) const {
    validate(*this, b);
    auto r = binary_broadcast(*this, b, std::minus<>());

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};
      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, sum_to_shape(dz, ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, sum_to_shape(-dz, bp->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::sub(const Tensor &a, const Tensor &b) {
    validate(a, b);
    return binary_broadcast(a, b, std::minus<>());
  }

  Tensor &Tensor::sub_(const dtype x) { return add_(-x); }

  Tensor &Tensor::sub_(const Tensor &b) {
    check_mutable();
    validate(*this, b);
    if (b.p->requires_grad or b.p->grad_fn)
      throw std::logic_error("In-place op involving autograd tensor.");
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::minus<>());
  }

  Tensor Tensor::mul(const Tensor &b) const {
    validate(*this, b);
    auto r = binary_broadcast(*this, b, std::multiplies<>());

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};
      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, sum_to_shape(binary_broadcast(dz, Tensor(bp), std::multiplies<>()), ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, sum_to_shape(binary_broadcast(dz, Tensor(ap), std::multiplies<>()), bp->shape));
      };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::mul(const dtype x) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    auto r = clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v *= x; });

    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p, x](const Tensor &dz) { accumulate(*ap, dz * x); };
      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
    validate(a, b);
    return binary_broadcast(a, b, std::multiplies<>());
  }

  Tensor &Tensor::mul_(const dtype x) {
    check_mutable();
    std::ranges::for_each(p->data, [x](dtype &v) { v *= x; });
    return *this;
  }

  Tensor &Tensor::mul_(const Tensor &b) {
    check_mutable();
    validate(*this, b);
    if (b.p->requires_grad or b.p->grad_fn)
      throw std::logic_error("In-place op involving autograd tensor.");
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::multiplies<>());
  }

  Tensor Tensor::h_matmul(const Tensor &b) const {
    validate(*this, b);
    auto &shape = p->shape, &bshape = b.p->shape;
    if (shape.size() < 2 or bshape.size() < 2)
      throw std::domain_error("Cant matmul with a scalar.");

    const size_t arank = shape.size(), brank = bshape.size();
    const size_t m = shape[arank - 2];
    const size_t k = shape[arank - 1];
    const size_t n = bshape[brank - 1];

    // A: [..., m, k]
    // B: [..., k, n]
    if (k != bshape[brank - 2])
      throw std::invalid_argument("Incompatible tensors for matmul.");

    auto a_shape = std::span(shape).first(shape.size() - 2);
    auto b_shape = std::span(bshape).first(bshape.size() - 2);
    auto batch = broadcast_shape(a_shape, b_shape);

    const size_t br = batch.size();
    const size_t a_pad = br - a_shape.size(), b_pad = br - b_shape.size();

    std::vector<size_t> a_scratch, b_scratch;
    auto &strides = p->strides, &bstrides = b.p->strides;
    auto a_bs = std::span<const size_t>(strides).first(br);
    auto b_bs = std::span<const size_t>(bstrides).first(br);
    if (a_pad == 0 and std::ranges::equal(a_shape, batch)) {
      a_bs = std::span<const size_t>(strides).first(br);
    } else {
      a_scratch.assign(br, 0);
      for (size_t i = 0; i < a_shape.size(); i++)
        if (a_shape[i] != 1)
          a_scratch[a_pad + i] = strides[i];
      a_bs = a_scratch;
    }
    if (b_pad == 0 and std::ranges::equal(b_shape, batch)) {
      b_bs = std::span<const size_t>(bstrides).first(br);
    } else {
      b_scratch.assign(br, 0);
      for (size_t i = 0; i < b_shape.size(); i++)
        if (b_shape[i] != 1)
          b_scratch[b_pad + i] = bstrides[i];
      b_bs = b_scratch;
    }

    size_t cnt = 1;
    for (const size_t d : batch)
      cnt *= d;

    batch.reserve(br + 2);
    batch.push_back(m);
    batch.push_back(n);

    std::vector<dtype> out(cnt * m * n);

    size_t a1 = strides[arank - 1], a2 = strides[arank - 2];
    size_t b1 = bstrides[brank - 1], b2 = bstrides[brank - 2];

    auto &data = p->data, &bdata = b.p->data;
    for (size_t idx = 0; idx < cnt; idx++) {
      size_t a_off = 0, b_off = 0, tmp = idx;
      for (size_t d = br; d-- > 0;) {
        const size_t c = tmp % batch[d];
        tmp /= batch[d];
        a_off += c * a_bs[d];
        b_off += c * b_bs[d];
      }
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
          dtype sum = 0;
          for (size_t t = 0; t < k; t++)
            sum += data[a_off + i * a2 + t * a1] * bdata[b_off + t * b2 + j * b1];
          out[idx * m * n + i * n + j] = sum;
        }
      }
    }

    Tensor ret(std::move(batch));
    ret.set_data(std::move(out));
    return ret;
  }

  Tensor Tensor::matmul(const Tensor &b) const {
    auto r = h_matmul(b);

    if (p->requires_grad or b.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p, b.p};

      fn->backward = [ap = p, bp = b.p](const Tensor &dz) {
        Tensor a(ap), b(bp);
        auto a_t = a.h_transpose(), b_t = b.h_transpose();
        if (ap->requires_grad)
          accumulate(*ap, sum_to_shape(dz.h_matmul(b_t), ap->shape));
        if (bp->requires_grad)
          accumulate(*bp, sum_to_shape(a_t.h_matmul(dz), bp->shape));
      };

      r.p->grad_fn = fn;
      r.p->requires_grad = true;
    }
    return r;
  }

  Tensor Tensor::matmul(const Tensor &a, const Tensor &b) { return a.h_matmul(b); }

  Tensor Tensor::operator-() const {
    auto r = clone();
    std::ranges::for_each(r.p->data, [](dtype &x) { x = -x; });
    if (p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {p};
      fn->backward = [ap = p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, -dz);
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
    Tensor r(a.p->shape);
    auto &rdata = r.p->data, &data = a.p->data;
    for (size_t i = 0, n = a.p->num_el; i < n; i++)
      rdata[i] = x - data[i];
    if (a.p->requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = {a.p};
      fn->backward = [ap = a.p](const Tensor &dz) {
        if (ap->requires_grad)
          accumulate(*ap, -dz);
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
} // namespace NN
