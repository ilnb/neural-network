#include "tensor/detail.hpp"
#include "nn/activations.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace NN::Detail {
  Tensor neg(const Tensor &t) {
    auto r = t.clone();
    std::ranges::for_each(r.p->data, [](dtype &x) { x = -x; });
    return r;
  }

  Tensor sqrt(const Tensor &t) {
    if (t.empty())
      throw std::domain_error("Empty tensor.");
    return Act::Detail::map(t, [](auto x) { return std::sqrt(x); });
  }

  Tensor sum(const Tensor &t) {
    if (t.empty())
      throw std::domain_error("Empty tensor.");
    return Tensor(std::ranges::fold_left(t.p->data, 0, std::plus<>()));
  }

  Tensor sum(const Tensor &t, size_t dim) { return reduce(t, dim, 0, std::plus<>()); }

  Tensor expand(const Tensor &t, size_t dim, size_t size) {
    if (t.empty())
      throw std::domain_error("Empty tensor.");
    auto &p = t.p;
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

  Tensor reshape(const Tensor &t, std::vector<size_t> &&v) {
    if (t.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &p = t.p;
    if (p->num_el != checked_num_el(v))
      throw std::invalid_argument("Incompatible target shape.");

    Tensor x(std::move(v));
    x.p->data = p->data;
    return x;
  }

  Tensor permute(const Tensor &t, const std::vector<size_t> &v) {
    if (t.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &p = t.p;
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

  Tensor transpose(const Tensor &t) {
    if (t.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &p = t.p;
    const size_t n = p->shape.size();
    if (n < 2)
      throw std::domain_error("Tensor needs to be rank >= 2 for transpose.");

    auto perm = std::views::iota(0zu, n) | std::ranges::to<std::vector>();
    std::swap(perm[n - 2], perm[n - 1]);
    return permute(t, perm);
  }

  Tensor cat(const Tensor &t, const Tensor &b, size_t dim) {
    if (t.empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &p = t.p;
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

  Tensor stack(const Tensor &t, const Tensor &b, size_t dim) {
    if (t.empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &p = t.p;
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

  Tensor add(const Tensor &t, const dtype x) {
    auto r = t.clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v += x; });
    return r;
  }

  Tensor mul(const Tensor &t, const dtype x) {
    auto r = t.clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v *= x; });
    return r;
  }

  Tensor matmul(const Tensor &t, const Tensor &b) {
    Tensor::validate(t, b);
    auto &p = t.p;
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
    std::span<const size_t> a_bs, b_bs;
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
    ret.p->data = std::move(out);
    return ret;
  }
} // namespace NN::Detail
