#include "tensor/tensor.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace NN {
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
    const size_t n = checked_num_el(dims);
    if (!p)
      p = std::make_shared<Impl>();
    p->shape = std::move(dims);
    finalize_shape(n, clear);
  }

  void Tensor::finalize_shape(const size_t n, const bool clear) {
    p->num_el = clear ? 0 : n;
    p->strides.resize(p->shape.size());
    for (size_t i = 0, tmp = n; i < p->shape.size(); i++) {
      tmp /= p->shape[i];
      p->strides[i] = tmp;
    }
    if (clear)
      p->data.clear();
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
    check_mutable();
    if (v.size() != checked_num_el(p->shape))
      throw std::invalid_argument("Data vectors are of different size.");
    p->data = std::move(v);
    p->num_el = p->data.size();
    return *this;
  }

  Tensor &Tensor::set_dims(std::vector<size_t> &&v, const bool clear) {
    check_mutable();
    assign_shape(std::move(v), clear);
    return *this;
  }

  Tensor Tensor::transpose() const { return h_transpose(); }

  Tensor Tensor::h_transpose() const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (p->shape.size() != 2)
      throw std::invalid_argument("Transpose requires 2D tensor.");
    if (p->data.size() != p->num_el)
      throw std::invalid_argument("Malformed tensor data.");

    const size_t r = p->shape[0], c = p->shape[1];
    Tensor ret(c, r);
    auto &data = p->data, &rdata = ret.p->data;
    auto num_el = p->num_el;
    for (size_t t = 0; t < num_el; t++)
      rdata[t] = data[(t % r) * c + t / r];
    return ret;
  }

  Tensor Tensor::cat(const Tensor &b, size_t dim) const { return h_cat(b, dim); }

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

  Tensor Tensor::stack(const Tensor &b, size_t dim) const { return h_stack(b, dim); }

  Tensor Tensor::h_stack(const Tensor &b, size_t dim) const {
    if (empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");
    auto &shape = p->shape;
    if (dim > shape.size())
      throw std::out_of_range("dim is out of bounds.");
    if (shape != b.p->shape)
      throw std::invalid_argument("b is not of the same shape.");

    const size_t axis = dim < shape.size() ? shape[dim] : 1;
    size_t inner = 1;
    for (size_t i = dim + 1; i < shape.size(); i++)
      inner *= shape[i];
    const size_t outer = p->num_el / inner / axis;
    const size_t block = axis * inner;

    std::vector<size_t> new_shape;
    new_shape.reserve(shape.size() + 1);
    for (size_t i = 0; i < dim; i++)
      new_shape.push_back(shape[i]);
    new_shape.push_back(2);
    for (size_t i = dim; i < shape.size(); i++)
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

  bool Tensor::empty() const { return !p or p->num_el == 0; }

  void Tensor::set_requires_grad(const bool x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (p->grad_fn)
      throw std::logic_error("requires_grad can only be changed on a leaf tensor.");
    p->requires_grad = x;
    if (!x)
      p.reset();
  }

  void Tensor::check_mutable() const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if ((p->requires_grad || p->grad_fn))
      throw std::logic_error("In-place op on a tensor that requires grad.");
  }

  void Tensor::validate_pair(const Tensor &a, const Tensor &b) {
    if (a.empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");

    if (not a.valid_strides() or not b.valid_strides())
      throw std::invalid_argument("Malformed strides.");

    if (a.p->data.size() != a.p->num_el or b.p->data.size() != b.p->num_el)
      throw std::logic_error("Malformed num_el or data.");
  }

  Tensor Tensor::clone() const { return Tensor(p->shape); }

  dtype Tensor::sum() const {
    if (empty())
      return 0;
    return std::ranges::fold_left(p->data, 0, [](dtype s, dtype x) { return s + x; });
  }

  dtype Tensor::mean() const {
    if (empty())
      return 0;
    return sum() / p->num_el;
  }

  dtype Tensor::min() const {
    if (empty())
      throw std::domain_error("Cant find minimum of empty tensor.");
    return *std::ranges::min_element(p->data);
  }

  dtype Tensor::max() const {
    if (empty())
      throw std::domain_error("Cant find maximum of empty tensor.");
    return *std::ranges::max_element(p->data);
  }

  Tensor Tensor::sum(size_t dim) const { return h_reduce(dim, 0.0f, std::plus<>()); }

  Tensor Tensor::mean(size_t dim) const { return h_mean(dim); }

  Tensor Tensor::h_mean(size_t dim) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (dim >= p->shape.size())
      throw std::out_of_range("Out of bounds axis.");
    const dtype f = p->shape[dim];
    auto ret = h_reduce(dim, dtype{0}, std::plus<>());
    std::ranges::for_each(ret.p->data, [f](dtype &x) { x /= f; });
    return ret;
  }

  Tensor Tensor::max(size_t dim) const {
    return h_reduce(dim, std::numeric_limits<dtype>::lowest(), [](dtype x, dtype v) { return std::max(x, v); });
  }

  Tensor Tensor::min(size_t dim) const {
    return h_reduce(dim, std::numeric_limits<dtype>::max(), [](dtype x, dtype v) { return std::min(x, v); });
  }

  Tensor Tensor::add(const dtype x) const {
    auto r = clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v += x; });
    return r;
  }

  Tensor Tensor::add(const Tensor &b) const {
    validate_pair(*this, b);
    return binary_broadcast(*this, b, std::plus<>());
  }

  Tensor Tensor::add(const Tensor &a, const Tensor &b) { return a.add(b); }

  Tensor &Tensor::add_(const dtype x) {
    check_mutable();
    std::ranges::for_each(p->data, [x](dtype &v) { v += x; });
    return *this;
  }

  Tensor &Tensor::add_(const Tensor &b) {
    check_mutable();
    validate_pair(*this, b);
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::plus<>());
  }

  Tensor Tensor::sub(const dtype x) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    auto r = clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v -= x; });
    return r;
  }

  Tensor Tensor::sub(const Tensor &b) const {
    validate_pair(*this, b);
    return binary_broadcast(*this, b, std::minus<>());
  }

  Tensor Tensor::sub(const Tensor &a, const Tensor &b) { return a.sub(b); }

  Tensor &Tensor::sub_(const dtype x) {
    check_mutable();
    std::ranges::for_each(p->data, [x](dtype &v) { v -= x; });
    return *this;
  }

  Tensor &Tensor::sub_(const Tensor &b) {
    check_mutable();
    validate_pair(*this, b);
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::minus<>());
  }

  Tensor Tensor::mul(const Tensor &b) const {
    validate_pair(*this, b);
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_broadcast(*this, b, std::multiplies<>());
  }

  Tensor Tensor::mul(const dtype x) const {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    auto r = clone();
    std::ranges::for_each(r.p->data, [x](dtype &v) { v *= x; });
    return r;
  }

  Tensor Tensor::mul(const Tensor &a, const Tensor &b) { return a.mul(b); }

  Tensor &Tensor::mul_(const dtype x) {
    check_mutable();
    std::ranges::for_each(p->data, [x](dtype &v) { v *= x; });
    return *this;
  }

  Tensor &Tensor::mul_(const Tensor &b) {
    check_mutable();
    validate_pair(*this, b);
    if (not broadcasts_into(p->shape, b.p->shape))
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::multiplies<>());
  }

  Tensor Tensor::matmul(const Tensor &b) const { return h_matmul(b); }

  Tensor Tensor::matmul(const Tensor &a, const Tensor &b) { return a.h_matmul(b); }

  Tensor Tensor::h_matmul(const Tensor &b) const {
    validate_pair(*this, b);
    auto &shape = p->shape, &bshape = b.p->shape;
    if (p->shape.size() < 2 or bshape.size() < 2)
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
    if (not std::ranges::equal(a_shape, batch)) {
      a_scratch.assign(br, 0);
      for (size_t i = 0; i < a_shape.size(); i++)
        if (a_shape[i] != 1)
          a_scratch[a_pad + i] = strides[i];
      a_bs = a_scratch;
    }
    if (not std::ranges::equal(b_shape, batch)) {
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

  Tensor Tensor::operator-() const {
    auto c = clone();
    std::ranges::for_each(c.p->data, [](dtype &x) { x = -x; });
    return c;
  }

  Tensor Tensor::operator+(const dtype x) const { return add(x); }

  Tensor operator+(const dtype x, const Tensor &a) { return a.add(x); }

  Tensor &Tensor::operator+=(const dtype x) { return add_(x); }

  Tensor Tensor::operator+(const Tensor &b) const { return add(b); }

  Tensor &Tensor::operator+=(const Tensor &b) { return add_(b); }

  Tensor Tensor::operator-(const dtype x) const { return sub(x); }

  Tensor operator-(const dtype x, const Tensor &a) {
    auto c = -a;
    c.add(x);
    return c;
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
