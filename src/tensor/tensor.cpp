#include "tensor/tensor.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace NN {
  Tensor &Tensor::operator=(const Tensor &b) {
    if (this != &b) {
      data = b.data;
      shape = b.shape;
      strides = b.strides;
      num_el = b.num_el;
    }
    return *this;
  }

  Tensor &Tensor::operator=(Tensor &&b) noexcept {
    if (this != &b) {
      data = std::move(b.data);
      shape = std::move(b.shape);
      strides = std::move(b.strides);
      num_el = b.num_el;
      b.num_el = 0;
    }
    return *this;
  }

  bool Tensor::valid_strides() const {
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

  void Tensor::valid_pair(const Tensor &b) const {
    if (not valid_strides() or not b.valid_strides())
      throw std::invalid_argument("Malformed strides.");

    if (empty() or b.empty())
      throw std::invalid_argument("Tensor data is not initialized.");

    if (data.size() != num_el or b.data.size() != b.num_el)
      throw std::logic_error("Malformed num_el or data.");
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
    shape = std::move(dims);
    finalize_shape(n, clear);
  }

  void Tensor::finalize_shape(const size_t n, const bool clear) {
    num_el = clear ? 0 : n;
    strides.resize(shape.size());
    for (size_t i = 0, tmp = n; i < shape.size(); i++) {
      tmp /= shape[i];
      strides[i] = tmp;
    }
    if (clear)
      data.clear();
    else
      data.resize(n);
  }

  std::vector<size_t> Tensor::aligned_strides(const Tensor &t, const std::vector<size_t> &out) {
    if (out.size() < t.shape.size())
      throw std::logic_error("Cant align strides to a lower rank shape.");
    std::vector<size_t> s(out.size());
    const size_t off = out.size() - t.shape.size();
    for (size_t i = 0; i < t.shape.size(); i++)
      if (t.shape[i] != 1)
        s[off + i] = t.strides[i];
    return s;
  }

  Tensor &Tensor::set_data(std::vector<dtype> &&v) {
    if (v.size() != checked_num_el(shape))
      throw std::invalid_argument("Data vectors are of different size.");
    data = std::move(v);
    num_el = data.size();
    return *this;
  }

  Tensor &Tensor::set_dims(std::vector<size_t> &&v, const bool clear) {
    assign_shape(std::move(v), clear);
    return *this;
  }

  Tensor &Tensor::transpose() {
    if (shape.size() != 2)
      throw std::invalid_argument("Transpose requires 2D tensor.");
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (data.size() != num_el)
      throw std::invalid_argument("Malformed tensor data.");
    const size_t r = shape[0], c = shape[1];
    std::vector<dtype> out(num_el);
    for (size_t i = 0; i < r; i++)
      for (size_t j = 0; j < c; j++)
        out[j * r + i] = data[i * c + j];
    data = std::move(out);
    std::ranges::reverse(shape);
    strides = compute_strides(shape);
    return *this;
  }

  Tensor Tensor::ctranspose() const {
    auto c = *this;
    c.transpose();
    return c;
  }

  bool Tensor::empty() const { return num_el == 0; }

  dtype Tensor::sum() const {
    if (empty())
      return 0;
    return std::ranges::fold_left(data, 0, [](dtype s, dtype x) { return s + x; });
  }

  dtype Tensor::mean() const {
    if (empty())
      return 0;
    return sum() / num_el;
  }

  dtype Tensor::min() const {
    if (empty())
      throw std::domain_error("Cant find minimum of empty tensor.");
    return *std::ranges::min_element(data);
  }

  dtype Tensor::max() const {
    if (empty())
      throw std::domain_error("Cant find maximum of empty tensor.");
    return *std::ranges::max_element(data);
  }

  Tensor &Tensor::sum(size_t dim) { return reduce(dim, 0.0f, std::plus<>()); }

  Tensor Tensor::csum(size_t dim) const {
    auto c = *this;
    c.sum(dim);
    return c;
  }

  Tensor &Tensor::mean(size_t dim) {
    if (dim >= shape.size())
      throw std::out_of_range("Out of bounds axis.");
    const size_t f = shape[dim];
    sum(dim);
    std::ranges::for_each(data, [f](dtype &x) { x /= f; });
    return *this;
  }

  Tensor Tensor::cmean(size_t dim) const {
    auto c = *this;
    c.mean(dim);
    return c;
  }

  Tensor &Tensor::max(size_t dim) {
    return reduce(dim, std::numeric_limits<dtype>::lowest(), [](dtype x, dtype v) { return std::max(x, v); });
  }

  Tensor Tensor::cmax(size_t dim) const {
    auto c = *this;
    c.max(dim);
    return c;
  }

  Tensor &Tensor::min(size_t dim) {
    return reduce(dim, std::numeric_limits<dtype>::max(), [](dtype x, dtype v) { return std::min(x, v); });
  }

  Tensor Tensor::cmin(size_t dim) const {
    auto c = *this;
    c.min(dim);
    return c;
  }

  Tensor &Tensor::add(const dtype x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    std::ranges::for_each(data, [x](dtype &v) { v += x; });
    return *this;
  }

  Tensor &Tensor::add(const Tensor &b) {
    valid_pair(b);
    if (broadcast_shape(shape, b.shape) != shape)
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::plus<>());
  }

  Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.add(b);
    return c;
  }

  Tensor Tensor::cadd(const dtype x) const {
    auto c = *this;
    c.add(x);
    return c;
  }

  Tensor Tensor::cadd(const Tensor &b) const {
    valid_pair(b);
    return binary_broadcast(*this, b, std::plus<>());
  }

  Tensor &Tensor::sub(const dtype x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    std::ranges::for_each(data, [x](dtype &v) { v -= x; });
    return *this;
  }

  Tensor &Tensor::sub(const Tensor &b) {
    valid_pair(b);
    if (broadcast_shape(shape, b.shape) != shape)
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::minus<>());
  }

  Tensor Tensor::sub(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.sub(b);
    return c;
  }

  Tensor Tensor::csub(const dtype x) const {
    auto c = *this;
    c.sub(x);
    return c;
  }

  Tensor Tensor::csub(const Tensor &b) const {
    valid_pair(b);
    return binary_broadcast(*this, b, std::minus<>());
  }

  Tensor &Tensor::mul(const Tensor &b) {
    valid_pair(b);
    if (broadcast_shape(shape, b.shape) != shape)
      throw std::invalid_argument("Operand doesnt broadcast into this tensor in-place.");
    return binary_inplace(b, std::multiplies<>());
  }

  Tensor &Tensor::mul(const dtype x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    std::ranges::for_each(data, [x](dtype &v) { v *= x; });
    return *this;
  }

  Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.mul(b);
    return c;
  }

  Tensor Tensor::cmul(const dtype x) const {
    auto c = *this;
    c.mul(x);
    return c;
  }

  Tensor Tensor::cmul(const Tensor &b) const {
    valid_pair(b);
    return binary_broadcast(*this, b, std::multiplies<>());
  }

  Tensor &Tensor::matmul(const Tensor &b) {
    if (empty() or b.empty())
      throw std::invalid_argument("Empty tensor.");
    if (shape.size() < 2 or b.shape.size() < 2)
      throw std::domain_error("Cant matmul with a scalar.");

    const size_t arank = shape.size(), brank = b.shape.size();
    const size_t m = shape[arank - 2];
    const size_t k = shape[arank - 1];
    const size_t n = b.shape[brank - 1];

    // A: [..., m, k]
    // B: [..., k, n]
    if (k != b.shape[brank - 2])
      throw std::invalid_argument("Incompatible tensors for matmul.");

    auto a_shape = std::span(shape).first(shape.size() - 2);
    auto b_shape = std::span(b.shape).first(b.shape.size() - 2);
    auto batch = broadcast_shape(a_shape, b_shape);

    const size_t a_pad = batch.size() - a_shape.size(), b_pad = batch.size() - b_shape.size();
    std::vector<size_t> a_bs(batch.size()), b_bs(batch.size());
    for (size_t i = 0; i < a_shape.size(); i++)
      if (a_shape[i] != 1)
        a_bs[a_pad + i] = strides[i];
    for (size_t i = 0; i < b_shape.size(); i++)
      if (b_shape[i] != 1)
        b_bs[b_pad + i] = b.strides[i];

    std::vector<size_t> out_shape = batch;
    out_shape.push_back(m);
    out_shape.push_back(n);
    Tensor ret(std::move(out_shape));

    size_t a1 = strides[arank - 1], a2 = strides[arank - 2];
    size_t b1 = b.strides[brank - 1], b2 = b.strides[brank - 2];

    for (size_t idx = 0, cnt = ret.num_el / m / n; idx < cnt; idx++) {
      size_t a_off = 0, b_off = 0, tmp = idx;
      for (size_t d = batch.size(); d-- > 0;) {
        const size_t c = tmp % batch[d];
        tmp /= batch[d];
        a_off += c * a_bs[d];
        b_off += c * b_bs[d];
      }
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
          dtype sum = 0;
          for (size_t t = 0; t < k; t++)
            sum += data[a_off + i * a2 + t * a1] * b.data[b_off + t * b2 + j * b1];
          ret.data[idx * m * n + i * n + j] = sum;
        }
      }
    }

    *this = std::move(ret);
    return *this;
  }

  Tensor Tensor::cmatmul(const Tensor &b) const {
    auto c = *this;
    c.matmul(b);
    return c;
  }

  Tensor Tensor::matmul(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.matmul(b);
    return c;
  }

  Tensor Tensor::operator-() const {
    auto c = *this;
    std::ranges::for_each(c.data, [](dtype &x) { x = -x; });
    return c;
  }

  Tensor Tensor::operator+(const dtype x) const { return cadd(x); }

  Tensor operator+(const dtype x, const Tensor &a) { return a.cadd(x); }

  Tensor &Tensor::operator+=(const dtype x) { return add(x); }

  Tensor Tensor::operator+(const Tensor &b) const { return cadd(b); }

  Tensor &Tensor::operator+=(const Tensor &b) { return add(b); }

  Tensor Tensor::operator-(const dtype x) const { return csub(x); }

  Tensor operator-(const dtype x, const Tensor &a) {
    auto c = -a;
    c.add(x);
    return c;
  }

  Tensor &Tensor::operator-=(const dtype x) { return sub(x); }

  Tensor &Tensor::operator-=(const Tensor &b) { return sub(b); }

  Tensor Tensor::operator-(const Tensor &b) const { return csub(b); }

  Tensor Tensor::operator*(const dtype x) const { return cmul(x); }

  Tensor operator*(const dtype x, const Tensor &a) { return a.cmul(x); }

  Tensor &Tensor::operator*=(const dtype x) { return mul(x); }

  Tensor Tensor::operator*(const Tensor &b) const { return cmul(b); }

  Tensor &Tensor::operator*=(const Tensor &b) { return mul(b); }

  Tensor Tensor::operator/(const dtype x) const {
    if (x == 0)
      throw std::domain_error("Division by zero.");
    return cmul(1 / x);
  }

  Tensor &Tensor::operator/=(const dtype x) {
    if (x == 0)
      throw std::domain_error("Division by zero.");
    return mul(1 / x);
  }
} // namespace NN
