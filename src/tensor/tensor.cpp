#include "tensor/tensor.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

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
    num_el = n;
    strides.resize(shape.size());
    for (size_t i = 0, tmp = num_el; i < shape.size(); i++) {
      tmp /= shape[i];
      strides[i] = tmp;
    }
    data.resize(num_el);
    if (clear)
      data.clear();
  }

  std::vector<size_t> Tensor::broadcast_shape(const std::vector<size_t> &a, const std::vector<size_t> &b) {
    const size_t n = std::max(a.size(), b.size());
    std::vector<size_t> out(n, 1);
    const size_t ao = n - a.size(), bo = n - b.size();
    for (size_t i = 0; i < n; i++) {
      const size_t x = i < ao ? 1 : a[i - ao];
      const size_t y = i < bo ? 1 : b[i - bo];
      if (x != y and x != 1 and y != 1)
        throw std::invalid_argument("Shapes not broadcastable.");
      out[i] = std::max(x, y);
    }
    return out;
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
    if (num_el != v.size())
      throw std::invalid_argument("Data vectors are of different size.");
    data = std::move(v);
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

  bool Tensor::empty() const { return num_el == 0 or data.size() == 0; }

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
    if (shape.size() != 2 or b.shape.size() != 2)
      throw std::invalid_argument("Tensor should be rank 2 for matmul");
    if (shape[1] != b.shape[0])
      throw std::invalid_argument("Incompatible tensors for matmul.");

    Tensor ret(shape[0], b.shape[1]);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < b.shape[1]; j++) {
        dtype sum = 0;
        for (size_t k = 0; k < shape[1]; k++)
          sum += at(i, k) * b.at(k, j);
        ret.at(i, j) = sum;
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
