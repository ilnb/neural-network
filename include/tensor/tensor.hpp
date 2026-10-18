#pragma once

#include <array>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace NN {
  template <typename R, typename T>
  concept ConvIntervalOf = std::ranges::random_access_range<R> and std::convertible_to<std::ranges::range_value_t<R>, T>;

  using dtype = float; // type to represent raw values

  struct Tensor {
    std::vector<dtype> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t num_el{};

    Tensor() = default;

    Tensor(const Tensor &b) = default;

    Tensor(Tensor &&b) noexcept : data(std::move(b.data)), shape(std::move(b.shape)), strides(std::move(b.strides)), num_el(b.num_el) { b.num_el = 0; }

    Tensor &operator=(const Tensor &b);

    Tensor &operator=(Tensor &&b) noexcept;

    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    explicit Tensor(R &&r) : shape(std::ranges::begin(r), std::ranges::end(r)) {
      const size_t n = checked_num_el(shape);
      finalize_shape(n, false);
    }

    explicit Tensor(std::vector<size_t> &&v) : shape(std::move(v)) {
      const size_t n = checked_num_el(shape);
      finalize_shape(n, false);
    }

    template <typename T>
      requires std::integral<T>
    explicit Tensor(std::initializer_list<T> l) : shape(l.begin(), l.end()) {
      const size_t n = checked_num_el(shape);
      finalize_shape(n, false);
    }

    template <typename... Args>
      requires(std::integral<Args> and ...)
    explicit Tensor(Args... dims) : shape{static_cast<size_t>(dims)...} {
      const size_t n = checked_num_el(shape);
      finalize_shape(n, false);
    }

    template <typename R>
      requires ConvIntervalOf<R, dtype> and std::ranges::sized_range<R>
    Tensor &set_data(R &&r) {
      if (std::ranges::size(r) != checked_num_el(shape))
        throw std::invalid_argument("Data ranges are of different size.");
      data.assign_range(r);
      num_el = data.size();
      return *this;
    }

    template <typename... Args>
      requires(std::convertible_to<Args, dtype> and ...)
    Tensor &set_data(Args... vals) {
      return set_data(std::vector<float>{static_cast<dtype>(vals)...});
    }

    Tensor &set_data(std::vector<dtype> &&v);

    // Clears data vector if @clear is set
    template <typename R>
      requires std::integral<std::ranges::range_value_t<R>>
    Tensor &set_dims(R &&r, const bool clear = true) {
      assign_shape(std::forward<R>(r), clear);
      return *this;
    }

    // Clears data vector if @clear is set
    Tensor &set_dims(std::vector<size_t> &&v, const bool clear = true);

    template <typename Self, typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    decltype(auto) at(this Self &&self, R &&r) {
      if (std::ranges::size(r) != self.shape.size())
        throw std::invalid_argument("Number of indices doesnt match tensor rank.");
      auto it = std::ranges::begin(r); // supports more types
      size_t offset = 0;
      for (size_t i = 0; i < self.shape.size(); i++) {
        if (it[i] >= self.shape[i])
          throw std::out_of_range("Tensor index out of bounds.");
        offset += it[i] * self.strides[i];
      }
      return self.data.at(offset);
    }

    template <typename Self, typename T>
      requires std::integral<T>
    decltype(auto) at(this Self &&self, std::initializer_list<T> l) {
      return self.template at<std::initializer_list<T> &>(l); // bounds above to itr-indexing
    }

    template <typename Self, typename... Args>
      requires(std::integral<Args> and ...)
    decltype(auto) at(this Self &&self, Args... idx) {
      return self.at(std::array<size_t, sizeof...(Args)>{static_cast<size_t>(idx)...});
    }

    template <typename Self, typename... Args>
      requires(std::integral<Args> and ...)
    decltype(auto) operator[](this Self &&self, Args... idx) {
      if (sizeof...(idx) != self.shape.size())
        throw std::invalid_argument("Number of indices doesnt match tensor rank.");
      std::array<size_t, sizeof...(Args)> tmp{static_cast<size_t>(idx)...};
      size_t offset = 0;
      for (size_t i = 0; i < self.shape.size(); i++)
        offset += tmp[i] * self.strides[i];
      return self.data[offset];
    }

    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    Tensor &reshape(R &&r) {
      const size_t n = checked_num_el(r);
      if (n != num_el)
        throw std::invalid_argument("Incompatible target shape.");
      shape.assign_range(r);
      finalize_shape(n, false);
      return *this;
    }

    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor &reshape(Args... dims) {
      return reshape(std::array<size_t, sizeof...(Args)>{static_cast<size_t>(dims)...});
    }

    template <typename T>
      requires std::integral<T>
    Tensor &reshape(std::initializer_list<T> l) {
      return reshape<std::initializer_list<T> &>(l);
    }

    Tensor &transpose();
    Tensor ctranspose() const;

    bool empty() const;
    dtype sum() const;
    dtype mean() const;
    dtype max() const;
    dtype min() const;

    Tensor &sum(size_t dim);
    Tensor &mean(size_t dim);
    Tensor &max(size_t dim);
    Tensor &min(size_t dim);

    Tensor csum(size_t dim) const;
    Tensor cmean(size_t dim) const;
    Tensor cmax(size_t dim) const;
    Tensor cmin(size_t dim) const;

    Tensor &add(const dtype x);
    Tensor &sub(const dtype x);
    Tensor &mul(const dtype x);
    Tensor &add(const Tensor &b);
    Tensor &sub(const Tensor &b);
    Tensor &mul(const Tensor &b);
    Tensor &matmul(const Tensor &b);

    // Copy based ops
    Tensor cadd(const dtype x) const;
    Tensor csub(const dtype x) const;
    Tensor cmul(const dtype x) const;
    Tensor cadd(const Tensor &b) const;
    Tensor csub(const Tensor &b) const;
    Tensor cmul(const Tensor &b) const;
    Tensor cmatmul(const Tensor &b) const;

    static Tensor add(const Tensor &a, const Tensor &b);
    static Tensor sub(const Tensor &a, const Tensor &b);
    static Tensor mul(const Tensor &a, const Tensor &b);
    static Tensor matmul(const Tensor &a, const Tensor &b);

    Tensor operator-() const;

    Tensor operator+(const dtype x) const;
    friend Tensor operator+(const dtype x, const Tensor &a);
    Tensor &operator+=(const dtype x);

    Tensor operator-(const dtype x) const;
    friend Tensor operator-(const dtype x, const Tensor &a);
    Tensor &operator-=(const dtype x);

    Tensor operator*(const dtype x) const;
    friend Tensor operator*(const dtype x, const Tensor &a);
    Tensor &operator*=(const dtype x);

    Tensor operator/(const dtype x) const;
    Tensor &operator/=(const dtype x);

    Tensor operator+(const Tensor &b) const;
    Tensor &operator+=(const Tensor &b);
    Tensor operator-(const Tensor &b) const;
    Tensor &operator-=(const Tensor &b);
    Tensor operator*(const Tensor &b) const;
    Tensor &operator*=(const Tensor &b);

  private:
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    void assign_shape(R &&dims, const bool clear) {
      const size_t n = checked_num_el(dims);
      shape.assign_range(dims);
      finalize_shape(n, clear);
    }

    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    static size_t checked_num_el(R &&s) {
      if (std::ranges::empty(s))
        throw std::invalid_argument("Empty size vector.");
      size_t n = 1;
      for (size_t x : s) {
        if (x == 0)
          throw std::invalid_argument("Tensor dimensions must be positive.");
        if (n > std::numeric_limits<size_t>::max() / x)
          throw std::overflow_error("Tensor size overflow");
        n *= x;
      }
      return n;
    }

    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    Tensor &reduce(size_t dim, dtype init, Op op) {
      if (empty())
        throw std::domain_error("Cant reduce an empty tensor.");
      const size_t n = shape.size();
      if (dim >= n)
        throw std::out_of_range("Out of bounds axis.");
      if (n == 1)
        throw std::domain_error("Cant reduce a scalar.");

      std::vector<size_t> new_size;
      new_size.reserve(n - 1);
      for (size_t i = 0; i < n; i++)
        if (i != dim)
          new_size.push_back(shape[i]);

      Tensor ret(new_size);
      for (size_t out = 0; out < ret.num_el; out++) {
        size_t offset = 0, tmp = out;

        for (size_t i = 0, j = 0; i < n; i++) {
          if (i == dim)
            continue;
          const size_t coord = tmp / ret.strides[j];
          tmp %= ret.strides[j];
          offset += coord * strides[i];
          j++;
        }

        auto val = init;
        for (size_t k = 0; k < shape[dim]; k++)
          val = op(val, data[offset + k * strides[dim]]);
        ret.data[out] = val;
      }

      *this = std::move(ret);
      return *this;
    }

    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    Tensor &binary_inplace(const Tensor &b, Op op) {
      const auto bs = aligned_strides(b, shape);

      for (size_t i = 0; i < num_el; i++) {
        size_t ob = 0, tmp = i;
        for (size_t k = 0; k < shape.size(); k++) {
          const size_t c = tmp / strides[k];
          tmp %= strides[k];
          ob += c * bs[k];
        }

        data[i] = op(data[i], b.data[ob]);
      }
      return *this;
    }

    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    static Tensor binary_broadcast(const Tensor &a, const Tensor &b, Op op) {
      auto out = broadcast_shape(a.shape, b.shape);
      const auto out_sz = out.size();
      const auto as = aligned_strides(a, out);
      const auto bs = aligned_strides(b, out);

      Tensor ret(std::move(out));
      for (size_t i = 0; i < ret.num_el; i++) {
        size_t oa = 0, ob = 0, tmp = i;
        for (size_t k = 0; k < out_sz; k++) {
          const size_t c = tmp / ret.strides[k];
          tmp %= ret.strides[k];
          oa += c * as[k];
          ob += c * bs[k];
        }

        ret.data[i] = op(a.data[oa], b.data[ob]);
      }
      return ret;
    }

    void valid_pair(const Tensor &b) const;
    bool valid_strides() const;
    static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);
    void assign_shape(std::vector<size_t> &&dims, const bool clear);
    void finalize_shape(size_t n, bool clear);
    static std::vector<size_t> broadcast_shape(const std::vector<size_t> &a, const std::vector<size_t> &b);
    // right aligned strides
    static std::vector<size_t> aligned_strides(const Tensor &t, const std::vector<size_t> &out);
  };
} // namespace NN
