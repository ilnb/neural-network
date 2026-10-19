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
  // Random-access range whose values convert to `T`.
  template <typename R, typename T>
  concept ConvIntervalOf = std::ranges::random_access_range<R> and std::convertible_to<std::ranges::range_value_t<R>, T>;

  // Element type for raw values.
  using dtype = float;

  // An N-dimensional, contiguous, row-major tensor.
  //
  // Arithmetic operators are element-wise and broadcast; `matmul()` is the
  // matrix product.
  //
  // **In-place vs copy**
  //
  // `cadd`/`csub`/`cmul` and `+`/`-`/`*` return a new, fully broadcast tensor.
  // `add`/`sub`/`mul` and `+=`/`-=`/`*=` mutate in-place and require the result
  // shape to equal the receiver's shape.
  //
  // **Broadcasting**
  //
  // Shapes are right-aligned (missing leading dims are 1); each aligned axis
  // must be equal or 1. Ranks may differ.
  //
  // ```
  // {2,3} +  {3}    -> {2,3}
  // {3}   +  {2,3}  -> {2,3}
  // {2,3} += {3}    -> ok
  // {3}   += {2,3}  -> throws
  // ```
  struct Tensor {
    // Flat row-major storage. `data.size() == num_el`.
    std::vector<dtype> data;
    // Dimensions. `shape.size()` is the rank.
    std::vector<size_t> shape;
    // Row-major strides of `shape`.
    std::vector<size_t> strides;
    // `product(shape)`.
    size_t num_el{};

    Tensor() = default;

    Tensor(const Tensor &b) = default;

    Tensor(Tensor &&b) noexcept : data(std::move(b.data)), shape(std::move(b.shape)), strides(std::move(b.strides)), num_el(b.num_el) { b.num_el = 0; }

    Tensor &operator=(const Tensor &b);

    Tensor &operator=(Tensor &&b) noexcept;

    // Constructs a tensor from a shape range. Data is zero-initialized.
    //
    // Throws `std::invalid_argument` if the range is empty or contains a 0,
    // and `std::overflow_error` if the element count overflows `size_t`.
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

    // Convenience constructor, directly takes values.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    explicit Tensor(Args... dims) : shape{static_cast<size_t>(dims)...} {
      const size_t n = checked_num_el(shape);
      finalize_shape(n, false);
    }

    // Copies `r` into `data`. `r` must have exactly `num_el` elements.
    //
    // Throws `std::invalid_argument` on a size mismatch or if the new shape is invalid,
    // and `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, dtype> and std::ranges::sized_range<R>
    Tensor &set_data(R &&r) {
      if (std::ranges::size(r) != checked_num_el(shape))
        throw std::invalid_argument("Data ranges are of different size.");
      data.assign_range(r);
      num_el = data.size();
      return *this;
    }

    // Convenience form of `set_data` taking the values directly.
    template <typename... Args>
      requires(std::convertible_to<Args, dtype> and ...)
    Tensor &set_data(Args... vals) {
      return set_data(std::vector<float>{static_cast<dtype>(vals)...});
    }

    // Moves `v` into `data`. `v` must have exactly `num_el` elements.
    //
    // Throws `std::invalid_argument` on a size mismatch.
    Tensor &set_data(std::vector<dtype> &&v);

    // Sets the shape from `r` and recomputes strides.
    // `data` is emptied if `clear` is set, resized otherwise.
    //
    // Throws `std::invalid_argument` if the range is empty or contains a 0,
    // and `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires std::integral<std::ranges::range_value_t<R>>
    Tensor &set_dims(R &&r, const bool clear = true) {
      assign_shape(std::forward<R>(r), clear);
      return *this;
    }

    // Same as the range overload, taking a shape vector.
    Tensor &set_dims(std::vector<size_t> &&v, const bool clear = true);

    // Element access with bounds checking. Accepts a sized random-access range of indices.
    //
    // Throws `std::invalid_argument` if the number of indices differs from the rank,
    // and `std::out_of_range` if an index is out of bounds.
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

    // Element access with bounds checking.
    template <typename Self, typename T>
      requires std::integral<T>
    decltype(auto) at(this Self &&self, std::initializer_list<T> l) {
      return self.template at<std::initializer_list<T> &>(l); // bounds above to itr-indexing
    }

    // Element access with bounds checking.
    template <typename Self, typename... Args>
      requires(std::integral<Args> and ...)
    decltype(auto) at(this Self &&self, Args... idx) {
      return self.at(std::array<size_t, sizeof...(Args)>{static_cast<size_t>(idx)...});
    }

    // Element access without bounds checking.
    //
    // Throws `std::invalid_argument` if the number of indices
    // differs from the rank. Indices must be in range.
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

    // Reshapes in-place. The total element count must be preserved.
    //
    // Throws `std::invalid_argument` if the new shape is invalid or its element count differs from `num_el`,
    // and `std::overflow_error` if the element count overflows `size_t`.
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

    // Directly takes values for new shape.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor &reshape(Args... dims) {
      return reshape(std::array<size_t, sizeof...(Args)>{static_cast<size_t>(dims)...});
    }

    // Reshapes in-place from an initializer list.
    template <typename T>
      requires std::integral<T>
    Tensor &reshape(std::initializer_list<T> l) {
      return reshape<std::initializer_list<T> &>(l);
    }

    // Transposes a rank-2 tensor in-place: permutes `data` and recomputes
    // strides. Not a view.
    //
    // Throws `std::invalid_argument` if the tensor is not rank 2 or is empty.
    Tensor &transpose();

    // Copy-transpose.
    Tensor ctranspose() const;

    // Concatenates `b` onto this tensor along the existing dimension `dim`.
    // Every dimension is kept, and `dim` holds `shape[dim] + b.shape[dim]`.
    //
    // The tensors must have equal rank and match on every dimension but `dim`.
    //
    // Throws `std::out_of_range` if `dim >= rank`, and `std::invalid_argument`
    // if the ranks differ, any dimension other than `dim` differs,
    // or either tensor is empty.
    Tensor &cat(const Tensor &b, size_t dim);

    // Copycat.
    Tensor ccat(const Tensor &b, size_t dim) const;

    // Stacks `b` onto this tensor along a new dimension `dim`. The result has
    // rank + 1, with a size-2 axis inserted at `dim`; this tensor and `b` are
    // its two slices. The two tensors must have identical shapes.
    //
    // `dim` may equal the rank, which appends the new axis at the end.
    //
    // Throws `std::out_of_range` if `dim > rank`, and `std::invalid_argument`
    // if the shapes differ or either tensor is empty.
    Tensor &stack(const Tensor &b, size_t dim);

    // Copy-stack.
    Tensor cstack(const Tensor &b, size_t dim) const;

    // True if the tensor holds no elements.
    bool empty() const;

    // Sum of all elements. Returns 0 for an empty tensor.
    dtype sum() const;

    // Mean of all elements. Returns 0 for an empty tensor.
    dtype mean() const;

    // Largest element.
    //
    // Throws `std::domain_error` for an empty tensor.
    dtype max() const;

    // Smallest element.
    //
    // Throws `std::domain_error` for an empty tensor.
    dtype min() const;

    // Reduces axis `dim` in-place. The axis is removed,
    // so the rank decreases by one.
    //
    // Throws `std::out_of_range` if `dim >= rank`,
    // and `std::domain_error` for empty tensor.
    Tensor &sum(size_t dim);

    // Reduces axis `dim` in-place by the mean.
    Tensor &mean(size_t dim);

    // Reduces axis `dim` in-place by the maximum.
    Tensor &max(size_t dim);

    // Reduces axis `dim` in-place by the minimum.
    Tensor &min(size_t dim);

    // Copy-reduces axis `dim` by the sum.
    Tensor csum(size_t dim) const;

    // Copy-reduces axis `dim` by the mean.
    Tensor cmean(size_t dim) const;

    // Copy-reduces axis `dim` by the maximum.
    Tensor cmax(size_t dim) const;

    // Copy-reduces axis `dim` by the minimum.
    Tensor cmin(size_t dim) const;

    // Adds scalar `x` to every element in-place.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor &add(const dtype x);

    // Subtracts scalar `x` from every element in-place.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor &sub(const dtype x);

    // Multiplies every element by scalar `x` in-place.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor &mul(const dtype x);

    // In-place element-wise add with broadcasting. `b` must broadcast into
    // `*this` without changing its shape.
    //
    // Throws `std::invalid_argument` if `b` does not broadcast into `shape`,
    // or if either tensor is empty or malformed.
    Tensor &add(const Tensor &b);

    // In-place element-wise subtract with broadcasting.
    Tensor &sub(const Tensor &b);

    // In-place element-wise multiply with broadcasting.
    Tensor &mul(const Tensor &b);

    // Rank-2 in-place matrix product.
    //
    // Throws `std::invalid_argument` if either tensor is not rank 2 or the
    // inner dimensions differ.
    Tensor &matmul(const Tensor &b);

    // Returns a copy with scalar `x` added.
    Tensor cadd(const dtype x) const;

    // Returns a copy with scalar `x` subtracted.
    Tensor csub(const dtype x) const;

    // Returns a copy with every element multiplied by scalar `x`.
    Tensor cmul(const dtype x) const;

    // Copy element-wise add with full broadcasting.
    Tensor cadd(const Tensor &b) const;

    // Copy element-wise subtract with full broadcasting.
    Tensor csub(const Tensor &b) const;

    // Copy element-wise multiply with full broadcasting.
    Tensor cmul(const Tensor &b) const;

    // Copy matrix product. Rank 2 only.
    Tensor cmatmul(const Tensor &b) const;

    // Returns `a + b` as a new tensor, broadcasting fully.
    static Tensor add(const Tensor &a, const Tensor &b);

    // Returns `a - b` as a new tensor, broadcasting fully.
    static Tensor sub(const Tensor &a, const Tensor &b);

    // Returns `a * b` element-wise as a new tensor, broadcasting fully.
    static Tensor mul(const Tensor &a, const Tensor &b);

    // Returns the matrix product `a * b`. Rank 2 only.
    static Tensor matmul(const Tensor &a, const Tensor &b);

    // Unary negation; returns a new tensor.
    Tensor operator-() const;

    // Adds scalar `x`; returns a new tensor.
    Tensor operator+(const dtype x) const;

    // Adds scalar `x`; returns a new tensor.
    friend Tensor operator+(const dtype x, const Tensor &a);

    // Adds scalar `x` in-place.
    Tensor &operator+=(const dtype x);

    // Subtracts scalar `x`; returns a new tensor.
    Tensor operator-(const dtype x) const;

    // Subtracts the tensor from scalar `x`; returns a new tensor.
    friend Tensor operator-(const dtype x, const Tensor &a);

    // Subtracts scalar `x` in-place.
    Tensor &operator-=(const dtype x);

    // Multiplies by scalar `x`; returns a new tensor.
    Tensor operator*(const dtype x) const;

    // Multiplies scalar `x` by the tensor; returns a new tensor.
    friend Tensor operator*(const dtype x, const Tensor &a);

    // Multiplies by scalar `x` in-place.
    Tensor &operator*=(const dtype x);

    // Divides every element by scalar `x`; returns a new tensor.
    //
    // Throws `std::domain_error` if `x == 0`.
    Tensor operator/(const dtype x) const;

    // Divides every element by scalar `x` in-place.
    //
    // Throws `std::domain_error` if `x == 0`.
    Tensor &operator/=(const dtype x);

    // Element-wise add with full broadcasting; returns a new tensor.
    Tensor operator+(const Tensor &b) const;

    // In-place element-wise add; `b` must broadcast into `*this`.
    Tensor &operator+=(const Tensor &b);

    // Element-wise subtract with full broadcasting; returns a new tensor.
    Tensor operator-(const Tensor &b) const;

    // In-place element-wise subtract; `b` must broadcast into `*this`.
    Tensor &operator-=(const Tensor &b);

    // Element-wise multiply with full broadcasting.
    Tensor operator*(const Tensor &b) const;

    // In-place element-wise multiply; `b` must broadcast into `*this`.
    Tensor &operator*=(const Tensor &b);

  private:
    // Sets `shape` from `dims` and recomputes strides and `data.size()`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    void assign_shape(R &&dims, const bool clear) {
      const size_t n = checked_num_el(dims);
      shape.assign_range(dims);
      finalize_shape(n, clear);
    }

    // Returns the product of the dimensions in `s`.
    //
    // Throws `std::invalid_argument` if `s` is empty or contains a 0, and
    // `std::overflow_error` if the product overflows `size_t`.
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

    // In-place reduction of axis `dim` using `op`, starting from `init`.
    // Removes the axis and rebuilds `*this`.
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
          const size_t c = tmp / ret.strides[j];
          tmp %= ret.strides[j];
          offset += c * strides[i];
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

    // Applies `op(data[i], b[offset])` in-place, with `b` broadcast into `*this`.
    // Requires `broadcast_shape(shape, b.shape) == shape`.
    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    Tensor &binary_inplace(const Tensor &b, Op op) {
      const auto bs = aligned_strides(b, shape);

      for (size_t i = 0; i < num_el; i++) {
        size_t off = 0, tmp = i;
        for (size_t k = 0; k < shape.size(); k++) {
          const size_t c = tmp / strides[k];
          tmp %= strides[k];
          off += c * bs[k];
        }

        data[i] = op(data[i], b.data[off]);
      }
      return *this;
    }

    // Returns a new tensor of `op(a, b)` with full broadcasting.
    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    static Tensor binary_broadcast(const Tensor &a, const Tensor &b, Op op) {
      auto out = broadcast_shape(a.shape, b.shape);
      const auto out_sz = out.size();
      const auto as = aligned_strides(a, out);
      const auto bs = aligned_strides(b, out);

      Tensor ret(std::move(out));
      for (size_t i = 0; i < ret.num_el; i++) {
        size_t a_off = 0, b_off = 0, tmp = i;
        for (size_t k = 0; k < out_sz; k++) {
          const size_t c = tmp / ret.strides[k];
          tmp %= ret.strides[k];
          a_off += c * as[k];
          b_off += c * bs[k];
        }

        ret.data[i] = op(a.data[a_off], b.data[b_off]);
      }
      return ret;
    }

    // Throws unless both tensors have valid strides and consistent storage,
    // and neither is empty. Does not compare shapes.
    void valid_pair(const Tensor &b) const;

    // True if `strides` are the contiguous row-major strides of `shape`.
    bool valid_strides() const;

    // Row-major strides for `shape`.
    static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);

    // Sets `shape` from `dims` and recomputes strides and `data.size()`.
    void assign_shape(std::vector<size_t> &&dims, const bool clear);

    // Sets `num_el`, `strides`, and `data.size()` from `shape`.
    void finalize_shape(size_t n, bool clear);

    // Right-aligned broadcast of two shapes.
    //
    // Throws `std::invalid_argument` if the shapes are incompatible.
    template <typename R1, typename R2>
      requires ConvIntervalOf<R1, size_t> and ConvIntervalOf<R2, size_t> and std::integral<std::ranges::range_value_t<R1>> and std::integral<std::ranges::range_value_t<R2>> and
               std::ranges::sized_range<R1> and std::ranges::sized_range<R2>
    static std::vector<size_t> broadcast_shape(R1 &&a, R2 &&b) {
      const auto asz = std::ranges::size(a);
      const auto bsz = std::ranges::size(b);
      const size_t n = std::max(asz, bsz);
      const size_t ao = n - asz, bo = n - bsz;

      std::vector<size_t> out(n, 1);
      auto ita = std::ranges::begin(a);
      auto itb = std::ranges::begin(b);
      for (size_t i = 0; i < n; i++) {
        const size_t x = i < ao ? 1 : ita[i - ao];
        const size_t y = i < bo ? 1 : itb[i - bo];
        if (x != y and x != 1 and y != 1)
          throw std::invalid_argument("Shapes not broadcastable.");
        out[i] = std::max(x, y);
      }
      return out;
    }

    // `t`'s strides right-aligned to `out`; 0 for broadcast axes.
    static std::vector<size_t> aligned_strides(const Tensor &t, const std::vector<size_t> &out);
  };
} // namespace NN
