#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace NN {
  // Random-access range whose values convert to `T`.
  template <typename R, typename T>
  concept ConvIntervalOf = std::ranges::random_access_range<R> and std::convertible_to<std::ranges::range_value_t<R>, T>;

  // Element type for raw values.
  using dtype = float;

  // forward decl
  namespace Autograd {
    struct GradFn;
  }
  struct Tensor;

  // Data members of Tensor
  struct Impl {
    // Flat row-major storage. `data.size() == num_el`.
    std::vector<dtype> data;
    // Dimensions. `shape.size()` is the rank.
    std::vector<size_t> shape;
    // Row-major strides of `shape`.
    std::vector<size_t> strides;
    // Number of stored elements; 0 for an empty tensor.
    size_t num_el = 0;
    // True if this tensor takes part in autograd. Always set when `grad_fn`
    // is non-null; settable on a leaf to make it a differentiable input.
    bool requires_grad = false;
    // Accumulated gradient, or null before the first backward pass. When set,
    // `grad->shape == shape`, and later passes add into it.
    std::shared_ptr<Tensor> grad;
    // Node that produced this tensor, or null for a leaf.
    std::shared_ptr<Autograd::GradFn> grad_fn;
  };

  // An N-dimensional, contiguous, row-major tensor.
  //
  // Arithmetic operators are element-wise and broadcast; `matmul()` is the
  // matrix product.
  //
  // **Copy semantics and in-place ops**
  //
  // `Tensor` is a handle: copying it (`Tensor b = a;`, `b = a;`) is O(1) and
  // shares the underlying storage, so writes through one handle are visible
  // through every other handle to the same data. Use `clone()` for an
  // independent deep copy. A default-constructed or moved-from tensor is
  // empty (`empty()` is true).
  //
  // `add`/`sub`/`mul`, `+`/`-`/`*`, `transpose`, `cat`, `reshape`, `sum`, ...
  // return a fresh, fully broadcast tensor and never alias their inputs.
  // `add_`/`sub_`/`mul_`, `+=`/`-=`/`*=` and `set_data` write through the
  // shared storage, so every alias observes the change. They require the
  // result shape to equal the receiver's shape.
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
    std::shared_ptr<Impl> p;

    Tensor() = default;

    Tensor(const Tensor &b) = default;

    Tensor(Tensor &&b) noexcept = default;

    Tensor &operator=(const Tensor &b) = default;

    Tensor &operator=(Tensor &&b) noexcept = default;

    // Constructs tensor from `std::shared_ptr<Impl>`.
    explicit Tensor(std::shared_ptr<Impl> r) noexcept : p(std::move(r)) {}

    // Constructs a scalar
    explicit Tensor(dtype x) : p(std::make_shared<Impl>()) {
      p->num_el = 1;
      p->data = {x};
    }

    // Constructs a tensor from a shape range. Data is zero-initialized.
    //
    // Throws `std::invalid_argument` if the range is empty or contains a 0,
    // and `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    explicit Tensor(R &&r) : p(std::make_shared<Impl>()) {
      p->shape.assign_range(std::forward<R>(r));
      const size_t n = checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    explicit Tensor(std::vector<size_t> &&v) : p(std::make_shared<Impl>()) {
      p->shape = std::move(v);
      const size_t n = checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    template <typename T>
      requires std::integral<T>
    explicit Tensor(std::initializer_list<T> l) : p(std::make_shared<Impl>()) {
      p->shape.assign_range(l.begin(), l.end());
      const size_t n = checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    // Convenience constructor, directly takes values.
    template <typename... Args>
      requires(sizeof...(Args) > 1) and (std::integral<Args> and ...)
    explicit Tensor(Args... dims) : p(std::make_shared<Impl>()) {
      p->shape = std::vector{static_cast<size_t>(dims)...};
      const size_t n = checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    // Copies `r` into `data`. `r` must have exactly `num_el` elements.
    // The source range must not alias the tensor's internal storage.
    //
    // Throws `std::invalid_argument` on a size mismatch or if the new shape is invalid,
    // and `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, dtype> and std::ranges::sized_range<R>
    Tensor &set_data(R &&r) {
      if (not empty())
        check_mutable();
      if (std::ranges::size(r) != checked_num_el(p->shape))
        throw std::invalid_argument("Data ranges are of different size.");
      p->data.assign_range(r);
      p->num_el = p->data.size();
      return *this;
    }

    // Convenience form of `set_data`.
    template <typename... Args>
      requires(std::convertible_to<Args, dtype> and ...)
    Tensor &set_data(Args... vals) {
      return set_data(std::vector<dtype>{static_cast<dtype>(vals)...});
    }

    // Moves `v` into `data`. `v` must have exactly `num_el` elements.
    // The source range must not alias the tensor's internal storage.
    //
    // Throws `std::invalid_argument` on a size mismatch.
    Tensor &set_data(std::vector<dtype> &&v);

    // Sets the shape from `r` and recomputes strides.
    // `data` is emptied if `clear` is set, resized otherwise.
    // The source range must not alias the tensor's internal storage.
    //
    // Throws `std::invalid_argument` if the range is empty or contains a 0,
    // and `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    Tensor &set_dims(R &&r, const bool clear = true) {
      if (not empty())
        check_mutable();
      assign_shape(std::forward<R>(r), clear);
      return *this;
    }

    // Convenience form of `set_dims`.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor &set_dims(Args... vals) {
      return set_dims(std::vector<size_t>{static_cast<size_t>(vals)...});
    }

    // Same as the range overload, taking a shape vector.
    Tensor &set_dims(std::vector<size_t> &&v, const bool clear = true);

    // Sets if the tensor needs gradient or not
    void set_requires_grad(const bool x);

    // Element access with bounds checking. Accepts a sized random-access range of indices.
    //
    // Throws `std::invalid_argument` if the number of indices differs from the rank,
    // and `std::out_of_range` if an index is out of bounds.
    template <typename Self, typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    decltype(auto) at(this Self &&self, R &&r) {
      if (std::ranges::size(r) != self.p->shape.size())
        throw std::invalid_argument("Number of indices doesnt match tensor rank.");
      auto it = std::ranges::begin(r); // supports more types
      size_t offset = 0;
      for (size_t i = 0; i < self.p->shape.size(); i++) {
        if (it[i] >= self.p->shape[i])
          throw std::out_of_range("Tensor index out of bounds.");
        offset += it[i] * self.p->strides[i];
      }
      return self.p->data.at(offset);
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
      if (sizeof...(idx) != self.p->shape.size())
        throw std::invalid_argument("Number of indices doesnt match tensor rank.");
      std::array<size_t, sizeof...(Args)> tmp{static_cast<size_t>(idx)...};
      size_t offset = 0;
      for (size_t i = 0; i < self.p->shape.size(); i++)
        offset += tmp[i] * self.p->strides[i];
      return self.p->data[offset];
    }

    // Returns a new tensor with same data, with the provided shape.
    // The total element count must be preserved.
    //
    // Throws `std::invalid_argument` if the new shape is invalid or
    // its element count differs from `num_el`, and `std::overflow_error`
    // if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    Tensor reshape(R &&r) const {
      return h_reshape(std::vector<size_t>(std::ranges::begin(r), std::ranges::end(r)));
    }

    // Takes the new shape as separate arguments.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor reshape(Args... dims) const {
      return h_reshape(std::vector{static_cast<size_t>(dims)...});
    }

    Tensor reshape(std::vector<size_t> &&v) const;

    // Transposes a rank-2 tensor.
    //
    // Throws `std::invalid_argument` if the tensor is not rank 2 or is empty.
    Tensor transpose() const;

    // Concatenates `b` with the tensor along the existing dimension `dim`.
    // Every dimension is kept, and `dim` holds `shape[dim] + b.shape[dim]`.
    //
    // The tensors must have equal rank and match on every dimension but `dim`.
    //
    // Throws `std::out_of_range` if `dim >= rank`, and `std::invalid_argument`
    // if the ranks differ, any dimension other than `dim` differs,
    // or either tensor is empty.
    Tensor cat(const Tensor &b, size_t dim) const;

    // Stacks `b` onto this tensor along a new dimension `dim`. The result has
    // rank + 1, with a size-2 axis inserted at `dim`; this tensor and `b` are
    // its two slices. The two tensors must have identical shapes.
    //
    // `dim` may equal the rank, which appends the new axis at the end.
    //
    // Throws `std::out_of_range` if `dim > rank`, and `std::invalid_argument`
    // if the shapes differ or either tensor is empty.
    Tensor stack(const Tensor &b, size_t dim) const;

    // Permutes tensor dimensions according to provided permutation.
    //
    // Throws `std::invalid_argument` if `perm` is not a permutation, or if
    // the size doesn't match.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    Tensor permute(R &&r) const {
      return h_permute(std::vector<size_t>(std::ranges::begin(r), std::ranges::end(r)));
    }

    // Convenience form of `permute`.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor permute(Args... args) const {
      return h_permute(std::vector{static_cast<size_t>(args)...});
    }

    Tensor permute(std::vector<size_t> &&v) const;

    // True if the tensor holds no elements.
    bool empty() const;

    // Checks if the tensor can be modified.
    void check_mutable() const;

    // Returns a deep copy of the data, shape and strides in a fresh Impl.
    // The result is a leaf: requires_grad = false.
    // Safe to mutate without affecting the source.
    Tensor clone() const;

    // Inserts a dimension of `size` at `dim`, expanding the tensor along that
    // dimension by repeating its data.
    //
    // Throws `std::invalid_argument` on empty tensor,
    // or `std::out_of_range` on out of bounds dim.
    Tensor expand(size_t dim, size_t size) const;

    // Wrapper around Autograd::backward
    void backward() const;

    // Validates the tensor's internal state.
    //
    // Throws `std::invalid_argument` if the tensor is empty, has invalid
    // strides, or its storage is inconsistent with its element count.
    void validate() const;

    // Validates multiple tensors.
    template <typename... Args>
    static void validate(const Args &...args) {
      (args.validate(), ...);
    }

    // Returns the sole element of the tensor.
    //
    // Throws `std::domain_error` if the tensor is empty or does not
    // contain exactly one element.
    dtype item() const;

    // Sum of all elements.
    //
    // Throws `std::domain_error` for an empty tensor.
    Tensor sum() const;

    Tensor h_sum() const;

    // Mean of all elements.
    //
    // Throws `std::domain_error` for an empty tensor.
    Tensor mean() const;

    // Largest element.
    //
    // Throws `std::domain_error` for an empty tensor.
    Tensor max() const;

    // Smallest element.
    //
    // Throws `std::domain_error` for an empty tensor.
    Tensor min() const;

    // Reduces axis `dim` in-place. The axis is removed,
    // so the rank decreases by one.
    //
    // Throws `std::out_of_range` if `dim >= rank`,
    // and `std::domain_error` for empty tensor.
    Tensor sum(size_t dim) const;

    Tensor h_sum(size_t dim) const;

    // Reduces axis `dim` in-place by the mean.
    Tensor mean(size_t dim) const;

    // Reduces axis `dim` in-place by the maximum.
    Tensor max(size_t dim) const;

    // Reduces axis `dim` in-place by the minimum.
    Tensor min(size_t dim) const;

    // Adds scalar `x` to every element in-place.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor add(const dtype x) const;

    // Subtracts scalar `x` from every element in-place.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor sub(const dtype x) const;

    // Multiplies every element by scalar `x` in-place.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor mul(const dtype x) const;

    // In-place element-wise add with broadcasting. `b` must broadcast into
    // `*this` without changing its shape.
    //
    // Throws `std::invalid_argument` if `b` does not broadcast into `shape`,
    // or if either tensor is empty or malformed.
    Tensor add(const Tensor &b) const;

    // In-place element-wise subtract with broadcasting.
    Tensor sub(const Tensor &b) const;

    // In-place element-wise multiply with broadcasting.
    Tensor mul(const Tensor &b) const;

    // In-place matrix product.
    //
    // `*this` and `b` must both be at least rank 2, with shapes `[..., m, k]`
    // and `[..., k, n]`. Leading batch dimensions broadcast right-aligned and
    // the result has shape `[...broadcast, m, n]`.
    //
    // Throws `std::domain_error` if either tensor is below rank 2, and
    // `std::invalid_argument` if the inner dimensions differ or the batch
    // dimensions cannot broadcast or if either tensor is empty or malformed.
    Tensor matmul(const Tensor &b) const;

    // In-place adds scalar `x` to the tensor.
    Tensor &add_(const dtype x);

    // In-place subtracts scalar `x` to the tensor.
    Tensor &sub_(const dtype x);

    // In-place multiplies scalar `x` to the tensor.
    Tensor &mul_(const dtype x);

    // In-place element-wise add with broadcasting.
    // `b` must broadcast into `this` without changing its shape.
    //
    // Throws `std::logic_error` if either tensor participates in autograd.
    Tensor &add_(const Tensor &b);

    // In-place element-wise sub with broadcasting.
    // `b` must broadcast into `this` without changing its shape.
    //
    // Throws `std::logic_error` if either tensor participates in autograd.
    Tensor &sub_(const Tensor &b);

    // In-place element-wise multiply with broadcasting.
    // `b` must broadcast into `this` without changing its shape.
    //
    // Throws `std::logic_error` if either tensor participates in autograd.
    Tensor &mul_(const Tensor &b);

    // Returns `a + b` as a new tensor with full broadcasting.
    static Tensor add(const Tensor &a, const Tensor &b);

    // Returns `a - b` as a new tensor with full broadcasting.
    static Tensor sub(const Tensor &a, const Tensor &b);

    // Returns `a * b` element-wise as a new tensor with full broadcasting.
    static Tensor mul(const Tensor &a, const Tensor &b);

    // Returns the matrix product `a * b`. See `matmul(const Tensor &)` for the
    // shape rules.
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
      if (!p)
        p = std::make_shared<Impl>();
      const size_t n = checked_num_el(dims);
      p->shape.assign_range(dims);
      finalize_shape(n, clear);
    }

    // Returns the product of the dimensions in `s`.
    //
    // Throws `std::invalid_argument` if `s` is empty or contains a 0, and
    // `std::overflow_error` if the product overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    static size_t checked_num_el(R &&s) {
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

    // HELPERS

    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    Tensor h_reduce(size_t dim, dtype init, Op op) const {
      if (empty())
        throw std::domain_error("Empty tensor.");
      const size_t n = p->shape.size();
      if (dim >= n)
        throw std::out_of_range("Out of bounds axis.");

      std::vector<size_t> new_size;
      new_size.reserve(n - 1);
      for (size_t i = 0; i < n; i++)
        if (i != dim)
          new_size.push_back(p->shape[i]);

      Tensor ret;
      if (new_size.empty())
        ret = Tensor(init);
      else
        ret.set_dims(std::move(new_size), false);
      auto &data = p->data, &rdata = ret.p->data;
      auto &strides = p->strides, &rstrides = ret.p->strides;
      auto &shape = p->shape, &rshape = ret.p->shape;

      for (size_t out = 0, itr = ret.p->num_el; out < itr; out++) {
        size_t offset = 0, tmp = out;

        for (size_t i = n, j = rshape.size(); i-- > 0;) {
          if (i == dim)
            continue;
          j--;
          const size_t c = tmp % rshape[j];
          tmp /= rshape[j];
          offset += c * strides[i];
        }

        auto val = init;
        for (size_t k = 0; k < shape[dim]; k++)
          val = op(val, data[offset + k * strides[dim]]);
        rdata[out] = val;
      }

      return ret;
    }

    template <typename Op>
      requires std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, bool> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    std::pair<Tensor, std::vector<size_t>> h_cmp(size_t dim, Op op) const {
      if (empty())
        throw std::domain_error("Empty tensor.");
      const size_t n = p->shape.size();
      if (dim >= n)
        throw std::out_of_range("Out of bounds axis.");

      std::vector<size_t> new_size;
      new_size.reserve(n - 1);
      for (size_t i = 0; i < n; i++)
        if (i != dim)
          new_size.push_back(p->shape[i]);

      Tensor ret;
      if (new_size.empty())
        ret = Tensor(p->data[0]);
      else
        ret.set_dims(std::move(new_size), false);
      std::vector<size_t> indices(ret.p->num_el);

      auto &data = p->data, &rdata = ret.p->data;
      auto &strides = p->strides, &rstrides = ret.p->strides;
      auto &shape = p->shape, &rshape = ret.p->shape;

      for (size_t out = 0, itr = ret.p->num_el; out < itr; out++) {
        size_t offset = 0, tmp = out;

        for (size_t i = n, j = rshape.size(); i-- > 0;) {
          if (i == dim)
            continue;
          j--;
          const size_t c = tmp % rshape[j];
          tmp /= rshape[j];
          offset += c * strides[i];
        }

        auto val = data[offset];
        size_t idx = 0;
        for (size_t k = 1; k < shape[dim]; k++) {
          if (auto nval = data[offset + k * strides[dim]]; op(nval, val)) {
            val = nval;
            idx = k;
          }
        }
        rdata[out] = val;
        indices[out] = idx;
      }

      return {std::move(ret), std::move(indices)};
    }

    Tensor h_reshape(std::vector<size_t> &&v) const;

    Tensor h_permute(std::vector<size_t> &&v) const;

    Tensor h_transpose() const;

    Tensor h_cat(const Tensor &b, size_t dim) const;

    Tensor h_stack(const Tensor &b, size_t dim) const;

    Tensor h_add(const dtype x) const;

    Tensor h_matmul(const Tensor &b) const;

    // Applies `op(data[i], b[offset])` in-place, with `b` broadcast into `*this`.
    // Requires `broadcasts_into(shape, b.shape)`.
    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    Tensor &binary_inplace(const Tensor &b, Op op) {
      std::vector<size_t> scratch;
      std::span<const size_t> bs = b.p->strides;
      if (b.p->shape != p->shape) {
        scratch = aligned_strides(b, p->shape);
        bs = scratch;
      }

      for (size_t i = 0; i < p->num_el; i++) {
        size_t off = 0, tmp = i;
        for (size_t k = p->shape.size(); k-- > 0;) {
          const size_t c = tmp % p->shape[k];
          tmp /= p->shape[k];
          off += c * bs[k];
        }

        p->data[i] = op(p->data[i], b.p->data[off]);
      }
      return *this;
    }

    // Returns a new tensor of `op(a, b)` with full broadcasting.
    template <typename Op>
      requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
    static Tensor binary_broadcast(const Tensor &a, const Tensor &b, Op op) {
      auto out = broadcast_shape(a.p->shape, b.p->shape);
      const auto out_sz = out.size();

      std::vector<size_t> a_scratch, b_scratch;
      std::span<const size_t> as = a.p->strides;
      std::span<const size_t> bs = b.p->strides;
      if (a.p->shape != out) {
        a_scratch = aligned_strides(a, out);
        as = a_scratch;
      }
      if (b.p->shape != out) {
        b_scratch = aligned_strides(b, out);
        bs = b_scratch;
      }

      Tensor ret(std::move(out));
      for (size_t i = 0; i < ret.p->num_el; i++) {
        size_t a_off = 0, b_off = 0, tmp = i;
        for (size_t k = out_sz; k-- > 0;) {
          const size_t c = tmp % ret.p->shape[k];
          tmp /= ret.p->shape[k];
          a_off += c * as[k];
          b_off += c * bs[k];
        }

        ret.p->data[i] = op(a.p->data[a_off], b.p->data[b_off]);
      }
      return ret;
    }

    // True if `b` broadcasts into `s` without changing its shape.
    static bool broadcasts_into(const std::vector<size_t> &s, const std::vector<size_t> &b);

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
