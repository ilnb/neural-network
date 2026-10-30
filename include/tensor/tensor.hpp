#pragma once

#include "tensor/shape.hpp"
#include <array>
#include <concepts>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace NN {
  // Raw, non-differentiable tensor arithmetic.
  //
  // These never attach a `GradFn` and never set `requires_grad`, so they are
  // safe to use inside backward closures and in code that must not build a
  // graph. Use the member ops / operators when gradients are wanted.
  namespace Raw {
    // Returns `a + b` with full broadcasting.
    Tensor add(const Tensor &a, const Tensor &b);

    // Returns `a - b` with full broadcasting.
    Tensor sub(const Tensor &a, const Tensor &b);

    // Returns `a * b` element-wise with full broadcasting.
    Tensor mul(const Tensor &a, const Tensor &b);

    // Returns the matrix product `a * b`.
    Tensor matmul(const Tensor &a, const Tensor &b);
  } // namespace Raw

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
  // **Storage**
  //
  // `p` is the shared handle to the backing `Impl`, and its fields are public:
  // `p->data` (flat row-major values), `p->shape`, `p->strides` and
  // `p->num_el` (`== p->data.size()`). A default-constructed or moved-from
  // tensor has no `Impl` at all; `empty()` is true for it. Shape and value
  // constructors always produce a shaped tensor with at least one element.
  //
  // **Autograd**
  //
  // A tensor takes part in autograd once it is marked with
  // `set_requires_grad(true)`. Every differentiable op applied to such a
  // tensor attaches a `GradFn` to its result and sets `requires_grad` on the
  // result, so the graph grows as an expression is built. Gradients accumulate
  // on the leaves that require grad; a leaf has no `grad_fn`.
  //
  // `backward()` starts at a scalar root, seeds its gradient with 1, walks the
  // graph in reverse topological order and adds each adjoint into `grad`.
  // Intermediate gradients are released as they are consumed, and every node's
  // `grad_fn` is dropped once the walk finishes, so a graph supports a single
  // backward pass. Leaf gradients stay behind, but a second call on the same
  // root only re-seeds the root and propagates nothing; to combine several
  // losses, add them into one expression before calling `backward()`.
  //
  // ```
  // Tensor x({2});
  // x.set_data(1, 2);
  // x.set_requires_grad(true);
  //
  // auto y = (x * 3 + 1).sum();   // y.p->grad_fn is set
  // y.backward();                 // x.p->grad == {3, 3}
  // ```
  //
  // `NN::Raw` and `clone()` never attach a node: their results are leaves.
  // In-place mutators (`set_data`, `set_dims`, `add_`, `+=`, ...) throw
  // `std::logic_error` on a tensor that participates in autograd.
  //
  // `backward()` itself throws `std::logic_error` if the root does not require
  // grad, and `std::invalid_argument` if the root does not hold exactly one
  // element.
  struct Tensor {
    std::shared_ptr<Impl> p;

    Tensor() = default;

    Tensor(const Tensor &b) = default;

    Tensor(Tensor &&b) noexcept = default;

    Tensor &operator=(const Tensor &b) = default;

    Tensor &operator=(Tensor &&b) noexcept = default;

    // Wraps an existing `Impl`, taking ownership of the handle.
    //
    // No validation is performed. `r` should be null (an empty tensor) or a
    // well-formed `Impl`.
    explicit Tensor(std::shared_ptr<Impl> r) noexcept : p(std::move(r)) {}

    // Constructs a scalar holding `x`.
    explicit Tensor(dtype x) : p(std::make_shared<Impl>()) {
      p->num_el = 1;
      p->data = { x };
    }

    // Constructs a tensor from a shape range. Data is zero-initialized.
    //
    // Throws `std::invalid_argument` if the range is empty or contains a 0,
    // and `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
    explicit Tensor(R &&r) : p(std::make_shared<Impl>()) {
      p->shape.assign_range(std::forward<R>(r));
      const size_t n = Detail::checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    explicit Tensor(std::vector<size_t> &&v) : p(std::make_shared<Impl>()) {
      p->shape = std::move(v);
      const size_t n = Detail::checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    template <typename T>
      requires std::integral<T>
    explicit Tensor(std::initializer_list<T> l) : p(std::make_shared<Impl>()) {
      p->shape.assign(l.begin(), l.end());
      const size_t n = Detail::checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    // Convenience constructor, directly takes values.
    template <typename... Args>
      requires(sizeof...(Args) > 1) and (std::integral<Args> and ...)
    explicit Tensor(Args... dims) : p(std::make_shared<Impl>()) {
      p->shape = std::vector{ static_cast<size_t>(dims)... };
      const size_t n = Detail::checked_num_el(p->shape);
      finalize_shape(n, false);
    }

    // Copies `r` into `data`. `r` must hold exactly `num_el` elements.
    // The source range must not alias the tensor's internal storage.
    //
    // Throws `std::invalid_argument` on a size mismatch or for an empty
    // tensor, and `std::logic_error` if the tensor participates in autograd.
    template <typename R>
      requires ConvIntervalOf<R, dtype> and std::ranges::sized_range<R>
    Tensor &set_data(R &&r) {
      check_mutable();
      if (std::ranges::size(r) != Detail::checked_num_el(p->shape))
        throw std::invalid_argument("Data ranges are of different size.");
      p->data.assign_range(r);
      p->num_el = p->data.size();
      return *this;
    }

    // Convenience form of `set_data`.
    template <typename... Args>
      requires(std::convertible_to<Args, dtype> and ...)
    Tensor &set_data(Args... vals) {
      return set_data(std::array<dtype, sizeof...(Args)>{ static_cast<dtype>(vals)... });
    }

    // Takes data as an initializer_list
    template <typename T>
      requires std::integral<T>
    Tensor &set_data(std::initializer_list<T> l) {
      return set_data(std::span<const T>(l.begin(), l.end()));
    }

    // Sets the shape from `r` and recomputes strides.
    //
    // When `clear` is true every element is set to 0. When it is false the
    // storage is resized (keeping existing values) and any new elements are
    // zero-filled. The source range must not alias the tensor's internal
    // storage.
    //
    // Throws `std::invalid_argument` if the range is empty or contains a 0,
    // `std::overflow_error` if the element count overflows `size_t`, and
    // `std::logic_error` if the tensor participates in autograd.
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
      return set_dims(std::vector<size_t>{ static_cast<size_t>(vals)... });
    }

    // Same as the range overload, taking a shape vector.
    Tensor &set_dims(std::vector<size_t> &&v, const bool clear = true);

    // Marks the tensor as requiring grad (or not). Only leaves may be
    // changed: a tensor produced by an op inherits its state from the graph.
    //
    // Throws `std::invalid_argument` if the tensor is empty, and
    // `std::logic_error` if it already has a `grad_fn`.
    void set_requires_grad(const bool x);

    // Element access with bounds checking. Accepts a sized random-access range of indices.
    //
    // Throws `std::invalid_argument` if the number of indices differs from the rank,
    // and `std::out_of_range` if an index is out of bounds.
    template <typename Self, typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    decltype(auto) at(this Self &&self, R &&r) {
      if (self.empty())
        throw std::invalid_argument("Empty tensor.");
      if (std::ranges::size(r) != self.p->shape.size())
        throw std::invalid_argument("Number of indices doesnt match tensor rank.");
      auto it = std::ranges::begin(r); // supports more types
      size_t offset = 0;
      for (size_t i = 0; i < self.p->shape.size(); i++) {
        if (it[i] >= self.p->shape[i])
          throw std::out_of_range("Tensor index out of bounds.");
        offset += it[i] * self.p->strides[i];
      }
      if constexpr (std::is_const_v<std::remove_reference_t<Self>>)
        return std::as_const(self.p->data).at(offset);
      else
        return self.p->data.at(offset);
    }

    // Element access with bounds checking.
    template <typename Self, typename T>
      requires std::integral<T>
    decltype(auto) at(this Self &&self, std::initializer_list<T> l) {
      return self.at(std::span<const T>(l.begin(), l.size()));
    }

    // Element access with bounds checking.
    template <typename Self, typename... Args>
      requires(std::integral<Args> and ...)
    decltype(auto) at(this Self &&self, Args... idx) {
      return self.at(std::array<size_t, sizeof...(Args)>{ static_cast<size_t>(idx)... });
    }

    // Element access without bounds checking.
    //
    // Throws `std::invalid_argument` if the number of indices
    // differs from the rank. Indices must be in range.
    template <typename Self, typename... Args>
      requires(std::integral<Args> and ...)
    decltype(auto) operator[](this Self &&self, Args... idx) {
      if (self.empty())
        throw std::invalid_argument("Empty tensor.");
      if (sizeof...(idx) != self.p->shape.size())
        throw std::invalid_argument("Number of indices doesnt match tensor rank.");
      std::array<size_t, sizeof...(Args)> tmp{ static_cast<size_t>(idx)... };
      size_t offset = 0;
      for (size_t i = 0; i < self.p->shape.size(); i++)
        offset += tmp[i] * self.p->strides[i];
      if constexpr (std::is_const_v<std::remove_reference_t<Self>>)
        return std::as_const(self.p->data)[offset];
      else
        return self.p->data[offset];
    }

    // Returns a new tensor with same data, with the provided shape.
    // The total element count must be preserved.
    //
    // Throws `std::invalid_argument` if the tensor is empty, if the new shape
    // is invalid or its element count differs from `num_el`, and
    // `std::overflow_error` if the element count overflows `size_t`.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    Tensor reshape(R &&r) const {
      return reshape(std::vector<size_t>(std::ranges::begin(r), std::ranges::end(r)));
    }

    // Takes the new shape as separate arguments.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor reshape(Args... dims) const {
      return reshape(std::vector{ static_cast<size_t>(dims)... });
    }

    Tensor reshape(std::vector<size_t> &&v) const;

    // Swaps the last two axes. Ranks above 2 are supported; only the trailing
    // axes are exchanged and leading axes are left in place.
    //
    // Throws `std::invalid_argument` if the tensor is empty, and
    // `std::domain_error` if its rank is below 2.
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
    // Throws `std::invalid_argument` if the tensor is empty, if `perm` is not
    // a permutation, or if its size doesn't match the rank.
    template <typename R>
      requires ConvIntervalOf<R, size_t> and std::ranges::sized_range<R> and std::integral<std::ranges::range_value_t<R>>
    Tensor permute(R &&r) const {
      return permute(std::vector<size_t>(std::ranges::begin(r), std::ranges::end(r)));
    }

    // Convenience form of `permute`.
    template <typename... Args>
      requires(std::integral<Args> and ...)
    Tensor permute(Args... args) const {
      return permute(std::vector{ static_cast<size_t>(args)... });
    }

    Tensor permute(std::vector<size_t> &&v) const;

    // True if the tensor has no backing storage, i.e. it was default
    // constructed or moved from.
    //
    // A shaped tensor is never empty: dimensions must be positive, so it
    // always holds at least one element.
    bool empty() const;

    // Checks if the tensor can be modified.
    //
    // Throws `std::invalid_argument` if the tensor is empty, and
    // `std::logic_error` if it participates in autograd.
    void check_mutable() const;

    // Returns a deep copy of the data, shape and strides in a fresh Impl.
    // The result is a leaf: requires_grad = false.
    // Safe to mutate without affecting the source.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor clone() const;

    // Inserts a dimension of `size` at `dim`, expanding the tensor along that
    // dimension by repeating its data. The result has rank + 1; `dim` may
    // equal the rank, which appends the new axis at the end.
    //
    // Throws `std::domain_error` for an empty tensor, `std::out_of_range` if
    // `dim > rank`, and `std::invalid_argument` if `size == 0`.
    Tensor expand(size_t dim, size_t size) const;

    // Runs backpropagation from this tensor.
    //
    // The tensor must require grad and hold exactly one element; it is seeded
    // with a gradient of 1. Every node's `grad_fn` is released by the pass, so
    // a graph is walked once.
    //
    // Throws `std::logic_error` if the tensor does not require grad or a node
    // on the path has no backward rule, and `std::invalid_argument` if the
    // tensor holds more than one element.
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

    // Return a copy of the tensor with all elememnts sqrt-ed.
    //
    // Throws `std::domain_error` for an empty tensor.
    Tensor sqrt() const;

    // Sum of all elements.
    //
    // Throws `std::domain_error` for an empty tensor.
    Tensor sum() const;

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

    // Reduces axis `dim` by summing. The axis is removed, so the rank
    // decreases by one. Returns a new tensor.
    //
    // Throws `std::out_of_range` if `dim >= rank`,
    // and `std::domain_error` for an empty tensor.
    Tensor sum(size_t dim) const;

    // Reduces axis `dim` by the mean. Returns a new tensor.
    Tensor mean(size_t dim) const;

    // Reduces axis `dim` by the maximum. Returns a new tensor.
    Tensor max(size_t dim) const;

    // Reduces axis `dim` by the minimum. Returns a new tensor.
    Tensor min(size_t dim) const;

    // Adds scalar `x` to every element; returns a new tensor.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor add(const dtype x) const;

    // Subtracts scalar `x` from every element; returns a new tensor.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor sub(const dtype x) const;

    // Multiplies every element by scalar `x`; returns a new tensor.
    //
    // Throws `std::invalid_argument` if the tensor is empty.
    Tensor mul(const dtype x) const;

    // Element-wise add with full broadcasting; returns a new tensor.
    //
    // Throws `std::invalid_argument` if either tensor is empty, malformed, or
    // the shapes cannot broadcast.
    Tensor add(const Tensor &b) const;

    // Element-wise subtract with full broadcasting; returns a new tensor.
    Tensor sub(const Tensor &b) const;

    // Element-wise multiply with full broadcasting; returns a new tensor.
    Tensor mul(const Tensor &b) const;

    // Matrix product; returns a new tensor.
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
      const size_t n = Detail::checked_num_el(dims);
      p->shape.assign_range(dims);
      finalize_shape(n, clear);
    }

    // Sets `shape` from `dims` and recomputes strides and `data.size()`.
    void assign_shape(std::vector<size_t> &&dims, const bool clear);

    // Sets `num_el`, `strides`, and `data.size()` from `shape`.
    void finalize_shape(size_t n, bool clear);
  };
} // namespace NN
