#pragma once

#include <cstdint>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

namespace NN {
  // Random-access range whose values convert to `T`.
  template <typename R, typename T>
  concept ConvIntervalOf = std::ranges::random_access_range<R> and std::convertible_to<std::ranges::range_value_t<R>, T>;

  // Element type for raw values.
#ifdef NN_DTYPE
  using dtype = NN_DTYPE;
#else
  using dtype = float;
#endif

  // sized types
  using i32 = std::int32_t;
  using u32 = std::uint32_t;
  using i64 = std::int64_t;
  using u64 = std::uint64_t;

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
    // Number of elements.
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

  // Casting helper
  template <typename To, typename From>
    requires requires(From &&f) { static_cast<To>(f); }
  constexpr To SC(From &&x) {
    return static_cast<To>(std::forward<From>(x));
  }
} // namespace NN
