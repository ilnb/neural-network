#pragma once

#include "tensor/impl.hpp"
#include <memory>
#include <ranges>
#include <vector>

// Shape helpers shared by the tensor kernels. None of these touch autograd.
namespace NN::Detail {
  // Returns the product of the dimensions in `s`.
  //
  // An empty range yields 1, which is how a rank-0 tensor is described.
  // Throws `std::invalid_argument` if any dimension is 0, and
  // `std::overflow_error` if the product overflows `size_t`.
  template <typename R>
    requires ConvIntervalOf<R, size_t> and std::integral<std::ranges::range_value_t<R>>
  inline size_t checked_num_el(R &&s) {
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

  // True if `strides` are the contiguous row-major strides of `shape`.
  inline bool valid_strides(const std::shared_ptr<Impl> &p) {
    const auto &strides = p->strides;
    const auto &shape = p->shape;
    if (strides.size() != shape.size())
      return false;
    for (size_t i = shape.size(), acc = 1; i-- > 0;) {
      if (strides[i] != acc)
        return false;
      acc *= shape[i];
    }
    return true;
  }

  // Row-major strides for `shape`.
  inline std::vector<size_t> compute_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides(shape.size());
    size_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      strides[i] = acc;
      acc *= shape[i];
    }
    return strides;
  }

  // `p`'s strides right-aligned to `out`; broadcast axes (size 1 in `p`) get
  // stride 0 so repeated reads land on the same element.
  //
  // Throws `std::logic_error` if `out` has lower rank than `p->shape`.
  inline std::vector<size_t> aligned_strides(const std::shared_ptr<Impl> &p, const std::vector<size_t> &out) {
    const size_t n = p->shape.size();
    if (out.size() < n)
      throw std::logic_error("Cant align strides to a lower rank shape.");
    std::vector<size_t> s(out.size());
    const size_t off = out.size() - n;
    for (size_t i = 0; i < n; i++)
      if (p->shape[i] != 1)
        s[off + i] = p->strides[i];
    return s;
  }

  // True if `b` broadcasts into `s` without changing its shape.
  inline bool broadcasts_into(const std::vector<size_t> &s, const std::vector<size_t> &b) {
    if (b.size() > s.size())
      return false;
    const size_t off = s.size() - b.size();
    for (size_t i = 0; i < b.size(); i++)
      if (b[i] != 1 and b[i] != s[off + i])
        return false;
    return true;
  }

  // Right-aligned broadcast of two shapes.
  //
  // Throws `std::invalid_argument` if the shapes are incompatible.
  template <typename R1, typename R2>
    requires ConvIntervalOf<R1, size_t> and ConvIntervalOf<R2, size_t> and std::integral<std::ranges::range_value_t<R1>> and std::integral<std::ranges::range_value_t<R2>> and
             std::ranges::sized_range<R1> and std::ranges::sized_range<R2>
  inline std::vector<size_t> broadcast_shape(R1 &&a, R2 &&b) {
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
} // namespace NN::Detail
