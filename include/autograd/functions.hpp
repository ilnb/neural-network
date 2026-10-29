#pragma once

#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Autograd {
  // Undoes broadcasting: sums `g` over every axis where `shape` is 1 or
  // missing, returning a tensor of `shape`. Used by binary-op backward rules
  // to turn a result-shaped adjoint into an input-shaped one.
  // Returns a copy of `g` when the shapes already match.
  //
  // Throws `std::invalid_argument` if `shape` has higher rank than `g` or if
  // the two are not broadcast-compatible.
  Tensor sum_to_shape(const Tensor &g, const std::vector<size_t> &shape);

  // Adds `g` into `p.grad`. `g` may carry a graph node; it is read as raw data,
  // so backward rules can pass composite results without building new nodes.
  //
  // Throws `std::invalid_argument` if `g` is empty or the shapes differ.
  void accumulate(Impl &p, const Tensor &g);

  // Backpropagates from `root`.
  //
  // `root` must require grad and hold exactly one element; it is seeded with a
  // gradient of 1. The graph is walked in reverse topological order, each
  // node's `backward` is called once with its output gradient, and the adjoint
  // is accumulated into the leaves. Intermediate gradients are dropped as they
  // are consumed, and each node's `grad_fn` is released when the walk finishes,
  // so a graph is good for a single pass. Leaf gradients persist, but a second
  // call re-seeds the root without reaching them; combine multiple losses into
  // one root first.
  //
  // Throws `std::logic_error` if `root` does not require grad or a node has no
  // backward rule, and `std::invalid_argument` if `root` is not a scalar.
  void backward(const Tensor &root);
} // namespace NN::Autograd
