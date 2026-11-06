#pragma once

#include "data/dataset.hpp"
#include "tensor/tensor.hpp"
#include <span>
#include <stdexcept>

namespace NN::Data {
  // A dataset backed by two tensors that share a leading sample axis.
  //
  // `data` has shape `{n, ...}` and `target` has shape `{n, ...}`, so `size()`
  // is `n`. The tensors are shared, not copied, so changes to them are visible
  // to the dataset.
  struct TensorDataset : Dataset {
    Tensor data, target;

    // Throws `std::invalid_argument` if either tensor is empty, has rank 0, or
    // their leading dimensions differ.
    TensorDataset(const Tensor &d, const Tensor &t) : data(d), target(t) {
      if (d.empty() || t.empty())
        throw std::invalid_argument("Empty dataset tensor.");
      if (d.p->shape.empty() || t.p->shape.empty())
        throw std::invalid_argument("Dataset tensors need a sample axis (rank >= 1).");
      if (d.p->shape[0] != t.p->shape[0])
        throw std::invalid_argument("data and target must have the same first dimension.");
    }

    size_t size() const override { return data.p->shape[0]; }

    // Returns {sample, target} at index `i`
    //
    // Throws `std::out_of_range` if `i >= size()`.
    std::pair<Tensor, Tensor> operator[](size_t i) const override;

    // Returns the samples at `idx` as one batch.
    //
    // Every index must be less than `size()`; indices are not bounds checked.
    //
    // Throws `std::invalid_argument` if `idx` is empty.
    std::pair<Tensor, Tensor> batch(std::span<const size_t> idx) const override;
  };
} // namespace NN::Data
