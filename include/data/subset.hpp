#pragma once

#include "data/dataset.hpp"
#include "data/tensor_dataset.hpp"
#include "tensor/tensor.hpp"
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace NN::Data {
  // A view over a parent dataset that exposes only the samples in `idx`.
  //
  // Indices refer to `ds` and may repeat. The parent is shared, so it stays
  // alive as long as the subset does. `batch` maps its indices into the
  // parent's space and forwards, so a parent's contiguous fast path is kept.
  //
  // Throws `std::invalid_argument` if `ds` is empty, and `std::out_of_range`
  // if an index in `idx` is `>= ds->size()`.
  struct Subset : Dataset {
    // The parent dataset, shared with every subset derived from it.
    std::shared_ptr<const Dataset> ds;
    // Sample indices into `ds`, in iteration order.
    std::vector<size_t> idx;

    Subset(std::shared_ptr<const Dataset> d, std::vector<size_t> v) : ds(std::move(d)), idx(std::move(v)) {
      if (!ds)
        throw std::invalid_argument("Subset needs a dataset.");
      for (const size_t i : idx)
        if (i >= ds->size())
          throw std::out_of_range("Subset index out of range.");
    }

    // Copies `d` into a shared `TensorDataset`, then takes `v`.
    Subset(const TensorDataset &d, std::vector<size_t> v) : Subset(std::make_shared<const TensorDataset>(d), std::move(v)) {}

    size_t size() const override { return idx.size(); }

    // Returns the parent's sample at `idx[i]`.
    //
    // Throws `std::out_of_range` if `i >= size()`.
    std::pair<Tensor, Tensor> operator[](size_t i) const override {
      if (i >= idx.size())
        throw std::out_of_range("Subset index out of range.");
      return (*ds)[idx[i]];
    }

    // Returns the parent's samples at the mapped indices.
    //
    // Throws `std::out_of_range` if an entry of `i` is not a valid subset
    // position, and `std::invalid_argument` if `i` is empty.
    std::pair<Tensor, Tensor> batch(std::span<const size_t> i) const override {
      std::vector<size_t> mapped;
      mapped.reserve(i.size());
      for (const size_t k : i) {
        if (k >= idx.size())
          throw std::out_of_range("Subset batch index out of range.");
        mapped.push_back(idx[k]);
      }
      return ds->batch(mapped);
    }
  };
} // namespace NN::Data
