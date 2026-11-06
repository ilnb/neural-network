#pragma once

#include "tensor/tensor.hpp"
#include <span>
#include <utility>

namespace NN::Data {
  // A random-access collection of `{sample, target}` pairs.
  //
  // Samples are returned by value; implementations decide how a sample is
  // materialized. Batches are built from `batch`, whose default requires every
  // sample to share a shape.
  struct Dataset {
    // Number of samples.
    virtual size_t size() const = 0;

    // Returns {sample, target} at index `i`.
    //
    // Implementations should throw `std::out_of_range` if `i >= size()`.
    virtual std::pair<Tensor, Tensor> operator[](size_t i) const = 0;

    virtual ~Dataset() = default;

    // Materializes the samples at `idx` as one batch along a new leading axis.
    //
    // The default implementation stacks `operator[]` and requires every sample
    // to share a shape. Datasets that can copy contiguously should override it.
    //
    // Throws `std::invalid_argument` for an empty `idx`, or for samples whose
    // shapes differ.
    virtual std::pair<Tensor, Tensor> batch(std::span<const size_t> idx) const;

    // Random-access iterator over individual samples.
    //
    // Holds the dataset by pointer, so the dataset must outlive the iterator.
    struct Iterator {
      const Dataset *ds;
      size_t i;

      Iterator(const Dataset *d, size_t idx) : ds(d), i(idx) {}

      std::pair<Tensor, Tensor> operator*() const { return (*ds)[i]; }

      Iterator &operator++() {
        ++i;
        return *this;
      }

      bool operator!=(const Iterator &o) const { return i != o.i; }
    };

    // Iterates over `[0, size())`.
    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, size()); }
  };
} // namespace NN::Data
