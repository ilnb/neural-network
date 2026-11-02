#pragma once

#include "tensor/tensor.hpp"
#include <span>
#include <utility>

namespace NN::Data {
  struct Dataset {
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> operator[](size_t i) const = 0;
    virtual ~Dataset() = default;

    virtual std::pair<Tensor, Tensor> batch(std::span<const size_t> idx) const;

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

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, size()); }
  };
} // namespace NN::Data
