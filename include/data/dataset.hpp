#pragma once

#include "tensor/tensor.hpp"
#include <utility>

namespace NN::Data {
  struct Dataset {
    virtual size_t size() const = 0;
    virtual std::pair<Tensor, Tensor> operator[](size_t i) const = 0;
    virtual ~Dataset() = default;
  };
} // namespace NN::Data
