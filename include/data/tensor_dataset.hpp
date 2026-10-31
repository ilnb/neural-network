#pragma once

#include "data/dataset.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace NN::Data {
  struct TensorDataset : Dataset {
    Tensor data, target;

    TensorDataset(const Tensor &d, const Tensor &t) : data(d), target(t) {
      if (d.empty() || t.empty())
        throw std::invalid_argument("Empty dataset tensor.");
      if (d.p->shape.empty() || t.p->shape.empty())
        throw std::invalid_argument("Dataset tensors need a sample axis (rank >= 1).");
      if (d.p->shape[0] != t.p->shape[0])
        throw std::invalid_argument("data and target must have the same first dimension.");
    }

    size_t size() const override { return data.p->shape[0]; }

    std::pair<Tensor, Tensor> operator[](size_t i) const override {
      if (i >= size())
        throw std::out_of_range("Index out of range.");

      auto shape_d = data.p->shape | std::views::drop(1);
      auto shape_t = target.p->shape | std::views::drop(1);
      Tensor rd(shape_d), rt(shape_t);

      size_t len_d = data.p->num_el / data.p->shape[0];
      size_t len_t = target.p->num_el / target.p->shape[0];

      std::copy_n(data.p->data.begin() + i * len_d, len_d, rd.p->data.begin());
      std::copy_n(target.p->data.begin() + i * len_t, len_t, rt.p->data.begin());

      return { rd, rt };
    }
  };
} // namespace NN::Data
