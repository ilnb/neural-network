#include "data/tensor_dataset.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <ranges>

namespace NN::Data {
  std::pair<Tensor, Tensor> TensorDataset::operator[](size_t i) const {
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
} // namespace NN::Data
