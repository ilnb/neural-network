#include "data/dataloader.hpp"
#include "data/dataset.hpp"
#include "data/detail.hpp"
#include "data/tensor_dataset.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace NN::Data {
  // Kept out of line: inlining the virtual `size()` call into the DataLoader
  // constructors makes GCC 16 emit a false `-Warray-bounds`.
  void DataLoader::check() const {
    if (cfg.batch_size == 0)
      throw std::invalid_argument("Batch size should be positive.");
    if (cfg.drop_last && cfg.batch_size > ds->size())
      throw std::invalid_argument("drop_last with batch_size > dataset size yields no batches.");
  }

  namespace Detail {
    Tensor stack(const std::vector<Tensor> &ts) {
      if (ts.empty())
        throw std::invalid_argument("Cannot stack zero tensors.");

      const auto &shape0 = ts.front().p->shape;
      const size_t len = ts.front().p->num_el;
      for (const auto &t : ts)
        if (t.p->shape != shape0)
          throw std::invalid_argument("Stacked tensors must share a shape.");

      std::vector<size_t> shape(shape0);
      shape.insert(shape.begin(), ts.size());
      Tensor r(std::move(shape));
      for (size_t j = 0; j < ts.size(); j++)
        std::copy_n(ts[j].p->data.begin(), len, r.p->data.begin() + j * len);
      return r;
    }
  } // namespace Detail

  std::pair<Tensor, Tensor> Dataset::batch(std::span<const size_t> idx) const {
    if (idx.empty())
      throw std::invalid_argument("Cannot batch zero samples.");

    std::vector<Tensor> d, t;
    d.reserve(idx.size());
    t.reserve(idx.size());
    for (size_t i : idx) {
      auto [a, b] = (*this)[i];
      d.push_back(std::move(a));
      t.push_back(std::move(b));
    }
    return { Detail::stack(d), Detail::stack(t) };
  }

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

  std::pair<Tensor, Tensor> TensorDataset::batch(std::span<const size_t> idx) const {
    if (idx.empty())
      throw std::invalid_argument("Cannot batch zero samples.");

    auto dshape = data.p->shape;
    dshape[0] = idx.size();
    auto tshape = target.p->shape;
    tshape[0] = idx.size();
    Tensor rdata(std::move(dshape)), rtarget(std::move(tshape));

    const size_t rd = data.p->num_el / data.p->shape[0];
    const size_t rt = target.p->num_el / target.p->shape[0];
    for (size_t j = 0; j < idx.size(); j++) {
      const size_t src = idx[j];
      std::copy_n(data.p->data.begin() + src * rd, rd, rdata.p->data.begin() + j * rd);
      std::copy_n(target.p->data.begin() + src * rt, rt, rtarget.p->data.begin() + j * rt);
    }

    return { std::move(rdata), std::move(rtarget) };
  }
} // namespace NN::Data
