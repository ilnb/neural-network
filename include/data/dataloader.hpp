#pragma once

#include "data/tensor_dataset.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace NN::Data {
  struct DataLoader {
    struct Config {
      size_t batch_size = 0;
      bool shuf = false;
      bool drop_last = false;
      uint32_t seed = 0;
    };

    std::shared_ptr<const TensorDataset> ds;
    Config cfg;
    std::mt19937 m;

    DataLoader(const TensorDataset &ds) : DataLoader(ds, Config{ .batch_size = ds.size() }) {}

    DataLoader(TensorDataset &&) = delete;

    DataLoader(TensorDataset &&, const Config &) = delete;

    DataLoader(const TensorDataset &ds, const Config &cfg) : ds(std::make_shared<const TensorDataset>(ds)), cfg(cfg), m(cfg.seed) {
      if (cfg.batch_size == 0)
        throw std::invalid_argument("Batch size should be positive.");
      if (cfg.drop_last && cfg.batch_size > ds.size())
        throw std::invalid_argument("drop_last with batch_size > dataset size yields no batches.");
    }

    struct Epoch {
      const DataLoader *dl;
      std::vector<size_t> order;
      size_t limit;

      Epoch(const DataLoader *dl, std::vector<size_t> &&o) : dl(dl), order(std::move(o)) {
        const size_t N = order.size(), B = dl->cfg.batch_size;
        limit = dl->cfg.drop_last ? N - N % B : N;
      }

      struct Iterator {
        const Epoch *e;
        size_t i;

        Iterator(const Epoch *e, size_t i) : e(e), i(i) {}

        bool operator!=(const Iterator &o) const { return i != o.i; }

        bool operator==(const Iterator &o) const { return i == o.i; }

        std::pair<Tensor, Tensor> operator*() const {
          auto &ds = e->dl->ds;
          const size_t n = std::min(e->dl->cfg.batch_size, e->limit - i);

          auto dshape = ds->data.p->shape;
          dshape[0] = n;
          auto tshape = ds->target.p->shape;
          tshape[0] = n;
          Tensor rdata(std::move(dshape)), rtarget(std::move(tshape));

          const auto &data = ds->data, &target = ds->target;
          const size_t rd = data.p->num_el / data.p->shape[0];
          const size_t rt = target.p->num_el / target.p->shape[0];
          for (size_t j = 0; j < n; j++) {
            const size_t src = e->order[i + j];
            std::copy_n(data.p->data.begin() + src * rd, rd, rdata.p->data.begin() + j * rd);
            std::copy_n(target.p->data.begin() + src * rt, rt, rtarget.p->data.begin() + j * rt);
          }

          return { std::move(rdata), std::move(rtarget) };
        }

        Iterator &operator++() {
          i = std::min(i + e->dl->cfg.batch_size, e->limit);
          return *this;
        }

        Iterator operator++(int) {
          auto r = *this;
          ++*this;
          return r;
        }
      };

      Iterator begin() const { return Iterator(this, 0); }
      Iterator end() const { return Iterator(this, limit); }
    };

    Epoch epoch() {
      auto order = std::views::iota(0zu, ds->size()) | std::ranges::to<std::vector>();
      if (cfg.shuf)
        std::ranges::shuffle(order, m);
      return Epoch(this, std::move(order));
    }
  };
} // namespace NN::Data
