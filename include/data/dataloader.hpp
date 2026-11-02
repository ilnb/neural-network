#pragma once

#include "data/dataset.hpp"
#include "data/tensor_dataset.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <ranges>
#include <span>
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
    std::shared_ptr<const Dataset> ds;
    Config cfg;
    std::mt19937 m;

    DataLoader(const TensorDataset &ds) : DataLoader(ds, Config{ .batch_size = ds.size() }) {}

    DataLoader(TensorDataset &&) = delete;

    DataLoader(TensorDataset &&, const Config &) = delete;
    DataLoader(std::shared_ptr<const Dataset> d, const Config &c) : ds(std::move(d)), cfg(c), m(c.seed) {
      if (cfg.batch_size == 0)
        throw std::invalid_argument("Batch size should be positive.");
      if (cfg.drop_last && cfg.batch_size > ds->size())
        throw std::invalid_argument("drop_last with batch_size > dataset size yields no batches.");
    }

    DataLoader(const TensorDataset &d, const Config &c) : DataLoader(std::make_shared<const TensorDataset>(d), c) {}

    DataLoader(const TensorDataset &d) : DataLoader(d, Config{ .batch_size = d.size() }) {}

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
          const size_t n = std::min(e->dl->cfg.batch_size, e->limit - i);
          return e->dl->ds->batch(std::span(e->order).subspan(i, n));
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
