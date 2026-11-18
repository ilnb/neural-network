#pragma once

#include "data/dataset.hpp"
#include "data/subset.hpp"
#include "data/tensor_dataset.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace NN::Data {
  // Iterates a `Dataset` in mini-batches.
  //
  // Batches come from `Dataset::batch`, so any dataset works; `TensorDataset`
  // has a contiguous fast path. Every batch has a new leading axis of size
  // `batch_size`, except the final one when `drop_last` is false.
  struct DataLoader {
    // Batch iteration options.
    struct Config {
      // Samples per batch. Must be positive.
      size_t batch_size = 0;
      // Shuffle the sample order in every epoch.
      bool shuf = false;
      // Discard a final partial batch.
      bool drop_last = false;
      // Seed of the shuffling RNG.
      uint32_t seed = std::random_device{}();
    };

    // The dataset, owned through the base so any `Dataset` can be passed.
    std::shared_ptr<const Dataset> ds;
    Config cfg;
    // RNG used by `epoch()`.
    std::mt19937 m;

    // Takes ownership of `d`.
    //
    // Throws `std::invalid_argument` if `cfg.batch_size` is 0, or if
    // `cfg.drop_last` is set and `cfg.batch_size > d->size()`,
    // which would yield no batches.
    DataLoader(std::shared_ptr<const Dataset> d, const Config &c) : ds(std::move(d)), cfg(c), m(c.seed) { check(); }

    // Copies `d` into the loader.
    DataLoader(const TensorDataset &d, const Config &c) : DataLoader(std::make_shared<const TensorDataset>(d), c) {}

    // One batch containing the whole dataset, in order.
    DataLoader(const TensorDataset &d) : DataLoader(d, Config{ .batch_size = d.size() }) {}

    // Batches over a subset, taking ownership of a copy of `s`.
    DataLoader(Subset s, const Config &c) : DataLoader(std::make_shared<const Subset>(std::move(s)), c) {}

    // One pass over the dataset, yielded as batches.
    //
    // Holds the loader by pointer, so the loader must outlive the epoch.
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

        bool operator!=(const Iterator &o) const { return e != o.e or i != o.i; }

        bool operator==(const Iterator &o) const { return e == o.e and i == o.i; }

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

    // Starts a new epoch over `[0, size())`.
    //
    // The order is shuffled when `cfg.shuf`, and each call advances the RNG, so
    // consecutive shuffled epochs differ. With `seed` fixed, a fresh loader
    // reproduces the same sequence of orders.
    Epoch epoch() {
      auto order = std::views::iota(0zu, ds->size()) | std::ranges::to<std::vector>();
      if (cfg.shuf)
        std::ranges::shuffle(order, m);
      return Epoch(this, std::move(order));
    }

  private:
    // Throws unless the dataset can be split into batches of `cfg.batch_size`.
    void check() const;
  };
} // namespace NN::Data
