#pragma once

#include "data/subset.hpp"

#include <memory>
#include <random>

namespace NN::Data {
  // Options for `train_test_split`.
  struct SplitConfig {
    // Fraction of the dataset assigned to the test subset, in `(0, 1)`.
    double test_size = 0.2;
    // Seed of the shuffling RNG. Ignored when `shuf` is false.
    uint32_t seed = std::random_device{}();
    // Shuffle the indices before splitting. When false, the first
    // `test_size * size()` indices go to the test subset.
    bool shuf = true;
  };

  // Splits `ds` into disjoint train and test subsets.
  //
  // The result is `{train, test}`. Each subset holds a reference to `ds`, so
  // the parent outlives both. The test subset has `test_size * size()` samples
  // rounded to nearest and clamped to `[1, size() - 1]`, so neither side is
  // empty.
  //
  // Throws `std::invalid_argument` if `ds` is null, has fewer than 2 samples,
  // or `cfg.test_size` is not in `(0, 1)`.
  std::pair<std::shared_ptr<Subset>, std::shared_ptr<Subset>> train_test_split(std::shared_ptr<const Dataset> ds, const SplitConfig &cfg = {});

  inline auto train_test_split(const TensorDataset &ds, const SplitConfig &cfg = {}) { return train_test_split(std::make_shared<const TensorDataset>(ds), cfg); }
} // namespace NN::Data
