#include "data/split.hpp"
#include "data/dataset.hpp"
#include "data/subset.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <ranges>

namespace NN::Data {
  std::pair<std::shared_ptr<Subset>, std::shared_ptr<Subset>> train_test_split(std::shared_ptr<const Dataset> ds, const SplitConfig &cfg) {
    if (!ds)
      throw std::invalid_argument("train_test_split needs a dataset.");

    const size_t N = ds->size();
    if (N < 2)
      throw std::invalid_argument("train_test_split needs at least 2 samples.");
    if (!(cfg.test_size > 0 and cfg.test_size < 1))
      throw std::invalid_argument("test_size should be in (0, 1).");

    const size_t n_test = std::clamp<size_t>(static_cast<size_t>(std::llround(N * cfg.test_size)), 1, N - 1);

    auto idx = std::views::iota(0zu, N) | std::ranges::to<std::vector>();
    if (cfg.shuf) {
      std::mt19937 m(cfg.seed);
      std::ranges::shuffle(idx, m);
    }

    auto train_idx = idx | std::views::drop(n_test) | std::ranges::to<std::vector>();
    auto test_idx = idx | std::views::take(n_test) | std::ranges::to<std::vector>();

    auto train = std::make_shared<Subset>(ds, std::move(train_idx));
    auto test = std::make_shared<Subset>(std::move(ds), std::move(test_idx));
    return { std::move(train), std::move(test) };
  }

} // namespace NN::Data
