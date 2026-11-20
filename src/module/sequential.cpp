#include "module/sequential.hpp"

#include <ranges>
#include <string>

namespace NN {
  Tensor Sequential::forward(const Tensor &x) {
    Tensor r = x;
    for (const auto &m : modules)
      r = m->forward(r);
    return r;
  }

  std::vector<NamedTensor> Sequential::named_params() const {
    std::vector<NamedTensor> out;
    for (auto [i, m] : modules | std::views::enumerate) {
      for (const auto &[name, t] : m->named_params())
        out.emplace_back(std::to_string(i) + "." + name, t);
    }
    return out;
  }

  std::vector<NamedTensor> Sequential::named_buffers() const {
    std::vector<NamedTensor> out;
    for (auto [i, m] : modules | std::views::enumerate) {
      for (const auto &[name, t] : m->named_buffers())
        out.emplace_back(std::to_string(i) + "." + name, t);
    }
    return out;
  }

  void Sequential::train(bool mode) {
    Module::train(mode);
    for (auto &m : modules)
      m->train(mode);
  }
} // namespace NN
