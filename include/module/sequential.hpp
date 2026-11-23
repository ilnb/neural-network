#pragma once

#include "module/base.hpp"

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace NN {
  struct Sequential : Module {
    std::vector<std::unique_ptr<Module>> modules;

    Sequential() = default;

    template <typename... M>
    Sequential(M &&...args) {
      (modules.push_back(std::make_unique<std::remove_cvref_t<M>>(std::forward<M>(args))), ...);
    }

    template <typename M>
    Sequential(std::initializer_list<M> l) {
      for (auto &&m : l)
        modules.push_back(std::make_unique<std::remove_cvref_t<M>>(m));
    }

    size_t size() const { return modules.size(); }

    Module &at(size_t i) { return *modules.at(i); }

    const Module &at(size_t i) const { return *modules.at(i); }

    Module &operator[](size_t i) { return *modules[i]; }

    const Module &operator[](size_t i) const { return *modules[i]; }

    Tensor operator()(const Tensor &x) { return forward(x); }

    [[nodiscard]] Tensor forward(const Tensor &x) override;

    std::vector<NamedTensor> named_params() const override;

    std::vector<NamedTensor> named_buffers() const override;

    void train(bool mode = true) override;
  };
} // namespace NN
