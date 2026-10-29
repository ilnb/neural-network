#pragma once

#include "nn/module/base.hpp"
#include "tensor/tensor.hpp"
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

    Tensor forward(const Tensor &x) const override {
      Tensor r = x;
      for (const auto &m : modules)
        r = m->forward(r);
      return r;
    }

    std::vector<Tensor> params() const override {
      std::vector<Tensor> ret;
      for (const auto &m : modules) {
        auto p = m->params();
        ret.append_range(p);
      }
      return ret;
    }
  };
} // namespace NN
