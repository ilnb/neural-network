#pragma once

#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Optim {
  struct Adam {
    struct HParams {
      dtype lr = 0.01, eps = 1e-8;
      dtype beta1 = 0.9, beta2 = 0.999;
    };

    std::vector<Tensor> params, mom, vel;
    HParams hp;
    size_t t = 0;

    explicit Adam(const std::vector<Tensor> &v) : Adam(v, HParams{}) {}

    Adam(const std::vector<Tensor> &v, const HParams &c) : params(v), hp(c) {
      mom.reserve(params.size());
      vel.reserve(params.size());
      for (const auto &p : params) {
        mom.emplace_back(Tensor(p.p->shape));
        vel.emplace_back(Tensor(p.p->shape));
      }
    }

    void zero_grad();

    void step();
  };
} // namespace NN::Optim
