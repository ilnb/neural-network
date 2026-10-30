#pragma once

#include "tensor/tensor.hpp"
#include <vector>

namespace NN::Optim {
  struct SGD {
    struct HParams {
      dtype lr = 0.01, beta = 0.9;
    };

    std::vector<Tensor> params, vel;
    HParams hp;

    explicit SGD(const std::vector<Tensor> &v) : SGD(v, HParams{}) {}

    SGD(const std::vector<Tensor> &v, const HParams &c) : params(v), hp(c) {
      vel.reserve(v.size());
      for (const auto &p : params)
        vel.emplace_back(Tensor(p.p->shape));
    }

    void zero_grad();

    void step();
  };
} // namespace NN::Optim
