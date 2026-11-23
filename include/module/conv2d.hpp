#pragma once

#include "module/base.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace NN {
  struct Conv2d : ParamOnly {
    struct Config {
      size_t stride = 1;
      size_t padding = 0;
      size_t dilation = 1;
      bool bias = true;
      uint32_t seed = std::random_device{}();
    };

    Tensor w, b;
    size_t in_channels, out_channels;
    size_t kernel_h, kernel_w;
    Config cfg;
    std::mt19937 rng;

    // Recycled im2col buffers. Forward takes from here and backward puts
    // it back. Shared so a backward that outlives the module cant dangle.
    std::shared_ptr<std::vector<std::shared_ptr<Tensor>>> col_pool = std::make_shared<std::vector<std::shared_ptr<Tensor>>>();

    // N x C_out x H x W initialization
    Conv2d(size_t in, size_t out, size_t kernel_h, size_t kernel_w, const Config &c)
        : in_channels(in), out_channels(out), kernel_h(kernel_h), kernel_w(kernel_w), cfg(c), rng(cfg.seed) {
      w.set_dims(out, in, kernel_h, kernel_w);
      if (cfg.bias)
        b.set_dims(out);

      std::normal_distribution dist(dtype(0), static_cast<dtype>(2) / static_cast<dtype>(in * kernel_h * kernel_w));
      std::ranges::generate(w.p->data, [&]() { return dist(rng); });

      w.set_requires_grad(true);
      if (cfg.bias)
        b.set_requires_grad(true);
    }

    Conv2d(size_t in, size_t out, size_t kernel, const Config &c) : Conv2d(in, out, kernel, kernel, c) {}

    Conv2d(size_t in, size_t out, size_t kernel_h, size_t kernel_w) : Conv2d(in, out, kernel_h, kernel_w, {}) {}

    Conv2d(size_t in, size_t out, size_t kernel) : Conv2d(in, out, kernel, kernel) {}

    [[nodiscard]] Tensor forward(const Tensor &x) override;

    std::vector<NamedTensor> named_params() const override;
  };
} // namespace NN
