#pragma once

#include "module/base.hpp"

namespace NN {
  struct MaxPool2d : Module {
    struct Config {
      // 0 means use kernel size
      size_t stride_h = 0;
      size_t stride_w = 0;
    };

    size_t kernel_h, kernel_w;
    size_t stride_h, stride_w;

    MaxPool2d(size_t kernel_h, size_t kernel_w, const Config &c)
        : kernel_h(kernel_h), kernel_w(kernel_w), stride_h(c.stride_h ? c.stride_h : kernel_h), stride_w(c.stride_w ? c.stride_w : kernel_w) {
      if (!kernel_h or !kernel_w)
        throw std::invalid_argument("MaxPool2d: kernel size should be > 0.");
    }

    MaxPool2d(size_t kernel_h, size_t kernel_w) : MaxPool2d(kernel_h, kernel_w, {}) {}

    explicit MaxPool2d(size_t kernel) : MaxPool2d(kernel, kernel) {}

    std::vector<Tensor> params() const override { return {}; }

    Tensor forward(const Tensor &x) override;
  };
} // namespace NN
