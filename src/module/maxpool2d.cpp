#include "module/maxpool2d.hpp"
#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "tensor/tensor.hpp"

using NN::Autograd::GradFn, NN::Autograd::accumulate;

namespace NN {
  Tensor MaxPool2d::forward(const Tensor &x) {
    if (x.empty())
      throw std::invalid_argument("MaxPool2d::forward: empty tensor.");

    const auto &shape = x.p->shape;
    if (shape.size() != 4)
      throw std::invalid_argument("MaxPool2d::forward: 4D NCHW tensor expected.");

    const size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    if (H < kernel_h or W < kernel_w)
      throw std::invalid_argument(std::format("MaxPool2d::forward: input {}x{} is smaller than kernel {}x{}.", H, W, kernel_h, kernel_w));

    const size_t H_out = (H - kernel_h) / stride_h + 1;
    const size_t W_out = (W - kernel_w) / stride_w + 1;

    Tensor r(N, C, H_out, W_out);
    const auto &input = x.p->data;
    auto &output = r.p->data;

    const bool requires_grad = Autograd::grad_enabled and x.p->requires_grad;
    std::vector<size_t> argmax;
    if (requires_grad)
      argmax.resize(r.p->num_el);

    for (size_t n = 0; n < N; n++) {
      for (size_t c = 0; c < C; c++) {
        for (size_t oh = 0; oh < H_out; oh++) {
          for (size_t ow = 0; ow < W_out; ow++) {
            const size_t ih0 = oh * stride_h;
            const size_t iw0 = ow * stride_w;

            size_t best_idx = ((n * C + c) * H + ih0) * W + iw0;
            dtype best = input[best_idx];

            for (size_t kh = 0; kh < kernel_h; kh++) {
              for (size_t kw = 0; kw < kernel_w; kw++) {
                const size_t xi = ((n * C + c) * H + ih0 + kh) * W + iw0 + kw;
                if (input[xi] > best) {
                  best_idx = xi;
                  best = input[best_idx];
                }
              }
            }

            const size_t oi = ((n * C + c) * H_out + oh) * W_out + ow;
            output[oi] = best;
            if (requires_grad)
              argmax[oi] = best_idx;
          }
        }
      }
    }

    if (requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { x.p };
      fn->backward = [ap = x.p, argmax = std::move(argmax)](const Tensor &dz) {
        if (!ap->requires_grad)
          return;

        Tensor dx(ap->shape);
        auto &xd = dx.p->data, &zd = dz.p->data;
        for (size_t i = 0, n = dz.p->num_el; i < n; i++)
          xd[argmax[i]] += zd[i];
        accumulate(*ap, dx);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }
} // namespace NN
