#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "module/conv2d.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

using NN::Autograd::GradFn, NN::Autograd::accumulate;

namespace NN {
  Tensor Conv2d::forward(const Tensor &x) {
    if (x.empty())
      throw std::invalid_argument("Empty tensor.");
    const auto &shape = x.p->shape;
    if (shape.size() != 4)
      throw std::invalid_argument("Conv2d expects a 4D NCHW tensor.");
    if (shape[1] != in_channels)
      throw std::invalid_argument("Conv2d input channel mismatch.");

    size_t N = shape[0], H = shape[2], W = shape[3];
    size_t S = cfg.stride, P = cfg.padding, D = cfg.dilation;

    size_t H_out = (H + 2 * P - D * (kernel_h - 1) - 1) / S + 1;
    size_t W_out = (W + 2 * P - D * (kernel_w - 1) - 1) / S + 1;

    Tensor r(N, out_channels, H_out, W_out);
    const auto &input = x.p->data;
    const auto &weight = w.p->data;
    const dtype *bias = cfg.bias ? b.p->data.data() : nullptr;
    auto &output = r.p->data;

    // Y[n,o,y,x] = b[o] +
    //   \sum_{c=0}^{C_{in}-1} \sum_{i=0}^{K_h-1}
    //   \sum_{j=0}^{K_W-1} X[n, c, y * S_h + i * D_h - P_h, x * S_w + j * D_w - P_w]
    //
    // The innermost loop walks one contiguous output row. For each kernel tap
    // the valid `ow` range is clipped once, so no iteration tests padding.
    std::vector<ptrdiff_t> lo(kernel_w), hi(kernel_w);
    for (size_t kw = 0; kw < kernel_w; kw++) {
      const auto off = static_cast<ptrdiff_t>(kw * D) - static_cast<ptrdiff_t>(P);
      lo[kw] = off >= 0 ? ptrdiff_t(0) : (-off + static_cast<ptrdiff_t>(S) - 1) / static_cast<ptrdiff_t>(S);
      hi[kw] = (static_cast<ptrdiff_t>(W) - 1 - off) / static_cast<ptrdiff_t>(S);
      if (hi[kw] > static_cast<ptrdiff_t>(W_out) - 1)
        hi[kw] = static_cast<ptrdiff_t>(W_out) - 1;
    }

    for (size_t n = 0; n < N; n++) {
      for (size_t oc = 0; oc < out_channels; oc++) {
        const dtype bv = bias ? bias[oc] : dtype(0);
        for (size_t oh = 0; oh < H_out; oh++) {
          dtype *out_row = &output[((n * out_channels + oc) * H_out + oh) * W_out];
          std::fill_n(out_row, W_out, bv);

          for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t kh = 0; kh < kernel_h; kh++) {
              const auto ih = static_cast<ptrdiff_t>(oh * S + kh * D) - static_cast<ptrdiff_t>(P);
              if (ih < 0 or ih >= static_cast<ptrdiff_t>(H))
                continue;

              const dtype *in_row = &input[((n * in_channels + ic) * H + static_cast<size_t>(ih)) * W];
              const dtype *w_row = &weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w];

              for (size_t kw = 0; kw < kernel_w; kw++) {
                const dtype wv = w_row[kw];
                const auto off = static_cast<ptrdiff_t>(kw * D) - static_cast<ptrdiff_t>(P);
                for (auto ow = lo[kw]; ow <= hi[kw]; ow++)
                  out_row[ow] += in_row[ow * static_cast<ptrdiff_t>(S) + off] * wv;
              }
            }
          }
        }
      }
    }

    bool requires_grad = x.p->requires_grad or w.p->requires_grad or (cfg.bias and b.p->requires_grad);
    if (Autograd::grad_enabled and requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { x.p, w.p };
      if (cfg.bias)
        fn->inputs.emplace_back(b.p);

      fn->backward = [xp = x.p, wp = w.p, bp = cfg.bias ? b.p : nullptr, in = in_channels, out = out_channels, kernel_h = kernel_h, kernel_w = kernel_w, S = cfg.stride,
                      P = cfg.padding, D = cfg.dilation](const Tensor &dz) {
        const auto &shape = xp->shape;
        const auto &grad = dz.p->data, &input = xp->data, &weight = wp->data;

        size_t N = shape[0], H = shape[2], W = shape[3];
        size_t H_out = dz.p->shape[2], W_out = dz.p->shape[3];

        Tensor dx(N, in, H, W);
        Tensor dw(out, in, kernel_h, kernel_w), db({ out });

        auto &xd = dx.p->data;
        auto &wd = dw.p->data;
        auto &bd = db.p->data;

        // Mirror the forward row walk: clip the `ow` range per tap and keep
        // the inner loop contiguous, so backward does no padding tests either.
        std::vector<ptrdiff_t> lo(kernel_w), hi(kernel_w);
        for (size_t kw = 0; kw < kernel_w; kw++) {
          const auto off = static_cast<ptrdiff_t>(kw * D) - static_cast<ptrdiff_t>(P);
          lo[kw] = off >= 0 ? ptrdiff_t(0) : (-off + static_cast<ptrdiff_t>(S) - 1) / static_cast<ptrdiff_t>(S);
          hi[kw] = (static_cast<ptrdiff_t>(W) - 1 - off) / static_cast<ptrdiff_t>(S);
          if (hi[kw] > static_cast<ptrdiff_t>(W_out) - 1)
            hi[kw] = static_cast<ptrdiff_t>(W_out) - 1;
        }

        for (size_t n = 0; n < N; n++) {
          for (size_t oc = 0; oc < out; oc++) {
            dtype db_acc = 0;
            for (size_t oh = 0; oh < H_out; oh++) {
              const dtype *grad_row = &grad[((n * out + oc) * H_out + oh) * W_out];
              if (bp)
                for (size_t ow = 0; ow < W_out; ow++)
                  db_acc += grad_row[ow];

              for (size_t ic = 0; ic < in; ic++) {
                for (size_t kh = 0; kh < kernel_h; kh++) {
                  const auto ih = static_cast<ptrdiff_t>(oh * S + kh * D) - static_cast<ptrdiff_t>(P);
                  if (ih < 0 or ih >= static_cast<ptrdiff_t>(H))
                    continue;

                  const size_t row = ((n * in + ic) * H + static_cast<size_t>(ih)) * W;
                  dtype *xd_row = &xd[row];
                  const dtype *in_row = &input[row];

                  for (size_t kw = 0; kw < kernel_w; kw++) {
                    const size_t wi = ((oc * in + ic) * kernel_h + kh) * kernel_w + kw;
                    const dtype wv = weight[wi];
                    const auto off = static_cast<ptrdiff_t>(kw * D) - static_cast<ptrdiff_t>(P);
                    dtype dw_acc = 0;
                    for (auto ow = lo[kw]; ow <= hi[kw]; ow++) {
                      const auto xi = ow * static_cast<ptrdiff_t>(S) + off;
                      xd_row[xi] += wv * grad_row[ow];
                      dw_acc += in_row[xi] * grad_row[ow];
                    }
                    wd[wi] += dw_acc;
                  }
                }
              }
            }
            if (bp)
              bd[oc] += db_acc;
          }
        }

        if (xp->requires_grad)
          accumulate(*xp, dx);
        if (wp->requires_grad)
          accumulate(*wp, dw);
        if (bp and bp->requires_grad)
          accumulate(*bp, db);
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    return r;
  }
} // namespace NN
