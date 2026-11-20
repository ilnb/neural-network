#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "autograd/grad_mode.hpp"
#include "module/conv2d.hpp"
#include "module/conv_detail.hpp"
#include "tensor/detail.hpp"

#include <format>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using NN::Autograd::GradFn, NN::Autograd::accumulate;
namespace CD = NN::Conv::Detail;

namespace NN {
  std::vector<NamedTensor> Conv2d::named_params() const {
    if (cfg.bias)
      return { { "w", w }, { "b", b } };
    return { { "w", w } };
  }

  Tensor Conv2d::forward(const Tensor &x) {
    if (x.empty())
      throw std::invalid_argument("Conv2d::forward: empty tensor.");
    const auto &shape = x.p->shape;
    if (shape.size() != 4)
      throw std::invalid_argument("Conv2d::forward: 4D NCHW tensor expected.");
    if (shape[1] != in_channels)
      throw std::invalid_argument(std::format("Conv2d::forward: input channel mismatch (shape[1] = {}, in_channels = {}).", shape[1], in_channels));

    // Output geometry and the column layout are computed once and shared by the
    // forward and backward passes.
    const auto g = CD::Geometry::make(x, out_channels, kernel_h, kernel_w, cfg.stride, cfg.padding, cfg.dilation);
    const size_t N = g.N, out = out_channels, K = g.K(), HW = g.HW();

    const auto acquire = [this] {
      if (col_pool->empty())
        return std::make_shared<Tensor>();
      auto t = std::move(col_pool->back());
      col_pool->pop_back();
      return t;
    };

    // Y[n] = W2 @ col[n], with W2: [out, K] and col[n]: [K, HW].
    // The product is written straight into the NCHW output,
    // so no reshape or transpose is needed afterwards.
    auto col = acquire();
    CD::im2col(x, g, *col);
    Tensor r(N, out, g.H_out, g.W_out);
    Detail::matmul_into(w.p->data.data(), col->p->data.data(), r.p->data.data(), N, out, K, HW, cfg.bias ? b.p->data.data() : nullptr);

    const bool requires_grad = x.p->requires_grad or w.p->requires_grad or (cfg.bias and b.p->requires_grad);
    if (Autograd::grad_enabled and requires_grad) {
      auto fn = std::make_shared<GradFn>();
      fn->inputs = { x.p, w.p };
      if (cfg.bias)
        fn->inputs.emplace_back(b.p);

      // `mutable`: the captures are const in a plain lambda, which would make
      // `std::move(col)` below bind to the copy constructor instead of the move.
      // Backward runs once, so consuming the capture is safe.
      fn->backward = [xp = x.p, wp = w.p, bp = cfg.bias ? b.p : nullptr, col, pool = col_pool, g](const Tensor &dz) mutable {
        const size_t N = g.N, out = g.out_channels, K = g.K(), HW = g.HW();
        const dtype *dy = dz.p->data.data(); // flat [N, out, HW]
        const auto acquire = [&pool] {
          if (pool->empty())
            return std::make_shared<Tensor>();
          auto t = std::move(pool->back());
          pool->pop_back();
          return t;
        };

        if (xp->requires_grad) {
          // dcol = W2^T @ dy, then scatter the windows back into dx. The weight
          // transpose is tiny, so it is rebuilt here instead of being kept.
          std::vector<dtype> w2t(K * out);
          const dtype *wd = wp->data.data();
          for (size_t o = 0; o < out; o++)
            for (size_t k = 0; k < K; k++)
              w2t[k * out + o] = wd[o * K + k];

          auto dcol = acquire();
          dcol->set_dims({ N, K, HW }, false);
          Detail::matmul_into(w2t.data(), dy, dcol->p->data.data(), N, K, out, HW);

          Tensor dx(N, g.C, g.H, g.W); // zeroed; col2im accumulates into it
          CD::col2im(dcol->p->data.data(), g, dx.p->data.data());
          accumulate(*xp, dx);
          pool->push_back(std::move(dcol));
        }

        if (wp->requires_grad) {
          // dW[o,k] = sum_{n,p} dy[n,o,p] * col[n,k,p]. The column buffer is
          // transposed once so the contraction's inner loop runs over `k`.
          auto colT = acquire();
          colT->set_dims({ N * HW, K }, false);
          CD::transpose_col(col->p->data.data(), N, K, HW, colT->p->data.data());

          Tensor dw(out, g.C, g.kh, g.kw); // zeroed; dw_accum accumulates into it
          CD::dw_accum(dy, colT->p->data.data(), g, dw.p->data.data());
          accumulate(*wp, dw);
          pool->push_back(std::move(colT));
        }

        if (bp and bp->requires_grad) {
          Tensor db({ out });
          CD::db_accum(dy, N, out, HW, db.p->data.data());
          accumulate(*bp, db);
        }

        // Hand the column buffer back for the next forward.
        pool->push_back(std::move(col));
      };
      r.p->grad_fn = std::move(fn);
      r.p->requires_grad = true;
    }

    if (not(Autograd::grad_enabled and requires_grad))
      col_pool->push_back(std::move(col));

    return r;
  }
} // namespace NN
