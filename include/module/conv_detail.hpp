#pragma once

#include "tensor/tensor.hpp"

#include <cstddef>

namespace NN::Conv::Detail {
  // Output geometry for one convolution, computed and validated once.
  struct Geometry {
    size_t N, C, H, W, out_channels;
    size_t kh, kw, H_out, W_out;
    size_t stride, padding, dilation;

    size_t K() const { return C * kh * kw; }
    size_t HW() const { return H_out * W_out; }

    static Geometry make(const Tensor &x, size_t out_channels, size_t kh, size_t kw, size_t stride, size_t padding, size_t dilation);
  };

  // x: [N, C, H, W] -> [N, K, H_out * W_out], zero padded, K = C*kh*kw.
  Tensor im2col(const Tensor &x, const Geometry &g);

  // Fills `col` in place, resizing if needed. Lets a caller recycle
  // one buffer across the steps instead of reallocing every forward.
  void im2col(const Tensor &x, const Geometry &g, Tensor &col);

  // Adjoint of `im2col`, accumulating into a zeroed `dx`: [N, C, H, W].
  Tensor col2im(const Tensor &dcol, const Geometry &g);

  // Pointer form, for callers that own the buffers.
  //
  // `dcol` has the shape `im2col` returns; `dx` must be zeroed by the caller,
  // because every window position adds into it.
  void col2im(const dtype *dcol, const Geometry &g, dtype *dx);

  // col: [N, K, HW] -> colT: [N, HW, K]. Tiled so both the reads and the writes
  // of a tile are contiguous runs.
  void transpose_col(const dtype *col, size_t N, size_t K, size_t HW, dtype *colT);

  // dw[o, k] += sum_{n, p} dy[n, o, p] * colT[n, p, k], with `dw` zeroed.
  //
  // Stored transposed so the inner loop runs over `k`, and blocked over `k` so
  // the tile of partial sums stays in L1. Each worker keeps a private copy of
  // `dw` and the partials are reduced in worker order, which keeps the result
  // fixed for a given thread count.
  void dw_accum(const dtype *dy, const dtype *colT, const Geometry &g, dtype *dw);

  // db[o] += sum_{n, p} dy[n, o, p], with `db` zeroed.
  void db_accum(const dtype *dy, size_t N, size_t channels, size_t HW, dtype *db);
} // namespace NN::Conv::Detail
