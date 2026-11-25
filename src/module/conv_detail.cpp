#include "module/conv_detail.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <omp.h>
#include <vector>

namespace NN::Conv::Detail {
  Geometry Geometry::make(const Tensor &x, size_t out_channels, size_t kh, size_t kw, size_t stride, size_t padding, size_t dilation) {
    if (x.empty() or x.p->shape.size() != 4)
      throw std::invalid_argument("Geometry::make: 4D NCHW tensor expected.");
    if (kh == 0 or kw == 0 or out_channels == 0)
      throw std::invalid_argument("Geometry::make: kernel and output channel counts must be positive.");
    if (stride == 0 or dilation == 0)
      throw std::invalid_argument("Geometry::make: stride and dilation must be positive.");

    Geometry g{};
    g.N = x.p->shape[0];
    g.C = x.p->shape[1];
    g.H = x.p->shape[2];
    g.W = x.p->shape[3];
    g.out_channels = out_channels;
    g.kh = kh;
    g.kw = kw;
    g.stride = stride;
    g.padding = padding;
    g.dilation = dilation;

    const ptrdiff_t P = padding, D = dilation, S = stride;

    // number of top-left positions the window can take along H, i.e. the
    // padded extent minus the kernel extent. H_out = floor((hs - 1) / S) + 1.
    const auto hs = SC<ptrdiff_t>(g.H) + 2 * P - D * (SC<ptrdiff_t>(kh) - 1);
    const auto ws = SC<ptrdiff_t>(g.W) + 2 * P - D * (SC<ptrdiff_t>(kw) - 1);
    if (hs < 1 or ws < 1)
      throw std::invalid_argument(std::format("Geometry::make: kernel {}x{} (dilation {}) does not fit input {}x{} with padding {}.", kh, kw, dilation, g.H, g.W, padding));

    g.H_out = SC<size_t>((hs - 1) / S + 1);
    g.W_out = SC<size_t>((ws - 1) / S + 1);
    return g;
  }

  void im2col(const Tensor &x, const Geometry &g, Tensor &col) {
    if (x.empty())
      throw std::invalid_argument("im2col: empty tensor.");
    if (not std::ranges::equal(x.p->shape, std::array{ g.N, g.C, g.H, g.W }))
      throw std::invalid_argument("im2col: geometry doesnt match input shape.");

    const size_t K = g.K(), HW = g.HW(), kh = g.kh, kw = g.kw;
    const size_t N = g.N, C = g.C, S = g.stride, P = g.padding, D = g.dilation;
    const size_t H_out = g.H_out, W_out = g.W_out, H = g.H, W = g.W;

    col.set_dims({ N, K, HW }, false);
    const dtype *src = x.p->data.data();
    dtype *dst = col.p->data.data();

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t n = 0; n < N; n++) {
      for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < kh; i++) {
          for (size_t j = 0; j < kw; j++) {
            // One plane per (n, k): k = (c * kh + i) * kw + j.
            dtype *plane = dst + ((n * K + (c * kh + i) * kw + j) * HW);

            // Height offset is constant per i, so hoist it out of the oh walk.
            const auto dh = SC<ptrdiff_t>(i * D) - SC<ptrdiff_t>(P);

            for (size_t oh = 0; oh < H_out; oh++) {
              dtype *drow = plane + oh * W_out;
              const auto ih = SC<ptrdiff_t>(oh * S) + dh;
              if (ih < 0 or ih >= SC<ptrdiff_t>(H)) {
                std::fill_n(drow, W_out, dtype(0));
                continue;
              }

              const dtype *srow = src + ((n * C + c) * H + SC<size_t>(ih)) * W;
              for (size_t ow = 0; ow < W_out; ow++) {
                const auto iw = SC<ptrdiff_t>(ow * S + j * D) - SC<ptrdiff_t>(P);
                drow[ow] = (iw < 0 or iw >= SC<ptrdiff_t>(W)) ? dtype(0) : srow[iw];
              }
            }
          }
        }
      }
    }
  }

  Tensor im2col(const Tensor &x, const Geometry &g) {
    Tensor col;
    im2col(x, g, col);
    return col;
  }

  void col2im(const dtype *dcol, const Geometry &g, dtype *dx) {
    const size_t K = g.K(), HW = g.HW(), kh = g.kh, kw = g.kw;
    const size_t N = g.N, C = g.C, S = g.stride, P = g.padding, D = g.dilation;
    const size_t H_out = g.H_out, W_out = g.W_out, H = g.H, W = g.W;

    const dtype *src = dcol;
    dtype *dst = dx;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t n = 0; n < N; n++) {
      for (size_t c = 0; c < C; c++) {
        for (size_t i = 0; i < kh; i++) {
          for (size_t j = 0; j < kw; j++) {
            const dtype *plane = src + ((n * K + (c * kh + i) * kw + j) * HW);
            const auto dh = SC<ptrdiff_t>(i * D) - SC<ptrdiff_t>(P);

            for (size_t oh = 0; oh < H_out; oh++) {
              const dtype *srow = plane + oh * W_out;
              const auto ih = SC<ptrdiff_t>(oh * S) + dh;
              if (ih < 0 or ih >= SC<ptrdiff_t>(H))
                continue;

              dtype *drow = dst + ((n * C + c) * H + SC<size_t>(ih)) * W;
              for (size_t ow = 0; ow < W_out; ow++) {
                const auto iw = SC<ptrdiff_t>(ow * S + j * D) - SC<ptrdiff_t>(P);
                if (iw >= 0 and iw < SC<ptrdiff_t>(W))
                  drow[iw] += srow[ow];
              }
            }
          }
        }
      }
    }
  }

  Tensor col2im(const Tensor &dcol, const Geometry &g) {
    if (dcol.empty())
      throw std::invalid_argument("col2im: empty tensor.");
    if (not std::ranges::equal(dcol.p->shape, std::array{ g.N, g.K(), g.HW() }))
      throw std::invalid_argument("col2im: geometry does not match column shape.");

    Tensor dx(g.N, g.C, g.H, g.W);
    col2im(dcol.p->data.data(), g, dx.p->data.data());
    return dx;
  }

  void transpose_col(const dtype *col, size_t N, size_t K, size_t HW, dtype *colT) {
    // Square tiles: reading a [T, T] block of `col` walks T runs of T, and
    // writing the transposed block does the same on `colT`, so neither side
    // strides through memory a cache line at a time.
    constexpr size_t T = 32;
    const size_t kb_blocks = (K + T - 1) / T, pb_blocks = (HW + T - 1) / T;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t n = 0; n < N; n++) {
      for (size_t kb = 0; kb < kb_blocks; kb++) {
        for (size_t pb = 0; pb < pb_blocks; pb++) {
          const size_t k0 = kb * T, p0 = pb * T;
          const size_t kn = std::min(T, K - k0), pn = std::min(T, HW - p0);

          const dtype *src = col + (n * K + k0) * HW + p0;
          dtype *dst = colT + (n * HW + p0) * K + k0;

          // Stage the tile so both the reads and the writes are contiguous
          // runs; the block itself is 8 KB and stays in L1.
          dtype tile[T][T];
          for (size_t i = 0; i < kn; i++) {
            const dtype *srow = src + i * HW;
            for (size_t j = 0; j < pn; j++)
              tile[i][j] = srow[j];
          }
          for (size_t j = 0; j < pn; j++) {
            dtype *drow = dst + j * K;
            for (size_t i = 0; i < kn; i++)
              drow[i] = tile[i][j];
          }
        }
      }
    }
  }

  void dw_accum(const dtype *dy, const dtype *colT, const Geometry &g, dtype *dw) {
    const size_t N = g.N, out = g.out_channels, K = g.K(), HW = g.HW();

    // Block the `k` axis so the tile of partial sums is a few tens of KB and
    // stays in L1 across the whole spatial walk.
    constexpr size_t KB = 32;
    const size_t kblocks = (K + KB - 1) / KB;

    const int workers = SC<int>(std::max<size_t>(1, std::min<size_t>(SC<size_t>(omp_get_max_threads()), N)));
    const size_t wsize = out * K;
    std::vector<dtype> partial(SC<size_t>(workers) * wsize, dtype(0));

#pragma omp parallel num_threads(workers)
    {
      const auto worker = SC<size_t>(omp_get_thread_num());
      dtype *dwl = partial.data() + worker * wsize;
      std::vector<dtype> tile(out * KB);

#pragma omp for schedule(static)
      for (size_t n = 0; n < N; n++) {
        const dtype *dyn = dy + n * out * HW;
        const dtype *ct = colT + n * HW * K;

        for (size_t kb = 0; kb < kblocks; kb++) {
          const size_t k0 = kb * KB;
          const size_t kn = std::min(KB, K - k0);

          std::fill_n(tile.begin(), out * KB, dtype(0));

          // Rank-1 accumulate: for each spatial position, `dy[:, p]` scales the
          // row `colT[p, k0 : k0 + kn]` and adds into the L1 tile.
          for (size_t p = 0; p < HW; p++) {
            const dtype *cr = ct + p * K + k0;
            for (size_t o = 0; o < out; o++) {
              const dtype av = dyn[o * HW + p];
              dtype *tr = tile.data() + o * KB;
              for (size_t kk = 0; kk < kn; kk++)
                tr[kk] += av * cr[kk];
            }
          }

          for (size_t o = 0; o < out; o++)
            for (size_t kk = 0; kk < kn; kk++)
              dwl[o * K + k0 + kk] += tile[o * KB + kk];
        }
      }
    }

    // Reduction of the per-worker partials, in worker order so a fixed thread
    // count always produces the same bits.
    for (int t = 0; t < workers; t++) {
      const dtype *dwl = partial.data() + SC<size_t>(t) * wsize;
      for (size_t i = 0; i < wsize; i++)
        dw[i] += dwl[i];
    }
  }

  void db_accum(const dtype *dy, size_t N, size_t channels, size_t HW, dtype *db) {
#pragma omp parallel for schedule(static)
    for (size_t o = 0; o < channels; o++) {
      dtype sum = 0;
      for (size_t n = 0; n < N; n++) {
        const dtype *row = dy + (n * channels + o) * HW;
        for (size_t p = 0; p < HW; p++)
          sum += row[p];
      }
      db[o] += sum;
    }
  }
} // namespace NN::Conv::Detail
