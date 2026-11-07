#include "module/conv2d.hpp"
#include "utils.hpp"

#include <cmath>
#include <print>
#include <vector>

using namespace NN;

static constexpr dtype TOL = sizeof(dtype) == 8 ? dtype(1e-10) : dtype(2e-5);

// Independent reference for one output element. Every index is written
// explicitly instead of reusing the layer's flat row walk.
static dtype naive(const dtype *xs, const dtype *ws, const dtype *bs, size_t C, size_t H, size_t W, size_t KH, size_t KW, size_t S, size_t P, size_t D, size_t oc, size_t oh,
                   size_t ow) {
  dtype sum = bs ? bs[oc] : 0;
  for (size_t ic = 0; ic < C; ic++)
    for (size_t kh = 0; kh < KH; kh++)
      for (size_t kw = 0; kw < KW; kw++) {
        auto ih = static_cast<ptrdiff_t>(oh * S + kh * D) - static_cast<ptrdiff_t>(P);
        auto iw = static_cast<ptrdiff_t>(ow * S + kw * D) - static_cast<ptrdiff_t>(P);
        if (ih < 0 or iw < 0 or ih >= static_cast<ptrdiff_t>(H) or iw >= static_cast<ptrdiff_t>(W))
          continue;
        sum += xs[(ic * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw)] * ws[((oc * C + ic) * KH + kh) * KW + kw];
      }
  return sum;
}

static void check_case(size_t N, size_t Cin, size_t H, size_t W, size_t Cout, size_t KH, size_t KW, Conv2d::Config cfg, const char *what) {
  Conv2d conv(Cin, Cout, KH, KW, cfg);
  for (size_t i = 0; i < conv.w.p->num_el; i++)
    conv.w.p->data[i] = dtype(0.1) * dtype(i % 7) - dtype(0.2);
  if (cfg.bias)
    for (size_t i = 0; i < Cout; i++)
      conv.b.p->data[i] = dtype(0.05) * dtype(i + 1);

  Tensor x = mk({ N, Cin, H, W }, std::vector<dtype>(N * Cin * H * W));
  for (size_t i = 0; i < x.p->num_el; i++)
    x.p->data[i] = dtype(0.07) * dtype(i % 11) - dtype(0.3);

  const Tensor y = conv.forward(x);
  const size_t Hout = (H + 2 * cfg.padding - cfg.dilation * (KH - 1) - 1) / cfg.stride + 1;
  const size_t Wout = (W + 2 * cfg.padding - cfg.dilation * (KW - 1) - 1) / cfg.stride + 1;
  check(y.p->shape == std::vector<size_t>{ N, Cout, Hout, Wout }, "conv: output shape");

  dtype worst = 0;
  for (size_t n = 0; n < N; n++) {
    const dtype *xs = x.p->data.data() + n * Cin * H * W;
    for (size_t oc = 0; oc < Cout; oc++)
      for (size_t oh = 0; oh < Hout; oh++)
        for (size_t ow = 0; ow < Wout; ow++) {
          const dtype want = naive(xs, conv.w.p->data.data(), cfg.bias ? conv.b.p->data.data() : nullptr, Cin, H, W, KH, KW, cfg.stride, cfg.padding, cfg.dilation, oc, oh, ow);
          const dtype got = y.p->data[((n * Cout + oc) * Hout + oh) * Wout + ow];
          worst = std::max(worst, std::abs(got - want));
        }
  }
  check(worst <= TOL, what);
}

static void test_forward() {
  check_case(2, 2, 5, 6, 3, 3, 2, { .stride = 1, .padding = 1, .dilation = 1, .seed = 1 }, "conv: stride 1, pad 1, non-square kernel");
  check_case(2, 2, 5, 6, 3, 3, 2, { .stride = 2, .padding = 0, .dilation = 1, .seed = 1 }, "conv: stride 2, no padding");
  check_case(2, 2, 7, 6, 2, 3, 3, { .stride = 1, .padding = 2, .dilation = 2, .seed = 1 }, "conv: dilation 2");
  check_case(2, 2, 7, 6, 2, 3, 3, { .stride = 2, .padding = 1, .dilation = 2, .bias = false, .seed = 1 }, "conv: dilation 2, no bias");
  check_case(1, 1, 8, 8, 4, 5, 5, { .stride = 1, .padding = 2, .dilation = 1, .seed = 1 }, "conv: 5x5 kernel");
}

static void test_grads() {
  Conv2d conv(1, 2, 3, 3, { .stride = 1, .padding = 1, .seed = 3 });
  for (size_t i = 0; i < conv.w.p->num_el; i++)
    conv.w.p->data[i] = dtype(0.11) * dtype(i % 5) - dtype(0.2);
  conv.b.p->data[0] = dtype(0.1);
  conv.b.p->data[1] = dtype(-0.2);

  Tensor x = mk_grad({ 2, 1, 4, 4 }, std::vector<dtype>(32));
  for (size_t i = 0; i < x.p->num_el; i++)
    x.p->data[i] = dtype(0.09) * dtype(i % 9) - dtype(0.3);

  const Tensor y = conv.forward(x);
  Tensor w = mk(y.p->shape, std::vector<dtype>(y.p->num_el));
  for (size_t i = 0; i < w.p->num_el; i++)
    w.p->data[i] = dtype(0.13) * dtype(i % 5) - dtype(0.2);

  auto f = [&] { return (conv.forward(x) * w).sum(); };
  check_fd(f, x, "conv: fd dx");
  check_fd(f, conv.w, "conv: fd dw");
  check_fd(f, conv.b, "conv: fd db");

  // A second configuration exercises the strided/dilated backward ranges.
  Conv2d c2(2, 2, 3, 3, { .stride = 2, .padding = 1, .seed = 4 });
  for (size_t i = 0; i < c2.w.p->num_el; i++)
    c2.w.p->data[i] = dtype(0.09) * dtype(i % 5) - dtype(0.1);
  Tensor x2 = mk_grad({ 1, 2, 5, 5 }, std::vector<dtype>(50));
  for (size_t i = 0; i < x2.p->num_el; i++)
    x2.p->data[i] = dtype(0.07) * dtype(i % 7) - dtype(0.2);
  const Tensor y2 = c2.forward(x2);
  Tensor w2 = mk(y2.p->shape, std::vector<dtype>(y2.p->num_el));
  for (size_t i = 0; i < w2.p->num_el; i++)
    w2.p->data[i] = dtype(0.1) * dtype(i % 3) - dtype(0.1);
  check_fd([&] { return (c2.forward(x2) * w2).sum(); }, x2, "conv: strided fd dx");
  check_fd([&] { return (c2.forward(x2) * w2).sum(); }, c2.w, "conv: strided fd dw");
}

static void test_params_and_tracking() {
  Conv2d conv(1, 2, 3, 3, { .padding = 1, .seed = 9 });
  const auto ps = conv.params();
  check(ps.size() == 2, "conv: params() returns weight and bias");
  check(ps[0].p == conv.w.p and ps[1].p == conv.b.p, "conv: params() order");
  check(conv.w.p->requires_grad and conv.b.p->requires_grad, "conv: weight and bias require grad");

  // A constant input must still produce a graph, since the parameters train.
  const Tensor x = mk({ 1, 1, 4, 4 }, std::vector<dtype>(16));
  const Tensor y = conv.forward(x);
  check(y.p->requires_grad, "conv: output tracks the parameters");

  const Tensor w = mk(y.p->shape, std::vector<dtype>(y.p->num_el, dtype(1)));
  (y * w).sum().backward();
  check(conv.w.p->grad != nullptr, "conv: weight receives a gradient");
  check(conv.b.p->grad != nullptr, "conv: bias receives a gradient");

  Conv2d nb(1, 2, 3, 3, { .bias = false, .seed = 9 });
  check(nb.params().size() == 1, "conv: bias-free layer exposes one parameter");
  check(nb.w.p->requires_grad, "conv: bias-free weight requires grad");
  check(nb.b.empty(), "conv: bias-free layer has no bias tensor");

  Conv2d r1(1, 2, 3, 3, { .seed = 7 }), r2(1, 2, 3, 3, { .seed = 7 });
  check(r1.w.p->data == r2.w.p->data, "conv: the same seed reproduces the weights");
}

static void test_errors() {
  Conv2d conv(1, 2, 3, 3, {});
  check_throws<std::invalid_argument>([&] { (void)conv.forward(Tensor()); }, "conv: empty input throws");
  check_throws<std::invalid_argument>([&] { (void)conv.forward(mk({ 1, 1, 4 }, { 1, 2, 3, 4 })); }, "conv: rank 3 throws");
  check_throws<std::invalid_argument>([&] { (void)conv.forward(mk({ 1, 3, 4, 4 }, std::vector<dtype>(48))); }, "conv: input channel mismatch throws");
}

int main() {
  test_forward();
  test_grads();
  test_params_and_tracking();
  test_errors();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("conv: all checks passed");
  return 0;
}
