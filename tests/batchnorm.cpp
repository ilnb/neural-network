#include "loss/loss.hpp"
#include "module/batchnorm1d.hpp"
#include "module/batchnorm2d.hpp"
#include "module/linear.hpp"
#include "module/relu.hpp"
#include "module/sequential.hpp"
#include "optim/sgd.hpp"
#include "utils.hpp"
#include <cmath>
#include <limits>
#include <print>
#include <vector>

using namespace NN;

static constexpr dtype TOL = sizeof(dtype) == 8 ? dtype(1e-9) : dtype(2e-4);

static void set(Tensor &t, std::initializer_list<dtype> v) { t.p->data.assign(v.begin(), v.end()); }

static void check_vec(const std::vector<dtype> &got, const std::vector<dtype> &want, const char *what) {
  if (got.size() != want.size()) {
    std::println(stderr, "FAIL: {} (size {} != {})", what, got.size(), want.size());
    failures++;
    return;
  }
  for (size_t i = 0; i < got.size(); i++)
    check_close(got[i], want[i], TOL, what);
}

// Independent reference: batch statistics are reduced with explicit
// (sample, channel, inner) coordinates instead of the layer's flat mapping.
static std::vector<dtype> ref_bn(const Tensor &x, const std::vector<dtype> &g, const std::vector<dtype> &b, const dtype eps, const std::vector<dtype> *rm = nullptr,
                                 const std::vector<dtype> *rv = nullptr) {
  const auto &shape = x.p->shape;
  const size_t N = shape[0], C = shape[1];
  size_t inner = 1;
  for (size_t d = 2; d < shape.size(); d++)
    inner *= shape[d];
  const size_t n = N * inner;

  std::vector<dtype> mu(C, 0), var(C, 0);
  if (rm) {
    mu = *rm;
    var = *rv;
  } else {
    for (size_t ni = 0; ni < N; ni++)
      for (size_t c = 0; c < C; c++)
        for (size_t k = 0; k < inner; k++)
          mu[c] += x.p->data[(ni * C + c) * inner + k];
    for (size_t c = 0; c < C; c++)
      mu[c] /= static_cast<dtype>(n);

    for (size_t ni = 0; ni < N; ni++)
      for (size_t c = 0; c < C; c++)
        for (size_t k = 0; k < inner; k++) {
          const dtype d = x.p->data[(ni * C + c) * inner + k] - mu[c];
          var[c] += d * d;
        }
    for (size_t c = 0; c < C; c++)
      var[c] /= static_cast<dtype>(n);
  }

  std::vector<dtype> y(x.p->num_el);
  for (size_t ni = 0; ni < N; ni++)
    for (size_t c = 0; c < C; c++)
      for (size_t k = 0; k < inner; k++) {
        const size_t i = (ni * C + c) * inner + k;
        y[i] = (x.p->data[i] - mu[c]) / std::sqrt(var[c] + eps) * g[c] + b[c];
      }
  return y;
}

static void test_bn1d_values() {
  BatchNorm1d bn(2);
  Tensor x = mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
  std::vector<dtype> g{ 0.7f, -1.3f }, b{ 0.2f, 0.4f };
  set(bn.gamma, { 0.7f, -1.3f });
  set(bn.beta, { 0.2f, 0.4f });

  const Tensor y = bn.forward(x);
  check_vec(y.p->data, ref_bn(x, g, b, bn.cfg.eps), "bn1d: train values");
  check(y.p->requires_grad, "bn1d: output requires grad");

  // Channel 1 of this input has a different spread; a swapped channel axis
  // would not match the reference.
  Tensor z = mk({ 2, 2 }, { 0, 100, 4, 102 });
  check_vec(bn.forward(z).p->data, ref_bn(z, g, b, bn.cfg.eps), "bn1d: per-channel statistics");
}

static void test_bn1d_rank3() {
  BatchNorm1d bn(2);
  Tensor x = mk({ 2, 2, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
  std::vector<dtype> g{ 0.5f, 2.f }, b{ -0.1f, 0.3f };
  set(bn.gamma, { 0.5f, 2.f });
  set(bn.beta, { -0.1f, 0.3f });

  check_vec(bn.forward(x).p->data, ref_bn(x, g, b, bn.cfg.eps), "bn1d: rank 3 pools N and L");
  check(bn.running_mean.p->data[0] != dtype(0), "bn1d: rank 3 updates running stats");
}

static void test_bn1d_fd() {
  for (const bool eval : { false, true }) {
    BatchNorm1d bn(2);
    set(bn.gamma, { 0.7f, -1.3f });
    set(bn.beta, { 0.2f, 0.4f });
    Tensor x = mk_grad({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
    const Tensor w = mk({ 3, 2 }, { 0.3f, -0.5f, 0.8f, 1.1f, -0.2f, 0.9f });

    if (eval) {
      (void)bn.forward(x);
      (void)bn.forward(x);
      bn.eval();
    }

    auto f = [&] { return (bn.forward(x) * w).sum(); };
    check_fd(f, x, eval ? "bn1d: eval fd dx" : "bn1d: train fd dx");
    check_fd(f, bn.gamma, eval ? "bn1d: eval fd dgamma" : "bn1d: train fd dgamma");
    check_fd(f, bn.beta, eval ? "bn1d: eval fd dbeta" : "bn1d: train fd dbeta");
  }

  // Rank 3 backward, through the same channel mapping.
  BatchNorm1d bn(2);
  set(bn.gamma, { 0.5f, -0.9f });
  set(bn.beta, { 0.1f, 0.2f });
  Tensor x = mk_grad({ 2, 2, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
  const Tensor w = mk({ 2, 2, 3 }, { 0.3f, -0.5f, 0.8f, 1.1f, -0.2f, 0.9f, 0.4f, -0.7f, 0.6f, 1.2f, -0.1f, 0.5f });
  check_fd([&] { return (bn.forward(x) * w).sum(); }, x, "bn1d: rank 3 fd dx");
}

static void test_running_stats() {
  BatchNorm1d bn(1, { .mom = 0.1, .eps = 1e-5 });

  // mean 3, biased var 5, unbiased var 20/3.
  const Tensor x = mk({ 4, 1 }, { 0, 2, 4, 6 });
  (void)bn.forward(x);
  check_close(bn.running_mean.p->data[0], dtype(0.3), TOL, "bn running_mean after one batch");
  check_close(bn.running_var.p->data[0], dtype(0.9) + dtype(0.1) * (dtype(20) / dtype(3)), TOL, "bn running_var uses unbiased variance");

  // Second batch has zero variance, so running_var only decays.
  const dtype rv1 = bn.running_var.p->data[0];
  (void)bn.forward(mk({ 2, 1 }, { 3, 3 }));
  check_close(bn.running_mean.p->data[0], dtype(0.57), TOL, "bn running_mean after two batches");
  check_close(bn.running_var.p->data[0], dtype(0.9) * rv1, TOL, "bn running_var decays on a zero-variance batch");

  // Eval uses and does not update the running buffers.
  const dtype rm = bn.running_mean.p->data[0];
  const dtype rv = bn.running_var.p->data[0];
  bn.eval();
  const Tensor in = mk({ 2, 1 }, { 1, 5 });
  const Tensor out = bn.forward(in);
  const std::vector<dtype> rm_v{ rm }, rv_v{ rv };
  check_vec(out.p->data, ref_bn(in, { 1 }, { 0 }, bn.cfg.eps, &rm_v, &rv_v), "bn1d: eval uses running stats");
  check_close(bn.running_mean.p->data[0], rm, TOL, "bn1d: eval leaves running_mean alone");
  check_close(bn.running_var.p->data[0], rv, TOL, "bn1d: eval leaves running_var alone");

  // A single element per channel skips the unbiased correction.
  BatchNorm1d one(1);
  (void)one.forward(mk({ 1, 1 }, { 5 }));
  check_close(one.running_mean.p->data[0], dtype(0.5), TOL, "bn: n == 1 running_mean");
  check_close(one.running_var.p->data[0], dtype(0.9), TOL, "bn: n == 1 skips the unbiased correction");
}

static void test_params_and_init() {
  BatchNorm1d bn(3);
  const auto ps = bn.params();
  check(ps.size() == 2, "bn: params() returns gamma and beta");
  check(ps[0].p == bn.gamma.p and ps[1].p == bn.beta.p, "bn: params() order");
  check(bn.gamma.p->requires_grad and bn.beta.p->requires_grad, "bn: gamma and beta require grad");
  check(!bn.running_mean.p->requires_grad and !bn.running_var.p->requires_grad, "bn: buffers do not require grad");

  for (size_t i = 0; i < 3; i++) {
    check_close(bn.gamma.p->data[i], 1, TOL, "bn: gamma starts at 1");
    check_close(bn.beta.p->data[i], 0, TOL, "bn: beta starts at 0");
    check_close(bn.running_mean.p->data[i], 0, TOL, "bn: running_mean starts at 0");
    check_close(bn.running_var.p->data[i], 1, TOL, "bn: running_var starts at 1");
  }
}

static void test_bn2d() {
  BatchNorm2d bn(2);
  Tensor x = mk({ 2, 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
  std::vector<dtype> g{ 0.6f, -0.4f }, b{ 0.1f, 0.2f };
  set(bn.gamma, { 0.6f, -0.4f });
  set(bn.beta, { 0.1f, 0.2f });

  check_vec(bn.forward(x).p->data, ref_bn(x, g, b, bn.cfg.eps), "bn2d: train values pool N, H, W");

  Tensor xg = mk_grad({ 2, 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
  const Tensor w = mk({ 2, 2, 2, 2 }, { 0.3f, -0.5f, 0.8f, 1.1f, -0.2f, 0.9f, 0.4f, -0.7f, 0.6f, 1.2f, -0.1f, 0.5f, 0.2f, -0.3f, 0.7f, 0.9f });
  auto f = [&] { return (bn.forward(xg) * w).sum(); };
  check_fd(f, xg, "bn2d: fd dx");
  check_fd(f, bn.gamma, "bn2d: fd dgamma");
  check_fd(f, bn.beta, "bn2d: fd dbeta");
}

static void test_errors() {
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(0); }, "bn1d: zero channels throws");
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(2, { .mom = -0.1 }); }, "bn: negative momentum throws");
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(2, { .mom = 1.1 }); }, "bn: momentum above 1 throws");
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(2, { .mom = std::numeric_limits<dtype>::quiet_NaN() }); }, "bn: NaN momentum throws");
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(2, { .eps = 0 }); }, "bn: zero eps throws");
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(2, { .eps = -1 }); }, "bn: negative eps throws");
  check_throws<std::invalid_argument>([] { (void)BatchNorm1d(2, { .eps = std::numeric_limits<dtype>::quiet_NaN() }); }, "bn: NaN eps throws");

  BatchNorm1d bn(2);
  check_throws<std::invalid_argument>([&] { (void)bn.forward(Tensor()); }, "bn1d: empty input throws");
  check_throws<std::invalid_argument>([&] { (void)bn.forward(mk({ 3 }, { 1, 2, 3 })); }, "bn1d: rank 1 throws");
  check_throws<std::invalid_argument>([&] { (void)bn.forward(mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })); }, "bn1d: wrong channel count throws");
  check_throws<std::invalid_argument>([&] { (void)bn.forward(mk({ 1, 2, 1, 1 }, { 1, 2 })); }, "bn1d: rank 4 throws");

  BatchNorm2d bn2(2);
  check_throws<std::invalid_argument>([&] { (void)bn2.forward(mk({ 2, 2 }, { 1, 2, 3, 4 })); }, "bn2d: rank 2 throws");
  check_throws<std::invalid_argument>([&] { (void)bn2.forward(mk({ 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8 })); }, "bn2d: rank 3 throws");
  check_throws<std::invalid_argument>([&] { (void)bn2.forward(mk({ 1, 3, 1, 1 }, { 1, 2, 3 })); }, "bn2d: wrong channel count throws");
}

static void test_sequential_training() {
  Sequential model(Linear(2, 8, 7), BatchNorm1d(8), ReLU(), Linear(8, 1, 11));
  const Tensor x = mk({ 8, 2 }, { -1, -1, -1, 1, 1, -1, 1, 1, -2, -2, -2, 2, 2, -2, 2, 2 });
  const Tensor y = mk({ 8, 1 }, { -2, 0, 0, 2, -4, 0, 0, 4 });

  Optim::SGD opt(model.params(), { .lr = 0.05 });
  dtype first = 0, last = 0;
  for (int step = 0; step < 200; step++) {
    const Tensor pred = model(x);
    const Tensor loss = Loss::mse(pred, y);
    if (step == 0)
      first = loss.item();
    opt.zero_grad();
    loss.backward();
    opt.step();
    last = loss.item();
  }
  check(last < first, "bn: Sequential(Linear, BatchNorm1d, ReLU) trains");
}

int main() {
  test_bn1d_values();
  test_bn1d_rank3();
  test_bn1d_fd();
  test_running_stats();
  test_params_and_init();
  test_bn2d();
  test_errors();
  test_sequential_training();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("batchnorm: all checks passed");
  return 0;
}
