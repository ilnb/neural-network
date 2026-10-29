#include "nn/loss.hpp"
#include <cmath>
#include <limits>
#include <print>
#include <stdexcept>
#include <vector>

using NN::dtype, NN::Tensor;
namespace Loss = NN::Loss;

static int failures = 0;

static void check(bool ok, const char *what) {
  if (not ok) {
    std::println(stderr, "FAIL: {}", what);
    failures++;
  }
}

static void check_close(dtype got, dtype want, dtype tol, const char *what) {
  if (std::abs(got - want) > tol) {
    std::println(stderr, "FAIL: {} (got {}, want {})", what, got, want);
    failures++;
  }
}

static Tensor mk(std::vector<size_t> shape, std::vector<dtype> vals) {
  Tensor t;
  t.set_dims(std::move(shape), false);
  t.set_data(vals);
  return t;
}

template <typename Ex, typename F>
static void check_throws(F f, const char *what) {
  try {
    f();
    std::println(stderr, "FAIL: {} (no exception)", what);
    failures++;
  } catch (const Ex &) {
  } catch (const std::exception &e) {
    std::println(stderr, "FAIL: {} (wrong exception: {})", what, e.what());
    failures++;
  }
}

static void test_mse() {
  check_close(Loss::mse(mk({ 3 }, { 1, 2, 3 }), mk({ 3 }, { 1, 1, 1 })).item(), 1.6666667f, 1e-6f, "mse: (1,2,3) vs (1,1,1)");
  check_close(Loss::mse(mk({ 2, 2 }, { 1, 2, 3, 4 }), mk({ 2, 2 }, { 1, 2, 3, 5 })).item(), 0.25f, 1e-6f, "mse: one element off");
  check_close(Loss::mse(mk({ 2 }, { 1, 1 }), mk({ 2 }, { 1, 1 })).item(), 0, 0, "mse: identical -> 0");
}

static void test_bce_values() {
  check_close(Loss::bce(mk({ 3 }, { 0.5f, 0.9f, 0.1f }), mk({ 3 }, { 0, 1, 0 })).item(), 0.3012894f, 1e-6f, "bce: mixed labels");
  check_close(Loss::bce(mk({ 2 }, { 0.7f, 0.1f }), mk({ 2 }, { 1, 0 })).item(), 0.2310177f, 1e-6f, "bce: both label directions");
  check_close(Loss::bce(mk({ 1 }, { 0.9f }), mk({ 1 }, { 1 })).item(), 0.1053605f, 1e-6f, "bce: y=1 uses log(p)");
  check_close(Loss::bce(mk({ 1 }, { 0.9f }), mk({ 1 }, { 0 })).item(), 2.3025851f, 1e-6f, "bce: y=0 uses log(1-p)");
}

static void test_bce_edges() {
  check(std::isinf(Loss::bce(mk({ 1 }, { 0 }), mk({ 1 }, { 1 })).item()), "bce: p=0, y=1 -> inf");
  check(std::isinf(Loss::bce(mk({ 1 }, { 1 }), mk({ 1 }, { 0 })).item()), "bce: p=1, y=0 -> inf");
  check_close(Loss::bce(mk({ 1 }, { 0 }), mk({ 1 }, { 0 })).item(), 0, 0, "bce: p=0, y=0 -> 0");
  check_close(Loss::bce(mk({ 1 }, { 1 }), mk({ 1 }, { 1 })).item(), 0, 0, "bce: p=1, y=1 -> 0");
  check_close(Loss::bce(mk({ 2 }, { 0.5f, 0.5f }), mk({ 2 }, { 0, 0 })).item(), 0.6931472f, 1e-6f, "bce: all-zero targets keep their contribution");
}

static void test_ce_values() {
  Tensor pred = mk({ 2, 3 }, { 0.09003057f, 0.24472847f, 0.66524096f, 0.3333333f, 0.3333333f, 0.3333333f });
  Tensor target = mk({ 2, 3 }, { 0, 0, 1, 1, 0, 0 });

  check_close(Loss::ce(pred, target).item(), 0.7531091f, 1e-6f, "ce: defaults to the last axis");
  check_close(Loss::ce(pred, target, 1).item(), 0.7531091f, 1e-6f, "ce: dim=1 explicit");
  check_close(Loss::ce(mk({ 2 }, { 0.3f, 0.7f }), mk({ 2 }, { 0, 1 })).item(), 0.3566749f, 1e-6f, "ce: rank-1 defaults to dim=0");
}

static void test_ce_axes() {
  Tensor pred = mk({ 2, 3 }, { 0.2f, 0.3f, 0.4f, 0.8f, 0.7f, 0.6f });
  Tensor target = mk({ 2, 3 }, { 1, 0, 1, 0, 1, 0 });
  check_close(Loss::ce(pred, target, 0).item(), 0.9608012f, 1e-6f, "ce: dim=0 sums over rows");
}

static void test_ce_rank3() {
  Tensor pred = mk({ 2, 3, 2 }, { 0.2f, 0.5f, 0.3f, 0.25f, 0.5f, 0.25f, 0.1f, 0.3333333f, 0.1f, 0.3333333f, 0.8f, 0.3333333f });
  Tensor target = mk({ 2, 3, 2 }, { 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1 });
  check_close(Loss::ce(pred, target, 1).item(), 1.1968729f, 1e-6f, "ce: dim=1 averages over the remaining axes");
}

static void test_ce_soft_labels() { check_close(Loss::ce(mk({ 1, 2 }, { 0.5f, 0.5f }), mk({ 1, 2 }, { 0.25f, 0.75f })).item(), 0.6931472f, 1e-6f, "ce: soft labels"); }

static void test_ce_binary_matches_bce() {
  Tensor pred = mk({ 2, 2 }, { 0.3f, 0.7f, 0.9f, 0.1f });
  Tensor target = mk({ 2, 2 }, { 0, 1, 1, 0 });
  const dtype ce = Loss::ce(pred, target, 1).item();
  const dtype bce = Loss::bce(mk({ 2 }, { 0.7f, 0.1f }), mk({ 2 }, { 1, 0 })).item();

  check_close(ce, bce, 1e-6f, "ce with C=2 equals bce on the positive class");
  check_close(Loss::ce(mk({ 1, 2 }, { 0.3f, 0.7f }), mk({ 1, 2 }, { 0, 1 }), 1).item(), 0.3566749f, 1e-6f, "ce with C=2 matches bce for a single sample");
}

static void test_ce_edges() {
  check(std::isinf(Loss::ce(mk({ 1, 2 }, { 0, 1 }), mk({ 1, 2 }, { 1, 0 }), 1).item()), "ce: p=0 with target mass -> inf");
  check_close(Loss::ce(mk({ 1, 2 }, { 0, 1 }), mk({ 1, 2 }, { 0, 1 }), 1).item(), 0, 0, "ce: p=0 with y=0 is ignored");
}

static void test_validation_order() {
  // An `inf` result must not hide malformed entries that follow it.
  check_throws<std::invalid_argument>([&] { (void)Loss::bce(mk({ 2 }, { 0, 0.5f }), mk({ 2 }, { 1, 0.3f })); }, "bce: inf must not hide a later malformed target");

  check_throws<std::invalid_argument>([&] { (void)Loss::ce(mk({ 1, 2 }, { 0, 0.5f }), mk({ 1, 2 }, { 1, 1.3f }), 1); }, "ce: inf must not hide a later malformed target");

  // Probabilities are validated even where the target mass is zero.
  check_throws<std::invalid_argument>([&] { (void)Loss::ce(mk({ 1, 2 }, { -0.5f, 1.5f }), mk({ 1, 2 }, { 0, 1 }), 1); }, "ce: validates probabilities on zero-target entries");

  // bce needs p in [0, 1] because it takes log(p) and log(1 - p).
  check_throws<std::invalid_argument>([&] { (void)Loss::bce(mk({ 2 }, { -0.5f, 1.5f }), mk({ 2 }, { 0, 1 })); }, "bce: probabilities outside [0, 1] throw");

  // ...but NaN must still propagate rather than throw.
  check(std::isnan(Loss::bce(mk({ 1 }, { std::numeric_limits<dtype>::quiet_NaN() }), mk({ 1 }, { 1 })).item()), "bce: NaN prediction propagates");
}

static void test_errors() {
  bool threw = false;
  try {
    (void)Loss::ce(mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }), mk({ 2, 3 }, { 0, 0, 0, 0, 0, 0 }), 2);
  } catch (const std::out_of_range &) {
    threw = true;
  }
  check(threw, "ce: dim >= rank throws out_of_range");

  threw = false;
  try {
    (void)Loss::ce(mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }), mk({ 3 }, { 0, 0, 0 }));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "ce: shape mismatch throws invalid_argument");

  threw = false;
  try {
    (void)Loss::ce(mk({ 1, 2 }, { 0.5f, 0.5f }), mk({ 1, 2 }, { 2, 0 }));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "ce: target > 1 throws invalid_argument");

  threw = false;
  try {
    (void)Loss::ce(Tensor(), Tensor());
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "ce: empty tensor throws invalid_argument");

  threw = false;
  try {
    (void)Loss::bce(mk({ 1 }, { 0.5f }), mk({ 1 }, { 0.5f }));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "bce: non-binary target throws invalid_argument");

  threw = false;
  try {
    (void)Loss::mse(mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }), mk({ 3 }, { 1, 2, 3 }));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "mse: shape mismatch throws invalid_argument");
}

static void test_scalar_results() {
  const auto mse = Loss::mse(mk({ 2, 2 }, { 1, 2, 3, 4 }), mk({ 2, 2 }, { 1, 2, 3, 5 }));
  check(mse.p->shape.empty(), "mse: result is rank 0");
  check(mse.p->num_el == 1, "mse: result has one element");

  const auto bce = Loss::bce(mk({ 2 }, { 0.7f, 0.1f }), mk({ 2 }, { 1, 0 }));
  check(bce.p->shape.empty(), "bce: result is rank 0");
  check(bce.p->num_el == 1, "bce: result has one element");

  const auto ce = Loss::ce(mk({ 1, 2 }, { 0.3f, 0.7f }), mk({ 1, 2 }, { 0, 1 }), 1);
  check(ce.p->shape.empty(), "ce: result is rank 0");
  check(ce.p->num_el == 1, "ce: result has one element");
}

int main() {
  test_mse();
  test_bce_values();
  test_bce_edges();
  test_ce_values();
  test_ce_axes();
  test_ce_rank3();
  test_ce_soft_labels();
  test_ce_binary_matches_bce();
  test_ce_edges();
  test_errors();
  test_validation_order();
  test_scalar_results();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("losses: all checks passed");
  return 0;
}
