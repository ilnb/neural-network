#include "nn/activations.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <vector>

using NN::dtype, NN::Tensor;
namespace Act = NN::Act;

static int failures = 0;

static void check(bool ok, const char *what) {
  if (not ok) {
    std::println(stderr, "FAIL: {}", what);
    failures++;
  }
}

static void check_close(dtype got, dtype want, dtype tol, const char *what) {
  if (not (std::abs(got - want) <= tol)) {
    std::println(stderr, "FAIL: {} (got {}, want {})", what, got, want);
    failures++;
  }
}

static Tensor mk(std::vector<size_t> shape, std::vector<dtype> vals) {
  Tensor t;
  t.set_dims(std::move(shape), false);
  t.set_data(std::move(vals));
  return t;
}

// Calls `f` with every index whose coordinate along `dim` is 0.
static void gen(std::vector<size_t> &idx, size_t k, const std::vector<size_t> &sh, size_t dim, auto &&f) {
  if (k == sh.size()) {
    f(idx);
    return;
  }
  if (k == dim) {
    idx[k] = 0;
    gen(idx, k + 1, sh, dim, f);
    return;
  }
  for (size_t i = 0; i < sh[k]; i++) {
    idx[k] = i;
    gen(idx, k + 1, sh, dim, f);
  }
}

static void check_slices_sum_to_one(const Tensor &t, size_t dim, const char *what) {
  std::vector<size_t> idx(t.shape.size(), 0);
  gen(idx, 0, t.shape, dim, [&](std::vector<size_t> &i) {
    auto j = i;
    dtype s = 0;
    for (size_t k = 0; k < t.shape[dim]; k++) {
      j[dim] = k;
      s += t.at(j);
    }
    check_close(s, 1, 1e-5f, what);
  });
}

static void check_slices_exp_sum_to_one(const Tensor &t, size_t dim, const char *what) {
  std::vector<size_t> idx(t.shape.size(), 0);
  gen(idx, 0, t.shape, dim, [&](std::vector<size_t> &i) {
    auto j = i;
    dtype s = 0;
    for (size_t k = 0; k < t.shape[dim]; k++) {
      j[dim] = k;
      s += std::exp(t.at(j));
    }
    check_close(s, 1, 1e-5f, what);
  });
}

static void test_relu() {
  Tensor x = mk({5}, {-1, 0, 2, -4, 5});
  const Tensor &orig = x;
  const auto r = Act::relu(x);

  check(r.shape == x.shape, "relu: shape is preserved");
  check_close(r.data[0], 0, 0, "relu: negative -> 0");
  check_close(r.data[1], 0, 0, "relu: zero -> 0");
  check_close(r.data[2], 2, 0, "relu: positive is kept");
  check_close(r.data[3], 0, 0, "relu: negative -> 0");
  check_close(r.data[4], 5, 0, "relu: positive is kept");
  check(x.data == orig.data, "relu: input is not modified");
}

static void test_sigmoid() {
  check_close(Act::sigmoid(0), 0.5f, 1e-6f, "sigmoid: 0 -> 0.5");
  check_close(Act::sigmoid(2), 0.88079708f, 1e-6f, "sigmoid: 2");
  check_close(Act::sigmoid(-2), 0.11920292f, 1e-6f, "sigmoid: -2");

  for (dtype v : {0.5f, 2.f, 5.f})
    check_close(Act::sigmoid(-v), 1 - Act::sigmoid(v), 1e-6f, "sigmoid: symmetry");

  check_close(Act::sigmoid(100), 1, 1e-6f, "sigmoid: large positive -> 1");
  check_close(Act::sigmoid(-100), 0, 1e-6f, "sigmoid: large negative -> 0");
  check(std::isnan(Act::sigmoid(std::numeric_limits<dtype>::quiet_NaN())), "sigmoid: NaN propagates");
}

static void test_sigmoid_tensor() {
  Tensor x = mk({2, 3}, {-3, -1, 0, 1, 2, 3});
  const Tensor &orig = x;
  const auto r = Act::sigmoid(x);

  check(r.shape == x.shape, "sigmoid(tensor): shape is preserved");
  for (size_t i = 0; i < x.num_el; i++)
    check_close(r.data[i], Act::sigmoid(x.data[i]), 0, "sigmoid(tensor): matches the scalar form");
  check(x.data == orig.data, "sigmoid(tensor): input is not modified");
}

static void test_softmax_values() {
  Tensor x = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor &orig = x;
  const auto r = Act::softmax(x, 1);

  check(r.shape == x.shape, "softmax: shape is preserved");
  check_close(r.at(size_t{0}, size_t{0}), 0.09003057f, 1e-6f, "softmax: row 0 col 0");
  check_close(r.at(size_t{0}, size_t{1}), 0.24472847f, 1e-6f, "softmax: row 0 col 1");
  check_close(r.at(size_t{0}, size_t{2}), 0.66524096f, 1e-6f, "softmax: row 0 col 2");
  check_close(r.at(size_t{1}, size_t{0}), 0.09003057f, 1e-6f, "softmax: row 1 col 0");
  check_slices_sum_to_one(r, 1, "softmax: dim=1 slices sum to 1");
  check(x.data == orig.data, "softmax: input is not modified");
}

static void test_softmax_axes() {
  Tensor x;
  x.set_dims({2, 3, 4}, false);
  for (size_t i = 0; i < x.num_el; i++)
    x.data[i] = static_cast<dtype>(i) / 4 - 3;

  for (size_t dim = 0; dim < 3; dim++) {
    const auto r = Act::softmax(x, dim);
    check(r.shape == x.shape, "softmax: shape is preserved for every dim");
    check_slices_sum_to_one(r, dim, "softmax: every slice sums to 1");
  }
}

static void test_softmax_size_one_axis() {
  Tensor x = mk({1, 5}, {1, 2, 3, 4, 5});
  const auto r = Act::softmax(x, 0);

  for (size_t j = 0; j < 5; j++)
    check_close(r.at(size_t{0}, j), 1, 0, "softmax: a size-1 axis normalizes to 1");
}

static void test_softmax_stability() {
  Tensor x = mk({3, 4}, {1000, 1001, 1002, 1003, -1000, -500, 0, 500, 700, 699, 698, 697});
  const auto r = Act::softmax(x, 1);

  check(std::ranges::all_of(r.data, [](dtype v) { return std::isfinite(v); }), "softmax: stays finite for large inputs");
  check_slices_sum_to_one(r, 1, "softmax: slices sum to 1 for large inputs");
  check_close(r.at(size_t{0}, size_t{3}), 0.6439f, 1e-3f, "softmax: dominant entry wins");
}

static void test_softmax_errors() {
  Tensor x = mk({2, 3}, {1, 2, 3, 4, 5, 6});

  bool threw = false;
  try {
    (void)Act::softmax(x, 2);
  } catch (const std::out_of_range &) {
    threw = true;
  }
  check(threw, "softmax: dim >= rank throws out_of_range");

  Tensor e;
  threw = false;
  try {
    (void)Act::softmax(e, 0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "softmax: an empty tensor throws invalid_argument");
}

static void test_log_softmax_values() {
  Tensor x = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor &orig = x;
  const auto r = Act::log_softmax(x, 1);

  check(r.shape == x.shape, "log_softmax: shape is preserved");
  check(r.num_el == x.num_el, "log_softmax: storage is allocated");
  check_close(r.at(size_t{0}, size_t{0}), -2.40760596f, 1e-6f, "log_softmax: row 0 col 0");
  check_close(r.at(size_t{0}, size_t{1}), -1.40760596f, 1e-6f, "log_softmax: row 0 col 1");
  check_close(r.at(size_t{0}, size_t{2}), -0.40760596f, 1e-6f, "log_softmax: row 0 col 2");
  check_close(r.at(size_t{1}, size_t{0}), -2.40760596f, 1e-6f, "log_softmax: row 1 col 0");
  check_close(r.at(size_t{1}, size_t{2}), -0.40760596f, 1e-6f, "log_softmax: row 1 col 2");
  check(x.data == orig.data, "log_softmax: input is not modified");
}

static void test_log_softmax_axes() {
  Tensor x;
  x.set_dims({2, 3, 4}, false);
  for (size_t i = 0; i < x.num_el; i++)
    x.data[i] = static_cast<dtype>(i) / 4 - 3;

  for (size_t dim = 0; dim < 3; dim++) {
    const auto r = Act::log_softmax(x, dim);
    const auto s = Act::softmax(x, dim);
    check(r.shape == x.shape, "log_softmax: shape is preserved for every dim");
    check(r.num_el == x.num_el, "log_softmax: storage is allocated for every dim");
    check_slices_exp_sum_to_one(r, dim, "log_softmax: every slice exponentiates to 1");

    std::vector<size_t> idx(x.shape.size(), 0);
    gen(idx, 0, x.shape, dim, [&](std::vector<size_t> &i) {
      auto j = i;
      for (size_t k = 0; k < x.shape[dim]; k++) {
        j[dim] = k;
        check_close(r.at(j), std::log(s.at(j)), 1e-5f, "log_softmax: matches log(softmax)");
      }
    });
  }
}

static void test_log_softmax_size_one_axis() {
  Tensor x = mk({1, 5}, {1, 2, 3, 4, 5});
  const auto r = Act::log_softmax(x, 0);

  for (size_t j = 0; j < 5; j++)
    check_close(r.at(size_t{0}, j), 0, 0, "log_softmax: a size-1 axis normalizes to 0");
}

static void test_log_softmax_stability() {
  Tensor x = mk({3, 4}, {1000, 1001, 1002, 1003, -1000, -500, 0, 500, 700, 699, 698, 697});
  const auto r = Act::log_softmax(x, 1);

  check(std::ranges::all_of(r.data, [](dtype v) { return std::isfinite(v); }), "log_softmax: stays finite for large inputs");
  check_close(r.at(size_t{0}, size_t{3}), -0.4401897f, 1e-5f, "log_softmax: dominant entry wins");
  check_close(r.at(size_t{1}, size_t{3}), 0, 1e-6f, "log_softmax: underflowed terms leave the max entry at 0");
  check_slices_exp_sum_to_one(r, 1, "log_softmax: slices exponentiate to 1 for large inputs");
}

static void test_log_softmax_nan() {
  Tensor x = mk({1, 3}, {1, std::numeric_limits<dtype>::quiet_NaN(), 2});
  const auto r = Act::log_softmax(x, 1);

  check(std::ranges::all_of(r.data, [](dtype v) { return std::isnan(v); }), "log_softmax: NaN poisons the slice");
}

static void test_log_softmax_errors() {
  Tensor x = mk({2, 3}, {1, 2, 3, 4, 5, 6});

  bool threw = false;
  try {
    (void)Act::log_softmax(x, 2);
  } catch (const std::out_of_range &) {
    threw = true;
  }
  check(threw, "log_softmax: dim >= rank throws out_of_range");

  Tensor e;
  threw = false;
  try {
    (void)Act::log_softmax(e, 0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "log_softmax: an empty tensor throws invalid_argument");
}

int main() {
  test_relu();
  test_sigmoid();
  test_sigmoid_tensor();
  test_softmax_values();
  test_softmax_axes();
  test_softmax_size_one_axis();
  test_softmax_stability();
  test_softmax_errors();
  test_log_softmax_values();
  test_log_softmax_axes();
  test_log_softmax_size_one_axis();
  test_log_softmax_stability();
  test_log_softmax_nan();
  test_log_softmax_errors();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("activations: all checks passed");
  return 0;
}
