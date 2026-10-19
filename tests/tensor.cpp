#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <print>
#include <vector>

using NN::dtype, NN::Tensor;

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

static void check_shape(const Tensor &t, std::initializer_list<size_t> want, const char *what) {
  const std::vector<size_t> w(want);
  if (t.shape != w) {
    std::println(stderr, "FAIL: {} (shape is {}, want {})", what, t.shape, w);
    failures++;
  }
}

static void check_strides(const Tensor &t, std::initializer_list<size_t> want, const char *what) {
  const std::vector<size_t> w(want);
  if (t.strides != w) {
    std::println(stderr, "FAIL: {} (strides are {}, want {})", what, t.strides, w);
    failures++;
  }
}

static void check_data(const Tensor &t, std::initializer_list<dtype> want, const char *what, dtype tol = 0) {
  const std::vector<dtype> w(want);
  if (t.data.size() != w.size()) {
    std::println(stderr, "FAIL: {} (data size is {}, want {})", what, t.data.size(), w.size());
    failures++;
    return;
  }
  for (size_t i = 0; i < w.size(); i++) {
    if (std::abs(t.data[i] - w[i]) > tol) {
      std::println(stderr, "FAIL: {} (data[{}] is {}, want {})", what, i, t.data[i], w[i]);
      failures++;
      return;
    }
  }
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

static Tensor mk(std::vector<size_t> shape, std::vector<dtype> vals) {
  Tensor t;
  t.set_dims(std::move(shape), false);
  t.set_data(std::move(vals));
  return t;
}

static void test_indexing() {
  Tensor a(2, 3);
  check_shape(a, {2, 3}, "ctor: shape");
  check_strides(a, {3, 1}, "ctor: strides");
  check_data(a, {0, 0, 0, 0, 0, 0}, "ctor: data is zero-initialized");
  check(a.num_el == 6, "ctor: num_el");

  std::vector<size_t> dims = {2, 3};
  a.set_dims(dims);
  check(a.empty(), "set_dims: tensor is empty");
  check(a.data.empty(), "set_dims: data is empty");
  check(a.num_el == 0, "set_dims: num_el");
  check_shape(a, {2, 3}, "set_dims: shape is kept");
  check_strides(a, {3, 1}, "set_dims: strides");

  a.set_data({3, 2, 10, 4, 5, 6});
  check(not a.empty(), "set_data: tensor is not empty");
  check(a.num_el == 6, "set_data: num_el");
  check_data(a, {3, 2, 10, 4, 5, 6}, "set_data: data");

  Tensor b = a;
  std::ranges::for_each(b.data, [](dtype &x) { x++; });

  const auto c = a + b;
  check_shape(c, {2, 3}, "a+b: shape");
  check_data(c, {7, 5, 21, 9, 11, 13}, "a+b: data");

  const auto d = Tensor::mul(c, a);
  check_data(d, {21, 10, 210, 36, 55, 78}, "Tensor::mul(c,a): data");
  check_data(a.cadd(b).mul(a), {21, 10, 210, 36, 55, 78}, "cadd/mul chain: data");

  check(a.at(1, 2) == 6, "at: read");
  a.at(1, 2) = 7;
  check_data(a, {3, 2, 10, 4, 5, 7}, "at: write");
  a[0, 1] = 8;
  check_data(a, {3, 8, 10, 4, 5, 7}, "operator[]: write");

  check_throws<std::out_of_range>([&] { (void)a.at(2, 0); }, "at: index out of bounds");
  check_throws<std::invalid_argument>([&] { (void)a.at(1); }, "at: wrong number of indices");
}

static void test_transpose() {
  Tensor t = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  t.transpose();
  check_shape(t, {3, 2}, "transpose: shape");
  check_strides(t, {2, 1}, "transpose: strides");
  check_data(t, {1, 4, 2, 5, 3, 6}, "transpose: data");

  Tensor x(2, 3, 4);
  check_throws<std::invalid_argument>([&] { x.transpose(); }, "transpose: rank is not 2");

  Tensor r = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  r.transpose().transpose();
  check_shape(r, {2, 3}, "double transpose: shape");
  check_strides(r, {3, 1}, "double transpose: strides");
  check_data(r, {1, 2, 3, 4, 5, 6}, "double transpose: data");

  const auto c = mk({2, 3}, {1, 2, 3, 4, 5, 6}).ctranspose();
  check_shape(c, {3, 2}, "ctranspose: shape");
  check_strides(c, {2, 1}, "ctranspose: strides");
  check_data(c, {1, 4, 2, 5, 3, 6}, "ctranspose: data");
}

static void test_reduce() {
  Tensor t = mk({2, 3, 2}, {3, 2, 5, 1, 6, 12, 9, 8, 10, 12, 14, 7});

  check_close(t.sum(), 89, 0, "sum: total");
  check_close(t.mean(), 89.0f / 12, 1e-6f, "mean: total");
  check_close(t.min(), 1, 0, "min: total");
  check_close(t.max(), 14, 0, "max: total");

  check_shape(t.csum(0), {3, 2}, "csum(0): shape");
  check_data(t.csum(0), {12, 10, 15, 13, 20, 19}, "csum(0): data");
  check_shape(t.csum(1), {2, 2}, "csum(1): shape");
  check_data(t.csum(1), {14, 15, 33, 27}, "csum(1): data");
  check_shape(t.csum(2), {2, 3}, "csum(2): shape");
  check_data(t.csum(2), {5, 6, 18, 17, 22, 21}, "csum(2): data");

  check_data(t.cmean(0), {6, 5, 7.5f, 6.5f, 10, 9.5f}, "cmean(0): data", 1e-6f);
  check_data(t.cmean(1), {14.0f / 3, 5, 11, 9}, "cmean(1): data", 1e-6f);
  check_data(t.cmean(2), {2.5f, 3, 9, 8.5f, 11, 10.5f}, "cmean(2): data", 1e-6f);

  check_data(t.cmax(0), {9, 8, 10, 12, 14, 12}, "cmax(0): data");
  check_data(t.cmax(1), {6, 12, 14, 12}, "cmax(1): data");
  check_data(t.cmax(2), {3, 5, 12, 9, 12, 14}, "cmax(2): data");

  check_data(t.cmin(0), {3, 2, 5, 1, 6, 7}, "cmin(0): data");
  check_data(t.cmin(1), {3, 1, 9, 7}, "cmin(1): data");
  check_data(t.cmin(2), {2, 1, 6, 8, 10, 7}, "cmin(2): data");

  Tensor u = t;
  u.sum(1);
  check_shape(u, {2, 2}, "sum(1): shape");
  check_data(u, {14, 15, 33, 27}, "sum(1): data");

  check_throws<std::out_of_range>([&] { (void)t.csum(3); }, "csum: dim out of range");
  check_throws<std::domain_error>([&] { (void)Tensor(3).csum(0); }, "csum: scalar tensor");
  check_throws<std::domain_error>([&] { (void)Tensor().csum(0); }, "csum: empty tensor");
}

static void test_broadcasting() {
  Tensor u = mk({2, 1, 2}, {4, 6, 1, 9});
  Tensor v = mk({2, 1}, {2, 5});

  const auto add = u + v;
  check_shape(add, {2, 2, 2}, "u+v: shape");
  check_data(add, {6, 8, 9, 11, 3, 11, 6, 14}, "u+v: data");

  const auto sub = u - v;
  check_shape(sub, {2, 2, 2}, "u-v: shape");
  check_data(sub, {2, 4, -1, 1, -1, 7, -4, 4}, "u-v: data");

  const auto mul = u * v;
  check_shape(mul, {2, 2, 2}, "u*v: shape");
  check_data(mul, {8, 12, 20, 30, 2, 18, 5, 45}, "u*v: data");

  const auto cadd = u.cadd(v);
  check_shape(cadd, {2, 2, 2}, "cadd: shape");
  check_data(cadd, {6, 8, 9, 11, 3, 11, 6, 14}, "cadd: data");

  Tensor t = u;
  t += Tensor(1).set_data(6);
  check_shape(t, {2, 1, 2}, "scalar +=: shape");
  check_data(t, {10, 12, 7, 15}, "scalar +=: data");

  Tensor s = mk({3}, {1, 2, 3});
  s += 5;
  check_data(s, {6, 7, 8}, "scalar += value: data");

  Tensor r = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  r += mk({3}, {10, 20, 30});
  check_data(r, {11, 22, 33, 14, 25, 36}, "row broadcast +=: data");

  Tensor bad = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  check_throws<std::invalid_argument>([&] { bad += mk({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}); }, "+=: operand does not broadcast in place");
}

static void test_matmul() {
  Tensor u = mk({1, 2, 1, 2}, {1, 2, 3, 4});
  Tensor v = mk({2, 2, 2, 1}, {8, 7, 6, 5, 4, 3, 2, 1});

  const auto c = Tensor::matmul(u, v);
  check_shape(c, {2, 2, 1, 1}, "matmul: shape");
  check_data(c, {22, 38, 10, 10}, "matmul: data");

  const auto d = u.cmatmul(v);
  check_shape(d, {2, 2, 1, 1}, "cmatmul: shape");
  check_data(d, {22, 38, 10, 10}, "cmatmul: data");

  Tensor w = mk({2, 2}, {1, 2, 3, 4});
  w.matmul(mk({2, 2}, {1, 0, 0, 1}));
  check_data(w, {1, 2, 3, 4}, "matmul: identity");

  check_throws<std::domain_error>([&] { (void)mk({3}, {1, 2, 3}).cmatmul(v); }, "matmul: rank below 2");
  check_throws<std::invalid_argument>([&] { (void)mk({2, 3}, {1, 2, 3, 4, 5, 6}).cmatmul(mk({2, 3}, {1, 2, 3, 4, 5, 6})); }, "matmul: inner dimensions");
  check_throws<std::invalid_argument>([&] { (void)mk({2, 4, 3}, std::vector<dtype>(24)).cmatmul(mk({3, 3, 5}, std::vector<dtype>(45))); }, "matmul: batch dimensions");
  check_throws<std::invalid_argument>([&] { (void)Tensor().cmatmul(v); }, "matmul: empty tensor");
}

static void test_cat() {
  Tensor a = mk({2, 2}, {1, 2, 3, 4});
  Tensor b = mk({2, 3}, {5, 6, 7, 8, 9, 10});

  const auto t = a.ccat(b, 1);
  check_shape(t, {2, 5}, "cat: shape");
  check_data(t, {1, 2, 5, 6, 7, 3, 4, 8, 9, 10}, "cat: data");

  Tensor u = t;
  u.cat(u, 1);
  check_shape(u, {2, 10}, "cat self: shape");
  check_data(u, {1, 2, 5, 6, 7, 1, 2, 5, 6, 7, 3, 4, 8, 9, 10, 3, 4, 8, 9, 10}, "cat self: data");

  const auto w = mk({3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}).ccat(mk({3, 1, 2}, {13, 14, 15, 16, 17, 18}), 1);
  check_shape(w, {3, 3, 2}, "cat 3d: shape");
  check_data(w, {1, 2, 3, 4, 13, 14, 5, 6, 7, 8, 15, 16, 9, 10, 11, 12, 17, 18}, "cat 3d: data");

  check_throws<std::invalid_argument>([&] { (void)a.ccat(mk({3, 2}, {1, 2, 3, 4, 5, 6}), 1); }, "cat: mismatched dimensions");
  check_throws<std::out_of_range>([&] { (void)a.ccat(b, 7); }, "cat: dim out of range");
}

static void test_stack() {
  Tensor a = mk({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = mk({2, 3}, {7, 8, 9, 10, 11, 12});

  const auto s0 = a.cstack(b, 0);
  check_shape(s0, {2, 2, 3}, "stack(0): shape");
  check_data(s0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, "stack(0): data");

  const auto s1 = a.cstack(b, 1);
  check_shape(s1, {2, 2, 3}, "stack(1): shape");
  check_data(s1, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}, "stack(1): data");

  const auto s2 = a.cstack(b, 2);
  check_shape(s2, {2, 3, 2}, "stack(rank): shape");
  check_data(s2, {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}, "stack(rank): data");

  Tensor c = mk({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  const auto s3 = c.cstack(c, 2);
  check_shape(s3, {2, 2, 2, 3}, "stack 3d: shape");
  check_data(s3, {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 10, 11, 12}, "stack 3d: data");

  check_throws<std::invalid_argument>([&] { (void)a.cstack(mk({3, 2}, {1, 2, 3, 4, 5, 6}), 0); }, "stack: mismatched shapes");
  check_throws<std::out_of_range>([&] { (void)a.cstack(b, 9); }, "stack: dim out of range");
}

static void test_permute() {
  Tensor t = mk({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const auto u = t.cpermute(2, 0, 1);
  check_shape(u, {2, 2, 3}, "permute(2,0,1): shape");
  check_data(u, {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12}, "permute(2,0,1): data");

  const auto v = t.cpermute({1, 0, 2});
  check_shape(v, {3, 2, 2}, "permute({1,0,2}): shape");
  check_data(v, {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}, "permute({1,0,2}): data");

  const std::vector<size_t> p = {0, 2, 1};
  const auto w = t.cpermute(p);
  check_shape(w, {2, 2, 3}, "permute(vector): shape");
  check_data(w, {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12}, "permute(vector): data");

  const auto id = t.cpermute(0, 1, 2);
  check_shape(id, {2, 3, 2}, "permute(identity): shape");
  check_data(id, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, "permute(identity): data");

  Tensor m = mk({2, 2}, {1, 2, 3, 4});
  m.permute(1, 0);
  check_shape(m, {2, 2}, "permute in place: shape");
  check_data(m, {1, 3, 2, 4}, "permute in place: data");

  check_throws<std::invalid_argument>([&] { (void)t.cpermute(0, 0, 1); }, "permute: not a permutation");
  check_throws<std::invalid_argument>([&] { (void)t.cpermute(0, 1); }, "permute: wrong size");
}

int main() {
  test_indexing();
  test_transpose();
  test_reduce();
  test_broadcasting();
  test_matmul();
  test_cat();
  test_stack();
  test_permute();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("tensor: all checks passed");
  return 0;
}
