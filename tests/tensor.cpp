#include "tensor/tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <print>
#include <vector>

static void check_shape(const Tensor &t, std::initializer_list<size_t> want, const char *what) {
  const std::vector<size_t> w(want);
  if (t.p->shape != w) {
    std::println(stderr, "FAIL: {} (shape is {}, want {})", what, t.p->shape, w);
    failures++;
  }
}

static void check_strides(const Tensor &t, std::initializer_list<size_t> want, const char *what) {
  const std::vector<size_t> w(want);
  if (t.p->strides != w) {
    std::println(stderr, "FAIL: {} (strides are {}, want {})", what, t.p->strides, w);
    failures++;
  }
}

static void check_data(const Tensor &t, std::initializer_list<dtype> want, const char *what, dtype tol = 0) {
  const std::vector<dtype> w(want);
  if (t.p->data.size() != w.size()) {
    std::println(stderr, "FAIL: {} (data size is {}, want {})", what, t.p->data.size(), w.size());
    failures++;
    return;
  }
  for (size_t i = 0; i < w.size(); i++) {
    if (std::abs(t.p->data[i] - w[i]) > tol) {
      std::println(stderr, "FAIL: {} (data[{}] is {}, want {})", what, i, t.p->data[i], w[i]);
      failures++;
      return;
    }
  }
}

// Compares shape and values against another tensor.
static void check_same(const Tensor &got, const Tensor &want, const char *what) { check(got.p->shape == want.p->shape and got.p->data == want.p->data, what); }

static void test_indexing() {
  Tensor a(2, 3);
  check_shape(a, { 2, 3 }, "ctor: shape");
  check_strides(a, { 3, 1 }, "ctor: strides");
  check_data(a, { 0, 0, 0, 0, 0, 0 }, "ctor: data is zero-initialized");
  check(a.p->num_el == 6, "ctor: num_el");

  std::vector<size_t> dims = { 2, 3 };
  a.set_dims(dims);
  a.set_data(3, 2, 10, 4, 5, 6);
  check_strides(a, { 3, 1 }, "set_dims: strides");
  check_data(a, { 3, 2, 10, 4, 5, 6 }, "set_data: data");

  Tensor b = a.clone();
  std::ranges::for_each(b.p->data, [](dtype &x) { x++; });

  const auto c = a + b;
  check_shape(c, { 2, 3 }, "a+b: shape");
  check_data(c, { 7, 5, 21, 9, 11, 13 }, "a+b: data");

  const auto d = NN::Raw::mul(c, a);
  check_data(d, { 21, 10, 210, 36, 55, 78 }, "Raw::mul(c,a): data");
  check_data(a.add(b).mul(a), { 21, 10, 210, 36, 55, 78 }, "cadd/mul chain: data");

  check(a.at(1, 2) == 6, "at: read");
  a.at(1, 2) = 7;
  check_data(a, { 3, 2, 10, 4, 5, 7 }, "at: write");
  a[0, 1] = 8;
  check_data(a, { 3, 8, 10, 4, 5, 7 }, "operator[]: write");

  check_throws<std::out_of_range>([&] { (void)a.at(2, 0); }, "at: index out of bounds");
  check_throws<std::invalid_argument>([&] { (void)a.at(1); }, "at: wrong number of indices");

  check(a.at({ 1, 2 }) == 7, "at(initializer_list): read");
  const Tensor &ca = a;
  check(ca.at(0, 0) == 3, "at const: read");
  check_throws<std::invalid_argument>([&] { (void)Tensor().at(0); }, "at: empty tensor");
  check_throws<std::invalid_argument>([&] { (void)Tensor()[0]; }, "operator[]: empty tensor");
}

static void test_transpose() {
  Tensor t = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  t = t.transpose();
  check_shape(t, { 3, 2 }, "transpose: shape");
  check_strides(t, { 2, 1 }, "transpose: strides");
  check_data(t, { 1, 4, 2, 5, 3, 6 }, "transpose: data");

  Tensor r = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  r = r.transpose().transpose();
  check_shape(r, { 2, 3 }, "double transpose: shape");
  check_strides(r, { 3, 1 }, "double transpose: strides");
  check_data(r, { 1, 2, 3, 4, 5, 6 }, "double transpose: data");

  const auto c = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }).transpose();
  check_shape(c, { 3, 2 }, "ctranspose: shape");
  check_strides(c, { 2, 1 }, "ctranspose: strides");
  check_data(c, { 1, 4, 2, 5, 3, 6 }, "ctranspose: data");
}

static void test_reduce() {
  Tensor t = mk({ 2, 3, 2 }, { 3, 2, 5, 1, 6, 12, 9, 8, 10, 12, 14, 7 });

  check_close(t.sum().item(), 89, 0, "sum: total");
  check_close(t.mean().item(), 89.0f / 12, 1e-6f, "mean: total");
  check_close(t.min().item(), 1, 0, "min: total");
  check_close(t.max().item(), 14, 0, "max: total");

  check_shape(t.sum(0), { 3, 2 }, "csum(0): shape");
  check_data(t.sum(0), { 12, 10, 15, 13, 20, 19 }, "csum(0): data");
  check_shape(t.sum(1), { 2, 2 }, "csum(1): shape");
  check_data(t.sum(1), { 14, 15, 33, 27 }, "csum(1): data");
  check_shape(t.sum(2), { 2, 3 }, "csum(2): shape");
  check_data(t.sum(2), { 5, 6, 18, 17, 22, 21 }, "csum(2): data");

  check_data(t.mean(0), { 6, 5, 7.5f, 6.5f, 10, 9.5f }, "cmean(0): data", 1e-6f);
  check_data(t.mean(1), { 14.0f / 3, 5, 11, 9 }, "cmean(1): data", 1e-6f);
  check_data(t.mean(2), { 2.5f, 3, 9, 8.5f, 11, 10.5f }, "cmean(2): data", 1e-6f);

  check_data(t.max(0), { 9, 8, 10, 12, 14, 12 }, "cmax(0): data");
  check_data(t.max(1), { 6, 12, 14, 12 }, "cmax(1): data");
  check_data(t.max(2), { 3, 5, 12, 9, 12, 14 }, "cmax(2): data");

  check_data(t.min(0), { 3, 2, 5, 1, 6, 7 }, "cmin(0): data");
  check_data(t.min(1), { 3, 1, 9, 7 }, "cmin(1): data");
  check_data(t.min(2), { 2, 1, 6, 8, 10, 7 }, "cmin(2): data");

  Tensor u = t;
  u = u.sum(1);
  check_shape(u, { 2, 2 }, "sum(1): shape");
  check_data(u, { 14, 15, 33, 27 }, "sum(1): data");

  check_throws<std::out_of_range>([&] { (void)t.sum(3); }, "csum: dim out of range");
  check_throws<std::out_of_range>([&] { (void)Tensor(3).sum(0); }, "csum: scalar tensor");
  check_throws<std::domain_error>([&] { (void)Tensor().sum(0); }, "csum: empty tensor");
}

static void test_broadcasting() {
  Tensor u = mk({ 2, 1, 2 }, { 4, 6, 1, 9 });
  Tensor v = mk({ 2, 1 }, { 2, 5 });

  const auto add = u + v;
  check_shape(add, { 2, 2, 2 }, "u+v: shape");
  check_data(add, { 6, 8, 9, 11, 3, 11, 6, 14 }, "u+v: data");

  const auto sub = u - v;
  check_shape(sub, { 2, 2, 2 }, "u-v: shape");
  check_data(sub, { 2, 4, -1, 1, -1, 7, -4, 4 }, "u-v: data");

  const auto mul = u * v;
  check_shape(mul, { 2, 2, 2 }, "u*v: shape");
  check_data(mul, { 8, 12, 20, 30, 2, 18, 5, 45 }, "u*v: data");

  const auto cadd = u.add(v);
  check_shape(cadd, { 2, 2, 2 }, "cadd: shape");
  check_data(cadd, { 6, 8, 9, 11, 3, 11, 6, 14 }, "cadd: data");

  Tensor t = u.clone();
  t += Tensor(6);
  check_shape(t, { 2, 1, 2 }, "scalar +=: shape");
  check_data(t, { 10, 12, 7, 15 }, "scalar +=: data");

  Tensor s = mk({ 3 }, { 1, 2, 3 });
  s += 5;
  check_data(s, { 6, 7, 8 }, "scalar += value: data");

  Tensor r = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  r += mk({ 3 }, { 10, 20, 30 });
  check_data(r, { 11, 22, 33, 14, 25, 36 }, "row broadcast +=: data");

  Tensor bad = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  check_throws<std::invalid_argument>([&] { bad += mk({ 2, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8 }); }, "+=: operand does not broadcast in place");
}

static void test_matmul() {
  Tensor u = mk({ 1, 2, 1, 2 }, { 1, 2, 3, 4 });
  Tensor v = mk({ 2, 2, 2, 1 }, { 8, 7, 6, 5, 4, 3, 2, 1 });

  const auto c = NN::Raw::matmul(u, v);
  check_shape(c, { 2, 2, 1, 1 }, "matmul: shape");
  check_data(c, { 22, 38, 10, 10 }, "matmul: data");

  const auto d = u.matmul(v);
  check_shape(d, { 2, 2, 1, 1 }, "cmatmul: shape");
  check_data(d, { 22, 38, 10, 10 }, "cmatmul: data");

  Tensor w = mk({ 2, 2 }, { 1, 2, 3, 4 });
  w = w.matmul(mk({ 2, 2 }, { 1, 0, 0, 1 }));
  check_data(w, { 1, 2, 3, 4 }, "matmul: identity");

  check_throws<std::domain_error>([&] { (void)mk({ 3 }, { 1, 2, 3 }).matmul(v); }, "matmul: rank below 2");
  check_throws<std::invalid_argument>([&] { (void)mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }).matmul(mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })); }, "matmul: inner dimensions");
  check_throws<std::invalid_argument>([&] { (void)mk({ 2, 4, 3 }, std::vector<dtype>(24)).matmul(mk({ 3, 3, 5 }, std::vector<dtype>(45))); }, "matmul: batch dimensions");
  check_throws<std::invalid_argument>([&] { (void)Tensor().matmul(v); }, "matmul: empty tensor");
}

static void test_cat() {
  Tensor a = mk({ 2, 2 }, { 1, 2, 3, 4 });
  Tensor b = mk({ 2, 3 }, { 5, 6, 7, 8, 9, 10 });

  const auto t = a.cat(b, 1);
  check_shape(t, { 2, 5 }, "cat: shape");
  check_data(t, { 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 }, "cat: data");

  Tensor u = t.cat(t, 1);
  check_shape(u, { 2, 10 }, "cat self: shape");
  check_data(u, { 1, 2, 5, 6, 7, 1, 2, 5, 6, 7, 3, 4, 8, 9, 10, 3, 4, 8, 9, 10 }, "cat self: data");

  const auto w = mk({ 3, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).cat(mk({ 3, 1, 2 }, { 13, 14, 15, 16, 17, 18 }), 1);
  check_shape(w, { 3, 3, 2 }, "cat 3d: shape");
  check_data(w, { 1, 2, 3, 4, 13, 14, 5, 6, 7, 8, 15, 16, 9, 10, 11, 12, 17, 18 }, "cat 3d: data");

  check_throws<std::invalid_argument>([&] { (void)a.cat(mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 }), 1); }, "cat: mismatched dimensions");
  check_throws<std::out_of_range>([&] { (void)a.cat(b, 7); }, "cat: dim out of range");
}

static void test_stack() {
  Tensor a = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  Tensor b = mk({ 2, 3 }, { 7, 8, 9, 10, 11, 12 });

  const auto s0 = a.stack(b, 0);
  check_shape(s0, { 2, 2, 3 }, "stack(0): shape");
  check_data(s0, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, "stack(0): data");

  const auto s1 = a.stack(b, 1);
  check_shape(s1, { 2, 2, 3 }, "stack(1): shape");
  check_data(s1, { 1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12 }, "stack(1): data");

  const auto s2 = a.stack(b, 2);
  check_shape(s2, { 2, 3, 2 }, "stack(rank): shape");
  check_data(s2, { 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12 }, "stack(rank): data");

  Tensor c = mk({ 2, 2, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
  const auto s3 = c.stack(c, 2);
  check_shape(s3, { 2, 2, 2, 3 }, "stack 3d: shape");
  check_data(s3, { 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 10, 11, 12 }, "stack 3d: data");

  check_throws<std::invalid_argument>([&] { (void)a.stack(mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 }), 0); }, "stack: mismatched shapes");
  check_throws<std::out_of_range>([&] { (void)a.stack(b, 9); }, "stack: dim out of range");
}

static void test_permute() {
  Tensor t = mk({ 2, 3, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

  const auto u = t.permute(2, 0, 1);
  check_shape(u, { 2, 2, 3 }, "permute(2,0,1): shape");
  check_data(u, { 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12 }, "permute(2,0,1): data");

  const auto v = t.permute({ 1, 0, 2 });
  check_shape(v, { 3, 2, 2 }, "permute({1,0,2}): shape");
  check_data(v, { 1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12 }, "permute({1,0,2}): data");

  const std::vector<size_t> p = { 0, 2, 1 };
  const auto w = t.permute(p);
  check_shape(w, { 2, 2, 3 }, "permute(vector): shape");
  check_data(w, { 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12 }, "permute(vector): data");

  const auto id = t.permute(0, 1, 2);
  check_shape(id, { 2, 3, 2 }, "permute(identity): shape");
  check_data(id, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, "permute(identity): data");

  Tensor m = mk({ 2, 2 }, { 1, 2, 3, 4 });
  m = m.permute(1, 0);
  check_shape(m, { 2, 2 }, "permute in place: shape");
  check_data(m, { 1, 3, 2, 4 }, "permute in place: data");

  check_throws<std::invalid_argument>([&] { (void)t.permute(0, 0, 1); }, "permute: not a permutation");
  check_throws<std::invalid_argument>([&] { (void)t.permute(0, 1); }, "permute: wrong size");
}

static void test_constructors() {
  Tensor s(3);
  check(s.p->shape.empty(), "ctor(3): rank 0");
  check(s.p->num_el == 1, "ctor(3): one element");
  check(s.item() == 3, "ctor(3): scalar value");

  Tensor i({ 3 });
  check_shape(i, { 3 }, "ctor({3}): shape");
  check_data(i, { 0, 0, 0 }, "ctor({3}): zero-initialized");

  Tensor v(std::vector<size_t>{ 2, 3 });
  check_shape(v, { 2, 3 }, "ctor(vector): shape");
  check_strides(v, { 3, 1 }, "ctor(vector): strides");

  Tensor d(2, 3);
  check_shape(d, { 2, 3 }, "ctor(2,3): shape");
  check_strides(d, { 3, 1 }, "ctor(2,3): strides");

  check(Tensor().empty(), "ctor(): empty");
  check(not s.empty(), "ctor(3): not empty");
}

static void test_set_data_forms() {
  Tensor a(2, 2);

  std::vector<dtype> v{ 1, 2, 3, 4 };
  a.set_data(v);
  check_data(a, { 1, 2, 3, 4 }, "set_data(lvalue vector): data");

  a.set_data(5, 6, 7, 8);
  check_data(a, { 5, 6, 7, 8 }, "set_data(variadic): data");

  std::array<dtype, 4> arr{ 9, 10, 11, 12 };
  a.set_data(arr);
  check_data(a, { 9, 10, 11, 12 }, "set_data(std::array): data");

  check_throws<std::invalid_argument>([&] { a.set_data(1, 2, 3); }, "set_data: size mismatch");
  check_throws<std::invalid_argument>([&] { Tensor().set_data(1, 2); }, "set_data: empty tensor");
}

static void test_raw() {
  Tensor a = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  Tensor b = mk({ 2, 3 }, { 10, 20, 30, 40, 50, 60 });
  Tensor c = mk({ 3 }, { 10, 20, 30 });

  check_same(NN::Raw::add(a, b), a + b, "Raw::add: matches operator+");
  check_same(NN::Raw::sub(a, b), a - b, "Raw::sub: matches operator-");
  check_same(NN::Raw::mul(a, b), a * b, "Raw::mul: matches operator*");

  const auto bc = NN::Raw::add(a, c);
  check_shape(bc, { 2, 3 }, "Raw::add broadcast: shape");
  check_data(bc, { 11, 22, 33, 14, 25, 36 }, "Raw::add broadcast: data");

  check_same(NN::Raw::sub(a, c), a.sub(c), "Raw::sub broadcast: matches member");

  Tensor x = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  Tensor y = mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
  const auto m = NN::Raw::matmul(x, y);
  check_shape(m, { 2, 2 }, "Raw::matmul: shape");
  check_data(m, { 22, 28, 49, 64 }, "Raw::matmul: data");
  check_same(m, x.matmul(y), "Raw::matmul: matches member");

  check_throws<std::invalid_argument>([&] { (void)NN::Raw::add(a, mk({ 2, 2 }, { 1, 2, 3, 4 })); }, "Raw::add: incompatible shapes");
  check_throws<std::invalid_argument>([&] { (void)NN::Raw::matmul(mk({ 2, 3 }, std::vector<dtype>(6)), mk({ 2, 3 }, std::vector<dtype>(6))); }, "Raw::matmul: inner dimensions");
}

static void test_clone() {
  Tensor a = mk({ 2, 2 }, { 1, 2, 3, 4 });
  Tensor b = a.clone();

  check(b.p != a.p, "clone: distinct storage");
  check_data(b, { 1, 2, 3, 4 }, "clone: data");
  check_strides(b, { 2, 1 }, "clone: strides");

  b.p->data[0] = 99;
  check_data(a, { 1, 2, 3, 4 }, "clone: source unchanged");
}

static void test_shape_ops() {
  Tensor t = mk({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });

  const auto r = t.reshape(3, 2);
  check_shape(r, { 3, 2 }, "reshape: shape");
  check_strides(r, { 2, 1 }, "reshape: strides");
  check_data(r, { 1, 2, 3, 4, 5, 6 }, "reshape: data");

  const std::vector<size_t> flat{ 6 };
  check_shape(t.reshape(flat), { 6 }, "reshape(vector): shape");
  check_data(t.reshape(flat), { 1, 2, 3, 4, 5, 6 }, "reshape(vector): data");

  check_throws<std::invalid_argument>([&] { (void)t.reshape(2, 2); }, "reshape: element count mismatch");

  const auto e0 = t.expand(0, 2);
  check_shape(e0, { 2, 2, 3 }, "expand(0,2): shape");
  check_data(e0, { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 }, "expand(0,2): data");

  const auto e1 = mk({ 2, 2 }, { 1, 2, 3, 4 }).expand(1, 2);
  check_shape(e1, { 2, 2, 2 }, "expand(1,2): shape");
  check_data(e1, { 1, 2, 1, 2, 3, 4, 3, 4 }, "expand(1,2): data");

  const auto e2 = mk({ 2, 2 }, { 1, 2, 3, 4 }).expand(2, 3);
  check_shape(e2, { 2, 2, 3 }, "expand(rank): shape");
  check_data(e2, { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 }, "expand(rank): data");

  check_throws<std::out_of_range>([&] { (void)t.expand(3, 2); }, "expand: dim out of range");
}

int main() {
  test_indexing();
  test_constructors();
  test_set_data_forms();
  test_transpose();
  test_reduce();
  test_broadcasting();
  test_raw();
  test_matmul();
  test_cat();
  test_stack();
  test_permute();
  test_clone();
  test_shape_ops();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("tensor: all checks passed");
  return 0;
}
