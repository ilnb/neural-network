#include "tensor/tensor.hpp"
#include <algorithm>
#include <print>

using NN::Tensor, std::println;

int main() {
  {
    println("INDEXING AND GENERAL OPS");
    Tensor a(2, 3);
    std::vector v = {2, 3};
    a.set_dims(v);
    if (a.data.size() != 0)
      println(stderr, "Test for set_dims failed: expected size(data) = 0, got {}", a.data.size());
    a.set_data({3, 2, 10, 4, 5, 6});

    Tensor b = a;
    std::ranges::for_each(b.data, [](auto &x) { x++; });

    auto c = a + b;
    auto d = Tensor::mul(c, a);
    println("        a : {}", a.data);
    println("        b : {}", b.data);
    println("c = a + b : {}", c.data);
    println("d = c * a : {}", d.data);
    println("with utils: {}", a.cadd(b).mul(a).data);
    println();

    a.at(1, 2) = 7;
    println("a[1,2] = 7: {}", a.data);
    a[0, 1] = 8;
    println("a[0,1] = 8: {}", a.data);
    println();
  }

  {
    println("TRANSPOSE");
    Tensor t(2, 3);
    t.set_data({1, 2, 3, 4, 5, 6});

    t.transpose();
    println("transpose: {}", t.data);
    println("shape    : {}", t.shape);
    println("strides  : {}", t.strides);
    println();

    Tensor x(2, 3, 4);
    try {
      x.transpose();
    } catch (const std::invalid_argument &e) {
      println("transpose rank test - shape: {} : {}", x.shape, e.what());
    }
    println();

    Tensor r(2, 3);
    r.set_data({1, 2, 3, 4, 5, 6});
    r.transpose().transpose();
    println("double transpose: {}", r.data);
    println("double transpose shape: {}", r.shape);
    println("double transpose strides: {}", r.strides);
    println();
  }

  {
    println("REDUCE");
    Tensor t(2, 3, 2);
    t.set_data(3, 2, 5, 1, 6, 12, 9, 8, 10, 12, 14, 7);
    println("data : {}", t.data);
    println("shape: {}", t.shape);
    for (size_t i = 0; i < 3; i++) {
      println("sum  axis = {}: {}", i, t.csum(i).data);
      println("mean axis = {}: {}", i, t.cmean(i).data);
      println("max  axis = {}: {}", i, t.cmax(i).data);
      println("min  axis = {}: {}", i, t.cmin(i).data);
    }
    println();
  }

  {
    println("BROADCASTING");
    Tensor u(2, 1, 2), v(2, 1);
    u.set_data(4, 6, 1, 9);
    v.set_data(2, 5);

    println("u shape: {}\nu  data: {}\n", u.shape, u.data);
    println("v shape: {}\nv  data: {}\n", v.shape, v.data);

    auto t = u;
    println("t shape: {}\nt  data: {}\n", t.shape, t.data);
    t += Tensor(1).set_data(6);
    println("t += [6]:\nshape: {}\ndata: {}\n", t.shape, t.data);

    Tensor tmp = u + v;
    println("u + v:");
    println("shape: {}", tmp.shape);
    println("data : {}\n", tmp.data);

    tmp = u - v;
    println("u - v:");
    println("shape: {}", tmp.shape);
    println("data : {}\n", tmp.data);

    tmp = u * v;
    println("u * v:");
    println("shape: {}", tmp.shape);
    println("data : {}", tmp.data);
    println();
  }

  {
    std::println("MATMUL");
    Tensor u(1, 2, 1, 2);
    u.set_data(1, 2, 3, 4);
    Tensor v(2, 2, 2, 1);
    v.set_data(8, 7, 6, 5, 4, 3, 2, 1);

    println("u shape: {}\nu  data: {}\n", u.shape, u.data);
    println("v shape: {}\nv  data: {}\n", v.shape, v.data);

    auto c = Tensor::matmul(u, v);
    std::println("a @ b:");
    std::println("shape: {}", c.shape);
    std::println(" data: {}", c.data);
    println();
  }

  {
    println("CONCAT");
    Tensor a(2, 2), b(2, 3);
    a.set_data(1, 2, 3, 4);
    b.set_data(5, 6, 7, 8, 9, 10);

    println("a shape: {}\na  data: {}\n", a.shape, a.data);
    println("b shape: {}\nb  data: {}\n", b.shape, b.data);

    auto t = a.ccat(b, 1);
    println("a.cat(b, 1):");
    println("shape: {}", t.shape);
    println("data : {}\n", t.data);

    t.cat(t, 1);
    println("t.cat(t, 1):");
    println("shape: {}", t.shape);
    println("data : {}\n", t.data);

    Tensor c(3, 2, 2), d(3, 1, 2);
    c.set_data(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    d.set_data(13, 14, 15, 16, 17, 18);

    auto u = c.ccat(d, 1);
    println("c.cat(d, 1):");
    println("shape: {}", u.shape);
    println("data : {}\n", u.data);

    Tensor e(3, 2);
    try {
      a.cat(e, 1);
    } catch (const std::invalid_argument &ex) {
      println("cat mismatch: {}", ex.what());
    }
    try {
      a.cat(b, 7);
    } catch (const std::out_of_range &ex) {
      println("cat dim oob: {}", ex.what());
    }
    println();
  }
  return 0;
}
