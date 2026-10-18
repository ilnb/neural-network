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
  }
  return 0;
}
