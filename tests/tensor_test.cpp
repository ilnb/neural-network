#include "tensor/tensor.hpp"
#include <algorithm>
#include <print>

using NN::Tensor;

int main() {
  {
    std::println("INDEXING AND GENERAL OPS");
    Tensor a(2, 3);
    std::vector v = {2, 3};
    a.set_dims(v);
    if (a.data.size() != 0)
      std::println(stderr, "Test for set_dims failed: expected size(data) = 0, got {}", a.data.size());
    a.set_data({3, 2, 10, 4, 5, 6});

    Tensor b = a;
    std::ranges::for_each(b.data, [](auto &x) { x++; });

    auto c = a + b;
    auto d = Tensor::mul(c, a);
    std::println("        a : {}", a.data);
    std::println("        b : {}", b.data);
    std::println("c = a + b : {}", c.data);
    std::println("d = c * a : {}", d.data);
    std::println("with utils: {}", a.cadd(b).mul(a).data);
    std::println();

    a.at(1, 2) = 7;
    std::println("a[1,2] = 7: {}", a.data);
    a[0, 1] = 8;
    std::println("a[0,1] = 8: {}", a.data);
    std::println();
  }

  {
    std::println("TRANSPOSE");
    Tensor t(2, 3);
    t.set_data({1, 2, 3, 4, 5, 6});

    t.transpose();
    std::println("transpose: {}", t.data);
    std::println("shape    : {}", t.shape);
    std::println("strides  : {}", t.strides);
    std::println();

    Tensor x(2, 3, 4);
    try {
      x.transpose();
    } catch (const std::invalid_argument &e) {
      std::println("transpose rank test - shape: {} : {}", x.shape, e.what());
    }
    std::println();

    Tensor r(2, 3);
    r.set_data({1, 2, 3, 4, 5, 6});
    r.transpose().transpose();
    std::println("double transpose: {}", r.data);
    std::println("double transpose shape: {}", r.shape);
    std::println("double transpose strides: {}", r.strides);
    std::println();
  }

  {
    std::println("REDUCE");
    Tensor t(2, 3, 2);
    t.set_data(3, 2, 5, 1, 6, 12, 9, 8, 10, 12, 14, 7);
    std::println("data : {}", t.data);
    std::println("shape: {}", t.shape);
    for (size_t i = 0; i < 3; i++) {
      std::println("sum  axis = {}: {}", i, t.csum(i).data);
      std::println("mean axis = {}: {}", i, t.cmean(i).data);
      std::println("max  axis = {}: {}", i, t.cmax(i).data);
      std::println("min  axis = {}: {}", i, t.cmin(i).data);
    }
    std::println();
  }

  {
    std::println("BROADCASTING");
    Tensor u(2, 1, 2), v(2, 1);
    u.set_data(4, 6, 1, 9);
    v.set_data(2, 5);

    std::println("u shape: {}\nu  data: {}\n", u.shape, u.data);
    std::println("v shape: {}\nv  data: {}\n", v.shape, v.data);

    auto t = u;
    std::println("t shape: {}\nt  data: {}\n", t.shape, t.data);
    t += Tensor(1).set_data(6);
    std::println("t += [6]:\nshape: {}\ndata: {}\n", t.shape, t.data);

    Tensor tmp = u + v;
    std::println("u + v:");
    std::println("shape: {}", tmp.shape);
    std::println("data : {}\n", tmp.data);

    tmp = u - v;
    std::println("u - v:");
    std::println("shape: {}", tmp.shape);
    std::println("data : {}\n", tmp.data);

    tmp = u * v;
    std::println("u * v:");
    std::println("shape: {}", tmp.shape);
    std::println("data : {}", tmp.data);
    std::println();
  }
  return 0;
}
