#include "optim/sgd.hpp"
#include "autograd/grad_mode.hpp"
#include "nn/linear.hpp"
#include "nn/loss.hpp"
#include "tensor/tensor.hpp"
#include <print>

int main() {
  NN::Linear l(3, 1);
  {
    NN::Autograd::NoGradGuard g;
    l.w.set_data(1.5, -0.5, 2.0);
    l.b.set_data(0.5);
  }

  NN::Tensor x(1, 3), target(1, 1);
  x.set_data(2.0, 4.0, 1.0);
  target.set_data(4.0);

  auto pred = l.forward(x);
  auto loss = NN::Loss::mse(pred, target);
  loss.backward();

  std::println("w.grad.data: {}", l.w.p->grad->p->data);
  std::println("b.grad.data: {}", l.b.p->grad->p->data);

  NN::Optim::SGD sgd(l.params(), 0.1);
  sgd.step();
  sgd.zero_grad();

  std::println("updated weights:");
  std::println("w: {}", l.w.p->data);
  std::println("b: {}", l.b.p->data);
  return 0;
}
