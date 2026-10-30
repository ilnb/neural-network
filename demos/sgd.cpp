#include "optim/sgd.hpp"
#include "autograd/grad_mode.hpp"
#include "nn/loss.hpp"
#include "nn/module/linear.hpp"
#include "nn/module/sequential.hpp"
#include "tensor/tensor.hpp"
#include <print>

int main() {
  NN::Linear l(3, 1);
  {
    NN::Autograd::NoGradGuard g;
    l.w.set_data(1.5, -0.5, 2.0);
    l.b.set_data(0.5);
  }

  NN::Sequential model(std::move(l));
  auto params = model.params();

  NN::Tensor x(1, 3), target(1, 1);
  x.set_data(2.0, 4.0, 1.0);
  target.set_data(4.0);

  auto pred = model.forward(x);
  auto loss = NN::Loss::mse(pred, target);
  loss.backward();

  std::println("grad_w : {}", params[0].p->grad->p->data);
  std::println("grad_b : {}", params[1].p->grad->p->data);

  NN::Optim::SGD sgd(model.params(), { .lr = 0.1, .beta = 0 });
  sgd.step();
  sgd.zero_grad();

  std::println("updated weights:");
  std::println("w : {}", params[0].p->data);
  std::println("b : {}", params[1].p->data);
  return 0;
}
