#include "optim/sgd.hpp"
#include "autograd/grad_mode.hpp"
#include "loss/loss.hpp"
#include "module/linear.hpp"
#include "module/sequential.hpp"

#include <print>

// One forward pass, one backward pass, one SGD step on a 3 -> 1 linear layer.
//
// There is no training loop on purpose: the weights are set by hand so the
// gradients can be checked without a calculator, and the step is printed so the
// update rule is visible.
int main() {
  NN::Linear l(3, 1);

  // Pin the weights. In-place writes are rejected while the tensor requires
  // grad, which the parameters do, hence the guard.
  {
    NN::Autograd::NoGradGuard g;
    l.w.set_data(1.5, -0.5, 2.0);
    l.b.set_data(0.5);
  }

  // Handles, not copies: the step below writes through these same tensors, so
  // the values printed at the end are live.
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

  // beta = 0 drops the momentum term, leaving plain gradient descent.
  NN::Optim::SGD sgd(model.params(), { .lr = 0.1, .beta = 0 });
  sgd.step();
  sgd.zero_grad();

  std::println("updated weights:");
  std::println("w : {}", params[0].p->data);
  std::println("b : {}", params[1].p->data);
  return 0;
}
