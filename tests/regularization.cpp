#include "regularization/regularization.hpp"
#include "autograd/grad_mode.hpp"
#include "loss/loss.hpp"
#include "utils.hpp"
#include <stdexcept>

namespace Reg = NN::Reg;

static void test_values() {
  Tensor w = mk({ 2, 2 }, { 3, -4, 0, 2 });

  check_close(Reg::l1(w).item(), 9, FD_TOL, "l1: sum of magnitudes");
  check_close(Reg::l2(w).item(), 14.5, FD_TOL, "l2: half the squared norm");
  check_close(Reg::elastic_net(w, 0.3).item(), 0.3 * 9 + 0.7 * 14.5, FD_TOL, "elastic_net: alpha=0.3");

  check_close(Reg::elastic_net(w, 0).item(), Reg::l2(w).item(), 0, "elastic_net: alpha=0 is l2");
  check_close(Reg::elastic_net(w, 1).item(), Reg::l1(w).item(), 0, "elastic_net: alpha=1 is l1");

  check(Reg::l1(w).p->shape.empty(), "l1: result is rank 0");
  check(Reg::l2(w).p->shape.empty(), "l2: result is rank 0");
  check(Reg::elastic_net(w, 0.3).p->shape.empty(), "elastic_net: result is rank 0");
}

static void test_gradients() {
  // Away from 0, where |w| is differentiable.
  Tensor w = mk_grad({ 5 }, { 0.7, -1.2, 0.05, 2.5, -0.3 });

  check_fd([&] { return Reg::l1(w); }, w, "l1: grad matches finite differences");
  check_fd([&] { return Reg::l2(w); }, w, "l2: grad matches finite differences");

  check_fd([&] { return Reg::elastic_net(w, 0); }, w, "elastic_net(alpha=0): grad matches finite differences");
  check_fd([&] { return Reg::elastic_net(w, 0.3); }, w, "elastic_net(alpha=0.3): grad matches finite differences");
  check_fd([&] { return Reg::elastic_net(w, 1); }, w, "elastic_net(alpha=1): grad matches finite differences");

  // Composed with a loss: exercises scalar add and gradient accumulation.
  Tensor p = mk_grad({ 3 }, { 0.2, 0.8, 0.5 });
  Tensor t = mk({ 3 }, { 0, 1, 1 });
  check_fd([&] { return NN::Loss::mse(p, t) + Reg::l2(p) * dtype(0.1); }, p, "mse + 0.1*l2: grad matches finite differences");

  // Composed through matmul: exercises a non-rank-0 weight.
  Tensor wm = mk_grad({ 2, 2 }, { 0.3, -0.4, 1.1, 0.2 });
  Tensor x = mk({ 1, 2 }, { 1, 2 });
  check_fd([&] { return x.matmul(wm).sum() + Reg::l1(wm) * dtype(0.05); }, wm, "matmul + 0.05*l1: grad matches finite differences");
}

static void test_l1_kink() {
  // `sign(0) = 0` is the chosen subgradient, so the kink contributes nothing.
  Tensor w = mk_grad({ 3 }, { -1, 0, 1 });
  Reg::l1(w).backward();
  check(w.p->grad and w.p->grad->p->data == std::vector<dtype>{ -1, 0, 1 }, "l1: sign(0)=0 at the kink");
}

static void test_autograd_contract() {
  Tensor w = mk_grad({ 3 }, { 1, -2, 3 });

  Tensor r = Reg::l1(w);
  check(r.p->requires_grad and r.p->grad_fn != nullptr, "l1: attaches a grad node");

  Tensor d = Reg::Detail::l1(w);
  check(not d.p->requires_grad and d.p->grad_fn == nullptr, "Detail::l1: no grad node");

  {
    NN::Autograd::NoGradGuard g;
    Tensor q = Reg::l2(w);
    check(not q.p->requires_grad and q.p->grad_fn == nullptr, "l2: no grad node under NoGradGuard");
  }
}

static void test_errors() {
  const Tensor e;

  check_throws<std::invalid_argument>([&] { (void)Reg::l1(e); }, "l1: empty tensor throws");
  check_throws<std::invalid_argument>([&] { (void)Reg::l2(e); }, "l2: empty tensor throws");
  check_throws<std::invalid_argument>([&] { (void)Reg::elastic_net(e, 0.5); }, "elastic_net: empty tensor throws");

  check_throws<std::invalid_argument>([&] { (void)Reg::Detail::l1(e); }, "Detail::l1: empty tensor throws");
  check_throws<std::invalid_argument>([&] { (void)Reg::Detail::l2(e); }, "Detail::l2: empty tensor throws");
  check_throws<std::invalid_argument>([&] { (void)Reg::Detail::elastic_net(e, 0.5); }, "Detail::elastic_net: empty tensor throws");

  Tensor w = mk({ 2 }, { 1, -1 });
  check_throws<std::invalid_argument>([&] { (void)Reg::elastic_net(w, -0.1); }, "elastic_net: alpha below 0 throws");
  check_throws<std::invalid_argument>([&] { (void)Reg::elastic_net(w, 1.1); }, "elastic_net: alpha above 1 throws");
}

int main() {
  test_values();
  test_gradients();
  test_l1_kink();
  test_autograd_contract();
  test_errors();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("regularization: all checks passed");
  return 0;
}
