#include "activations/activations.hpp"
#include "loss/loss.hpp"
#include "tensor/tensor.hpp"
#include "utils.hpp"
#include <cmath>
#include <initializer_list>
#include <print>
#include <stdexcept>
#include <vector>

namespace Act = NN::Act;
namespace Loss = NN::Loss;

// Checks the accumulated gradient of `t` against `want`.
static void check_grad(const Tensor &t, std::initializer_list<dtype> want, const char *what, dtype tol = 1e-5f) {
  if (!t.p->grad) {
    std::println(stderr, "FAIL: {} (grad is null)", what);
    failures++;
    return;
  }
  const std::vector<dtype> w(want);
  const auto &g = t.p->grad->p->data;

  if (t.p->grad->p->shape != t.p->shape) {
    std::println(stderr, "FAIL: {} (grad shape differs from tensor shape)", what);
    failures++;
    return;
  }
  if (g.size() != w.size()) {
    std::println(stderr, "FAIL: {} (grad has {} elements, want {})", what, g.size(), w.size());
    failures++;
    return;
  }
  for (size_t i = 0; i < w.size(); i++) {
    if (not(std::abs(g[i] - w[i]) <= tol)) {
      std::println(stderr, "FAIL: {} (grad[{}] is {}, want {})", what, i, g[i], w[i]);
      failures++;
      return;
    }
  }
}

static void test_chain() {
  Tensor x = mk_grad({ 2 }, { 1, 2 });
  (x * 3 + 1).sum().backward();
  check_grad(x, { 3, 3 }, "chain: y = sum(3x + 1)");
}

static void test_seed_and_roots() {
  {
    Tensor s(2.0f);
    s.set_requires_grad(true);
    s.backward();
    check_grad(s, { 1 }, "backward: leaf root seeds 1");
  }
  {
    Tensor x = mk_grad({ 1, 1 }, { 4 });
    (x * 2).backward();
    check_grad(x, { 2 }, "backward: rank-1 single-element root");
  }
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    check_throws<std::invalid_argument>([&] { (x * 2).backward(); }, "backward: non-scalar root throws");
  }
  {
    Tensor x(1.0f);
    check_throws<std::logic_error>([&] { x.backward(); }, "backward: root without requires_grad throws");
  }
  {
    check_throws<std::logic_error>([&] { Tensor().backward(); }, "backward: empty tensor throws");
  }
}

static void test_accumulation() {
  {
    // Several losses over a shared subgraph must be summed into one root
    // before backward, since the graph is released by the pass.
    Tensor x = mk_grad({ 2 }, { 1, 1 });
    auto u = x * 2;
    (u.sum() + (u * 3).sum()).backward(); // d/dx = 2 + 6
    check_grad(x, { 8, 8 }, "accumulation: summed losses share the subgraph");
  }
  {
    // The graph is released after a pass: a second backward seeds the root
    // again but cannot reach the leaves.
    Tensor x = mk_grad({ 2 }, { 1, 1 });
    auto y = (x * 2 + x).sum(); // d/dx = 3
    y.backward();
    check_grad(x, { 3, 3 }, "accumulation: first pass");
    y.backward();
    check_grad(x, { 3, 3 }, "accumulation: second pass propagates nothing");
  }
  {
    // The same tensor used twice in one expression.
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    (x * x + x).sum().backward(); // d/dx = 2x + 1
    check_grad(x, { 3, 5 }, "accumulation: x*x + x");
  }
}

static void test_broadcast() {
  {
    Tensor a = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    Tensor b = mk_grad({ 3 }, { 10, 20, 30 });
    (a + b).sum().backward();
    check_grad(a, { 1, 1, 1, 1, 1, 1 }, "broadcast add: a grad");
    check_grad(b, { 2, 2, 2 }, "broadcast add: b grad is summed over rows");
  }
  {
    Tensor a = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    Tensor b = mk_grad({ 3 }, { 10, 20, 30 });
    (a * b).sum().backward();
    check_grad(a, { 10, 20, 30, 10, 20, 30 }, "broadcast mul: a grad");
    check_grad(b, { 5, 7, 9 }, "broadcast mul: b grad is summed over rows");
  }
}

static void test_signs() {
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    (-x).sum().backward();
    check_grad(x, { -1, -1 }, "sign: unary minus");
  }
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    (5 - x).sum().backward();
    check_grad(x, { -1, -1 }, "sign: dtype - tensor");
  }
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    (x - 5).sum().backward();
    check_grad(x, { 1, 1 }, "sign: tensor - dtype");
  }
  {
    Tensor a = mk_grad({ 2 }, { 1, 2 });
    Tensor b = mk_grad({ 2 }, { 3, 4 });
    (a - b).sum().backward();
    check_grad(a, { 1, 1 }, "sign: a - b, a grad");
    check_grad(b, { -1, -1 }, "sign: a - b, b grad");
  }
}

static void test_transforms() {
  {
    Tensor x = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    x.reshape(3, 2).sum().backward();
    check_grad(x, { 1, 1, 1, 1, 1, 1 }, "reshape: identity adjoint");
  }
  {
    Tensor x = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    Tensor w = mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
    (x.permute(1, 0) * w).sum().backward();
    check_grad(x, { 1, 3, 5, 2, 4, 6 }, "permute: inverse permutation");
  }
  {
    Tensor x = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    Tensor w = mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
    (x.transpose() * w).sum().backward();
    check_grad(x, { 1, 3, 5, 2, 4, 6 }, "transpose: transpose adjoint");
  }
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    Tensor w = mk({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
    (x.expand(0, 3) * w).sum().backward();
    check_grad(x, { 9, 12 }, "expand: sums the inserted axis");
  }
}

static void test_cat_stack() {
  {
    Tensor a = mk_grad({ 2, 2 }, { 1, 2, 3, 4 });
    Tensor b = mk_grad({ 2, 3 }, { 5, 6, 7, 8, 9, 10 });
    Tensor w = mk({ 2, 5 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    (a.cat(b, 1) * w).sum().backward();
    check_grad(a, { 1, 2, 6, 7 }, "cat: a grad");
    check_grad(b, { 3, 4, 5, 8, 9, 10 }, "cat: b grad");
  }
  {
    Tensor a = mk_grad({ 2, 2 }, { 1, 2, 3, 4 });
    Tensor b = mk_grad({ 2, 2 }, { 5, 6, 7, 8 });
    Tensor w = mk({ 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8 });
    (a.stack(b, 1) * w).sum().backward();
    check_grad(a, { 1, 2, 5, 6 }, "stack: a grad");
    check_grad(b, { 3, 4, 7, 8 }, "stack: b grad");
  }
  {
    Tensor a = mk_grad({ 2 }, { 1, 2 });
    Tensor b = mk_grad({ 2 }, { 3, 4 });
    (a.stack(b, 1)).sum().backward();
    check_grad(a, { 1, 1 }, "stack(rank): a grad");
    check_grad(b, { 1, 1 }, "stack(rank): b grad");
  }
}

static void test_reductions() {
  {
    Tensor x = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    x.sum(1).sum().backward();
    check_grad(x, { 1, 1, 1, 1, 1, 1 }, "sum(dim): ones");
  }
  {
    Tensor x = mk_grad({ 2, 2 }, { 1, 2, 3, 4 });
    x.mean().backward();
    check_grad(x, { 0.25f, 0.25f, 0.25f, 0.25f }, "mean: whole tensor");
  }
  {
    Tensor x = mk_grad({ 2, 2 }, { 1, 2, 3, 4 });
    x.mean(0).sum().backward();
    check_grad(x, { 0.5f, 0.5f, 0.5f, 0.5f }, "mean(dim): each element gets 1/n");
  }
  {
    Tensor x = mk_grad({ 2, 3 }, { 1, 5, 2, 4, 3, 0 });
    Tensor w = mk({ 2 }, { 10, 20 });
    (x.max(1) * w).sum().backward();
    check_grad(x, { 0, 10, 0, 20, 0, 0 }, "max(dim): routes to the argmax");
  }
  {
    Tensor x = mk_grad({ 2, 3 }, { 1, 5, 2, 4, 0, 6 });
    Tensor w = mk({ 2 }, { 10, 20 });
    (x.min(1) * w).sum().backward();
    check_grad(x, { 10, 0, 0, 0, 20, 0 }, "min(dim): routes to the argmin");
  }
  {
    Tensor x = mk_grad({ 3 }, { 1, 5, 2 });
    x.max().backward();
    check_grad(x, { 0, 1, 0 }, "max: routes to the maximum");
  }
  {
    Tensor x = mk_grad({ 3 }, { 1, 5, 2 });
    x.min().backward();
    check_grad(x, { 1, 0, 0 }, "min: routes to the minimum");
  }
}

static void test_matmul() {
  Tensor a = mk_grad({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
  Tensor b = mk_grad({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
  a.matmul(b).sum().backward();
  check_grad(a, { 3, 7, 11, 3, 7, 11 }, "matmul: a grad");
  check_grad(b, { 5, 5, 7, 7, 9, 9 }, "matmul: b grad");
}

static void test_activations() {
  {
    Tensor x = mk_grad({ 3 }, { -1, 0, 2 });
    Tensor w = mk({ 3 }, { 1, 1, 3 });
    (Act::relu(x) * w).sum().backward();
    check_grad(x, { 0, 0, 3 }, "relu: zero gradient at and below 0");
  }
  {
    Tensor x = mk_grad({ 1 }, { 0 });
    Act::sigmoid(x).sum().backward();
    check_grad(x, { 0.25f }, "sigmoid: s'(0) = 0.25");
  }
  {
    Tensor x = mk_grad({ 1, 3 }, { 1, 2, 3 });
    auto p = Act::softmax(x, 1);
    auto q = Act::softmax(x, 1);
    (p * q).sum().backward();
    check_grad(x, { -0.075718f, -0.130105f, 0.205823f }, "softmax: shared-output backward", 1e-5f);
  }
  {
    Tensor x = mk_grad({ 1, 3 }, { 1, 2, 3 });
    Tensor w = mk({ 1, 3 }, { 1, 2, 3 });
    (Act::log_softmax(x, 1) * w).sum().backward();
    // d/dx_i = w_i - softmax_i * sum(w)
    check_grad(x, { 0.4598166f, 0.5316292f, -0.9914458f }, "log_softmax: weighted backward", 1e-5f);
  }
}

static void test_rank0_activations() {
  {
    Tensor x(1.0f);
    x.set_requires_grad(true);
    const auto y = Act::softmax(x, 0);
    check(y.p->shape.empty(), "softmax rank 0: result stays rank 0");
    y.backward();
    check_grad(x, { 0 }, "softmax rank 0: grad is 0");
  }
  {
    Tensor x(2.5f);
    x.set_requires_grad(true);
    const auto y = Act::log_softmax(x, 0);
    check(y.p->shape.empty(), "log_softmax rank 0: result stays rank 0");
    y.backward();
    check_grad(x, { 0 }, "log_softmax rank 0: grad is 0");
  }
  {
    Tensor x(1.0f);
    check_throws<std::out_of_range>([&] { (void)Act::softmax(x, 1); }, "softmax rank 0: dim 1 throws");
    check_throws<std::out_of_range>([&] { (void)Act::log_softmax(x, 1); }, "log_softmax rank 0: dim 1 throws");
  }
}

static void test_loss_targets() {
  {
    Tensor p = mk_grad({ 2 }, { 1, 2 });
    Tensor t = mk_grad({ 2 }, { 1, 1 });
    Loss::mse(p, t).backward();
    check_grad(p, { 0, 1 }, "mse target grad: pred grad unchanged");
    check_grad(t, { 0, -1 }, "mse target grad: negated pred grad");
  }
  {
    Tensor p = mk_grad({ 2 }, { 0.5f, 0.9f });
    Tensor t = mk_grad({ 2 }, { 0, 1 });
    Loss::bce(p, t).backward();
    check_grad(t, { 0, -1.0986123f }, "bce target grad: log((1-p)/p)/n", 1e-5f);
  }
  {
    Tensor p = mk_grad({ 2, 3 }, { 0.09003057f, 0.24472847f, 0.66524096f, 0.3333333f, 0.3333333f, 0.3333333f });
    Tensor t = mk_grad({ 2, 3 }, { 0, 0, 1, 1, 0, 0 });
    Loss::ce(p, t).backward();
    check_grad(t, { 1.203803f, 0.703803f, 0.203803f, 0.5493062f, 0.5493062f, 0.5493062f }, "ce target grad: -log(p)/slices", 1e-5f);
  }
  {
    Tensor p = mk({ 1 }, { 0.5f });
    Tensor t = mk_grad({ 1 }, { 1 });
    Loss::mse(p, t).backward();
    check(not p.p->grad, "mse target-only: pred without requires_grad gets no grad");
    check_grad(t, { 1 }, "mse target-only: target still gets its grad");
  }
}

static void test_losses() {
  {
    Tensor p = mk_grad({ 2 }, { 1, 2 });
    Tensor t = mk({ 2 }, { 1, 1 });
    auto loss = Loss::mse(p, t);
    check_close(loss.item(), 0.5f, 1e-6f, "mse: value");
    loss.backward();
    check_grad(p, { 0, 1 }, "mse: grad is 2(p-t)/n");
  }
  {
    Tensor p = mk_grad({ 2 }, { 0.5f, 0.9f });
    Tensor t = mk({ 2 }, { 0, 1 });
    auto loss = Loss::bce(p, t);
    check_close(loss.item(), 0.399254f, 1e-6f, "bce: value");
    loss.backward();
    check_grad(p, { 1, -0.5555556f }, "bce: grad is (p-t)/(n p (1-p))");
  }
  {
    Tensor p = mk_grad({ 2, 3 }, { 0.09003057f, 0.24472847f, 0.66524096f, 0.3333333f, 0.3333333f, 0.3333333f });
    Tensor t = mk({ 2, 3 }, { 0, 0, 1, 1, 0, 0 });
    auto loss = Loss::ce(p, t);
    check_close(loss.item(), 0.7531091f, 1e-6f, "ce: value");
    loss.backward();
    check_grad(p, { 0, 0, -0.751615f, -1.5f, 0, 0 }, "ce: grad is -t/(C p) averaged over slices", 1e-5f);
  }
}

static void test_raw() {
  Tensor a = mk_grad({ 2 }, { 2, 3 });
  Tensor b = mk_grad({ 2 }, { 4, 5 });
  const auto r = NN::Raw::mul(a, b);

  check(not r.p->requires_grad and not r.p->grad_fn, "raw: result carries no graph");
  check(not a.p->grad and not b.p->grad, "raw: forward leaves inputs without gradients");

  auto loss = r.sum();
  check_throws<std::logic_error>([&] { loss.backward(); }, "raw: result is not differentiable");
}

static void test_leaf_and_mutation_guards() {
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    Tensor c = x.clone();
    check(not c.p->requires_grad and not c.p->grad_fn, "clone: result is a leaf");
  }
  {
    Tensor x = mk_grad({ 2 }, { 1, 2 });
    check_throws<std::logic_error>([&] { x.set_data(3, 4); }, "set_data: rejects autograd tensor");
    check_throws<std::logic_error>([&] { x.add_(1); }, "in-place: rejects autograd tensor");
    check_throws<std::logic_error>([&] { x += 1; }, "in-place operator: rejects autograd tensor");
  }
}

int main() {
  test_chain();
  test_seed_and_roots();
  test_accumulation();
  test_broadcast();
  test_signs();
  test_transforms();
  test_cat_stack();
  test_reductions();
  test_matmul();
  test_activations();
  test_rank0_activations();
  test_losses();
  test_loss_targets();
  test_raw();
  test_leaf_and_mutation_guards();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("autograd: all checks passed");
  return 0;
}
