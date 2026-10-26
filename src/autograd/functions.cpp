#include "autograd/functions.hpp"
#include "autograd/grad_fn.hpp"
#include "tensor/tensor.hpp"
#include <functional>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace NN::Autograd {
  Tensor sum_to_shape(const Tensor &g, const std::vector<size_t> &shape) {
    const auto &gs = g.p->shape;
    if (gs == shape)
      return g.clone();

    const size_t n = gs.size();
    if (shape.size() > n)
      throw std::invalid_argument("sum_to_shape: target rank exceeds gradient rank.");
    const size_t off = n - shape.size();
    for (size_t i = 0; i < shape.size(); i++)
      if (shape[i] != 1 and shape[i] != gs[off + i])
        throw std::invalid_argument("sum_to_shape: target doesn't broadcast to gradient shape.");

    Tensor out(shape);

    std::vector<size_t> ostr(n);
    size_t s = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      ostr[off + i] = (shape[i] == 1) ? 0 : s;
      s *= shape[i];
    }

    const auto &in = g.p->data;
    auto &o = out.p->data;
    std::vector<size_t> coord(n);
    size_t oi = 0;
    for (size_t k = 0; k < g.p->num_el; k++) {
      o[oi] += in[k];
      for (size_t a = n; a-- > 0;) {
        coord[a]++;
        oi += ostr[a];
        if (coord[a] < gs[a])
          break;
        oi -= ostr[a] * gs[a];
        coord[a] = 0;
      }
    }
    return out;
  }

  void accumulate(Impl &p, const Tensor &g) {
    if (!g.p)
      throw std::invalid_argument("Empty gradient.");
    if (p.shape != g.p->shape)
      throw std::invalid_argument("Incompatible shape.");
    if (!p.grad)
      p.grad = std::make_shared<Tensor>(g.clone());
    else {
      // In-place op would reject g if it's a graph node.
      auto &pd = p.grad->p->data;
      const auto &gd = g.p->data;
      for (size_t i = 0; i < pd.size(); i++)
        pd[i] += gd[i];
    }
  }

  void backward(const Tensor &root) {
    if (root.empty() or !root.p->requires_grad)
      throw std::logic_error("backward on a tensor that doesnt require grad.");
    if (root.p->num_el != 1)
      throw std::invalid_argument("backward requires a scalar root.");

    std::vector<Impl *> order;
    std::unordered_set<Impl *> seen;
    std::function<void(Impl *)> visit = [&](Impl *p) {
      if (!seen.insert(p).second)
        return;
      if (p->grad_fn)
        for (auto &v : p->grad_fn->inputs)
          visit(v.get());
      order.push_back(p);
    };
    visit(root.p.get());

    Tensor seed(root.p->shape);
    seed.p->data.assign(seed.p->num_el, 1);
    accumulate(*root.p, seed);

    for (auto v : order | std::views::reverse) {
      if (!v->grad_fn)
        continue;
      if (v->grad) {
        if (!v->grad_fn->backward)
          throw std::logic_error("Requires backward rule, not present.");
        v->grad_fn->backward(*v->grad);
      }
      v->grad.reset();
    }
  }
} // namespace NN::Autograd
