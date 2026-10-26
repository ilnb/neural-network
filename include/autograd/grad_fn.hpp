#pragma once

#include "tensor/tensor.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace NN::Autograd {
  // A node in the autograd graph.
  //
  // `inputs` holds the tensors consumed by the forward op; the graph is
  // discovered through these handles. `backward(dz)` receives the gradient of
  // the node's output and must accumulate the adjoint of every input that
  // requires grad, normally through `accumulate`.
  //
  // `backward` is called once per `Tensor::backward()` pass. Rules that need
  // the forward output keep a weak reference to it, so they never form a
  // reference cycle; the output is still reachable because its consumers hold
  // it as an input.
  struct GradFn {
    std::vector<std::shared_ptr<Impl>> inputs;
    std::function<void(const Tensor &)> backward;
  };
} // namespace NN::Autograd
