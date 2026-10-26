#pragma once

namespace NN::Autograd {
  inline thread_local bool grad_enabled = true;

  struct NoGradGuard {
    bool prev;
    NoGradGuard() : prev(grad_enabled) { grad_enabled = false; }
    ~NoGradGuard() { grad_enabled = prev; }
    NoGradGuard(const NoGradGuard &) = delete;
    NoGradGuard &operator=(const NoGradGuard &) = delete;
  };
} // namespace NN::Autograd
