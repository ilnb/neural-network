#pragma once

// Shared testing utilities.

#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <print>
#include <utility>
#include <vector>

using NN::dtype, NN::Tensor;

// Number of failed checks in the current test binary.
inline int failures = 0;

inline void check(bool ok, const char *what) {
  if (not ok) {
    std::println(stderr, "FAIL: {}", what);
    failures++;
  }
}

inline void check_close(dtype got, dtype want, dtype tol, const char *what) {
  if (not(std::abs(got - want) <= tol)) {
    std::println(stderr, "FAIL: {} (got {}, want {})", what, got, want);
    failures++;
  }
}

template <typename Ex, typename F>
inline void check_throws(F f, const char *what) {
  try {
    f();
    std::println(stderr, "FAIL: {} (no exception)", what);
    failures++;
  } catch (const Ex &) {
  } catch (const std::exception &e) {
    std::println(stderr, "FAIL: {} (wrong exception: {})", what, e.what());
    failures++;
  }
}

inline Tensor mk(std::vector<size_t> shape, std::vector<dtype> vals) {
  Tensor t;
  t.set_dims(std::move(shape), false);
  t.set_data(vals);
  return t;
}

inline Tensor mk_grad(std::vector<size_t> shape, std::vector<dtype> vals) {
  Tensor t = mk(std::move(shape), std::move(vals));
  t.set_requires_grad(true);
  return t;
}

// Central-difference step, about `cbrt(epsilon)` for the active dtype.
inline constexpr dtype FD_EPS = sizeof(dtype) == 8 ? dtype(6.1e-6) : dtype(4.9e-3);

// Relative error that still counts as agreement.
inline constexpr dtype FD_TOL = sizeof(dtype) == 8 ? dtype(1e-6) : dtype(1e-2);

// Floor of the relative-error denominator, so near-zero gradients don't
// produce meaningless ratios.
inline constexpr dtype FD_FLOOR = std::numeric_limits<dtype>::epsilon();

// Compares the analytic gradient of `f` w.r.t. every element of `w` against a
// central difference `(f(w + e) - f(w - e)) / 2e`. `f` must return a rank-0
// tensor that depends on `w`.
//
// The analytic pass runs before any perturbation, and gradients are cleared
// afterwards, since leaves accumulate across backward calls. Elements are
// perturbed through `p->data` because `set_data` rejects autograd tensors.
inline void check_fd(const std::function<Tensor()> &f, Tensor &w, const char *what) {
  w.p->grad.reset();
  f().backward();
  if (!w.p->grad) {
    std::println(stderr, "FAIL: {} (grad is null)", what);
    failures++;
    return;
  }
  const std::vector<dtype> ana = w.p->grad->p->data;
  w.p->grad.reset();

  for (size_t i = 0; i < w.p->num_el; i++) {
    const dtype orig = w.p->data[i];
    w.p->data[i] = orig + FD_EPS;
    const dtype plus = f().item();
    w.p->data[i] = orig - FD_EPS;
    const dtype minus = f().item();
    w.p->data[i] = orig;

    const dtype num = (plus - minus) / (2 * FD_EPS);
    const dtype rel = std::abs(ana[i] - num) / std::max(FD_FLOOR, std::abs(ana[i]) + std::abs(num));
    if (rel > FD_TOL) {
      std::println(stderr, "FAIL: {} (element {}: analytic {}, numeric {}, rel {})", what, i, ana[i], num, rel);
      failures++;
      return;
    }
  }
}
