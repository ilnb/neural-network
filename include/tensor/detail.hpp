#pragma once

#include "tensor/impl.hpp"
#include "tensor/shape.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <concepts>

// Raw tensor kernels.
//
// Everything here performs the forward computation only: no `GradFn` is
// attached and `requires_grad` is never consulted. The `Tensor` methods are
// the differentiable wrappers; their backward rules call these directly so
// that adjoint computation stays off the graph.
namespace NN::Detail {
  // Returns a new tensor with every element negated.
  //
  // Throws `std::invalid_argument` if `t` is empty.
  Tensor neg(const Tensor &t);

  Tensor sum(const Tensor &t);

  Tensor sum(const Tensor &t, size_t dim);

  Tensor expand(const Tensor &t, size_t dim, size_t size);

  Tensor reshape(const Tensor &t, std::vector<size_t> &&v);

  Tensor permute(const Tensor &t, const std::vector<size_t> &v);

  Tensor transpose(const Tensor &t);

  Tensor cat(const Tensor &t, const Tensor &b, size_t dim);

  Tensor stack(const Tensor &t, const Tensor &b, size_t dim);

  Tensor add(const Tensor &t, const dtype x);

  Tensor mul(const Tensor &t, const dtype x);

  Tensor matmul(const Tensor &t, const Tensor &b);

  // Returns a new tensor of `op(a, b)` with full broadcasting.
  template <typename Op>
    requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
  Tensor binary_broadcast(const Tensor &a, const Tensor &b, Op op) {
    auto out = Detail::broadcast_shape(a.p->shape, b.p->shape);
    const auto out_sz = out.size();

    std::vector<size_t> a_scratch, b_scratch;
    std::span<const size_t> as = a.p->strides;
    std::span<const size_t> bs = b.p->strides;
    if (a.p->shape != out) {
      a_scratch = Detail::aligned_strides(a.p, out);
      as = a_scratch;
    }
    if (b.p->shape != out) {
      b_scratch = Detail::aligned_strides(b.p, out);
      bs = b_scratch;
    }

    Tensor ret(std::move(out));
    for (size_t i = 0; i < ret.p->num_el; i++) {
      size_t a_off = 0, b_off = 0, tmp = i;
      for (size_t k = out_sz; k-- > 0;) {
        const size_t c = tmp % ret.p->shape[k];
        tmp /= ret.p->shape[k];
        a_off += c * as[k];
        b_off += c * bs[k];
      }

      ret.p->data[i] = op(a.p->data[a_off], b.p->data[b_off]);
    }
    return ret;
  }

  // Reduces `dim` with `op`, starting from `init` and removing that axis.
  // `op` is applied left to right, so non-commutative operations are well
  // defined. When `dim` is the only axis, the result is rank 0.
  //
  // Throws `std::domain_error` if `t` is empty, and `std::out_of_range` if
  // `dim >= rank`.
  template <typename Op>
    requires std::invocable<Op, dtype, dtype> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
  Tensor reduce(const Tensor &t, size_t dim, dtype init, Op op) {
    if (t.empty())
      throw std::domain_error("Empty tensor.");
    auto &p = t.p;
    const size_t n = p->shape.size();
    if (dim >= n)
      throw std::out_of_range("Out of bounds axis.");

    std::vector<size_t> new_size;
    new_size.reserve(n - 1);
    for (size_t i = 0; i < n; i++)
      if (i != dim)
        new_size.push_back(p->shape[i]);

    Tensor ret;
    if (new_size.empty())
      ret = Tensor(init);
    else
      ret.set_dims(std::move(new_size), false);
    auto &data = p->data, &rdata = ret.p->data;
    auto &strides = p->strides;
    auto &shape = p->shape, &rshape = ret.p->shape;

    for (size_t out = 0, itr = ret.p->num_el; out < itr; out++) {
      size_t offset = 0, tmp = out;

      for (size_t i = n, j = rshape.size(); i-- > 0;) {
        if (i == dim)
          continue;
        j--;
        const size_t c = tmp % rshape[j];
        tmp /= rshape[j];
        offset += c * strides[i];
      }

      auto val = init;
      for (size_t k = 0; k < shape[dim]; k++)
        val = op(val, data[offset + k * strides[dim]]);
      rdata[out] = val;
    }

    return ret;
  }

  // Reduces `dim` with the comparison `op`, returning the reduced tensor and,
  // for every output element, the index along `dim` that won.
  //
  // Ties keep the first winning index. Throws `std::domain_error` if `t` is
  // empty, and `std::out_of_range` if `dim >= rank`.
  template <typename Op>
    requires std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, bool> and std::convertible_to<std::invoke_result_t<Op, dtype, dtype>, dtype>
  std::pair<Tensor, std::vector<size_t>> cmp(const Tensor &t, size_t dim, Op op) {
    if (t.empty())
      throw std::domain_error("Empty tensor.");
    auto &p = t.p;
    const size_t n = p->shape.size();
    if (dim >= n)
      throw std::out_of_range("Out of bounds axis.");

    std::vector<size_t> new_size;
    new_size.reserve(n - 1);
    for (size_t i = 0; i < n; i++)
      if (i != dim)
        new_size.push_back(p->shape[i]);

    Tensor ret;
    if (new_size.empty())
      ret = Tensor(p->data[0]);
    else
      ret.set_dims(std::move(new_size), false);
    std::vector<size_t> indices(ret.p->num_el);

    auto &data = p->data, &rdata = ret.p->data;
    auto &strides = p->strides;
    auto &shape = p->shape, &rshape = ret.p->shape;

    for (size_t out = 0, itr = ret.p->num_el; out < itr; out++) {
      size_t offset = 0, tmp = out;

      for (size_t i = n, j = rshape.size(); i-- > 0;) {
        if (i == dim)
          continue;
        j--;
        const size_t c = tmp % rshape[j];
        tmp /= rshape[j];
        offset += c * strides[i];
      }

      auto val = data[offset];
      size_t idx = 0;
      for (size_t k = 1; k < shape[dim]; k++) {
        if (auto nval = data[offset + k * strides[dim]]; op(nval, val)) {
          val = nval;
          idx = k;
        }
      }
      rdata[out] = val;
      indices[out] = idx;
    }

    return {std::move(ret), std::move(indices)};
  }

  // Applies `op(data[i], b[offset])` in place, with `b` broadcast into `*this`.
  // Checks if `b` can be broadcasted into `this`; no shape change.
  template <typename Op>
    requires requires(Op op, dtype x) {
      { op(x, x) } -> std::convertible_to<dtype>;
    }
  Tensor &binary_inplace(Tensor &a, const Tensor &b, Op op) {
    std::vector<size_t> scratch;
    std::span<const size_t> bs = b.p->strides;
    if (b.p->shape != a.p->shape) {
      scratch = Detail::aligned_strides(b.p, a.p->shape);
      bs = scratch;
    }

    auto &ashape = a.p->shape;
    auto &adata = a.p->data, &bdata = b.p->data;
    for (size_t i = 0, n = a.p->num_el; i < n; i++) {
      size_t off = 0, tmp = i;
      for (size_t k = ashape.size(); k-- > 0;) {
        const size_t c = tmp % ashape[k];
        tmp /= ashape[k];
        off += c * bs[k];
      }

      adata[i] = op(adata[i], bdata[off]);
    }
    return a;
  }

  // Returns a copy of `t` with `op(v, x)` applied to every element.
  //
  // Throws `std::invalid_argument` if `t` is empty.
  template <typename Op>
    requires requires(Op op, dtype x) {
      { op(x, x) } -> std::convertible_to<dtype>;
    }
  Tensor binary_scalar(const Tensor &t, const dtype x, Op op) {
    auto r = t.clone();
    std::ranges::for_each(r.p->data, [&](dtype &v) { v = op(v, x); });
    return r;
  }

  // Applies `op(v, x)` to every element of `t` in-place and returns `t`.
  // No autograd checks are performed here; callers must have validated that
  // `t` is mutable.
  template <typename Op>
    requires requires(Op op, dtype x) {
      { op(x, x) } -> std::convertible_to<dtype>;
    }
  Tensor &binary_inplace(Tensor &t, const dtype x, Op op) {
    std::ranges::for_each(t.p->data, [&](dtype &v) { v = op(v, x); });
    return t;
  }
} // namespace NN::Detail
