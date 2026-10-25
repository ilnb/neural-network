#include "nn/loss.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace NN::Loss {
  dtype mse(const Tensor &pred, const Tensor &target) {
    Tensor::validate(pred, target);
    if (pred.p->shape != target.p->shape)
      throw std::invalid_argument("Incompatible shape.");

    double sum = 0;
    auto num_el = pred.p->num_el;
    auto &pdata = pred.p->data, &tdata = target.p->data;
    for (size_t i = 0; i < num_el; i++) {
      const double d = static_cast<double>(pdata[i]) - static_cast<double>(tdata[i]);
      sum += d * d;
    }
    return static_cast<dtype>(sum / static_cast<double>(num_el));
  }

  dtype bce(const Tensor &pred, const Tensor &target) {
    Tensor::validate(pred, target);
    if (pred.p->shape != target.p->shape)
      throw std::invalid_argument("Incompatible shape.");

    double sum = 0;
    auto num_el = pred.p->num_el;
    auto &pdata = pred.p->data, &tdata = target.p->data;
    for (size_t i = 0; i < num_el; i++) {
      double y = tdata[i], p = pdata[i];
      if (y != 0 and y != 1)
        throw std::invalid_argument("Malformed tensor data. Entries should be binary.");
      if (p == 0) {
        if (y == 1)
          return std::numeric_limits<dtype>::infinity();
        else
          continue;
      } else if (p == 1) {
        if (y == 0)
          return std::numeric_limits<dtype>::infinity();
        else
          continue;
      }
      sum -= y * std::log(p);
      sum -= (1 - y) * std::log(1 - p);
    }
    return static_cast<dtype>(sum / static_cast<double>(num_el));
  }

  dtype ce(const Tensor &pred, const Tensor &target, size_t dim) {
    Tensor::validate(pred, target);
    if (pred.p->shape != target.p->shape)
      throw std::invalid_argument("Incompatible shape.");
    if (dim >= pred.p->shape.size())
      throw std::out_of_range("Class axis out of range.");

    auto num_el = pred.p->num_el;
    auto &pdata = pred.p->data, &tdata = target.p->data;
    const auto slices = static_cast<double>(num_el) / pred.p->shape[dim];
    double sum = 0;

    for (size_t i = 0; i < num_el; i++) {
      const double y = tdata[i], p = pdata[i];
      if (y < 0 or y > 1)
        throw std::invalid_argument("Malformed tensor data. Targets must lie in [0, 1].");
      if (y == 0)
        continue;
      if (p < 0)
        throw std::invalid_argument("Malformed tensor data. Probabilities must be non-negative.");
      if (p == 0)
        return std::numeric_limits<dtype>::infinity();
      sum -= y * std::log(p);
    }

    return static_cast<dtype>(sum / slices);
  }
} // namespace NN::Loss
