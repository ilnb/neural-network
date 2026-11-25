#pragma once

#include "tensor/tensor.hpp"

#include <string>
#include <vector>

namespace NN {
  // One named tensor in a module's state.
  struct NamedTensor {
    std::string name;
    Tensor t;
  };

  // Shared base for modules.
  struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor &x) = 0;
    virtual void train(bool mode = true) { training = mode; }
    void eval() { train(false); }

    // Trainable tensors, keyed by a stable name.
    // Default: no params.
    virtual std::vector<NamedTensor> named_params() const = 0;

    // Non-trainable state that must survive a checkpoint.
    // Default: none. Never passed to an optimizer.
    virtual std::vector<NamedTensor> named_buffers() const = 0;

    // Flattened named_params, for optimizers.
    std::vector<Tensor> params() const {
      std::vector<Tensor> out;
      for (auto &[_, t] : named_params())
        out.push_back(t);
      return out;
    }

  protected:
    bool training = true;
  };

  struct ParamOnly : Module {
    std::vector<NamedTensor> named_buffers() const final { return {}; }
  };

  struct Stateless : ParamOnly {
    std::vector<NamedTensor> named_params() const final { return {}; }
  };
} // namespace NN
