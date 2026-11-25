#pragma once

#include "module/base.hpp"

#include <filesystem>

namespace NN::Serial {
  // Saves the model's parameters and buffers to `path`.
  //
  // Writes every tensor from `Module::named_params()` followed by every tensor
  // from `Module::named_buffers()`, keyed by the names the module reports. For
  // a `Sequential`, those keys are `<slot>.<local name>`, such as `8.w`.
  //
  // The file is written under `path + ".tmp"` and renamed into place, so a
  // failed save leaves an existing checkpoint untouched. All tensor state is
  // validated before the file is opened: non-empty storage, a name of 1..64
  // bytes from `[A-Za-z0-9_.]` that is unique across parameters and buffers,
  // rank at most 8 (rank 0 is a scalar), and each dimension in `1..u32::max()`.
  // The file ends with a 64-bit FNV-1a checksum over every preceding byte.
  //
  // Throws `std::invalid_argument` if the model fails those checks, and
  // `std::runtime_error` if the file cannot be opened or written.
  void save(const Module &mod, const std::filesystem::path &path);

  // Restores the model's parameters and buffers from `path`.
  //
  // The model must already have the same keys, shapes and build dtype as the
  // checkpoint. Records are matched by name, so their order in the file does not
  // matter, but every key must appear exactly once. Nothing is created, resized
  // or converted.
  //
  // The checksum, header, keys, shapes and payload bounds are all verified
  // before any tensor is modified, so a failed load leaves the model exactly as
  // it was. Only element values are copied; shapes, strides, `requires_grad` and
  // gradients are left untouched.
  //
  // Throws `std::invalid_argument` if the model cannot hold the checkpoint (for
  // example two of its tensors share a name), and `std::runtime_error` for a
  // missing, corrupt, truncated or incompatible file.
  void load(Module &mod, const std::filesystem::path &path);
} // namespace NN::Serial
