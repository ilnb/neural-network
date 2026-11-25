# Checkpoints

`NN::Serial` saves and restores the state of a model to a single binary file
with the `.nnst` extension. A checkpoint holds every **parameter** and every
**buffer** of the model. It does not hold the optimizer.

> **Status:** `save` and `load` are implemented. The format and guarantees below
> are what this build writes and accepts.

## What is saved

| state | in the file | examples |
|---|---|---|
| parameters | yes | conv and linear weights, biases, batchnorm `gamma` / `beta` |
| buffers | yes | batchnorm `running_mean` / `running_var` |
| optimizer state | no | Adam moments and step counter (planned, separate file) |
| RNG, dataloader position, train/eval mode | no | |

Parameters are the tensors an optimizer updates. Buffers belong to the model's
state but are never optimized. Both have to survive a restart, which is why
both are in the file.

Tensors are identified by a **positional path**: `Sequential` prepends each
child's slot number, so a layer's keys stay the same as long as the number of
layers before it does not change. Reordering layers, or inserting one, shifts
every key after it. For the MNIST CNN demo (`demos/mnist_cnn.cpp`) the keys look
like this:

```text
0.w   0.b   1.gamma   1.beta   1.running_mean   1.running_var
4.w   4.b   5.gamma   5.beta   5.running_mean   5.running_var
8.w   8.b   9.gamma   9.beta   9.running_mean   9.running_var
13.w  13.b
```

The gaps are the slots taken by layers without state (`MaxPool2d`, `ReLU`,
`Flatten`, `Softmax`).

## Usage

```cpp
#include "serial/checkpoint.hpp"

NN::Serial::save(model, "mnist.nnst");   // write a checkpoint
NN::Serial::load(model, "mnist.nnst");   // restore into an existing model
```

`load` writes into the model you already built. It never creates, replaces or
reshapes tensors, so every handle an optimizer captured at construction stays
valid.

Both functions accept any `Module`, not just `Sequential`, so a single layer
can be saved on its own:

```cpp
NN::Serial::save(conv, "conv.nnst");     // keys "w" and "b"
```

The planned demo flags are:

```sh
./demo_mnist_cnn --save mnist.nnst                 # train, then write
./demo_mnist_cnn --load mnist.nnst --epochs 0      # restore, evaluate
```

Loading requires the same architecture: the same set of keys with the same
shapes, and the same build dtype. Nothing is converted or partially applied.

## Guarantees

- **Save is atomic.** The file is written to a temporary name and renamed into
  place. An interrupted or failed save leaves an existing checkpoint untouched;
  a half-written file is never visible under the real name.
- **Load is all-or-nothing.** The file and the model are fully validated before
  any tensor is modified. If loading fails for any reason, the model is exactly
  as it was.
- **Tensor identity is preserved.** Only element values are copied. Shapes,
  strides, `requires_grad` and gradients are not touched.
- **Matching is strict.** Every key the model reports must appear exactly once,
  and nothing else may appear. A missing key, an unknown key, a duplicate key, a
  shape mismatch or trailing bytes before the trailer is an error. The loader
  does not guess, skip or default.
- **Record order does not matter.** Records are matched by name, not by
  position, so a file whose records are shuffled still loads into the same
  model.
- **Round trips are stable.** `save` → `load` → `save` produces byte-identical
  files.
- **Corruption is detected.** Every file ends with a checksum, verified before
  anything is parsed, so a truncated or edited file fails with one clear
  message instead of loading garbage.

## Errors

| exception | raised when |
|---|---|
| `std::invalid_argument` | the model is wrong: a tensor with no storage, a name outside the allowed charset or longer than 64 bytes, a duplicate name, or a rank above 8 |
| `std::runtime_error` | the file is wrong: I/O error, bad magic, unsupported version, dtype mismatch, truncation, checksum failure, missing / unknown / duplicate record, shape mismatch, or trailing bytes |

File-side messages are prefixed `Serial::save:` or `Serial::load:` so the
failing side is obvious. Errors raised by a model's own `Tensor::validate()`
keep its `validate:` prefix. Representative messages (keys and numbers vary):

```text
Serial::load: checksum mismatch.
Serial::load: unsupported version 2.
Serial::load: the savefile is with a different dtype.
Serial::load: the savefile has fewer records than the model.
Serial::load: duplicate record for '8.w'.
Serial::load: file rank 3 doesn't match model rank 2.
Serial::load: file dimension 64 doesn't match model dimension 32.
Serial::load: trailing bytes after the last record.
Serial::save: duplicate tensor name '8.w'.
```

The path appears only in the open and write failures (`unable to open`,
`not able to open save file`, `unable to write`, `failed to write`); the other
messages identify the offending record instead.

## File format

The format is intentionally small: a fixed header, one record per tensor, a
trailer with a checksum. All integers are little-endian, and tensor data is raw
IEEE-754 in the build's `dtype`. There is no compression and no padding.

### Header (20 bytes)

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `"NNST"` |
| 4 | 4 | version, `1` |
| 8 | 4 | dtype tag, `1` = float32, `2` = float64 |
| 12 | 4 | number of parameter records |
| 16 | 4 | number of buffer records |

### Records

One record per tensor, parameters first, then buffers:

```text
u32     name_len          1 .. 64
bytes   name[name_len]    ASCII [A-Za-z0-9_.]
u32     rank              0 .. 8
u32     dims[rank]        omitted when rank is 0; each >= 1 otherwise
dtype   data[...]         product of the dims
```

The element count is the product of the dims, so it is not stored separately.
An empty product is 1, which is how a rank-0 tensor (a scalar) is written: no
dims, one value.

A reader matches records against the model's keys and requires the record
stream to end exactly at the trailer, so there is no padding between the last
record and the checksum. What the file describes is never trusted over the
model: the recorded rank and dims must equal the model's shape, and the payload
length is taken from the model, not computed from the file.

### Trailer

The last 8 bytes are a 64-bit FNV-1a hash of every preceding byte. It detects
truncation and accidental edits; it is not a signature and provides no
tamper protection.

A model with no state produces a 28-byte file: the 20-byte header plus the
8-byte trailer. The MNIST CNN demo's 20 tensors (62,154 elements) make a
497,686-byte checkpoint in a `double` build.

## Compatibility

- **Checkpoints are build-specific.** A `float` build writes float32 values, a
  `double` build writes float64, and neither can read the other's files. The
  dtype is recorded in the header so the mismatch is reported instead of being
  silently reinterpreted.
- **Versions are exact.** This build reads version 1 and rejects anything else
  by name. Older files are never guessed at.
- **Keys are positional.** Reordering two layers of the same shape cannot be
  detected from the file. Inserting a layer shifts the keys after it, which
  fails the matching check (unknown or missing key) rather than loading the
  wrong weights.
- **No cross-framework loading.** PyTorch, safetensors and GGUF files are not
  read, and `.nnst` is not consumed by any other tool.

## Not included

- Optimizer state (Adam moments, step counter). Planned as a separate file, so
  weights and training state can be versioned independently.
- RNG state, dataloader position and train/eval mode. A reloaded model resumes
  with its weights and buffers, not with the exact position of a run.
- Quantization, mixed precision and dtype conversion.
