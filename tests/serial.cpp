// Checkpoint format: round trip, key enumeration, and the loader's validation
// surface. Corrupt files are forged by editing a known-good checkpoint and
// recomputing the trailer, so each test exercises one check at a time.
#include "module/batchnorm2d.hpp"
#include "module/conv2d.hpp"
#include "module/flatten.hpp"
#include "module/linear.hpp"
#include "module/maxpool2d.hpp"
#include "module/relu.hpp"
#include "module/sequential.hpp"
#include "serial/checkpoint.hpp"
#include "utils.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

using namespace NN;
using Bytes = std::vector<char>;

namespace fs = std::filesystem;

static fs::path dir;

static fs::path tmp(const char *name) { return dir / name; }

static Bytes slurp(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  return Bytes(std::istreambuf_iterator<char>(f), {});
}

static void spit(const fs::path &p, const Bytes &b) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(b.data(), SC<std::streamsize>(b.size()));
}

static u64 fnv(const char *d, size_t n) {
  u64 h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= SC<unsigned char>(d[i]);
    h *= 0x100000001b3ULL;
  }
  return h;
}

// Recomputes the trailer after an edit, so the test reaches the intended check
// instead of stopping at the checksum.
static void refix(Bytes &b) {
  const u64 h = fnv(b.data(), b.size() - 8);
  std::memcpy(b.data() + b.size() - 8, &h, 8);
}

static u32 get32(const Bytes &b, size_t off) {
  u32 v;
  std::memcpy(&v, b.data() + off, 4);
  return v;
}

static void put32(Bytes &b, size_t off, u32 v) { std::memcpy(b.data() + off, &v, 4); }

// Byte ranges of one record, located by walking the header.
struct Rec {
  size_t name, rank, dims, payload, end;
};

static Rec parse(const Bytes &b, size_t off) {
  const size_t len = get32(b, off);
  const size_t rank = get32(b, off + 4 + len);
  size_t n = 1;
  for (size_t i = 0; i < rank; i++)
    n *= get32(b, off + 4 + len + 4 + 4 * i);

  Rec r{};
  r.name = off + 4;
  r.rank = off + 4 + len;
  r.dims = off + 4 + len + 4;
  r.payload = r.dims + 4 * rank;
  r.end = r.payload + n * sizeof(dtype);
  return r;
}

static std::vector<dtype> values(const Module &m) {
  std::vector<dtype> v;
  for (const auto &nt : m.named_params())
    v.insert(v.end(), nt.t.p->data.begin(), nt.t.p->data.end());
  for (const auto &nt : m.named_buffers())
    v.insert(v.end(), nt.t.p->data.begin(), nt.t.p->data.end());
  return v;
}

static std::vector<const void *> handles(const Module &m) {
  std::vector<const void *> v;
  for (const auto &nt : m.named_params())
    v.push_back(nt.t.p.get());
  for (const auto &nt : m.named_buffers())
    v.push_back(nt.t.p.get());
  return v;
}

// A checkpoint the test can edit without guessing offsets: two params and one
// buffer, with every value distinct.
struct Toy : Module {
  Tensor w, b, rv;

  explicit Toy(const std::vector<size_t> &wshape = { 4, 3 }) : w(Tensor(wshape)), b(Tensor(std::vector<size_t>{ 3 })), rv(Tensor(std::vector<size_t>{ 3 })) {
    dtype x = 1;
    for (auto &d : w.p->data)
      d = x++;
    for (auto &d : b.p->data)
      d = -x++;
    for (auto &d : rv.p->data)
      d = x++ * dtype(0.5);
  }

  Tensor forward(const Tensor &x) override { return x; }
  std::vector<NamedTensor> named_params() const override { return { { "w", w }, { "b", b } }; }
  std::vector<NamedTensor> named_buffers() const override { return { { "running_var", rv } }; }
};

static void expect_error(const char *what, const char *needle, const std::string &got) {
  if (got.find(needle) == std::string::npos) {
    std::println(stderr, "FAIL: {} (got '{}')", what, got);
    failures++;
  }
}

// Loads `b` into a fresh Toy, requires the message to contain `needle`, and
// requires the model to be untouched afterwards.
static void expect_reject(const Bytes &b, const char *what, const char *needle) {
  spit(tmp("bad.nnst"), b);

  Toy m;
  const auto before = values(m);
  std::string got;
  try {
    Serial::load(m, tmp("bad.nnst"));
    got = "<no throw>";
  } catch (const std::exception &e) {
    got = e.what();
  }

  expect_error(what, needle, got);
  check(values(m) == before, what);
}

static void test_round_trip() {
  Toy a;
  Serial::save(a, tmp("toy.nnst"));

  Toy b;
  for (auto &nt : b.named_params())
    for (auto &d : nt.t.p->data)
      d = 0;
  const auto hs = handles(b);

  Serial::load(b, tmp("toy.nnst"));
  check(values(a) == values(b), "round trip restores every value");
  check(handles(b) == hs, "round trip keeps tensor handles");
  check(b.w.p->data == a.w.p->data, "round trip restores parameters");
  check(b.rv.p->data == a.rv.p->data, "round trip restores buffers");

  Serial::save(b, tmp("toy2.nnst"));
  check(slurp(tmp("toy.nnst")) == slurp(tmp("toy2.nnst")), "save -> load -> save is byte-identical");
}

static void test_requires_grad() {
  Toy a;
  Serial::save(a, tmp("toy.nnst"));

  Toy b;
  b.w.set_requires_grad(true);
  Serial::load(b, tmp("toy.nnst"));
  check(b.w.p->requires_grad, "load leaves requires_grad alone");
  check(not b.b.p->requires_grad, "load does not add requires_grad");
}

static void test_key_set() {
  Sequential model(
      // clang-format off
      Conv2d(1, 32, 3, { .padding = 1, .seed = 0 }),
      BatchNorm2d(32),
      MaxPool2d(2),
      ReLU(),
      Conv2d(32, 64, 3, { .padding = 1, .seed = 0 }),
      BatchNorm2d(64),
      MaxPool2d(2),
      ReLU(),
      Conv2d(64, 64, 3, { .padding = 1, .seed = 0 }),
      BatchNorm2d(64),
      MaxPool2d(2),
      ReLU(),
      Flatten(),
      Linear(64 * 3 * 3, 10, 0)
      // clang-format on
  );

  const std::vector<std::string> want{
    "0.w",
    "0.b",
    "1.gamma",
    "1.beta",
    "4.w",
    "4.b",
    "5.gamma",
    "5.beta",
    "8.w",
    "8.b",
    "9.gamma",
    "9.beta",
    "13.w",
    "13.b",
    "1.running_mean",
    "1.running_var",
    "5.running_mean",
    "5.running_var",
    "9.running_mean",
    "9.running_var",
  };

  std::vector<std::string> got;
  for (const auto &nt : model.named_params())
    got.push_back(nt.name);
  for (const auto &nt : model.named_buffers())
    got.push_back(nt.name);

  check(got == want, "mnist_cnn key set and order");

  Serial::save(model, tmp("cnn.nnst"));

  Sequential copy(
      // clang-format off
      Conv2d(1, 32, 3, { .padding = 1, .seed = 1 }),
      BatchNorm2d(32),
      MaxPool2d(2),
      ReLU(),
      Conv2d(32, 64, 3, { .padding = 1, .seed = 1 }),
      BatchNorm2d(64),
      MaxPool2d(2),
      ReLU(),
      Conv2d(64, 64, 3, { .padding = 1, .seed = 1 }),
      BatchNorm2d(64),
      MaxPool2d(2),
      ReLU(),
      Flatten(),
      Linear(64 * 3 * 3, 10, 1)
      // clang-format on
  );
  Serial::load(copy, tmp("cnn.nnst"));
  check(values(model) == values(copy), "mnist_cnn checkpoint round trips");
}

// A module with no state at all: only the header and the trailer are written.
static void test_empty_module() {
  ReLU m;
  Serial::save(m, tmp("empty.nnst"));
  check(fs::file_size(tmp("empty.nnst")) == 28, "empty model writes 28 bytes");

  ReLU n;
  bool ok = true;
  try {
    Serial::load(n, tmp("empty.nnst"));
  } catch (const std::exception &e) {
    std::println(stderr, "FAIL: empty model loads ({})", e.what());
    failures++;
    ok = false;
  }
  check(ok, "empty model loads");
}

// Rank 0 is a scalar and is written without a dims array.
struct Scalar : ParamOnly {
  Tensor s{ dtype(2.5) };

  Tensor forward(const Tensor &x) override { return x; }
  std::vector<NamedTensor> named_params() const override { return { { "s", s } }; }
};

static void test_scalar() {
  Scalar a;
  a.s.p->data[0] = dtype(2.5);
  Serial::save(a, tmp("scalar.nnst"));

  Scalar b;
  b.s.p->data[0] = 0;
  Serial::load(b, tmp("scalar.nnst"));
  check_close(b.s.p->data[0], dtype(2.5), dtype(0), "rank 0 scalar round trips");
  check(b.s.p->shape.empty(), "rank 0 keeps an empty shape");
}

static void test_order_insensitive() {
  Toy a;
  Serial::save(a, tmp("toy.nnst"));
  Bytes b = slurp(tmp("toy.nnst"));

  const Rec r0 = parse(b, 20), r1 = parse(b, r0.end), r2 = parse(b, r1.end);

  // Reorder the trailing records: 'running_var' before 'b'.
  Bytes swapped(b.begin(), b.begin() + SC<std::ptrdiff_t>(r0.end));
  swapped.insert(swapped.end(), b.begin() + SC<std::ptrdiff_t>(r2.name - 4), b.begin() + SC<std::ptrdiff_t>(r2.end));
  swapped.insert(swapped.end(), b.begin() + SC<std::ptrdiff_t>(r1.name - 4), b.begin() + SC<std::ptrdiff_t>(r1.end));
  swapped.insert(swapped.end(), b.end() - 8, b.end());
  refix(swapped);

  Toy t;
  spit(tmp("swapped.nnst"), swapped);
  Serial::load(t, tmp("swapped.nnst"));
  check(values(t) == values(a), "records may appear in any order");
}

static void test_header() {
  Toy a;
  Serial::save(a, tmp("toy.nnst"));
  const Bytes orig = slurp(tmp("toy.nnst"));

  Bytes b = orig;
  b[0] = 'X';
  refix(b);
  expect_reject(b, "bad magic", "magic");

  b = orig;
  put32(b, 4, 2);
  refix(b);
  expect_reject(b, "unsupported version", "unsupported version");

  b = orig;
  put32(b, 8, get32(b, 8) == 1 ? 2 : 1);
  refix(b);
  expect_reject(b, "dtype mismatch", "different dtype");

  b = orig;
  put32(b, 12, get32(b, 12) + 1);
  refix(b);
  expect_reject(b, "param count mismatch", "#params");

  b = orig;
  put32(b, 16, get32(b, 16) + 1);
  refix(b);
  expect_reject(b, "buffer count mismatch", "#buffers");

  b.assign(4, 0);
  expect_reject(b, "too small", "file too small");
}

static void test_records() {
  Toy a;
  Serial::save(a, tmp("toy.nnst"));
  const Bytes orig = slurp(tmp("toy.nnst"));
  const Rec r0 = parse(orig, 20), r1 = parse(orig, r0.end), r2 = parse(orig, r1.end);
  check(r2.end == orig.size() - 8, "records end exactly at the trailer");

  Bytes b;

  b = orig; // 'b' renamed to 'w'
  b[r1.name] = 'w';
  refix(b);
  expect_reject(b, "duplicate record", "duplicate record");

  b = orig;
  b[r1.name] = 'z';
  refix(b);
  expect_reject(b, "unknown name", "unknown tensor name");

  b = orig;
  b[r1.name] = '/';
  refix(b);
  expect_reject(b, "invalid name character", "invalid character");

  b = orig;
  put32(b, r1.name - 4, 0);
  refix(b);
  expect_reject(b, "zero name length", "bad name length");

  b = orig;
  put32(b, r1.name - 4, 65); // MAX_NAME + 1
  refix(b);
  expect_reject(b, "name too long", "bad name length");

  b = orig;
  put32(b, r0.rank, 9);
  refix(b);
  expect_reject(b, "rank above MAX_RANK", "MAX_RANK");

  b = orig;
  put32(b, r0.rank, 1);
  refix(b);
  expect_reject(b, "rank mismatch", "model rank");

  b = orig;
  put32(b, r0.dims, 5);
  refix(b);
  expect_reject(b, "dimension mismatch", "model dimension");

  b = orig;
  b.erase(b.begin() + SC<std::ptrdiff_t>(r2.name - 4), b.begin() + SC<std::ptrdiff_t>(r2.end));
  refix(b);
  expect_reject(b, "missing record", "fewer records");

  b = orig;
  b.insert(b.begin() + SC<std::ptrdiff_t>(r2.end), 0);
  refix(b);
  expect_reject(b, "trailing bytes", "trailing bytes");

  b = orig;
  b.erase(b.begin() + SC<std::ptrdiff_t>(r1.name - 4), b.begin() + SC<std::ptrdiff_t>(r2.end));
  b.insert(b.begin() + SC<std::ptrdiff_t>(r1.name - 4), orig.begin() + 20, orig.begin() + SC<std::ptrdiff_t>(r1.name - 4));
  refix(b);
  expect_reject(b, "duplicate and missing", "duplicate record");
}

static void test_checksum() {
  Toy a;
  Serial::save(a, tmp("toy.nnst"));
  const Bytes orig = slurp(tmp("toy.nnst"));
  const Rec r0 = parse(orig, 20);

  Bytes b = orig;
  b.pop_back();
  expect_reject(b, "truncated", "checksum mismatch");

  b = orig;
  b[r0.payload] ^= 1;
  expect_reject(b, "flipped payload byte", "checksum mismatch");

  b = orig;
  b[3] ^= 1;
  expect_reject(b, "flipped header byte", "checksum mismatch");
}

// The model side: names, shapes and emptiness are checked before the file is
// opened, so a model that cannot be saved is reported as a model error.
struct LongName : ParamOnly {
  Tensor t = Tensor(std::vector<size_t>{ 1 });

  Tensor forward(const Tensor &x) override { return x; }
  std::vector<NamedTensor> named_params() const override { return { { std::string(65, 'a'), t } }; }
};

struct DuplicateNames : ParamOnly {
  Tensor t = Tensor(std::vector<size_t>{ 1 });

  Tensor forward(const Tensor &x) override { return x; }
  std::vector<NamedTensor> named_params() const override { return { { "t", t }, { "t", t } }; }
};

struct EmptyTarget : ParamOnly {
  Tensor t; // p == nullptr

  Tensor forward(const Tensor &x) override { return x; }
  std::vector<NamedTensor> named_params() const override { return { { "t", t } }; }
};

struct OneParam : ParamOnly {
  Tensor t = Tensor(std::vector<size_t>{ 1 });

  Tensor forward(const Tensor &x) override { return x; }
  std::vector<NamedTensor> named_params() const override { return { { "t", t } }; }
};

static void test_model_errors() {
  LongName ln;
  check_throws<std::invalid_argument>([&] { Serial::save(ln, tmp("long.nnst")); }, "name over 64 bytes rejected");

  // A failed save must not damage an existing checkpoint.
  Toy keep;
  Serial::save(keep, tmp("keep.nnst"));
  const Bytes before = slurp(tmp("keep.nnst"));
  check_throws<std::invalid_argument>([&] { Serial::save(ln, tmp("keep.nnst")); }, "invalid model rejected on save");
  check(slurp(tmp("keep.nnst")) == before, "failed save leaves the previous checkpoint intact");

  OneParam one;
  Serial::save(one, tmp("one.nnst"));

  DuplicateNames dn;
  check_throws<std::invalid_argument>([&] { Serial::save(dn, tmp("dup.nnst")); }, "duplicate names rejected on save");
  check_throws<std::invalid_argument>([&] { Serial::load(dn, tmp("one.nnst")); }, "duplicate names rejected on load");

  EmptyTarget et;
  check_throws<std::invalid_argument>([&] { Serial::load(et, tmp("one.nnst")); }, "empty target tensor rejected");

  Toy a;
  check_throws<std::runtime_error>([&] { Serial::load(a, tmp("does-not-exist.nnst")); }, "missing file rejected");
}

static void test_save_errors() {
  Toy a;
  const auto missing_dir = dir / "no-such-dir" / "x.nnst";
  check_throws<std::runtime_error>([&] { Serial::save(a, missing_dir); }, "unwritable path rejected");
  check(not fs::exists(missing_dir), "failed save leaves no file");
  check(not fs::exists(fs::path(missing_dir.string() + ".tmp")), "failed save leaves no .tmp");
}

int main() {
  dir = fs::temp_directory_path() / "nnst_test";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  test_round_trip();
  test_requires_grad();
  test_key_set();
  test_empty_module();
  test_scalar();
  test_order_insensitive();
  test_header();
  test_records();
  test_checksum();
  test_model_errors();
  test_save_errors();

  fs::remove_all(dir, ec);

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("serial: all checks passed");
  return 0;
}
