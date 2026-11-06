#include "data/dataloader.hpp"
#include "data/detail.hpp"
#include "data/subset.hpp"
#include "data/tensor_dataset.hpp"
#include "utils.hpp"
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Data = NN::Data;

static void check_shape(const Tensor &t, std::initializer_list<size_t> want, const char *what) {
  const std::vector<size_t> w(want);
  if (t.p->shape != w) {
    std::println(stderr, "FAIL: {} (shape is {}, want {})", what, t.p->shape, w);
    failures++;
  }
}

static void check_data(const Tensor &t, std::initializer_list<dtype> want, const char *what) {
  const std::vector<dtype> w(want);
  if (t.p->data != w) {
    std::println(stderr, "FAIL: {} (data is {}, want {})", what, t.p->data, w);
    failures++;
  }
}

static Data::TensorDataset make_ds() { return Data::TensorDataset(mk({ 5, 2 }, { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5 }), mk({ 5, 1 }, { 0, 1, 2, 3, 4 })); }

// Concatenates every sample of one epoch, in batch order.
static std::vector<dtype> epoch_values(Data::DataLoader &dl) {
  std::vector<dtype> out;
  for (auto [a, b] : dl.epoch())
    out.insert(out.end(), a.p->data.begin(), a.p->data.end());
  return out;
}

static void test_dataset_indexing() {
  auto ds = make_ds();
  check(ds.size() == 5, "TensorDataset: size");

  auto [a, b] = ds[2];
  check_shape(a, { 2 }, "TensorDataset: sample shape");
  check_shape(b, { 1 }, "TensorDataset: target shape");
  check_data(a, { 2, 3 }, "TensorDataset: sample values");
  check_data(b, { 2 }, "TensorDataset: target values");

  auto [first_a, first_b] = ds[0];
  check_data(first_a, { 0, 1 }, "TensorDataset: first sample");
  check_data(first_b, { 0 }, "TensorDataset: first target");

  check_throws<std::out_of_range>([&] { (void)ds[5]; }, "TensorDataset: index out of range");

  // A rank-1 dataset yields scalar samples.
  Data::TensorDataset flat(mk({ 4 }, { 1, 2, 3, 4 }), mk({ 4 }, { 0, 1, 0, 1 }));
  auto [fa, fb] = flat[1];
  check(fa.p->shape.empty(), "TensorDataset: rank-1 sample is rank 0");
  check_data(fa, { 2 }, "TensorDataset: rank-1 sample value");
  check_data(fb, { 1 }, "TensorDataset: rank-1 target value");
}

static void test_dataset_iterator() {
  auto ds = make_ds();
  size_t i = 0;
  for (auto [a, b] : ds) {
    check_data(a, { dtype(i), dtype(i + 1) }, "Dataset: iterator sample order");
    check_data(b, { dtype(i) }, "Dataset: iterator target order");
    i++;
  }
  check(i == 5, "Dataset: iterator visits every sample");
}

static void test_stack() {
  Tensor s = Data::Detail::stack({ mk({ 2 }, { 1, 2 }), mk({ 2 }, { 3, 4 }), mk({ 2 }, { 5, 6 }) });
  check_shape(s, { 3, 2 }, "stack: adds a leading axis");
  check_data(s, { 1, 2, 3, 4, 5, 6 }, "stack: row-major values");

  Tensor s0 = Data::Detail::stack({ mk({ 1 }, { 1 }).sum(), mk({ 1 }, { 2 }).sum() });
  check_shape(s0, { 2 }, "stack: rank-0 inputs give a rank-1 result");
  check_data(s0, { 1, 2 }, "stack: rank-0 values");

  check_throws<std::invalid_argument>([] { (void)Data::Detail::stack({}); }, "stack: empty input throws");
  check_throws<std::invalid_argument>([] { (void)Data::Detail::stack({ mk({ 2 }, { 1, 2 }), mk({ 3 }, { 1, 2, 3 }) }); }, "stack: shape mismatch throws");
}

static void test_batches() {
  auto ds = make_ds();
  Data::DataLoader dl(ds, { .batch_size = 2 });

  std::vector<size_t> sizes;
  std::vector<dtype> values;
  for (auto [a, b] : dl.epoch()) {
    const size_t n = a.p->shape[0];
    check_shape(a, { n, 2 }, "DataLoader: batch sample shape");
    check_shape(b, { n, 1 }, "DataLoader: batch target shape");
    sizes.push_back(n);
    values.insert(values.end(), a.p->data.begin(), a.p->data.end());
  }

  check(sizes == std::vector<size_t>{ 2, 2, 1 }, "DataLoader: last batch is partial");
  check(values == std::vector<dtype>{ 0, 1, 1, 2, 2, 3, 3, 4, 4, 5 }, "DataLoader: batches preserve order");
}

static void test_drop_last() {
  auto ds = make_ds();
  Data::DataLoader dl(ds, { .batch_size = 2, .drop_last = true });

  size_t batches = 0, samples = 0;
  for (auto [a, b] : dl.epoch()) {
    batches++;
    samples += a.p->shape[0];
  }

  check(batches == 2, "DataLoader: drop_last removes the partial batch");
  check(samples == 4, "DataLoader: drop_last keeps only complete batches");
}

static void test_default_config() {
  auto ds = make_ds();
  Data::DataLoader dl(ds);

  size_t batches = 0;
  for (auto [a, b] : dl.epoch()) {
    batches++;
    check_shape(a, { 5, 2 }, "DataLoader: default batch covers the dataset");
  }
  check(batches == 1, "DataLoader: default config yields one batch");
}

static void test_shuffle() {
  Data::TensorDataset ds(mk({ 8, 1 }, { 0, 1, 2, 3, 4, 5, 6, 7 }), mk({ 8, 1 }, { 0, 1, 2, 3, 4, 5, 6, 7 }));

  Data::DataLoader a(ds, { .batch_size = 3, .shuf = true, .seed = 7 });
  Data::DataLoader b(ds, { .batch_size = 3, .shuf = true, .seed = 7 });

  const auto first = epoch_values(a);
  check(first == epoch_values(b), "DataLoader: same seed reproduces the order");

  const auto second = epoch_values(a);
  check(first != second, "DataLoader: consecutive epochs reshuffle");

  auto sorted = first;
  std::ranges::sort(sorted);
  check(sorted == std::vector<dtype>{ 0, 1, 2, 3, 4, 5, 6, 7 }, "DataLoader: shuffle is a permutation");

  Data::DataLoader c(ds, { .batch_size = 8 });
  check(epoch_values(c) == std::vector<dtype>{ 0, 1, 2, 3, 4, 5, 6, 7 }, "DataLoader: shuf=false keeps the order");
}

// Uses the default `Dataset::batch`: rank-2 samples and rank-0 targets.
struct Custom : Data::Dataset {
  size_t size() const override { return 4; }

  std::pair<Tensor, Tensor> operator[](size_t i) const override { return { mk({ 2, 2 }, { dtype(i), dtype(i), dtype(i), dtype(i) }), mk({ 1 }, { dtype(i) }).sum() }; }
};

static void test_generic_dataset() {
  Custom c;
  Data::DataLoader dl(std::make_shared<const Custom>(c), { .batch_size = 3 });

  size_t batches = 0;
  for (auto [a, b] : dl.epoch()) {
    batches++;
    if (batches == 1) {
      check_shape(a, { 3, 2, 2 }, "generic batch: sample shape is stacked");
      check_shape(b, { 3 }, "generic batch: rank-0 targets gain a leading axis");
      check_data(b, { 0, 1, 2 }, "generic batch: target values are stacked in order");
    }
  }
  check(batches == 2, "generic batch: ceil(4/3) batches");

  auto [a, b] = static_cast<const Data::Dataset &>(c).batch(std::vector<size_t>{ 3, 1 });
  check_shape(a, { 2, 2, 2 }, "generic batch: explicit index list");
  check_data(b, { 3, 1 }, "generic batch: explicit index order");
}

static void test_loader_owns_dataset() {
  auto p = std::make_shared<Data::TensorDataset>(make_ds());
  Data::DataLoader dl(p, { .batch_size = 5 });
  p.reset();

  size_t n = 0;
  for (auto [a, b] : dl.epoch())
    n += a.p->shape[0];
  check(n == 5, "DataLoader: keeps the dataset alive");
}

static void test_errors() {
  auto ds = make_ds();

  check_throws<std::invalid_argument>([&] { (void)Data::DataLoader(ds, { .batch_size = 0 }); }, "DataLoader: zero batch_size throws");
  check_throws<std::invalid_argument>([&] { (void)Data::DataLoader(ds, { .batch_size = 6, .drop_last = true }); }, "DataLoader: drop_last larger than the dataset throws");
  check_throws<std::invalid_argument>([&] { (void)ds.batch({}); }, "batch: empty index list throws");
  check_throws<std::invalid_argument>([&] { (void)static_cast<const Data::Dataset &>(ds).batch({}); }, "Dataset::batch: empty index list throws");

  const Tensor empty;
  Tensor scalar = mk({ 1 }, { 1 }).sum();
  check_throws<std::invalid_argument>([&] { (void)Data::TensorDataset(empty, empty); }, "TensorDataset: empty tensors throw");
  check_throws<std::invalid_argument>([&] { (void)Data::TensorDataset(scalar, scalar); }, "TensorDataset: rank-0 throws");
  check_throws<std::invalid_argument>([] { (void)Data::TensorDataset(mk({ 3, 2 }, { 0, 0, 0, 0, 0, 0 }), mk({ 2, 1 }, { 0, 0 })); },
                                      "TensorDataset: mismatched sample axis throws");
}

static void test_subset() {
  auto ds = make_ds();
  auto p = std::make_shared<const Data::TensorDataset>(ds);
  Data::Subset s(p, { 4, 0, 2 });

  check(s.size() == 3, "Subset: size");
  check(s.ds == p, "Subset: keeps the parent handle");

  auto [a, b] = s[1];
  check_data(a, { 0, 1 }, "Subset: operator[] maps through idx");
  check_data(b, { 0 }, "Subset: operator[] target maps through idx");

  auto [ba, bb] = s.batch(std::vector<size_t>{ 2, 0 });
  check_shape(ba, { 2, 2 }, "Subset: batch sample shape");
  check_data(ba, { 2, 3, 4, 5 }, "Subset: batch preserves the requested order");
  check_data(bb, { 2, 4 }, "Subset: batch preserves target order");

  // Nested subsets compose their mappings.
  Data::Subset s2(std::make_shared<const Data::Subset>(s), { 2, 0 });
  check(s2.size() == 2, "Subset: nested size");
  auto [n0, n0t] = s2[0];
  auto [n1, n1t] = s2[1];
  check_data(n0, { 2, 3 }, "Subset: nested index 0");
  check_data(n1, { 4, 5 }, "Subset: nested index 1");
  check_data(n0t, { 2 }, "Subset: nested target 0");

  // The DataLoader convenience constructor copies the subset.
  Data::DataLoader dl(s, { .batch_size = 2 });
  std::vector<dtype> vals;
  for (auto [x, y] : dl.epoch())
    vals.insert(vals.end(), x.p->data.begin(), x.p->data.end());
  check(vals == std::vector<dtype>{ 4, 5, 0, 1, 2, 3 }, "Subset: DataLoader batches in subset order");

  // A subset keeps its parent alive.
  auto q = std::make_shared<Data::TensorDataset>(make_ds());
  Data::Subset owned(q, { 0 });
  q.reset();
  check_data(owned[0].first, { 0, 1 }, "Subset: keeps the parent alive");

  check_throws<std::invalid_argument>([] { (void)Data::Subset(std::shared_ptr<const Data::Dataset>(), {}); }, "Subset: null parent throws");
  check_throws<std::out_of_range>([&] { (void)Data::Subset(p, { 5 }); }, "Subset: index >= parent size throws");
  check_throws<std::out_of_range>([&] { (void)s[3]; }, "Subset: operator[] out of range throws");
  check_throws<std::out_of_range>([&] { (void)s.batch(std::vector<size_t>{ 3 }); }, "Subset: batch position out of range throws");
  check_throws<std::invalid_argument>([&] { (void)s.batch({}); }, "Subset: empty batch throws");
}

int main() {
  test_dataset_indexing();
  test_dataset_iterator();
  test_stack();
  test_batches();
  test_drop_last();
  test_default_config();
  test_shuffle();
  test_generic_dataset();
  test_loader_owns_dataset();
  test_subset();
  test_errors();

  if (failures != 0) {
    std::println(stderr, "{} check(s) failed", failures);
    return 1;
  }

  std::println("data: all checks passed");
  return 0;
}
