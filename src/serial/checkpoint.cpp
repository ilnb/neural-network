#include "serial/checkpoint.hpp"
#include "tensor/impl.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <format>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static_assert(std::endian::native == std::endian::little);
static_assert(std::numeric_limits<NN::dtype>::is_iec559);

using std::ios;

namespace NN::Serial {
  inline constexpr u64 FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
  inline constexpr u64 FNV_PRIME = 0x100000001b3ULL;

  namespace {
    void fnv(u64 &hash, std::span<const std::byte> b) {
      for (std::byte x : b) {
        hash ^= std::to_integer<u64>(x);
        hash *= FNV_PRIME;
      }
    }

    template <typename T>
    constexpr u32 dtype_label() {
      static_assert(sizeof(T) == 4 or sizeof(T) == 8, "Only float and double are supported.");
      return sizeof(T) == 4 ? 1 : 2;
    }

    // clang-format off
    constexpr bool allowed_name_char(unsigned char c) {
      return (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z')
        or (c >= '0' and c <= '9') or c == '_' or c == '.';
    }
    // clang-format on
  } // namespace

  inline constexpr size_t MAX_NAME = 64;
  inline constexpr size_t MAX_RANK = 8; // NOTE: to be changed, maybe
  inline constexpr std::string_view MAGIC = "NNST";
  inline constexpr u32 FORMAT_VERSION = 1;
  inline constexpr u32 DTYPE_LABEL = dtype_label<dtype>();              // TODO: add quantization
  inline constexpr size_t HEADER_SIZE = MAGIC.size() + 4 * sizeof(u32); // magic, version, dtype, n_params, n_buffers

  void save(const Module &mod, const std::filesystem::path &path) {
    const auto params = mod.named_params(), buffers = mod.named_buffers();
    const size_t n_params = params.size(), n_buffers = buffers.size();
    if (auto max = std::numeric_limits<u32>::max(); n_params > max or n_buffers > max)
      throw std::invalid_argument("Serial::save: too many tensors.");

    std::unordered_set<std::string_view> names;
    auto validate = [&names](const auto &v) {
      for (const auto &[name, t] : v) {
        if (name.empty() or name.size() > MAX_NAME)
          throw std::invalid_argument(std::format("Serial::save: tensor name length {} exceeds MAX_NAME ({}).", name.size(), MAX_NAME));

        for (unsigned char c : name)
          if (not allowed_name_char(c))
            throw std::invalid_argument("Serial::save: invalid tensor name.");

        if (not names.insert(name).second)
          throw std::invalid_argument(std::format("Serial::save: duplicate tensor name '{}'.", name));

        t.validate();
        if (auto s = t.p->shape.size(); s > MAX_RANK)
          throw std::invalid_argument(std::format("Serial::save: tensor rank {} exceeds MAX_RANK ({}).", s, MAX_RANK));

        for (size_t d : t.p->shape) {
          if (d == 0 or d > std::numeric_limits<u32>::max())
            throw std::invalid_argument("Serial::save: invalid tensor dimension.");
        }
      }
    };

    validate(params), validate(buffers);

    auto tmp = path;
    tmp += ".tmp";

    std::error_code ec;
    try {
      std::ofstream f;
      f.exceptions(ios::badbit | ios::failbit);
      try {
        f.open(tmp, ios::binary | ios::trunc);
      } catch (const std::ios_base::failure &) {
        throw std::runtime_error(std::format("Serial::save: not able to open save file {}.", tmp.c_str()));
      }

      u64 hash = FNV_OFFSET_BASIS;

      auto write_raw = [&f](std::span<const std::byte> b) { f.write(reinterpret_cast<const char *>(b.data()), SC<std::streamsize>(b.size())); };

      auto write_bytes = [&f, &hash, &write_raw](std::span<const std::byte> b) {
        fnv(hash, b);
        write_raw(b);
      };

      auto write_t = [&write_bytes]<typename T>(T x)
        requires std::is_arithmetic_v<T>
      {
        const auto bytes = std::bit_cast<std::array<char, sizeof(T)>>(x);
        write_bytes(std::as_bytes(std::span{ bytes }));
      };

      write_bytes(std::as_bytes(std::span{ MAGIC }));
      write_t(FORMAT_VERSION);
      write_t(DTYPE_LABEL);

      write_t(SC<u32>(n_params)), write_t(SC<u32>(n_buffers));

      for (const auto &[name, t] : params) {
        write_t(SC<u32>(name.size()));
        write_bytes(std::as_bytes(std::span{ name }));

        const auto &p = t.p;
        write_t(SC<u32>(p->shape.size()));
        for (size_t d : p->shape)
          write_t(SC<u32>(d));
        write_bytes(std::as_bytes(std::span{ p->data }));
      }

      for (const auto &[name, t] : buffers) {
        write_t(SC<u32>(name.size()));
        write_bytes(std::as_bytes(std::span{ name }));

        const auto &p = t.p;
        write_t(SC<u32>(p->shape.size()));
        for (size_t d : p->shape)
          write_t(SC<u32>(d));
        write_bytes(std::as_bytes(std::span{ p->data }));
      }

      const auto trailer = std::bit_cast<std::array<char, 8>>(hash);
      write_raw(std::as_bytes(std::span{ trailer }));

      f.close();

      std::filesystem::rename(tmp, path, ec);
      if (ec)
        throw std::runtime_error(std::format("Serial::save: unable to write {}: {}.", path.c_str(), ec.message()));
    } catch (const std::ios_base::failure &e) {
      std::filesystem::remove(tmp, ec);
      throw std::runtime_error(std::format("Serial::save: failed to write {}: {}.", tmp.c_str(), e.what()));
    } catch (...) {
      std::filesystem::remove(tmp, ec);
      throw;
    }
  }

  void load(Module &mod, const std::filesystem::path &path) {
    std::ifstream f;
    try {
      f.exceptions(ios::badbit | ios::failbit);
      f.open(path, ios::binary);
    } catch (const std::ios_base::failure &) {
      throw std::runtime_error(std::format("Serial::load: unable to open {}.", path.c_str()));
    }

    f.seekg(0, ios::end);
    const std::streamoff size = f.tellg();
    if (size < SC<std::streamoff>(HEADER_SIZE) + 8)
      throw std::runtime_error("Serial::load: file too small.");

    // Pass 1: verify the trailer over [0, size - 8).
    {
      f.seekg(0, ios::beg);
      u64 hash = FNV_OFFSET_BASIS;
      std::array<std::byte, 64 * 1024> chunk{};
      for (auto left = size - 8; left > 0;) {
        const auto n = std::min<std::streamoff>(left, std::ssize(chunk));
        f.read(reinterpret_cast<char *>(chunk.data()), SC<std::streamsize>(n));
        fnv(hash, std::span{ chunk }.first(SC<size_t>(n)));
        left -= n;
      }

      std::array<char, 8> trailer{};
      f.read(trailer.data(), SC<std::streamsize>(trailer.size()));
      if (std::bit_cast<u64>(trailer) != hash)
        throw std::runtime_error("Serial::load: checksum mismatch.");
    }

    auto read_t = [&f]<typename T>()
      requires std::is_arithmetic_v<T>
    {
      std::array<char, sizeof(T)> b{};
      f.read(b.data(), SC<std::streamsize>(sizeof(T)));
      return std::bit_cast<T>(b);
    };

    // NOTE: the returned view is valid until the next call.
    std::array<char, MAX_NAME> name_buf{};
    auto read_name = [&f, &read_t, &name_buf]() -> std::string_view {
      const auto len = read_t.operator()<u32>();
      if (len == 0 or len > MAX_NAME)
        throw std::runtime_error(std::format("Serial::load: bad name length {}.", len));

      f.read(name_buf.data(), SC<std::streamsize>(len));
      const std::string_view name(name_buf.data(), len);
      for (unsigned char c : name)
        if (not allowed_name_char(c))
          throw std::runtime_error(std::format("Serial::load: invalid character in tensor name '{}'.", name));

      return name;
    };

    f.seekg(0, ios::beg);
    auto params = mod.named_params(), buffers = mod.named_buffers();

    struct Slot {
      Tensor &t;
      bool seen = false;
    };

    std::unordered_map<std::string_view, Slot> index;
    for (auto &[name, t] : params) {
      t.validate();
      if (not index.try_emplace(name, t).second)
        throw std::invalid_argument(std::format("Serial::load: duplicate key '{}'.", name));
    }
    for (auto &[name, t] : buffers) {
      t.validate();
      if (not index.try_emplace(name, t).second)
        throw std::invalid_argument(std::format("Serial::load: duplicate key '{}'.", name));
    }

    {
      std::array<char, 4> magic{};
      f.read(magic.data(), SC<std::streamsize>(magic.size()));
      if (std::string_view(magic.data(), magic.size()) != MAGIC)
        throw std::runtime_error("Serial::load: magic header is incorrect.");
    }

    if (const u32 version = read_t.operator()<u32>(); version != FORMAT_VERSION)
      throw std::runtime_error(std::format("Serial::load: unsupported version {}.", version));

    if (const u32 label = read_t.operator()<u32>(); label != DTYPE_LABEL)
      throw std::runtime_error("Serial::load: the savefile is with a different dtype.");

    if (const u32 n_params = read_t.operator()<u32>(); n_params != params.size())
      throw std::runtime_error("Serial::load: #params dont match.");

    if (const u32 n_buffers = read_t.operator()<u32>(); n_buffers != buffers.size())
      throw std::runtime_error("Serial::load: #buffers dont match.");

    struct Record {
      Slot &slot;
      std::streamoff payload_at;
      std::streamoff bytes;
    };

    // Pass 2: validate metadata only, recording where each payload lives.
    const size_t total = params.size() + buffers.size();
    std::vector<Record> records;
    records.reserve(total);

    for (size_t i = 0; i < total; ++i) {
      if (f.tellg() + SC<std::streamoff>(sizeof(u32)) > size - 8)
        throw std::runtime_error("Serial::load: the savefile has fewer records than the model.");

      const auto name = std::string(read_name());

      auto it = index.find(name);
      if (it == index.end())
        throw std::runtime_error(std::format("Serial::load: unknown tensor name '{}'.", name));

      auto &slot = it->second;
      if (slot.seen)
        throw std::runtime_error(std::format("Serial::load: duplicate record for '{}'.", name));
      slot.seen = true;

      const auto &shape = slot.t.p->shape;
      const auto rank = read_t.operator()<u32>();
      if (rank > MAX_RANK)
        throw std::runtime_error(std::format("Serial::load: rank {} exceeds MAX_RANK ({}).", rank, MAX_RANK));
      if (rank != shape.size())
        throw std::runtime_error(std::format("Serial::load: file rank {} doesn't match model rank {}.", rank, shape.size()));

      for (size_t d = 0; d < rank; ++d)
        if (const auto dim = read_t.operator()<u32>(); dim != shape[d])
          throw std::runtime_error(std::format("Serial::load: file dimension {} doesn't match model dimension {} for tensor {}.", dim, shape[d], name));

      const auto payload_at = f.tellg();
      const auto bytes = SC<std::streamoff>(slot.t.p->data.size() * sizeof(dtype));
      if (payload_at + bytes > size - 8)
        throw std::runtime_error("Serial::load: record payload overruns the file.");

      f.seekg(bytes, ios::cur);
      records.emplace_back(slot, payload_at, bytes);
    }

    if (f.tellg() != size - 8)
      throw std::runtime_error("Serial::load: trailing bytes after the last record.");

    for (const auto &[name, slot] : index)
      if (not slot.seen)
        throw std::runtime_error(std::format("Serial::load: no record for '{}'.", name));

    // Pass 3: apply the payloads now that the whole file is known to be valid.
    for (const auto &[slot, payload_at, bytes] : records) {
      f.seekg(payload_at, ios::beg);
      auto &data = slot.t.p->data;
      f.read(reinterpret_cast<char *>(data.data()), SC<std::streamsize>(bytes));
    }
  }

} // namespace NN::Serial
