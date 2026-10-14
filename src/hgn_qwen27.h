// Qwen3.8-27B Halogen checkpoint semantics. The HGN container version alone
// does NOT identify the quantization: dtype 5/8 differ from Flash-Next.
#pragma once
#include "hgn.h"
#include <algorithm>
#include <fstream>
#include <initializer_list>

namespace qwen27 {
constexpr int hidden = 5120, intermediate = 17408, layers = 64, vocab = 248320;
constexpr int q_heads = 24, kv_heads = 4, head_dim = 256;
constexpr int key_heads = 16, value_heads = 48, linear_dim = 128;

inline bool identifies(const char* path) {
  std::ifstream f(path, std::ios::binary);
  hgn::Header h{};
  if (!f.read(reinterpret_cast<char*>(&h), sizeof h)) return false;
  return !memcmp(h.magic, "HGN1", 4) &&
         std::string(h.model_name, strnlen(h.model_name, 64)).rfind("Qwen3.8-27B", 0) == 0;
}

inline void validate_header(const char* path) {
  std::ifstream f(path, std::ios::binary);
  hgn::Header h{};
  if (!f.read(reinterpret_cast<char*>(&h), sizeof h) || memcmp(h.magic, "HGN1", 4) ||
      h.version != 2 || std::string(h.model_name, strnlen(h.model_name, 64)) != "Qwen3.8-27B-p1-d2")
    throw std::runtime_error("unsupported Qwen27 HGN version or recipe (requires Qwen3.8-27B-p1-d2 v2)");
}

inline void validate(const hgn::Tensor& t) {
  const uint64_t n = t.numel();
  if (t.dtype == 0 && t.qparam == 0 && t.data_size == n * 2) return;
  if (t.ndims == 2 && t.qparam == 0) {
    const uint64_t r = t.dims[0], c = t.dims[1];
    if (t.dtype == 6 && t.data_size == n + 2 * r) return;
    // NVFP4: codebook[16], then per row [C/2 nibbles][C/16 E4M3 scales].
    if (t.dtype == 5 && c % 256 == 0 && t.data_size == 64 + r * (c / 2 + c / 16)) return;
  }
  // MTP/draft NF4: each row has packed nibbles followed by FP16 G32 scales.
  // Its byte count matches NVFP4; qparam is essential to distinguish them.
  if (t.ndims == 2 && t.dtype == 5 && t.qparam == 1 && t.dims[1] % 32 == 0 &&
      t.data_size == 64 + t.dims[0] * (t.dims[1] / 2 + t.dims[1] / 16)) return;
  throw std::runtime_error("Qwen27 unsupported tensor layout: " + t.name +
                           " dtype=" + std::to_string(t.dtype) +
                           " qparam=" + std::to_string(t.qparam));
}

inline float value(const hgn::Tensor& t, uint64_t i) {
  if (t.dtype == 0) return hgn::bf16_to_f32(hgn_v2::u16(t.data + i * 2));
  const uint64_t c = t.dims[1], r = i / c, col = i % c;
  if (t.dtype == 6)
    return hgn::fp8e4m3_to_f32(t.data[i]) *
           hgn::bf16_to_f32(hgn_v2::u16(t.data + t.numel() + r * 2));
  if (t.dtype == 5) {
    const uint8_t* row = t.data + 64 + r * (c / 2 + c / 16);
    const uint8_t nib = (row[col / 2] >> ((col & 1) * 4)) & 15;
    float cb;
    memcpy(&cb, t.data + nib * 4, 4);
    return cb * (t.qparam == 1 ? hgn::fp16_to_f32(hgn_v2::u16(row+c/2+2*(col/32))) :
                                hgn::fp8e4m3_to_f32(row[c / 2 + col / 16]));
  }
  throw std::runtime_error("Qwen27 value called on unsupported dtype");
}

inline std::vector<float> dequant(const hgn::Tensor& t) {
  validate(t);
  std::vector<float> out(t.numel());
  for (uint64_t i = 0; i < t.numel(); ++i) out[i] = value(t, i);
  return out;
}

// Only canonical NVFP4 permits moving the common scale outside a dot product.
// Other valid codebooks still use the FP32 reference path.
inline bool can_factor(const hgn::Tensor& t) {
  if (t.dtype == 0 || t.dtype == 6) return true;
  if (t.dtype != 5 || t.qparam != 0) return false;
  constexpr float fp4[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
  float cb[16];memcpy(cb,t.data,sizeof cb);
  if (!std::isfinite(cb[2]) || cb[2] <= 0) return false;
  for (int i=0;i<16;++i) if (cb[i] != fp4[i]*cb[2]) return false;
  return true;
}

inline const hgn::Tensor& required(const hgn::Checkpoint& ck, const std::string& name,
                                  std::initializer_list<uint64_t> dims) {
  const auto& t = ck.at(name);
  if (t.ndims != dims.size() || !std::equal(dims.begin(), dims.end(), t.dims))
    throw std::runtime_error("Qwen27 wrong tensor shape: " + name);
  validate(t);
  return t;
}
}  // namespace qwen27
