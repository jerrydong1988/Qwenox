// The Python driver extracts prefill_batch from production, without HIP.
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#define CK(x) (void)(x)
[[maybe_unused]] static int g_str;
[[maybe_unused]] static int hipStreamSynchronize(int) { return 0; }
namespace ple_prefetch {
struct Hash {
  std::array<uint64_t, 3> mult{};
  std::array<int, 16> sizes{};
  std::array<uint64_t, 16> offsets{};
  int pad = 0;
  uint64_t table = 0;
  unsigned row_bytes = 0;
};
static int chunk_size(int remaining, int batch, int cap) {
  return std::min(remaining, std::min(batch, cap));
}
}
// Maintained-only prefetch/diagnostic interfaces are host stubs. The Python
// driver still includes the entire production method without rewriting it.
struct PleLookahead {
  template<class M> PleLookahead(M&, const std::vector<int>&) {}
};
struct PleWinPrefetch {
  bool setup(int, const std::vector<int>&, const std::vector<int>&,
             const ple_prefetch::Hash&) { return false; }
  void select(int, int, int) {}
};
[[maybe_unused]] static unsigned g_ple_rb = 160;
[[maybe_unused]] static bool ple_uring_ready() { return false; }
[[maybe_unused]] static struct { int d = 2560, branches = 4; } g_cfg;
static std::atomic<bool> g_borrow_light{false};
struct ReqCancelled : std::runtime_error { using std::runtime_error::runtime_error; };
struct Model {
  int maxbatch = 2, maxbatch_cap = 2, conc_rmax_ = 8, pf_end = 0, heavy_rem_ = 0;
  int prog_total = 6, chunks = 0;
  bool prof = false, mtp_tap_capture = false;
  void* d_mtap = nullptr;
  int maxctx = 1024, pad_id = 0, mtp_tail_begin = 0, nslots = 1;
  long long prog_req = 0;
  bool ple_on = false, spec_snap = false, spec_fp32_active = false;
  bool ngram_verify_active = false, mrope_active_ = false;
  void* d_Rb = nullptr;
  uint64_t ple_mult[3]{}, ple_voff[16]{};
  int ple_vsz[16]{};
  std::vector<int> chist;
  struct Table { void* data = nullptr; } table;
  Table* ple_table = &table;
  static bool env_zero(const char*) { return true; }
  static bool ple_lookahead_enabled() { return false; }
  static const char* ple_lookahead_ineligible(int, int, bool, bool) { return "test stub"; }
  static bool prefill_skip_intermediate_head_enabled() { return false; }
  static bool mtp_tap_norm_fused_enabled() { return false; }
  static bool request_timing_enabled() { return false; }
  static bool batch_bf16() { return false; }
  struct { bool on = false; int cap_mode = 0; } df2;
  void (*borrow_skip_fn)(Model&) = nullptr;
  bool (*heavy_gate_fn)(Model&, int) = nullptr;
  std::function<void()> on_chunk, on_yield;
  // Present only in the old source. The test driver models gen_one's old
  // arm/disarm operations to reproduce another slot mutating the shared model.
#if !CALL_LOCAL_CANCEL
  std::atomic<bool>* abort_flag = nullptr;
#endif
  void yield_point() { if (on_yield) on_yield(); }
  int prefill_chunk(const std::vector<int>&, int, bool, bool,
                    const std::vector<int>*, const std::vector<int>*) {
    ++chunks; if (on_chunk) on_chunk(); return 42;
  }
  int prefill_chunk(const std::vector<int>&, int, bool, bool, bool,
                    PleLookahead*, bool, PleWinPrefetch*) {
    ++chunks; if (on_chunk) on_chunk(); return 42;
  }
  void df2_inject_chunk(int, int) {}
  void mtp_ingest_b(void*, const std::vector<int>&, int, int) {}
  void mtp_ingest_b(void*, const std::vector<int>&, int, int, const uint16_t*, bool) {}
#include "prefill_batch_under_test.inc"
  int ingest(const std::atomic<bool>& flag) {
#if CALL_LOCAL_CANCEL
    return prefill_batch({1, 2, 3, 4, 5, 6}, 0, true, false, &flag);
#else
    abort_flag = const_cast<std::atomic<bool>*>(&flag);
    return prefill_batch({1, 2, 3, 4, 5, 6});
#endif
  }
};
static int failures = 0;
static void check(bool ok, const char* label) {
  printf("%s: %s\n", ok ? "PASS" : "FAIL", label);
  failures += !ok;
}
static bool cancelled(const std::function<void()>& f) {
  try { f(); return false; } catch (const ReqCancelled&) { return true; }
}
int main() {
  {
    Model m; std::atomic<bool> a{false};
    check(!cancelled([&] { m.ingest(a); }) && m.chunks == 3, "serial ingest");
  }
  {
    Model m; std::atomic<bool> a{true};
    check(cancelled([&] { m.ingest(a); }) && m.chunks == 0, "pre-cancel skips all chunks");
  }
  {
    Model m; std::atomic<bool> a{false}; m.on_chunk = [&] { a = true; };
    check(cancelled([&] { m.ingest(a); }) && m.chunks == 1, "cancel at completed chunk boundary");
  }
  {
    Model m; std::atomic<bool> a{false}; [[maybe_unused]] std::atomic<bool> b{true};
    m.on_yield = [&] {
#if !CALL_LOCAL_CANCEL
      m.abort_flag = &b;
#endif
    };
    check(!cancelled([&] { m.ingest(a); }) && m.chunks == 3,
          "cancelled peer cannot abort this request");
  }
  {
    Model m; std::atomic<bool> a{false};
    m.on_yield = [&] {
      a = true;
#if !CALL_LOCAL_CANCEL
      m.abort_flag = nullptr;
#endif
    };
    check(cancelled([&] { m.ingest(a); }) && m.chunks == 1,
          "peer disarm cannot hide this request's cancellation after yield");
  }
  {
    Model m; [[maybe_unused]] std::atomic<bool> peer{true};
#if !CALL_LOCAL_CANCEL
    m.abort_flag = &peer;
#endif
    check(!cancelled([&] { m.prefill_batch({1, 2, 3, 4}, 0, false, true); }) && m.chunks == 2,
          "default-null verifier cannot inherit peer cancellation");
  }
  {
    Model m; std::atomic<bool> a{false};
    m.on_yield = [&] { m.prog_total = 6; };
    check(!cancelled([&] { m.ingest(a); }) && m.chunks == 3, "uncancelled slot handoff");
  }
  return failures ? 1 : 0;
}
