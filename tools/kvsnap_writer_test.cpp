// Compile the production writer with real file I/O and a fault at file open.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include "../src/gpu/os_win32.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "posix_io.inc"
#endif

static constexpr int KV_PAGE = 256;
static const char* log_ts() { return "test"; }
static std::atomic<bool> fail_open{false};
static int test_open_wr(const char* path) {
  if (fail_open.exchange(false)) throw std::bad_alloc();
  return os_open_wr(path);
}
#define os_open_wr test_open_wr
#include "writer.inc"
#undef os_open_wr

// Simulate KV eviction throwing while a matched checkpoint is held.
struct GpuModel {};
enum { RS_OK, RS_CKPT, RS_PAGE, RS_OTHER };
static int restore_body(GpuModel&, const CkEnt&, const std::vector<int>&,
                        bool*, uint64_t*, int*) { throw std::bad_alloc(); }
#include "restore.inc"

static void require(bool condition, const char* what) {
  if (!condition) throw std::runtime_error(what);
}
static void queue_page(uint64_t key) {
  Item it;
  it.key = key;
  it.buf = buf_get(sizeof(ChHdr));
  std::memset(it.buf.data(), 0, it.buf.n);
  {
    std::lock_guard<std::mutex> lock(g_mtx);
    g_ch[key].refs = 1;
  }
  enqueue(std::move(it));
}

static void check_settled(uint64_t key, bool success) {
  sync_line();  // Must return even if the writer threw.
  std::lock_guard<std::mutex> lock(g_mtx);
  require(!g_wbusy && g_q_bytes == 0 && g_q.empty(), "writer accounting leaked");
  require(g_ch.at(key).st == (success ? ST_PUB : ST_FAIL), "wrong page state");
  require(!std::filesystem::exists(path_of("tmp_" + ch_name(key))), "temporary file leaked");
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected cache directory");
    g_dir = argv[1];
    g_cap = g_qcap = 1 << 20;
    std::thread(writer).detach();
    auto matched = std::make_shared<CkEnt>();
    matched->busy = 1;
    GpuModel model;
    std::vector<int> hist;
    bool mtp = false, threw = false;
    try { restore(model, matched, {}, hist, &mtp); }
    catch (const std::bad_alloc&) { threw = true; }
    require(threw && matched->busy == 0, "exception leaked a snapshot pin");

    fail_open = true;
    queue_page(1);
    check_settled(1, false);

    // A directory with the final filename forces the real rename to fail.
    std::filesystem::create_directory(path_of(ch_name(2)));
    queue_page(2);
    check_settled(2, false);

    queue_page(3);
    check_settled(3, true);
    require(std::filesystem::file_size(path_of(ch_name(3))) == sizeof(ChHdr), "page not written");

    auto ck = std::make_shared<CkEnt>();
    ck->n = 1;
    ck->name = "failed.kck";
    ck->chunks = {1};
    Item it;
    it.kind = 2;
    it.ck = ck;
    it.buf = buf_get(sizeof(CkHdr));
    std::memset(it.buf.data(), 0, it.buf.n);
    {
      std::lock_guard<std::mutex> lock(g_mtx);
      g_ck.push_back(ck);
    }
    enqueue(std::move(it));
    sync_line();
    {
      std::lock_guard<std::mutex> lock(g_mtx);
      require(g_ck.empty() && !g_ch.count(1), "failed checkpoint retained its reference");
      require(g_total == sizeof(ChHdr), "published byte count drifted");
    }
    std::puts("PASS writer allocation/rename failure, queue drain, continued publication");
    std::fflush(stdout);
    std::_Exit(0);  // The production writer lives until process exit.
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    std::_Exit(1);
  }
}
