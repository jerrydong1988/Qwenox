// Deterministic ring simulation around the extracted production PleUring::run.
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

struct io_uring_sqe {
  int opcode, fd;
  uint64_t addr, off, user_data;
  unsigned len;
};
struct io_uring_cqe { uint64_t user_data; int res; };
static constexpr int IORING_OP_READ = 22, IORING_ENTER_GETEVENTS = 1;
static constexpr int __NR_io_uring_enter = 426;
static int g_ple_fd = 1;
static size_t g_ple_rb = 16;
static uint64_t g_ple_foff = 0;

struct Ring {
  unsigned entries = 4, sh = 0, st = 0, ch = 0, ct = 0, sm = 3, cm = 7;
  unsigned *sq_head = &sh, *sq_tail = &st, *cq_head = &ch, *cq_tail = &ct;
  unsigned *sq_mask = &sm, *cq_mask = &cm;
  unsigned sq_array[4]{};
  io_uring_sqe sqes[4]{};
  io_uring_cqe cqes[8]{};
  int fd = 1, calls = 0, maximum = 0;
  bool short_submit = false, interrupt = false, short_read = false;
  const uint8_t* source = nullptr;
  std::deque<io_uring_sqe> pending;

  long enter(unsigned count) {
    if (++calls > 1000) throw std::runtime_error("submitted rows lost");
    if (count != st - sh) throw std::runtime_error("pending SQ entries omitted");
    unsigned accepted = short_submit ? std::min(count, 1u) : count;
    if (interrupt && calls == 1) accepted = std::min(count, 2u);
    for (unsigned i = 0; i < accepted; ++i) pending.push_back(sqes[sq_array[sh++ & sm]]);
    maximum = std::max(maximum, int(pending.size()));
    if (pending.size() > entries) throw std::runtime_error("in-flight depth exceeded");
    // Submission frees SQ slots immediately. Reads complete later, one at a time.
    if (calls > 1 && !pending.empty()) {
      auto request = pending.front();
      pending.pop_front();
      bool partial = short_read && request.user_data % 2 == 0;
      size_t bytes = partial ? request.len / 2 : request.len;
      memcpy(reinterpret_cast<void*>(uintptr_t(request.addr)), source + request.off, bytes);
      cqes[ct++ & cm] = {request.user_data, int(bytes)};
    }
    if (interrupt && calls == 1) { errno = EINTR; return -1; }
    return accepted;
  }
};

static Ring* active;
static long fake_enter(int, int, unsigned count, unsigned, int, void*, int) {
  return active->enter(count);
}
#define syscall fake_enter
struct Reader : Ring {
#include "ple_uring_run.inc"
};
#undef syscall

int main(int argc, char** argv) {
  try {
    assert(argc == 2);
    std::string mode = argv[1];
    Reader reader;
    reader.short_submit = mode == "short-submit";
    reader.interrupt = mode == "eintr";
    reader.short_read = mode == "short-read";
    active = &reader;
    std::vector<uint8_t> input(37 * g_ple_rb), output(input.size(), 0xcc);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i * 73 + 11);
    reader.source = input.data();
    std::vector<uint64_t> addresses(37);
    for (size_t i = 0; i < addresses.size(); ++i)
      addresses[i] = uint64_t(uintptr_t(input.data() + i * g_ple_rb));
    reader.run(addresses.data(), output.data(), uint64_t(uintptr_t(input.data())), addresses.size());
    assert(input == output && reader.pending.empty() && reader.sh == reader.st);
    reader.run(nullptr, nullptr, 0, 0);
    printf("PASS %s, maximum in flight %d, calls %d\n", argv[1], reader.maximum, reader.calls);
    return 0;
  } catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
