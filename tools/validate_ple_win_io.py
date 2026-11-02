"""Exercise the production Windows IOCP classes without initializing a GPU.

The compiled excerpt comes directly from 10_ple_io.inc, rather than a second
implementation. Fixtures and binaries are written only beneath --output.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

HARNESS = r'''
static void require(bool ok, const char* label) {
  if (!ok) throw std::runtime_error(label);
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "fixture path required");
    const size_t nr = 8193;
    const uint64_t table = 0x100000000ULL;
    g_ple_foff = 4093;
    std::vector<uint8_t> bytes(g_ple_foff + nr * g_ple_rb);
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = (uint8_t)((i * 37 + i / 160 + i / 4096) % 251);
    { std::ofstream f(argv[1], std::ios::binary);
      f.write((const char*)bytes.data(), bytes.size());
      require(f.good(), "fixture write"); }
    DWORD handles_before = 0;
    GetProcessHandleCount(GetCurrentProcess(), &handles_before);
    auto check = [&](auto& reader, size_t n, unsigned seed) {
      std::mt19937 rng(seed);
      std::vector<uint64_t> addresses(n);
      for (size_t i = 0; i < n; ++i)
        addresses[i] = table + (rng() % nr) * g_ple_rb;
      std::vector<uint8_t> output(n * g_ple_rb + 512, 0xCD);
      reader.run(addresses.data(), output.data() + 256, table, n);
      for (size_t i = 0; i < n; ++i)
        require(!memcmp(output.data() + 256 + i * g_ple_rb,
                        bytes.data() + g_ple_foff + addresses[i] - table,
                        g_ple_rb), "row bytes differ");
      for (size_t i = 0; i < 256; ++i)
        require(output[i] == 0xCD && output[output.size()-1-i] == 0xCD,
                "output guard changed");
    };
    unsigned cases = 0;
    for (int rings : {1, 8}) {
      _putenv_s("GDEC_PLE_RINGS", std::to_string(rings).c_str());
      {
        PleWinReaders pool, other;
        PleWin decode;
        require(pool.setup(argv[1]) && other.setup(argv[1]) && decode.setup(argv[1]),
                "reader setup");
        require(pool.readers.size() == (size_t)rings, "actual pump count");
        for (size_t n : {0,1,16,511,512,513,4095,4096,4097,8193}) {
          check(pool, n, (unsigned)n); ++cases;
        }
        // Shared-pool serialization plus independent decode/prefill ownership.
        std::exception_ptr errors[4];
        std::thread threads[4];
        for (int t = 0; t < 4; ++t) threads[t] = std::thread([&,t] {
          try {
            for (int r = 0; r < 6; ++r) {
              if (t < 2) check(pool, 4097, 100+t*10+r);
              else if (t == 2) check(other, 8193, 200+r);
              else check(decode, 16, 300+r);
            }
          } catch (...) { errors[t] = std::current_exception(); }
        });
        for (auto& t : threads) t.join();
        for (auto& e : errors) if (e) std::rethrow_exception(e);
        cases += 24;
        if (const char* page = getenv("GDEC_PLE_PAGE_IO"); page && !strcmp(page,"1")) {
          // One worker rejects before issuing I/O; all peer workers must join
          // before the exception reaches the caller, and the pool stays usable.
          std::vector<uint64_t> addresses(8193, table);
          std::vector<uint8_t> output(addresses.size() * g_ple_rb);
          addresses[4500] = table-1;
          bool threw = false;
          try { pool.run(addresses.data(), output.data(), table, addresses.size()); }
          catch (const std::out_of_range&) { threw = true; }
          require(threw, "worker exception must propagate after join");
          check(pool, 8193, 991); ++cases;
        }
      }
      // Repeated request-owned pool creation/destruction must release handles.
      for (int i = 0; i < 40; ++i) {
        PleWinReaders pool;
        require(pool.setup(argv[1]), "repeated setup");
        check(pool, 513, 400+i); ++cases;
      }
    }
    {
      PleWinReaders empty;
      bool threw = false;
      try { empty.run(nullptr, nullptr, table, 0); }
      catch (const std::runtime_error&) { threw = true; }
      require(threw, "empty pool must reject");
      require(!empty.setup("no-such-ple-test-file.invalid"), "missing file setup");
      ++cases;
    }
    DWORD handles_after = 0;
    GetProcessHandleCount(GetCurrentProcess(), &handles_after);
    require(handles_after == handles_before, "reader handle leak");
    require(DeleteFileA(argv[1]), "fixture cleanup");
    printf("PASS cases=%u handles_before=%lu handles_after=%lu\n", cases,
           (unsigned long)handles_before, (unsigned long)handles_after);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code = (root/'src/gpu/parts/10_ple_io.inc').read_text(encoding='utf-8')
    excerpt = code[code.index('struct PleWin {'):code.index('static PleWin g_ple_win;')]
    args.output.mkdir(parents=True, exist_ok=True)
    src, exe = args.output/'ple_win_io_test.cpp', args.output/'ple_win_io_test.exe'
    prefix = '''#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "ple_page_plan.h"
static uint64_t g_ple_foff = 0;
static uint32_t g_ple_rb = 160;
static bool g_ple_off_ok = true;
'''
    src.write_text(prefix+excerpt+HARNESS, encoding='utf-8')
    subprocess.run([args.cxx,'-O2','-std=c++17','-D_CRT_SECURE_NO_WARNINGS',
                    '-I'+str(root/'src/gpu'),str(src),'-o',str(exe)],check=True)
    rows=[]
    for pages in ['0','1']:
        env=os.environ.copy()
        env.update(GDEC_PLE_PAGE_IO=pages,GDEC_PLE_PAGE_VERIFY='1')
        p=subprocess.run([str(exe),str(args.output/('fixture-'+pages+'.dat'))],
                         env=env,capture_output=True,text=True,timeout=120)
        (args.output/('io-pages-'+pages+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
        rows.append({'pages':pages,'exit_code':p.returncode,'stdout':p.stdout})
        print(rows[-1],flush=True)
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if any(r['exit_code'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
