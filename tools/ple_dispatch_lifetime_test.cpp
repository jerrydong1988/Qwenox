// Host-only execution of the production gather dispatch, including delayed
// thread scheduling past the scope that owns the pump-count local variable.
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../src/gpu/ple_page_plan.h"
static uint64_t g_ple_foff = 4093;
static unsigned g_ple_rb = 160;
static bool g_ple_off_ok = true;
#include "production_readers.inc"
static void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
struct DeferredThread {
  std::function<void()> call;
  std::thread worker;
  template<class F> explicit DeferredThread(F&& f):call(std::forward<F>(f)){}
  DeferredThread(DeferredThread&&)=default;
  void start(){worker=std::thread(std::move(call));}
  void join(){worker.join();}
};
int main(int argc,char**argv){
 try {
  need(argc==2,"fixture required");
  std::vector<uint8_t> bytes(g_ple_foff+8193*g_ple_rb);
  for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t((i*37+i/4096)%251);
  {std::ofstream f(argv[1],std::ios::binary);f.write((char*)bytes.data(),bytes.size());need(f.good(),"fixture write");}
  unsigned checks=0;
  for(int configured:{1,8}){
   _putenv_s("GDEC_PLE_RINGS",std::to_string(configured).c_str());ple_uring_open(argv[1]);
   for(const int P:{1,16,32,64,512,1024,8192}){
    const int nt=std::max(1,std::min(16,P));
    int ple_base0=0,pad_id=248044;
    int64_t ple_mult[3]={7919,104729,15485863},ple_vsz[16],ple_voff[16];
    for(int h=0;h<16;++h){ple_vsz[h]=512;ple_voff[h]=h*512;}
    struct Table{const uint8_t*data;} table{bytes.data()+g_ple_foff};auto*ple_table=&table;
    std::vector<int64_t> chist(P);for(int i=0;i<P;++i)chist[i]=100+i%43;
    std::vector<uint64_t> ple_addrs;
    std::vector<uint8_t> storage(size_t(P)*16*g_ple_rb+512,0xcd);auto*ple_fp8=storage.data()+256;
    std::vector<DeferredThread> ple_pool;ple_pool.reserve(nt);
    std::atomic<int> ple_workers{0},ple_barrier{0};std::chrono::steady_clock::time_point ple_done;
    {
#include "production_dispatch.inc"
    }
    // No task can execute until AFTER the production branch's locals expire.
    for(auto& worker:ple_pool)worker.start();
    for(auto& worker:ple_pool)worker.join();
    need(ple_workers.load()==0,"all pump workers completed");
    for(size_t i=0;i<ple_addrs.size();++i)
      need(!memcmp(ple_fp8+i*g_ple_rb,reinterpret_cast<const void*>(ple_addrs[i]),g_ple_rb),"row bytes");
    for(size_t i=0;i<256;++i)need(storage[i]==0xcd&&storage[storage.size()-1-i]==0xcd,"output guard");
    ++checks;
   }
   g_ple_win.close();for(auto& r:g_ple_win_fls)r.close();g_ple_win_nfl=0;
  }
  need(DeleteFileA(argv[1]),"fixture cleanup");printf("PASS delayed-dispatch cases=%u\n",checks);return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
