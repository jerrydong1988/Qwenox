// Exact HT reference decoder checks and buffer-reuse regression stress.
// --stress N repeats the 2560x6144 case with a poisoned output buffer.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "../src/hgn_v2.h"
#include "../src/gpu/parts/28_kernels_hgn_v2.inc"
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); } while(0)
__global__ void init_codes(uint32_t* p,size_t n,unsigned seed){size_t i=(size_t)blockIdx.x*256+threadIdx.x;if(i<n)p[i]=(i*1664525ull+1013904223ull)^(i>>9)^seed;}
__global__ void init_scale(__half* h,int n,unsigned seed){int i=blockIdx.x*256+threadIdx.x;if(i<n)h[i]=__float2half(float(int((i*17+seed)%251)-125)/71.f);}
__global__ void poison(uint16_t* p,size_t n){size_t i=(size_t)blockIdx.x*256+threadIdx.x;if(i<n)p[i]=(i&1)?0xbf80:0x3f80;}
__global__ void cmp(const uint16_t* a,const uint16_t* b,size_t n,unsigned* d){size_t i=(size_t)blockIdx.x*256+threadIdx.x;if(i<n && a[i]!=b[i])atomicAdd(d,1u);}
static uint16_t host_bf16(float v){uint32_t u;std::memcpy(&u,&v,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static void verify_tile(const std::vector<uint8_t>& codes,const std::vector<uint16_t>& su,
                        const std::vector<uint16_t>& sv,const std::vector<uint16_t>& actual,
                        int cols,int row0,int col0) {
  float tile[128][128];
  for(int r=0;r<128;r++)for(int c=0;c<128;c++)tile[r][c]=hgn_v2::ht_at(codes.data(),cols,row0+r,col0+c);
  // Independent host butterfly, without intermediate normalized-H rounding.
  for(int axis=0;axis<2;axis++)for(int h=1;h<128;h*=2)
    for(int line=0;line<128;line++)for(int b=0;b<128;b+=2*h)for(int j=0;j<h;j++){
      float& a=axis?tile[b+j][line]:tile[line][b+j];
      float& z=axis?tile[b+j+h][line]:tile[line][b+j+h];float x=a,y=z;a=x+y;z=x-y;
    }
  for(int r=0;r<128;r++)for(int c=0;c<128;c++){
    float v=((tile[r][c]*(1.f/128.f))*hgn_v2::half(su[col0+c]))*hgn_v2::half(sv[row0+r]);
    if(host_bf16(v)!=actual[64+(size_t)(row0+r)*cols+col0+c])throw std::runtime_error("CPU weight decode mismatch");
  }
}
static void run(int rows,int cols,unsigned seed,int stress){
  size_t n=(size_t)rows*cols;uint32_t* q;__half *su,*sv;uint16_t *a,*b;unsigned* diff;
  CK(hipMalloc(&q,n/2));CK(hipMalloc(&su,cols*2));CK(hipMalloc(&sv,rows*2));
  CK(hipMalloc(&a,(n+128)*2));CK(hipMalloc(&b,(n+128)*2));CK(hipMalloc(&diff,4));
  init_codes<<<(n/8+255)/256,256>>>(q,n/8,seed);init_scale<<<(cols+255)/256,256>>>(su,cols,seed);init_scale<<<(rows+255)/256,256>>>(sv,rows,seed+57);
  CK(hipMemset(a,0x35,(n+128)*2));CK(hipMemset(b,0x35,(n+128)*2));
  auto decode=[&](uint16_t* out){hgnv2gpu::ht_dense<<<dim3(cols/128,rows/128),256>>>(q,su,sv,out+64,cols);};
  decode(a);
  std::vector<uint16_t> actual(n+128),hu(cols),hv(rows);std::vector<uint8_t> codes(n/2);
  CK(hipMemcpy(actual.data(),a,(n+128)*2,hipMemcpyDeviceToHost));
  CK(hipMemcpy(hu.data(),su,cols*2,hipMemcpyDeviceToHost));CK(hipMemcpy(hv.data(),sv,rows*2,hipMemcpyDeviceToHost));
  CK(hipMemcpy(codes.data(),q,n/2,hipMemcpyDeviceToHost));
  for(int i=0;i<64;i++)if(actual[i]!=0x3535 || actual[64+n+i]!=0x3535)throw std::runtime_error("guard overwrite");
  verify_tile(codes,hu,hv,actual,cols,0,0);
  if(rows>128 || cols>128)verify_tile(codes,hu,hv,actual,cols,rows-128,cols-128);
  for(int i=0;i<stress;i++){
    poison<<<(n+255)/256,256>>>(b+64,n);decode(b);
    CK(hipMemsetAsync(diff,0,4,0));cmp<<<(n+128+255)/256,256>>>(a,b,n+128,diff);
    unsigned unequal;CK(hipMemcpy(&unequal,diff,4,hipMemcpyDeviceToHost));
    if(unequal){printf("stress_iteration=%d unequal=%u\n",i,unequal);throw std::runtime_error("buffer-reuse decode mismatch");}
  }
  hipEvent_t start,end;CK(hipEventCreate(&start));CK(hipEventCreate(&end));
  CK(hipEventRecord(start));for(int i=0;i<10;i++)decode(b);CK(hipEventRecord(end));CK(hipEventSynchronize(end));
  float ms;CK(hipEventElapsedTime(&ms,start,end));CK(hipEventDestroy(start));CK(hipEventDestroy(end));
  printf("rows=%d cols=%d seed=%u cpu_tiles=pass stress=%d unequal=0 decode_ms=%.6f\n",rows,cols,seed,stress,ms/10);fflush(stdout);
  for(void* p:{(void*)q,(void*)su,(void*)sv,(void*)a,(void*)b,(void*)diff})CK(hipFree(p));
}
int main(int argc,char**argv){try{
  int stress=0;if(argc==3 && std::string(argv[1])=="--stress")stress=std::stoi(argv[2]);else if(argc!=1)throw std::runtime_error("usage: v2_exact_dense_bench [--stress N]");
  if(stress<0)throw std::runtime_error("stress must be nonnegative");
  for(unsigned seed:{0u,131u,0xdeadbeefu}){
    run(128,128,seed,0);run(640,2560,seed,0);run(2560,6144,seed,seed==0?stress:0);run(12288,2560,seed,0);
  }puts("ALL PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
