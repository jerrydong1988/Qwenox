// Native GGUF WMMA: CPU-reference accuracy, row invariance, and output guards.
// Host dispatch below is extracted from production 30_host_util.inc at build time;
// tools/gguf_wmma_test.py prepares it, so no second dispatch implementation exists.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>
#include "../src/gguf.h"
#define CK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %s at %d\n",hipGetErrorString(e),__LINE__);return_exit();} } while(0)
static void return_exit(){std::exit(2);}
__device__ uint16_t f2bf(float f){uint32_t u=__builtin_bit_cast(uint32_t,f);u+=0x7fff+((u>>16)&1);return u>>16;}
namespace moegg {
using v16h = _Float16 __attribute__((ext_vector_type(16)));
using v8f = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ v8f wmma(v16h a,v16h b,v8f c) {
 return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c);
}
}
#include "../src/gpu/parts/26_kernels_dense_gguf.inc"
struct WView {int dtype; const uint8_t* data; uint64_t rows,cols,scale_stride;};
hipStream_t g_str = nullptr;
#include "gguf_dense_host_under_test.inc"
template<class T> T* upload(const std::vector<T>& h){T* d=nullptr;CK(hipMalloc(&d,h.size()*sizeof(T)));CK(hipMemcpy(d,h.data(),h.size()*sizeof(T),hipMemcpyHostToDevice));return d;}
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float unbf(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;memcpy(&f,&u,4);return f;}
int main(){
 std::mt19937 rng(7351);std::uniform_real_distribution<float> dist(-1,1);
 int cases=0, failures=0;double worst=0;
 for(int t:{12,13,14,23,20}) for(int xbf:{0,1}) for(auto shape:std::vector<std::pair<int,int>>{{512,2560},{640,2560},{2560,6144},{6144,2560},{12288,2560},{320,10240},{48,2560},{10240,2560}}){
  int R=shape.first,C=shape.second,Pmax=48;uint32_t be,bb;gguf::block_info(t,be,bb);auto rb=gguf::row_bytes(t,C);
  std::vector<uint8_t> w(R*rb);for(auto&v:w)v=uint8_t(rng());
  for(int r=0;r<R;r++)for(int b=0;b<C/int(be);b++){auto p=w.data()+r*rb+b*bb;auto h=__float2half(.004f+.002f*dist(rng));memcpy(p+(t==14?208:0),&h,2);if(t==12||t==13)memcpy(p+2,&h,2);}
  std::vector<float> x(Pmax*C);for(auto&v:x)v=dist(rng);std::vector<uint16_t> xb(x.size());for(size_t i=0;i<x.size();i++)xb[i]=bf(x[i]);
  auto dw=upload(w);auto dx=upload(x);auto dxb=upload(xb);float*dy=nullptr;CK(hipMalloc(&dy,(size_t(Pmax)*R+64)*4));
  WView view{100+t,dw,uint64_t(R),uint64_t(C),0};std::vector<float> base(Pmax*R),got(Pmax*R+64);
  auto run=[&](int P,int row){CK(hipMemset(dy,0xee,(size_t(Pmax)*R+64)*4));const void*xp=xbf?(const void*)(dxb+row*C):(const void*)(dx+row*C);if(!ggd_wmma(view,xp,xbf,C,dy,R,P,0,R))return false;CK(hipDeviceSynchronize());CK(hipMemcpy(got.data(),dy,got.size()*4,hipMemcpyDeviceToHost));return true;};
  bool ok=true;
  for(int row=0;row<Pmax;row++){ok=run(1,row)&&ok;memcpy(base.data()+size_t(row)*R,got.data(),R*4);}
  for(int P:{2,3,4,7,8,9,15,16,17,24,32,33,48}){ok=run(P,0)&&ok;ok=ok&&!memcmp(base.data(),got.data(),size_t(P)*R*4);for(size_t i=size_t(P)*R;i<got.size();i++){uint32_t v;memcpy(&v,&got[i],4);if(v!=0xeeeeeeee)ok=false;}}
  std::vector<float> wr(C);double error=0;
  for(int p:{0,1,47})for(int row:{0,R/3,R-1}){
   gguf::dequant_row(t,w.data()+row*rb,wr.data(),C);double sum=0;
   for(int k=0;k<C;k++)sum+=double(wr[k])*__half2float(__float2half(xbf?unbf(xb[p*C+k]):x[p*C+k]));
   double e=std::fabs(sum-base[p*R+row])/std::max(1.,std::fabs(sum));error=std::max(error,e);if(!std::isfinite(base[p*R+row])||e>1e-2)ok=false;
  }
  worst=std::max(worst,error);cases++;failures+=!ok;
  printf("type=%d xbf=%d shape=%dx%d rel=%.7g %s\n",t,xbf,R,C,error,ok?"PASS":"FAIL");fflush(stdout);
  CK(hipFree(dw));CK(hipFree(dx));CK(hipFree(dxb));CK(hipFree(dy));
 }
 printf("RESULT %s cases=%d failures=%d worst_relative=%.7g\n",failures?"FAIL":"PASS",cases,failures,worst);return failures?1:0;
}
