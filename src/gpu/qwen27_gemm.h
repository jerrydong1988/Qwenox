// Compensated BF16 matrix products, retaining FP32 accumulation.
// The residual terms repair most BF16 rounding error; this is not bit-exact FP32.
#pragma once
#include "qwen27_gemv.h"
#include <hipblaslt/hipblaslt.h>
#include <unordered_map>
namespace qwen27_gemm {
__device__ __forceinline__ uint16_t round_bf16(float f) {
  uint32_t u=__float_as_uint(f);
  return uint16_t((u+0x7fff+((u>>16)&1))>>16);
}
template<int Dtype> __global__ void unpack(const uint8_t* w,float* out,
                                          uint16_t* hi,uint16_t* lo,int rows,int cols) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,n=size_t(rows)*cols;
  if(i>=n)return;
  float f=0;
  if constexpr(Dtype==0) f=qwen27_fast::bf16(reinterpret_cast<const uint16_t*>(w)[i]);
  if constexpr(Dtype==6) f=qwen27_fast::e4m3(w[i])*qwen27_fast::bf16(reinterpret_cast<const uint16_t*>(w+n)[i/cols]);
  if constexpr(Dtype==5){
    size_t c=i%cols;const uint8_t* r=w+64+(i/cols)*(cols/2+cols/16);
    f=reinterpret_cast<const float*>(w)[(r[c/2]>>((c&1)*4))&15]*qwen27_fast::e4m3(r[cols/2+c/16]);
  }
  if(out)out[i]=f;
  if(hi){uint16_t h=round_bf16(f);hi[i]=h;lo[i]=round_bf16(f-qwen27_fast::bf16(h));}
}
__global__ void split(const float* x,uint16_t* hi,uint16_t* lo,size_t n){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n){uint16_t h=round_bf16(x[i]);hi[i]=h;lo[i]=round_bf16(x[i]-qwen27_fast::bf16(h));}
}
// Move the existing per-row/global scale outside the dot product. BF16 holds
// every FP8 code and FP4*FP8 product exactly, so weights need only one plane.
template<int Dtype> __global__ void factor(const uint8_t* w,uint16_t* out,int rows,int cols){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,n=size_t(rows)*cols;
  if(i>=n)return;
  float f=0;
  if constexpr(Dtype==0){out[i]=reinterpret_cast<const uint16_t*>(w)[i];return;}
  if constexpr(Dtype==6)f=qwen27_fast::e4m3(w[i]);
  if constexpr(Dtype==5){
    constexpr float fp4[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    size_t c=i%cols;const uint8_t* r=w+64+(i/cols)*(cols/2+cols/16);
    f=fp4[(r[c/2]>>((c&1)*4))&15]*qwen27_fast::e4m3(r[cols/2+c/16]);
  }
  out[i]=round_bf16(f);
}
__global__ void rescale(const uint8_t* w,float* y,int dtype,int rows,int cols,int count){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(rows)*count)return;
  float scale=dtype==5?reinterpret_cast<const float*>(w)[2]:
      qwen27_fast::bf16(reinterpret_cast<const uint16_t*>(w+size_t(rows)*cols)[i%rows]);
  y[i]*=scale;
}
class Lt {
  struct Plan {hipblasLtMatmulDesc_t op=nullptr;hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;hipblasLtMatmulAlgo_t algo;};
  hipblasLtHandle_t handle=nullptr;
  void* workspace=nullptr;
  size_t workspace_bytes=64*1024*1024;
  std::unordered_map<uint64_t,Plan> plans;
  static void check(hipblasStatus_t s){if(s!=HIPBLAS_STATUS_SUCCESS)throw std::runtime_error("Qwen27 hipBLASLt error "+std::to_string(int(s)));}
public:
  Lt(){check(hipblasLtCreate(&handle));if(hipMalloc(&workspace,workspace_bytes)!=hipSuccess)throw std::runtime_error("Qwen27 LT workspace allocation failed");}
  ~Lt(){for(auto& kv:plans){auto& v=kv.second;hipblasLtMatmulDescDestroy(v.op);hipblasLtMatrixLayoutDestroy(v.a);hipblasLtMatrixLayoutDestroy(v.b);hipblasLtMatrixLayoutDestroy(v.c);}if(workspace)(void)hipFree(workspace);if(handle)hipblasLtDestroy(handle);}
  void product(const uint16_t* w,const uint16_t* x,float* out,int n,int p,int k,float beta){
    const uint64_t key=(uint64_t(n)<<40)|(uint64_t(k)<<20)|uint64_t(p);
    auto it=plans.find(key);
    if(it==plans.end()){
      Plan e;check(hipblasLtMatmulDescCreate(&e.op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
      hipblasOperation_t t=HIPBLAS_OP_T,normal=HIPBLAS_OP_N;
      check(hipblasLtMatmulDescSetAttribute(e.op,HIPBLASLT_MATMUL_DESC_TRANSA,&t,sizeof t));
      check(hipblasLtMatmulDescSetAttribute(e.op,HIPBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof normal));
      check(hipblasLtMatrixLayoutCreate(&e.a,HIP_R_16BF,k,n,k));check(hipblasLtMatrixLayoutCreate(&e.b,HIP_R_16BF,k,p,k));check(hipblasLtMatrixLayoutCreate(&e.c,HIP_R_32F,n,p,n));
      hipblasLtMatmulPreference_t pref;check(hipblasLtMatmulPreferenceCreate(&pref));
      check(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof workspace_bytes));
      hipblasLtMatmulHeuristicResult_t candidates[8];int count=0;
      check(hipblasLtMatmulAlgoGetHeuristic(handle,e.op,e.a,e.b,e.c,e.c,pref,8,candidates,&count));hipblasLtMatmulPreferenceDestroy(pref);
      if(!count)throw std::runtime_error("Qwen27 LT: no BF16 algorithm");
      int chosen=count>4&&candidates[4].state==HIPBLAS_STATUS_SUCCESS&&candidates[4].workspaceSize<=workspace_bytes?4:0;
      e.algo=candidates[chosen].algo;it=plans.emplace(key,e).first;
    }
    const auto& e=it->second;const float one=1;
    check(hipblasLtMatmul(handle,e.op,&one,w,e.a,x,e.b,&beta,out,e.c,out,e.c,&e.algo,workspace,workspace_bytes,nullptr));
  }
};
inline void compensated(Lt& handle,const uint16_t* wh,const uint16_t* wl,
                        const uint16_t* xh,const uint16_t* xl,float* y,int n,int p,int k,bool fourth=false){
  handle.product(wh,xl,y,n,p,k,0);
  handle.product(wl,xh,y,n,p,k,1);
  if(fourth)handle.product(wl,xl,y,n,p,k,1);
  handle.product(wh,xh,y,n,p,k,1);
}
} // namespace qwen27_gemm
