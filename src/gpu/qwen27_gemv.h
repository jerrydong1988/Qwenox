// Vector loads for the Qwen27 FP8-row/NVFP4 recipes. Float32 weights and
// activations throughout; no activation quantization or reduced-precision MAC.
#pragma once
#include <hip/hip_fp16.h>
namespace qwen27_fast {
__device__ __forceinline__ float bf16(uint16_t v){return __uint_as_float(uint32_t(v)<<16);}
__device__ __forceinline__ float e4m3(uint8_t v){
  unsigned e=(v>>3)&15,m=v&7;
  float f=e?__uint_as_float((e+120)<<23|m<<20):float(m)*0x1p-9f;
  if(e==15&&m==7)f=NAN;
  return v&128?-f:f;
}
template<int Dtype,int Rows=2>
__global__ void gemv(const uint8_t* __restrict__ weight,const float* __restrict__ x,
                     float* __restrict__ out,int n,int k){
  __shared__ float cb[16];
  if constexpr(Dtype==5||Dtype==51){if(threadIdx.x<16)cb[threadIdx.x]=reinterpret_cast<const float*>(weight)[threadIdx.x];__syncthreads();}
  int lane=threadIdx.x&31,first=(blockIdx.x*4+threadIdx.x/32)*Rows;
  if(first>=n)return;
  const uint8_t* ptr[Rows];float scale[Rows],acc[Rows]={};
  #pragma unroll
  for(int r=0;r<Rows;++r){
    if constexpr(Dtype==0)ptr[r]=weight+size_t(first+r)*k*2;
    if constexpr(Dtype==6){ptr[r]=weight+size_t(first+r)*k;scale[r]=first+r<n?bf16(reinterpret_cast<const uint16_t*>(weight+size_t(n)*k)[first+r]):0;}
    if constexpr(Dtype==5||Dtype==51)ptr[r]=weight+64+size_t(first+r)*(k/2+k/16);
  }
  for(int c=lane*8;c<k;c+=256){
    float4 a=*reinterpret_cast<const float4*>(x+c),b=*reinterpret_cast<const float4*>(x+c+4);
    float xv[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
    #pragma unroll
    for(int r=0;r<Rows;++r){
      if(first+r>=n)continue;
      if constexpr(Dtype==0){
        uint4 packed=*reinterpret_cast<const uint4*>(ptr[r]+c*2);
        const uint16_t* values=reinterpret_cast<const uint16_t*>(&packed);
        #pragma unroll
        for(int j=0;j<8;++j)acc[r]=fmaf(bf16(values[j]),xv[j],acc[r]);
      }
      if constexpr(Dtype==6){
        uint2 packed=*reinterpret_cast<const uint2*>(ptr[r]+c);
        const uint8_t* values=reinterpret_cast<const uint8_t*>(&packed);
        #pragma unroll
        for(int j=0;j<8;++j)acc[r]=fmaf(e4m3(values[j])*scale[r],xv[j],acc[r]);
      }
      if constexpr(Dtype==5||Dtype==51){
        uint32_t packed=*reinterpret_cast<const uint32_t*>(ptr[r]+c/2);
        float s;
        if constexpr(Dtype==51)s=__half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(ptr[r]+k/2)[c/32]));
        else s=e4m3(ptr[r][k/2+c/16]);
        #pragma unroll
        for(int j=0;j<8;++j){acc[r]=fmaf(cb[packed&15]*s,xv[j],acc[r]);packed>>=4;}
      }
    }
  }
  #pragma unroll
  for(int r=0;r<Rows;++r){
    for(int d=16;d;d>>=1)acc[r]+=__shfl_down(acc[r],d,32);
    if(lane==0&&first+r<n)out[first+r]=acc[r];
  }
}

// Short verification batches reuse each decoded weight across all token rows.
// Each row keeps the same FP32 FMA and reduction order as the serial GEMV.
template<int Dtype,int P> __global__ void batch(const uint8_t* weight,const float* x,float* out,int n,int k){
  __shared__ float cb[16];
  if constexpr(Dtype==5||Dtype==51){if(threadIdx.x<16)cb[threadIdx.x]=reinterpret_cast<const float*>(weight)[threadIdx.x];__syncthreads();}
  int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
  if(row>=n)return;
  const uint8_t* w=weight;
  if constexpr(Dtype==0)w+=size_t(row)*k*2;
  if constexpr(Dtype==6)w+=size_t(row)*k;
  if constexpr(Dtype==5||Dtype==51)w+=64+size_t(row)*(k/2+k/16);
  float acc[P]={},scale=1;
  if constexpr(Dtype==6)scale=bf16(reinterpret_cast<const uint16_t*>(weight+size_t(n)*k)[row]);
  for(int c=lane*8;c<k;c+=256){
    float values[8];
    if constexpr(Dtype==0){uint4 a=*reinterpret_cast<const uint4*>(w+2*c);const auto* b=reinterpret_cast<const uint16_t*>(&a);
      #pragma unroll
      for(int j=0;j<8;++j)values[j]=bf16(b[j]);}
    if constexpr(Dtype==6){uint2 a=*reinterpret_cast<const uint2*>(w+c);const auto* b=reinterpret_cast<const uint8_t*>(&a);
      #pragma unroll
      for(int j=0;j<8;++j)values[j]=e4m3(b[j])*scale;}
    if constexpr(Dtype==5||Dtype==51){uint32_t packed=*reinterpret_cast<const uint32_t*>(w+c/2);float s;
      if constexpr(Dtype==51)s=__half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(w+k/2)[c/32]));
      else s=e4m3(w[k/2+c/16]);
      #pragma unroll
      for(int j=0;j<8;++j){values[j]=cb[packed&15]*s;packed>>=4;}}
    #pragma unroll
    for(int p=0;p<P;++p){
      float4 a=*reinterpret_cast<const float4*>(x+size_t(p)*k+c),b=*reinterpret_cast<const float4*>(x+size_t(p)*k+c+4);
      float xv[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
      #pragma unroll
      for(int j=0;j<8;++j)acc[p]=fmaf(values[j],xv[j],acc[p]);
    }
  }
  #pragma unroll
  for(int p=0;p<P;++p){for(int d=16;d;d>>=1)acc[p]+=__shfl_down(acc[p],d,32);if(lane==0)out[size_t(p)*n+row]=acc[p];}
}
} // namespace qwen27_fast
