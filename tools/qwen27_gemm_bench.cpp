#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "../src/hgn_qwen27.h"
#include "../src/gpu/qwen27_gemm.h"
#include <chrono>
#include <iostream>
static void ck(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
template<class F> double time_ms(F f){ck(hipDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int i=0;i<3;++i)f();ck(hipDeviceSynchronize());return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/3;}
int main(int argc,char** argv)try{
  if(argc<2)return 2;int p=argc>2?atoi(argv[2]):256;hgn::Checkpoint file(argv[1]);rocblas_handle handle;rocblas_create_handle(&handle);
  qwen27_gemm::Lt lt;
  for(const char* name:{"layers.0.linear_attn.in_proj_qkv.weight","layers.0.mlp.gate_proj.weight"}){
    const auto& t=file.at(name);qwen27::validate(t);int n=int(t.dims[0]),k=int(t.dims[1]);
    uint8_t* w;float *wf,*x,*y;uint16_t *wh,*wl,*xh,*xl;
    ck(hipMalloc(&w,t.data_size));ck(hipMemcpy(w,t.data,t.data_size,hipMemcpyHostToDevice));
    ck(hipMalloc(&wf,size_t(n)*k*4));ck(hipMalloc(&wh,size_t(n)*k*2));ck(hipMalloc(&wl,size_t(n)*k*2));
    ck(hipMalloc(&x,size_t(p)*k*4));ck(hipMalloc(&xh,size_t(p)*k*2));ck(hipMalloc(&xl,size_t(p)*k*2));ck(hipMalloc(&y,size_t(p)*n*4));
    std::vector<float> input(size_t(p)*k),ref(size_t(p)*n),actual(ref.size());
    for(size_t i=0;i<input.size();++i)input[i]=sinf(float(i)*0.01731f)*0.43f+cosf(float(i)*0.00137f);
    ck(hipMemcpy(x,input.data(),input.size()*4,hipMemcpyHostToDevice));
    auto unpack=[&](){if(t.dtype==5)qwen27_gemm::unpack<5><<<(size_t(n)*k+255)/256,256>>>(w,wf,wh,wl,n,k);else qwen27_gemm::unpack<6><<<(size_t(n)*k+255)/256,256>>>(w,wf,wh,wl,n,k);};
    unpack();qwen27_gemm::split<<<(input.size()+255)/256,256>>>(x,xh,xl,input.size());ck(hipDeviceSynchronize());
    float one=1,zero=0;
    auto full=[&](){auto s=rocblas_sgemm(handle,rocblas_operation_transpose,rocblas_operation_none,n,p,k,&one,wf,k,x,k,&zero,y,n);if(s!=rocblas_status_success)throw std::runtime_error("sgemm failed");};
    full();double fms=time_ms(full);ck(hipMemcpy(ref.data(),y,ref.size()*4,hipMemcpyDeviceToHost));
    for(bool fourth:{false,true}){
      auto fast=[&](){qwen27_gemm::compensated(lt,wh,wl,xh,xl,y,n,p,k,fourth);};fast();double bms=time_ms(fast);
      ck(hipMemcpy(actual.data(),y,actual.size()*4,hipMemcpyDeviceToHost));double aa=0,ee=0,maxe=0;
      for(size_t i=0;i<ref.size();++i){double e=actual[i]-ref[i];aa+=double(ref[i])*ref[i];ee+=e*e;maxe=std::max(maxe,fabs(e));}
      std::cout<<name<<" p="<<p<<" terms="<<(fourth?4:3)<<" fp32_ms="<<fms<<" compensated_ms="<<bms<<" relative_l2="<<sqrt(ee/aa)<<" max_abs="<<maxe<<std::endl;
      if(!std::isfinite(ee)||sqrt(ee/aa)>1e-4)throw std::runtime_error("GEMM precision gate failed");
    }
    std::cout<<"unpack_split_ms="<<time_ms(unpack)<<std::endl;
    if(t.dtype==5)qwen27_gemm::factor<5><<<(size_t(n)*k+255)/256,256>>>(w,wh,n,k);else qwen27_gemm::factor<6><<<(size_t(n)*k+255)/256,256>>>(w,wh,n,k);
    auto factored=[&](){lt.product(wh,xl,y,n,p,k,0);lt.product(wh,xh,y,n,p,k,1);qwen27_gemm::rescale<<<(actual.size()+255)/256,256>>>(w,y,int(t.dtype),n,k,p);};
    factored();double factored_ms=time_ms(factored);ck(hipMemcpy(actual.data(),y,actual.size()*4,hipMemcpyDeviceToHost));
    double aa=0,ee=0,maxe=0;for(size_t i=0;i<ref.size();++i){double e=actual[i]-ref[i];aa+=double(ref[i])*ref[i];ee+=e*e;maxe=std::max(maxe,fabs(e));}
    std::cout<<"factored_two_ms="<<factored_ms<<" relative_l2="<<sqrt(ee/aa)<<" max_abs="<<maxe<<std::endl;
    if(!std::isfinite(ee)||sqrt(ee/aa)>1e-4)throw std::runtime_error("factored precision gate failed");
    ck(hipFree(w));ck(hipFree(wf));ck(hipFree(wh));ck(hipFree(wl));ck(hipFree(x));ck(hipFree(xh));ck(hipFree(xl));ck(hipFree(y));
  }
  rocblas_destroy_handle(handle);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
