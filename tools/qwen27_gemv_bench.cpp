#include <hip/hip_runtime.h>
#include "../src/hgn_qwen27.h"
#include "../src/gpu/qwen27_gemv.h"
#include <chrono>
#include <iostream>

static void ck(hipError_t x){if(x!=hipSuccess)throw std::runtime_error(hipGetErrorString(x));}
template<int D,int R>static float bench(const uint8_t* w,const float* x,float* y,int n,int k){
  for(int i=0;i<3;++i)qwen27_fast::gemv<D,R><<<(n+4*R-1)/(4*R),128>>>(w,x,y,n,k);
  ck(hipDeviceSynchronize());auto start=std::chrono::steady_clock::now();
  for(int i=0;i<15;++i)qwen27_fast::gemv<D,R><<<(n+4*R-1)/(4*R),128>>>(w,x,y,n,k);
  ck(hipDeviceSynchronize());
  return float(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/15);
}
int main(int argc,char** argv){
  if(argc!=2)return 2;hgn::Checkpoint file(argv[1]);
  for(const char* name:{"layers.0.linear_attn.in_proj_a.weight","layers.0.linear_attn.in_proj_qkv.weight","layers.0.mlp.gate_proj.weight","layers.0.mlp.down_proj.weight","lm_head.weight"}){
    const auto& t=file.at(name);qwen27::validate(t);int n=int(t.dims[0]),k=int(t.dims[1]);
    uint8_t* w;float *x,*y;ck(hipMalloc(&w,t.data_size));ck(hipMalloc(&x,k*4));ck(hipMalloc(&y,n*4));
    std::vector<float> input(k),output(n);for(int i=0;i<k;++i)input[i]=sinf(float(i)*0.017f);
    ck(hipMemcpy(w,t.data,t.data_size,hipMemcpyHostToDevice));ck(hipMemcpy(x,input.data(),k*4,hipMemcpyHostToDevice));
    float a=0,b=0,c=0;
    #define RUN(D) a=bench<D,1>(w,x,y,n,k);b=bench<D,2>(w,x,y,n,k);c=bench<D,4>(w,x,y,n,k)
    if(t.dtype==0){RUN(0);}else if(t.dtype==5){RUN(5);}else{RUN(6);}
    ck(hipMemcpy(output.data(),y,n*4,hipMemcpyDeviceToHost));
    double max_error=0;
    for(int r:{0,n/2,n-1}){
      double expected=0,abs_sum=0;for(int j=0;j<k;++j){double v=double(qwen27::value(t,size_t(r)*k+j))*input[j];expected+=v;abs_sum+=fabs(v);}
      max_error=std::max(max_error,fabs(output[r]-expected)/std::max(abs_sum,1e-20));
    }
    std::cout<<name<<" rows1_ms="<<a<<" rows2_ms="<<b<<" rows4_ms="<<c<<" normalized_error="<<max_error<<std::endl;
    if(max_error>1e-5)throw std::runtime_error("GEMV mismatch");
    ck(hipFree(w));ck(hipFree(x));ck(hipFree(y));
  }
}
