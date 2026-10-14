#include <hip/hip_runtime.h>
#include "../src/hgn_qwen27.h"
#include "../src/gpu/qwen27_gemv.h"
#include <chrono>
#include <iostream>
static void ck(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<int D,int P>void run(const uint8_t* w,const float* x,float* y,int n,int k){qwen27_fast::batch<D,P><<<(n+3)/4,128>>>(w,x,y,n,k);}
template<int D>void test(const hgn::Tensor& t){
  int n=int(t.dims[0]),k=int(t.dims[1]);uint8_t* w;float *x,*y;
  ck(hipMalloc(&w,t.data_size));ck(hipMalloc(&x,4*k*sizeof(float)));ck(hipMalloc(&y,4*n*sizeof(float)));
  ck(hipMemcpy(w,t.data,t.data_size,hipMemcpyHostToDevice));std::vector<float> input(4*k),serial(4*n),batch(4*n);
  for(size_t i=0;i<input.size();++i)input[i]=sinf(float(i)*.017f)+.1f*cosf(float(i)*.031f);
  ck(hipMemcpy(x,input.data(),input.size()*4,hipMemcpyHostToDevice));
  for(int p=0;p<4;++p)qwen27_fast::gemv<D,1><<<(n+3)/4,128>>>(w,x+p*k,y+p*n,n,k);
  ck(hipMemcpy(serial.data(),y,serial.size()*4,hipMemcpyDeviceToHost));
  double worst=0;
  for(int p=1;p<=4;++p){
    auto invoke=[&](){if(p==1)run<D,1>(w,x,y,n,k);if(p==2)run<D,2>(w,x,y,n,k);if(p==3)run<D,3>(w,x,y,n,k);if(p==4)run<D,4>(w,x,y,n,k);};
    invoke();ck(hipMemcpy(batch.data(),y,p*n*4,hipMemcpyDeviceToHost));
    if(memcmp(serial.data(),batch.data(),p*n*4))throw std::runtime_error("batch differs from serial GEMV");
    auto start=std::chrono::steady_clock::now();for(int j=0;j<10;++j)invoke();ck(hipDeviceSynchronize());
    std::cout<<t.name<<" P="<<p<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/10<<" bitwise=true\n";
  }
  for(int r:{0,n/2,n-1}){double expected=0,scale=0;for(int c=0;c<k;++c){double v=double(qwen27::value(t,size_t(r)*k+c))*input[c];expected+=v;scale+=fabs(v);}worst=std::max(worst,fabs(serial[r]-expected)/std::max(scale,1e-20));}
  if(worst>1e-5)throw std::runtime_error("CPU decoder mismatch");
  std::cout<<"cpu_normalized_error="<<worst<<std::endl;
  ck(hipFree(w));ck(hipFree(x));ck(hipFree(y));
}
int main(int argc,char** argv){
  if(argc!=2)return 2;hgn::Checkpoint ckpt(argv[1]);
  test<6>(ckpt.at("layers.0.linear_attn.in_proj_qkv.weight"));test<5>(ckpt.at("layers.0.mlp.gate_proj.weight"));
  test<0>(ckpt.at("mtp.fc.weight"));test<51>(ckpt.at("mtp.layers.0.mlp.gate_proj.weight"));
}
