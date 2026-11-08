// Independent text backend for the Qwen3.8-27B Halogen HGN recipe.
// Included after the common kernels and protocol sampler, before main.
#include "../../hgn_qwen27.h"
#include "../qwen27_gemv.h"
#include "../qwen27_gemm.h"

namespace qwen27 {
static void check(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
struct Allocations {
  std::vector<void*> pointers;
  ~Allocations() { for (void* p : pointers) (void)hipFree(p); }
  template<class T> T* get(size_t count) {
    T* p = nullptr;
    check(hipMalloc(&p, count * sizeof(T)));
    pointers.push_back(p);
    return p;
  }
};
__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ float fp8(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? m * 0x1p-9f : __uint_as_float(uint32_t(e+120)<<23 | uint32_t(m)<<20);
  if (e == 15 && m == 7) v = NAN;
  return b & 128 ? -v : v;
}
template<int Dtype>
__global__ void unpack_matrix(const uint8_t* w,float* out,int rows,int cols){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,n=size_t(rows)*cols;
  if(i>=n)return;
  if constexpr(Dtype==0)out[i]=bf(reinterpret_cast<const uint16_t*>(w)[i]);
  if constexpr(Dtype==6)out[i]=fp8(w[i])*bf(reinterpret_cast<const uint16_t*>(w+n)[i/cols]);
  if constexpr(Dtype==5||Dtype==51){
    size_t c=i%cols;const uint8_t* row=w+64+(i/cols)*(cols/2+cols/16);
    float scale;
    if constexpr(Dtype==51)scale=__half2float(__ushort_as_half(reinterpret_cast<const uint16_t*>(row+cols/2)[c/32]));
    else scale=fp8(row[cols/2+c/16]);
    out[i]=reinterpret_cast<const float*>(w)[(row[c/2]>>((c&1)*4))&15]*scale;
  }
}
__global__ void embedding_batch(const uint16_t* w,const int* tokens,float* out){
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c<hidden)out[size_t(blockIdx.y)*hidden+c]=bf(w[size_t(tokens[blockIdx.y])*hidden+c]);
}
__global__ void gated_norm_batch(const float* y,const float* z,const float* w,float* out){
  __shared__ float sums[128];
  int i=threadIdx.x,h=blockIdx.x;size_t token=blockIdx.y;
  float value=y[token*10240+4096+h*128+i];
  sums[i]=value*value;__syncthreads();
  for(int d=64;d;d>>=1){if(i<d)sums[i]+=sums[i+d];__syncthreads();}
  float gate=z[token*6144+h*128+i];
  out[token*6144+h*128+i]=value*rsqrtf(sums[0]/128+1e-6f)*w[i]*gate/(1+expf(-gate));
}
__global__ void embedding(const uint16_t* w, float* out, int token) {
  int c = blockIdx.x*blockDim.x+threadIdx.x;
  if (c < hidden) out[c] = bf(w[size_t(token)*hidden+c]);
}
__global__ void embedding_device(const uint16_t* w,float* out,const int* input){
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c<hidden)out[c]=bf(w[size_t(input[0])*hidden+c]);
}
__global__ void cache_values(const float* v,float* cache,const int* pos){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<1024)cache[size_t(*pos)*1024+i]=v[i];
}
__global__ void fill_positions(int* positions,int start,int count){int i=threadIdx.x;if(i<count)positions[i]=start+i;}
__global__ void add_residual(float* x, const float* y, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < n) x[i] += y[i];
}
__global__ void gated_norm(const float* y, const float* z, const float* w, float* out) {
  __shared__ float sums[128];
  int i = threadIdx.x, h = blockIdx.x;
  float v = y[h*128+i];
  sums[i] = v*v;
  __syncthreads();
  for (int d=64; d; d>>=1) { if(i<d) sums[i]+=sums[i+d]; __syncthreads(); }
  float gate = z[h*128+i];
  out[h*128+i] = v*rsqrtf(sums[0]/128+1e-6f)*w[i]*(gate/(1+expf(-gate)));
}
// Q is stored per head as [query(256), gate(256)]. Q/K norms are zero-centered.
__global__ void prepare_qk(const float* raw, const float* w, float* out,
                            float* gate, int stride, int pos,const int* position=nullptr,bool cache=false) {
  if(position)pos=*position;
  if(cache)out+=size_t(pos)*1024;
  raw+=size_t(blockIdx.y)*gridDim.x*stride;
  out+=size_t(blockIdx.y)*gridDim.x*256;
  if(gate)gate+=size_t(blockIdx.y)*gridDim.x*256;
  pos+=blockIdx.y;
  __shared__ float sums[256], values[256];
  int i=threadIdx.x, h=blockIdx.x;
  float v=raw[h*stride+i];
  sums[i]=v*v;
  __syncthreads();
  for(int d=128;d;d>>=1){ if(i<d)sums[i]+=sums[i+d]; __syncthreads(); }
  values[i]=v*rsqrtf(sums[0]/256+1e-6f)*(1+w[i]);
  __syncthreads();
  v=values[i];
  if(i<64){
    int j=i%32;
    double angle=double(pos)*pow(1e7,-double(j)/32);
    float co=float(cos(angle)),si=float(sin(angle));
    v=v*co+(i<32?-values[i+32]:values[i-32])*si;
  }
  out[h*256+i]=v;
  if(gate)gate[h*256+i]=raw[h*stride+256+i];
}
__global__ void attention_scores(const float* q,const float* keys,float* score,int pos,int capacity) {
  int lane=threadIdx.x&31, t=blockIdx.x*4+threadIdx.x/32, h=blockIdx.y;
  if(t>pos)return;
  float s=0;
  for(int i=lane;i<256;i+=32)s=fmaf(q[h*256+i],keys[size_t(t)*1024+(h/6)*256+i],s);
  for(int d=16;d;d>>=1)s+=__shfl_down(s,d,32);
  if(lane==0)score[size_t(h)*capacity+t]=s/16;
}
__global__ void attention_output(const float* score,const float* values,const float* gate,
                                 float* out,int pos,int capacity) {
  __shared__ float scratch[256];
  int i=threadIdx.x,h=blockIdx.x;
  const float* s=score+size_t(h)*capacity;
  float m=-INFINITY;
  for(int t=i;t<=pos;t+=256)m=fmaxf(m,s[t]);
  scratch[i]=m; __syncthreads();
  for(int d=128;d;d>>=1){if(i<d)scratch[i]=fmaxf(scratch[i],scratch[i+d]);__syncthreads();}
  m=scratch[0]; __syncthreads();
  float den=0;
  for(int t=i;t<=pos;t+=256)den+=expf(s[t]-m);
  scratch[i]=den; __syncthreads();
  for(int d=128;d;d>>=1){if(i<d)scratch[i]+=scratch[i+d];__syncthreads();}
  float inv=1/scratch[0], v=0;
  for(int t=0;t<=pos;++t)v=fmaf(expf(s[t]-m)*inv,values[size_t(t)*1024+(h/6)*256+i],v);
  out[h*256+i]=v/(1+expf(-gate[h*256+i]));
}

struct Matrix { const uint8_t* data=nullptr; int dtype=0, rows=0, cols=0; bool factorizable=false; };
struct Blas {
  rocblas_handle handle=nullptr;
  Blas(){if(rocblas_create_handle(&handle)!=rocblas_status_success)throw std::runtime_error("rocBLAS initialization failed");}
  ~Blas(){if(handle)(void)rocblas_destroy_handle(handle);}
};
struct DecodeGraph {
  hipStream_t capture=nullptr;
  hipGraph_t graph=nullptr;
  hipGraphExec_t executable=nullptr;
  ~DecodeGraph(){
    if(executable)(void)hipGraphExecDestroy(executable);
    if(graph)(void)hipGraphDestroy(graph);
    if(capture)(void)hipStreamDestroy(capture);
  }
};
struct Layer {
  float *norm=nullptr,*post=nullptr,*conv=nullptr,*alog=nullptr,*dt=nullptr,*gnorm=nullptr;
  float *qnorm=nullptr,*knorm=nullptr,*state=nullptr,*history=nullptr,*keys=nullptr,*values=nullptr;
  Matrix qkv,z,a,b,out,q,k,v,up,gate,down;
};
#include "54_qwen27_mtp.inc"
#include "55_qwen27_dflash.inc"
class Model {
  Allocations memory;
  hgn::Checkpoint ck;
  Matrix embeddings,head;
  Layer layer[64];
  float *x,*normalized,*tmp,*qkv,*conv,*z,*a,*b,*g,*beta,*attn,*q,*k,*v,*qgate,*scores,*ffgate,*ffup,*logits,*finalnorm;
  Blas blas;
  std::unique_ptr<qwen27_gemm::Lt> lt;
  uint16_t *factored=nullptr,*input_hi=nullptr,*input_lo=nullptr;
  float *bx,*bn,*bt,*bqkv,*bc,*bz,*ba,*bb,*bg,*bbe,*bo,*bq,*bk,*bv,*bqg,*bfg,*bfu,*unpacked;
  int* btokens;
  int* device_input;
  int* batch_positions;
  float *partial_attention,*partial_ml;
  hipStream_t stream=nullptr;
  DecodeGraph decode_graph;
  std::unique_ptr<Mtp> mtp;
  std::unique_ptr<Dflash> dflash;
  float *last_hidden=nullptr,*previous_hidden=nullptr,*hidden_batch=nullptr,*verify_logits=nullptr;
  float *state_snap=nullptr,*conv_snap=nullptr;
  int verified_count=0;
public:
  static constexpr int batch_size=512;
  int prefill_chunk=512;
  bool serial_prefill=false;
  bool fast_prefill=true;
  bool mtp_enabled=false;
  bool dflash_enabled=false;
  int dflash_gamma=4;
  int mtp_gamma=3;
  bool force_batch=false;
  bool graph_enabled=true;
  int maxctx;
  std::vector<int> history;
  std::string trace;
  std::vector<int> score_targets;
  void score_at(int pos){
    if(score_targets.empty())return;
    auto v=read_logits();int top=int(std::max_element(v.begin(),v.end())-v.begin());double maximum=v[top],sum=0;
    for(float x:v)sum+=std::exp(double(x)-maximum);
    printf("score %d %d %d %.12g\n",pos,score_targets.at(pos),top,double(v[score_targets.at(pos)])-maximum-std::log(sum));fflush(stdout);
  }
  explicit Model(const char* path,int capacity,bool enable_mtp=false,bool enable_dflash=false,bool df_full_vocab=true):ck(path),maxctx(capacity) {
    if(maxctx<1||maxctx>262144)throw std::runtime_error("Qwen27 context must be 1..262144");
    auto mat=[&](const std::string& name,int r,int c){
      const auto& t=required(ck,name,{uint64_t(r),uint64_t(c)});
      auto* p=memory.get<uint8_t>(t.data_size);
      check(hipMemcpy(p,t.data,t.data_size,hipMemcpyHostToDevice));
      return Matrix{p,t.dtype==5&&t.qparam==1?51:int(t.dtype),r,c,can_factor(t)};
    };
    auto vec=[&](const std::string& name,std::initializer_list<uint64_t> dims){
      auto host=dequant(required(ck,name,dims));
      float* p=memory.get<float>(host.size());
      check(hipMemcpy(p,host.data(),host.size()*4,hipMemcpyHostToDevice));return p;
    };
    embeddings=mat("embed_tokens.weight",vocab,hidden);
    if(embeddings.dtype!=0)throw std::runtime_error("Qwen27 embedding must be BF16");
    head=mat("lm_head.weight",vocab,hidden);
    finalnorm=vec("norm.weight",{hidden});
    for(int l=0;l<64;++l){
      Layer& w=layer[l];std::string p="layers."+std::to_string(l)+".";
      w.norm=vec(p+"input_layernorm.weight",{hidden});
      w.post=vec(p+"post_attention_layernorm.weight",{hidden});
      w.gate=mat(p+"mlp.gate_proj.weight",intermediate,hidden);
      w.up=mat(p+"mlp.up_proj.weight",intermediate,hidden);
      w.down=mat(p+"mlp.down_proj.weight",hidden,intermediate);
      if(l%4!=3){
        p+="linear_attn.";
        w.qkv=mat(p+"in_proj_qkv.weight",10240,hidden); w.z=mat(p+"in_proj_z.weight",6144,hidden);
        w.a=mat(p+"in_proj_a.weight",48,hidden);w.b=mat(p+"in_proj_b.weight",48,hidden);
        w.out=mat(p+"out_proj.weight",hidden,6144);
        w.conv=vec(p+"conv1d.weight",{10240,1,4});w.alog=vec(p+"A_log",{48});
        w.dt=vec(p+"dt_bias",{48});w.gnorm=vec(p+"norm.weight",{128});
        w.state=memory.get<float>(48*128*128);w.history=memory.get<float>(3*10240);
      }else{
        p+="self_attn.";
        w.q=mat(p+"q_proj.weight",12288,hidden);w.k=mat(p+"k_proj.weight",1024,hidden);
        w.v=mat(p+"v_proj.weight",1024,hidden);w.out=mat(p+"o_proj.weight",hidden,6144);
        w.qnorm=vec(p+"q_norm.weight",{256});w.knorm=vec(p+"k_norm.weight",{256});
        w.keys=memory.get<float>(size_t(maxctx)*1024);w.values=memory.get<float>(size_t(maxctx)*1024);
      }
      fprintf(stderr,"qwen27: loaded layer %d/64\n",l+1);
    }
    x=memory.get<float>(hidden);normalized=memory.get<float>(hidden);tmp=memory.get<float>(hidden);
    qkv=memory.get<float>(12288);conv=memory.get<float>(10240);z=memory.get<float>(6144);
    a=memory.get<float>(48);b=memory.get<float>(48);g=memory.get<float>(48);beta=memory.get<float>(48);
    attn=memory.get<float>(6144);q=memory.get<float>(6144);k=memory.get<float>(1024);v=memory.get<float>(1024);
    qgate=memory.get<float>(6144);scores=memory.get<float>(size_t(24)*maxctx);
    ffgate=memory.get<float>(intermediate);ffup=memory.get<float>(intermediate);logits=memory.get<float>(vocab);
    bx=memory.get<float>(batch_size*hidden);bn=memory.get<float>(batch_size*hidden);bt=memory.get<float>(batch_size*hidden);
    bqkv=memory.get<float>(batch_size*12288);bc=memory.get<float>(batch_size*10240);bz=memory.get<float>(batch_size*6144);
    ba=memory.get<float>(batch_size*48);bb=memory.get<float>(batch_size*48);bg=memory.get<float>(batch_size*48);bbe=memory.get<float>(batch_size*48);
    bo=memory.get<float>(batch_size*6144);bq=memory.get<float>(batch_size*6144);bk=memory.get<float>(batch_size*1024);bv=memory.get<float>(batch_size*1024);
    bqg=memory.get<float>(batch_size*6144);bfg=memory.get<float>(batch_size*intermediate);bfu=memory.get<float>(batch_size*intermediate);
    btokens=memory.get<int>(batch_size);unpacked=memory.get<float>(size_t(hidden)*intermediate);
    device_input=memory.get<int>(2);batch_positions=memory.get<int>(batch_size);partial_attention=memory.get<float>(24*32*256);partial_ml=memory.get<float>(24*32*2);
    mtp_enabled=enable_mtp;dflash_enabled=enable_dflash;
    if(enable_mtp)mtp=std::make_unique<Mtp>(ck,embeddings,head,capacity);
    if(enable_dflash)dflash=std::make_unique<Dflash>(ck,embeddings,head,capacity,df_full_vocab);
    if(enable_mtp||enable_dflash){
      last_hidden=memory.get<float>(hidden);previous_hidden=memory.get<float>(hidden);
      hidden_batch=memory.get<float>(batch_size*hidden);verify_logits=memory.get<float>((enable_dflash?8:4)*vocab);
      state_snap=memory.get<float>(size_t(enable_dflash?7:3)*48*48*128*128);
      conv_snap=memory.get<float>(size_t(enable_dflash?7:3)*48*3*10240);
    }
    reset();memstats_log("qwen27-loaded");
  }
  void reset(){
    for(int l=0;l<64;++l)if(l%4!=3){
      check(hipMemset(layer[l].state,0,48*128*128*4));
      check(hipMemset(layer[l].history,0,3*10240*4));
    }
    history.clear();
    if(last_hidden)check(hipMemset(last_hidden,0,hidden*4));
  }
  void mv(const Matrix& w,const float* input,float* output){
    dim3 grid((w.rows+3)/4);
    if(w.dtype==0)qwen27_fast::gemv<0,1><<<grid,128,0,stream>>>(w.data,input,output,w.rows,w.cols);
    else if(w.dtype==5)qwen27_fast::gemv<5,1><<<grid,128,0,stream>>>(w.data,input,output,w.rows,w.cols);
    else if(w.dtype==6)qwen27_fast::gemv<6,1><<<grid,128,0,stream>>>(w.data,input,output,w.rows,w.cols);
    else if(w.dtype==51)qwen27_fast::gemv<51,1><<<grid,128,0,stream>>>(w.data,input,output,w.rows,w.cols);
    else throw std::runtime_error("unsupported Qwen27 matrix");
  }
  void dump_at(int position,const std::string& name,const float* ptr,size_t n){
    if(trace.empty())return;
    std::vector<float> host(n);check(hipMemcpy(host.data(),ptr,n*4,hipMemcpyDeviceToHost));
    std::filesystem::create_directories(trace);
    std::ofstream f(trace+"/t"+std::to_string(position)+"-"+name+".f32",std::ios::binary);
    f.write(reinterpret_cast<const char*>(host.data()),n*4);
    if(!f)throw std::runtime_error("failed to write Qwen27 trace");
  }
  void dump(const std::string& name,const float* ptr,size_t n){dump_at(int(history.size()),name,ptr,n);}
  void mm(const Matrix& w,const float* input,float* output,int count){
    // The vocabulary head exceeds the bounded matrix scratch. Always keep its
    // short batches on GEMV, including when a diagnostic forces batched prefill.
    if(count<=8&&(!force_batch||w.rows>intermediate)){
      #define Q27_BATCH(D,P) qwen27_fast::batch<D,P><<<(w.rows+3)/4,128>>>(w.data,input,output,w.rows,w.cols)
      #define Q27_SIZE(D) switch(count){case 1:Q27_BATCH(D,1);break;case 2:Q27_BATCH(D,2);break;case 3:Q27_BATCH(D,3);break;case 4:Q27_BATCH(D,4);break;case 5:Q27_BATCH(D,5);break;case 6:Q27_BATCH(D,6);break;case 7:Q27_BATCH(D,7);break;case 8:Q27_BATCH(D,8);break;}
      if(w.dtype==0){Q27_SIZE(0);}else if(w.dtype==5){Q27_SIZE(5);}else if(w.dtype==6){Q27_SIZE(6);}else if(w.dtype==51){Q27_SIZE(51);}else throw std::runtime_error("invalid Qwen27 matrix dtype");
      #undef Q27_SIZE
      #undef Q27_BATCH
      return;
    }
    if(size_t(w.rows)*w.cols>size_t(hidden)*intermediate)
      throw std::runtime_error("Qwen27 matrix exceeds bounded GEMM scratch");
    dim3 grid(unsigned((size_t(w.rows)*w.cols+255)/256));
    if(fast_prefill&&w.factorizable&&w.rows>=256){
      if(!lt){
        lt=std::make_unique<qwen27_gemm::Lt>();
        factored=memory.get<uint16_t>(size_t(hidden)*intermediate);
        input_hi=memory.get<uint16_t>(batch_size*intermediate);
        input_lo=memory.get<uint16_t>(batch_size*intermediate);
      }
      if(w.dtype==0)qwen27_gemm::factor<0><<<grid,256>>>(w.data,factored,w.rows,w.cols);
      else if(w.dtype==5)qwen27_gemm::factor<5><<<grid,256>>>(w.data,factored,w.rows,w.cols);
      else if(w.dtype==6)qwen27_gemm::factor<6><<<grid,256>>>(w.data,factored,w.rows,w.cols);
      size_t n=size_t(count)*w.cols;
      qwen27_gemm::split<<<unsigned((n+255)/256),256>>>(input,input_hi,input_lo,n);
      lt->product(factored,input_lo,output,w.rows,count,w.cols,0);
      lt->product(factored,input_hi,output,w.rows,count,w.cols,1);
      if(w.dtype!=0)qwen27_gemm::rescale<<<(w.rows*count+255)/256,256>>>(w.data,output,w.dtype,w.rows,w.cols,count);
      return;
    }
    if(w.dtype==0)unpack_matrix<0><<<grid,256>>>(w.data,unpacked,w.rows,w.cols);
    else if(w.dtype==5)unpack_matrix<5><<<grid,256>>>(w.data,unpacked,w.rows,w.cols);
    else if(w.dtype==6)unpack_matrix<6><<<grid,256>>>(w.data,unpacked,w.rows,w.cols);
    else if(w.dtype==51)unpack_matrix<51><<<grid,256>>>(w.data,unpacked,w.rows,w.cols);
    const float one=1,zero=0;
    if(rocblas_sgemm(blas.handle,rocblas_operation_transpose,rocblas_operation_none,
         w.rows,count,w.cols,&one,unpacked,w.cols,input,w.cols,&zero,output,w.rows)!=rocblas_status_success)
      throw std::runtime_error("Qwen27 FP32 GEMM failed");
  }
  void prefill(const int* tokens,int count,const std::atomic<bool>* cancel=nullptr,bool verify=false){
    if(count<1||count>batch_size||history.size()+count>size_t(maxctx))throw std::runtime_error("invalid Qwen27 prefill size");
    for(int i=0;i<count;++i)if(tokens[i]<0||tokens[i]>=vocab)throw std::runtime_error("invalid Qwen27 token");
    if(!verify&&!force_batch&&(serial_prefill||(count<=8&&trace.empty()))){for(int i=0;i<count;++i){if(cancel&&*cancel)return;forward(tokens[i],i+1==count||!trace.empty()||!score_targets.empty());}return;}
    if(verify&&(!(mtp||dflash)||count<2||count>(dflash?8:4)))throw std::runtime_error("invalid Qwen27 verification batch");
    const int pos=int(history.size());
    if(verify)fill_positions<<<1,512>>>(batch_positions,pos,count);
    if(mtp)check(hipMemcpyAsync(previous_hidden,last_hidden,hidden*4,hipMemcpyDeviceToDevice));
    check(hipMemcpy(btokens,tokens,count*sizeof(int),hipMemcpyHostToDevice));
    embedding_batch<<<dim3(20,count),256>>>(reinterpret_cast<const uint16_t*>(embeddings.data),btokens,bx);
    for(int t=0;t<count&&!trace.empty();++t)dump_at(pos+t,"embedding",bx+size_t(t)*hidden,hidden);
    for(int l=0;l<64;++l){
      if(cancel&&*cancel)return;
      Layer& w=layer[l];
      k_rmsnorm_zc_grouped<<<count,256>>>(bx,w.norm,bn,hidden,1e-6f,1);
      if(l%4!=3){
        mm(w.qkv,bn,bqkv,count);mm(w.z,bn,bz,count);mm(w.a,bn,ba,count);mm(w.b,bn,bb,count);
        k_gdn_conv_b_serial_math<<<(count*10240+255)/256,256>>>(bqkv,w.conv,w.history,bc,count,10240);
        const int gl=l-l/4;
        if(verify)k_convst_snap<<<count-1,256>>>(bqkv,w.history,conv_snap+size_t(gl)*3*10240,count-1,10240,size_t(48)*3*10240);
        k_convst_update<<<40,256>>>(w.history,bqkv,count,10240);
        k_l2norm_qk_b<<<dim3(16,count),128>>>(bc,128,1/sqrtf(128),10240,16);
        k_gdn_gates<<<count,64>>>(ba,bb,w.alog,w.dt,bg,bbe,48,48);
        if(verify)k_gdn_snap<<<48,512>>>(bc,bg,bbe,w.state,state_snap+size_t(gl)*48*128*128,count-1,10240,size_t(48)*48*128*128);
        k_gdn_verify<<<48,512>>>(bc,bg,bbe,w.state,bc,count,10240);
        gated_norm_batch<<<dim3(48,count),128>>>(bc,bz,w.gnorm,bo);
      }else{
        mm(w.q,bn,bqkv,count);mm(w.k,bn,bk,count);mm(w.v,bn,bv,count);
        prepare_qk<<<dim3(24,count),256>>>(bqkv,w.qnorm,bq,bqg,512,pos);
        prepare_qk<<<dim3(4,count),256>>>(bk,w.knorm,w.keys+size_t(pos)*1024,nullptr,256,pos);
        check(hipMemcpyAsync(w.values+size_t(pos)*1024,bv,size_t(count)*4096,hipMemcpyDeviceToDevice));
        if(verify){
          for(int t=0;t<count;++t){
            k_qsa_flash<true,false,float,true><<<dim3(32,4),128>>>(bq+size_t(t)*6144,w.keys,w.values,bo+size_t(t)*6144,1,256,4,24,nullptr,0,batch_positions+t,partial_attention,partial_ml);
            k_qsa_flash_combine<<<dim3(6,4),64>>>(partial_ml,partial_attention,bo+size_t(t)*6144,6,256,batch_positions+t);
          }
        }else k_qsa_flash<false,false,float><<<dim3((count+15)/16,24),128>>>(bq,w.keys,w.values,bo,count,256,4,24,nullptr,0,nullptr,nullptr,nullptr,pos,nullptr);
        k_sigmoid_gate<<<(count*6144+255)/256,256>>>(bo,bqg,bo,count*6144);
      }
      mm(w.out,bo,bt,count);add_residual<<<(count*hidden+255)/256,256>>>(bx,bt,count*hidden);
      k_rmsnorm_zc_grouped<<<count,256>>>(bx,w.post,bn,hidden,1e-6f,1);
      mm(w.gate,bn,bfg,count);mm(w.up,bn,bfu,count);
      k_silu_mul<<<(count*intermediate+255)/256,256>>>(bfg,bfu,bfg,count*intermediate);
      mm(w.down,bfg,bt,count);add_residual<<<(count*hidden+255)/256,256>>>(bx,bt,count*hidden);
      if(dflash)dflash->capture(l,bx,count);
      for(int t=0;t<count&&!trace.empty();++t)dump_at(pos+t,"layer"+std::to_string(l),bx+size_t(t)*hidden,hidden);
    }
    if(mtp||dflash)k_rmsnorm_zc_grouped<<<count,256>>>(bx,finalnorm,hidden_batch,hidden,1e-6f,1);
    if(verify){
      mm(head,hidden_batch,verify_logits,count);
      check(hipMemcpyAsync(logits,verify_logits+size_t(count-1)*vocab,vocab*4,hipMemcpyDeviceToDevice));
    }
    for(int t=verify?count:((trace.empty()&&score_targets.empty())?count-1:0);t<count;++t){
      k_rmsnorm_zc<<<1,256>>>(bx+size_t(t)*hidden,finalnorm,normalized,hidden,1e-6f);
      mv(head,normalized,logits);dump_at(pos+t,"logits",logits,vocab);score_at(pos+t);
    }
    if(mtp&&!verify){
      mtp->sync(btokens,hidden_batch,previous_hidden,count,pos,[&](const Matrix& w,const float* in,float* out,int n){mm(w,in,out,n);});
      check(hipMemcpyAsync(last_hidden,hidden_batch+size_t(count-1)*hidden,hidden*4,hipMemcpyDeviceToDevice));
    }
    if(dflash&&!verify)dflash->sync(count,pos);
    verified_count=verify?count:0;
    check(hipGetLastError());history.insert(history.end(),tokens,tokens+count);
  }
  void launch_decoder(int token,int pos,bool readout,bool dynamic=false){
    if(dynamic)embedding_device<<<20,256,0,stream>>>(reinterpret_cast<const uint16_t*>(embeddings.data),x,device_input);
    else embedding<<<20,256,0,stream>>>(reinterpret_cast<const uint16_t*>(embeddings.data),x,token);
    dump("embedding",x,hidden);
    for(int l=0;l<64;++l){
      Layer& w=layer[l];
      k_rmsnorm_zc<<<1,256,0,stream>>>(x,w.norm,normalized,hidden,1e-6f);
      if(l%4!=3){
        mv(w.qkv,normalized,qkv);mv(w.z,normalized,z);mv(w.a,normalized,a);mv(w.b,normalized,b);
        k_gdn_conv<<<40,256,0,stream>>>(qkv,w.conv,w.history,conv,10240);
        k_l2norm_qk<<<16,128,0,stream>>>(conv,conv+2048,128,1/sqrtf(128));
        k_gdn_gates<<<1,64,0,stream>>>(a,b,w.alog,w.dt,g,beta,48,48);
        k_gdn_step<<<48,512,0,stream>>>(conv,conv+2048,conv+4096,g,beta,w.state,attn,128,128,48,16);
        gated_norm<<<48,128,0,stream>>>(attn,z,w.gnorm,attn);
      }else{
        mv(w.q,normalized,qkv);mv(w.k,normalized,k);mv(w.v,normalized,v);
        if(dynamic){
          const int* position=device_input+1;
          prepare_qk<<<24,256,0,stream>>>(qkv,w.qnorm,q,qgate,512,0,position);
          prepare_qk<<<4,256,0,stream>>>(k,w.knorm,w.keys,nullptr,256,0,position,true);
          cache_values<<<4,256,0,stream>>>(v,w.values,position);
          k_qsa_flash<true,false,float,true><<<dim3(32,4),128,0,stream>>>(q,w.keys,w.values,attn,1,256,4,24,nullptr,0,position,partial_attention,partial_ml);
          k_qsa_flash_combine<<<dim3(6,4),64,0,stream>>>(partial_ml,partial_attention,attn,6,256,position);
          k_sigmoid_gate<<<24,256,0,stream>>>(attn,qgate,attn,6144);
        }else{
          prepare_qk<<<24,256,0,stream>>>(qkv,w.qnorm,q,qgate,512,pos);
          prepare_qk<<<4,256,0,stream>>>(k,w.knorm,w.keys+size_t(pos)*1024,nullptr,256,pos);
          check(hipMemcpyAsync(w.values+size_t(pos)*1024,v,4096,hipMemcpyDeviceToDevice,stream));
          attention_scores<<<dim3((pos+4)/4,24),128,0,stream>>>(q,w.keys,scores,pos,maxctx);
          attention_output<<<24,256,0,stream>>>(scores,w.values,qgate,attn,pos,maxctx);
        }
      }
      mv(w.out,attn,tmp);add_residual<<<20,256,0,stream>>>(x,tmp,hidden);
      k_rmsnorm_zc<<<1,256,0,stream>>>(x,w.post,normalized,hidden,1e-6f);
      mv(w.gate,normalized,ffgate);mv(w.up,normalized,ffup);
      k_silu_mul<<<68,256,0,stream>>>(ffgate,ffup,ffgate,intermediate);
      mv(w.down,ffgate,tmp);add_residual<<<20,256,0,stream>>>(x,tmp,hidden);
      if(dflash)dflash->capture(l,x,1,stream);
      dump("layer"+std::to_string(l),x,hidden);
    }
    if(readout||mtp){
      k_rmsnorm_zc<<<1,256,0,stream>>>(x,finalnorm,normalized,hidden,1e-6f);
      if(mtp)check(hipMemcpyAsync(last_hidden,normalized,hidden*4,hipMemcpyDeviceToDevice,stream));
      mv(head,normalized,logits);dump("logits",logits,vocab);
    }
    check(hipGetLastError());
  }
  void forward(int token,bool readout=true){
    int pos=int(history.size());
    if(token<0||token>=vocab||pos>=maxctx)throw std::runtime_error("Qwen27 token/context outside bounds");
    if(mtp&&pos>0)mtp->step(token,pos-1,last_hidden,false);
    if(graph_enabled&&readout&&trace.empty()){
      if(!decode_graph.executable){
        check(hipStreamCreateWithFlags(&decode_graph.capture,hipStreamNonBlocking));
        stream=decode_graph.capture;
        try {
          check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
          launch_decoder(0,0,true,true);
          check(hipStreamEndCapture(stream,&decode_graph.graph));
        } catch (...) {
          hipGraph_t abandoned=nullptr;
          (void)hipStreamEndCapture(stream,&abandoned);
          if(abandoned)(void)hipGraphDestroy(abandoned);
          (void)hipStreamDestroy(decode_graph.capture);decode_graph.capture=nullptr;
          stream=nullptr;graph_enabled=false;throw;
        }
        stream=nullptr;
        check(hipGraphInstantiate(&decode_graph.executable,decode_graph.graph,nullptr,nullptr,0));
      }
      int input[2]={token,pos};check(hipMemcpy(device_input,input,sizeof input,hipMemcpyHostToDevice));
      check(hipGraphLaunch(decode_graph.executable,nullptr));
    }else launch_decoder(token,pos,readout);
    if(dflash)dflash->sync(1,pos);
    score_at(pos);
    history.push_back(token);
  }
  std::vector<float> read_logits(){
    std::vector<float> host(vocab);check(hipMemcpy(host.data(),logits,host.size()*4,hipMemcpyDeviceToHost));
    for(float f:host)if(!std::isfinite(f))throw std::runtime_error("nonfinite Qwen27 logits");
    return host;
  }
  int draft(int token){
    int result=mtp->step(token,int(history.size())-1,last_hidden,true);
    dump_at(int(history.size())-1,"mtp-logits",mtp->logits,vocab);return result;
  }
  std::vector<int> draft_sequence(int token,int gamma){
    if(dflash)return dflash->propose(token,int(history.size()),gamma,[&](const std::string& name,const float* p,size_t n){dump(name,p,n);});
    std::vector<int> result;const float* h=last_hidden;
    for(int i=0;i<gamma;++i){token=mtp->step(token,int(history.size())-1+i,h,true);result.push_back(token);h=mtp->normalized;}
    return result;
  }
  std::vector<float> verification_scores(){
    std::vector<float> host(size_t(verified_count)*vocab);
    check(hipMemcpy(host.data(),verify_logits,host.size()*4,hipMemcpyDeviceToHost));
    for(float f:host)if(!std::isfinite(f))throw std::runtime_error("nonfinite verification logits");
    return host;
  }
  void commit_verified(int accepted){
    if(!(mtp||dflash)||accepted<1||accepted>verified_count)throw std::runtime_error("invalid verification commit");
    int start=int(history.size())-verified_count;
    if(accepted<verified_count){
      for(int l=0;l<64;++l)if(l%4!=3){int gl=l-l/4;
        check(hipMemcpyAsync(layer[l].state,state_snap+(size_t(accepted-1)*48+gl)*48*128*128,48*128*128*4,hipMemcpyDeviceToDevice));
        check(hipMemcpyAsync(layer[l].history,conv_snap+(size_t(accepted-1)*48+gl)*3*10240,3*10240*4,hipMemcpyDeviceToDevice));
      }
    }
    if(dflash)dflash->sync(accepted,start);
    if(mtp)mtp->sync(btokens,hidden_batch,previous_hidden,accepted,start,[&](const Matrix& w,const float* in,float* out,int n){mm(w,in,out,n);});
    check(hipMemcpyAsync(last_hidden,hidden_batch+size_t(accepted-1)*hidden,hidden*4,hipMemcpyDeviceToDevice));
    check(hipMemcpyAsync(logits,verify_logits+size_t(accepted-1)*vocab,vocab*4,hipMemcpyDeviceToDevice));
    history.resize(start+accepted);verified_count=0;
  }
};
static double milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
static std::mutex model_mutex;
static std::atomic<int> active_requests{0};
static void generate(Model& m,serve::Conn& c,serve::GenReq g) {
  std::unique_lock<std::mutex> lock(model_mutex,std::try_to_lock);
  auto finish=[&](const std::string& text){c.in_flight=false;if(!c.send(text))sock_shutdown(c.fd);};
  auto error=[&](){finish("D "+std::to_string(g.req)+" error "+std::to_string(g.ids.size())+" 0 0 0\n");};
  if(!lock.owns_lock()||g.reject||g.drafter>2||(g.drafter==1&&!m.mtp_enabled)||(g.drafter==2&&!m.dflash_enabled)||!g.mrope_grids.empty()||g.vimg||
     g.ids.empty()||g.ids.size()>size_t(m.maxctx)||g.max_tokens<1||
     uint64_t(g.ids.size())+uint64_t(g.max_tokens)-1>uint64_t(m.maxctx)) {error();return;}
  for(int id:g.ids)if(id<0||id>=vocab){error();return;}
  if(g.sample&&(!std::isfinite(g.temp)||!std::isfinite(g.top_p)||!std::isfinite(g.min_p)||
      !std::isfinite(g.presence)||!std::isfinite(g.frequency))){error();return;}
  for(auto b:g.bias)if(!std::isfinite(b.second)){error();return;}
  active_requests=1;
  try {
    auto started=std::chrono::steady_clock::now();
    size_t cached=m.history.size();
    if(cached>g.ids.size()||!std::equal(m.history.begin(),m.history.end(),g.ids.begin())){m.reset();cached=0;}
    auto progress=[&](bool done){
      if(!g.progress)return;
      size_t n=m.history.size()-cached,total=g.ids.size()-cached;
      std::ostringstream s;s<<"P "<<g.req<<' '<<int(done)<<' '<<n<<' '<<n<<' '<<total<<' '<<cached<<' '<<milliseconds(started)<<'\n';
      if(!c.send(s.str()))c.cancel=true;
    };
    progress(false);
    for(size_t i=cached;i<g.ids.size()&&!c.cancel;){
      int count=int(std::min(g.ids.size()-i,size_t(m.prefill_chunk)));
      m.prefill(g.ids.data()+i,count,&c.cancel);i+=count;
      check(hipDeviceSynchronize());progress(false);
    }
    check(hipDeviceSynchronize());
    const double prefill_ms=milliseconds(started);
    if(!c.cancel)progress(true);
    auto decode_start=std::chrono::steady_clock::now();
    serve::HostSampler sampler{};
    sampler.temp=g.temp;sampler.top_p=g.top_p;sampler.min_p=g.min_p;sampler.top_k=g.top_k;
    sampler.rng=g.seed;sampler.presence=g.presence;sampler.frequency=g.frequency;sampler.bias=g.bias;
    int generated=0,drafted=0,accepted=0,rounds=0;
    std::string reason="length";
    auto emit=[&](int token,float lp){
      std::ostringstream out;out<<"T "<<g.req<<' '<<token;if(g.logprobs)out<<' '<<lp;out<<'\n';
      if(!c.send(out.str())){c.cancel=true;return false;}++generated;return true;
    };
    const bool speculative=(m.mtp_enabled||m.dflash_enabled)&&g.drafter!=0;
    while(generated<g.max_tokens&&!c.cancel){
      auto scores=m.read_logits();float lp=0;
      int token=g.sample?sampler.sample(scores,&lp):int(std::max_element(scores.begin(),scores.end())-scores.begin());
      if(std::find(g.eos.begin(),g.eos.end(),token)!=g.eos.end()){reason="done";break;}
      if(!emit(token,lp)||generated==g.max_tokens)break;
      if(speculative&&g.max_tokens-generated>1&&m.history.size()+2<=size_t(m.maxctx)){
        int gamma=std::min({m.dflash_enabled?m.dflash_gamma:m.mtp_gamma,g.max_tokens-generated-1,int(m.maxctx-m.history.size())-1});
        auto proposals=m.draft_sequence(token,gamma);std::vector<int> batch{token};batch.insert(batch.end(),proposals.begin(),proposals.end());
        m.prefill(batch.data(),int(batch.size()),nullptr,true);auto verify=m.verification_scores();
        int keep=1;bool stop=false;std::vector<float> proposal_lp(gamma,0);
        for(int j=0;j<gamma;++j){
          auto begin=verify.begin()+size_t(j)*vocab;
          int expected;
          if(g.sample){
            // Draw only from the target distribution. A rejected draft restores
            // RNG and emission counts so the next serial step repeats this draw.
            // Accepted drafts consume exactly one ordinary target sample each.
            auto rng=sampler.rng;auto hist=sampler.hist;
            std::vector<float> target(begin,begin+vocab);
            expected=sampler.sample(target,&proposal_lp[j]);
            if(proposals[j]!=expected){sampler.rng=rng;sampler.hist=std::move(hist);break;}
          }else expected=int(std::max_element(begin,begin+vocab)-begin);
          if(proposals[j]!=expected)break;
          if(std::find(g.eos.begin(),g.eos.end(),expected)!=g.eos.end()){stop=true;break;}
          ++keep;
        }
        m.commit_verified(keep);drafted+=gamma;++rounds;accepted+=keep-1;
        for(int j=0;j<keep-1&&!c.cancel;++j)emit(proposals[j],proposal_lp[j]);
        if(stop){reason="done";break;}
      }else m.forward(token);
    }
    if(c.cancel){reason="cancel";m.reset();}
    std::ostringstream done;
    done<<"D "<<g.req<<' '<<reason<<' '<<g.ids.size()<<' '<<generated<<' '<<prefill_ms<<' '
        <<milliseconds(decode_start)<<' '<<(speculative?(m.dflash_enabled?2:1):0)<<' '<<rounds<<' '<<rounds+accepted<<' '<<cached<<' '<<drafted<<'\n';
    if(speculative)fprintf(stderr,"qwen27 %s: drafted=%d accepted=%d rounds=%d\n",m.dflash_enabled?"DFlash2":"MTP",drafted,accepted,rounds);
    active_requests=0;finish(done.str());
  }catch(const std::exception& e){
    fprintf(stderr,"qwen27 request: %s\n",e.what());
    try{m.reset();}catch(...){}
    active_requests=0;error();
  }
}
static void connection(Model& m,sockfd_t fd) {
  serve::Conn c;c.fd=fd;
  serve::LineBuf lines{fd,{}};std::thread worker;std::string line;
  while(lines.readline(line)){
    if(line.size()>size_t(m.maxctx)*12+1048576)break;
    std::istringstream in(line);std::string verb;in>>verb;
    if(verb=="PING"){if(!c.send("PONG\n"))break;}
    else if(verb=="INFO"){
      std::ostringstream out;out<<"I "<<(m.dflash_enabled?2:int(m.mtp_enabled))<<" 0 "<<m.maxctx<<' '<<(m.dflash_enabled?m.dflash_gamma:m.mtp_gamma)+1<<' '<<int(m.mtp_enabled||m.dflash_enabled)<<' '<<int(m.mtp_enabled)<<" 0 0 0 1 "<<m.maxctx<<" 1 1 "<<m.maxctx<<'\n';
      if(!c.send(out.str()))break;
    }else if(verb=="CSTAT"){
      if(!c.send("C 1 "+std::to_string(active_requests.load())+" 0 0\n"))break;
    }else if(verb=="MEM"){
      auto ms=memstats_snapshot();std::ostringstream out;
      out<<"M 1 "<<ms.device_current<<' '<<ms.device_peak<<' '<<ms.registered_current<<' '<<ms.pinned_current<<' '
         <<ms.pinned_peak<<' '<<ms.committed_current<<' '<<ms.hip_free<<' '<<ms.hip_total<<' '
         <<ms.hip_used_since_start<<' '<<ms.process_rss<<' '<<ms.process_locked<<'\n';
      if(!c.send(out.str()))break;
    }else if(verb=="KVSYNC"){if(!c.send("K 0 0 0 0\n"))break;}
    else if(verb=="X"){
      int64_t req=0;in>>req;
      if(c.in_flight&&c.req==req)c.cancel=true;
      else if(!c.send("D "+std::to_string(req)+" cancel 0 0 0 0\n"))break;
    }else if(verb=="GEN"){
      // Preflight the count before the shared parser allocates its ID vector.
      std::istringstream bounds(line);std::string tag;int64_t req;int nt,ne,ni=-1;
      bool valid=bool(bounds>>tag>>req>>nt>>ne)&&ne>=0&&ne<=64;
      for(int i=0;valid&&i<ne;++i){int eos;valid=bool(bounds>>eos);}
      valid=valid&&bool(bounds>>ni)&&ni>0&&ni<=m.maxctx;
      if(!valid)break;
      auto request=serve::parse_gen(in);
      // Text-only connections cannot safely interpret binary vision frames.
      if(request.vimg)break;
      if(c.in_flight){if(!c.send("D "+std::to_string(request.req)+" error 0 0 0 0\n"))break;continue;}
      if(worker.joinable())worker.join();
      c.cancel=false;c.req=request.req;c.in_flight=true;
      worker=std::thread(generate,std::ref(m),std::ref(c),std::move(request));
    }else if(!c.send("E unknown command\n"))break;
  }
  c.cancel=true;if(worker.joinable())worker.join();sock_close(fd);
}
static int listen(Model& m,const std::string& host,int port){
  sockaddr_in addr{};
  if(!engine_net::address(host,port,&addr))throw std::runtime_error("invalid engine listen address");
  sockfd_t fd=sock_tcp();
  if(fd<0)throw std::runtime_error("socket creation failed");
  int one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,reinterpret_cast<const char*>(&one),sizeof one);
  if(bind(fd,reinterpret_cast<sockaddr*>(&addr),sizeof addr)||::listen(fd,16)){
    sock_close(fd);throw std::runtime_error("engine bind/listen failed");
  }
  fprintf(stderr,"qwen27: listening on %s:%d (context %d, one slot, %s, gamma %d, prefill %s/%d)\n",host.c_str(),port,m.maxctx,m.dflash_enabled?"DFlash2":(m.mtp_enabled?"MTP":"serial"),m.dflash_enabled?m.dflash_gamma:(m.mtp_enabled?m.mtp_gamma:0),m.fast_prefill?"compensated":"FP32",m.prefill_chunk);
  for(;;){sockfd_t client=accept(fd,nullptr,nullptr);if(client<0)continue;
    configure_send_timeout(client);
    setsockopt(client,IPPROTO_TCP,TCP_NODELAY,reinterpret_cast<const char*>(&one),sizeof one);
    std::thread(connection,std::ref(m),client).detach();}
}
static int main(int argc,char** argv){
  try{
    int capacity=8192,gen=8,port=8730,prefill_chunk=512,mtp_probe=-1,mtp_gamma=3,df_probe=-1,df_gamma=4;bool serving=false,serial_prefill=false,graph_enabled=true,fast_prefill=true,mtp_enabled=false,force_batch=false,df_enabled=false,score=false,df_full_vocab=true;
    if(const char* configured=getenv("GDEC_PREFILL_CHUNK"))if(*configured)prefill_chunk=std::stoi(configured);
    if(const char* configured=getenv("GDEC_QWEN27_MTP")){std::string value=configured;if(value!="0"&&value!="1")throw std::runtime_error("GDEC_QWEN27_MTP must be 0 or 1");mtp_enabled=value=="1";}
    if(const char* configured=getenv("GDEC_QWEN27_MTP_GAMMA"))if(*configured)mtp_gamma=std::stoi(configured);
    if(const char* configured=getenv("GDEC_QWEN27_PREFILL_MODE")){std::string value=configured;if(value!="fp32"&&value!="compensated")throw std::runtime_error("invalid Qwen27 prefill mode");fast_prefill=value=="compensated";}
    if(const char* configured=getenv("GDEC_QWEN27_DFLASH")){std::string v=configured;if(v!="0"&&v!="1")throw std::runtime_error("GDEC_QWEN27_DFLASH must be 0 or 1");df_enabled=v=="1";}
    if(const char* configured=getenv("GDEC_QWEN27_DFLASH_GAMMA"))if(*configured)df_gamma=std::stoi(configured);
    std::string host="127.0.0.1",trace;std::vector<int> tokens;
    auto parse_tokens=[&](std::string s){std::replace(s.begin(),s.end(),',',' ');std::istringstream in(s);int t;while(in>>t)tokens.push_back(t);if(!in.eof())throw std::runtime_error("invalid token list");};
    for(int i=2;i<argc;++i){
      std::string key=argv[i];
      auto next=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+key);return std::string(argv[++i]);};
      if(key=="--serve")serving=true;
      else if(key=="--qwen27-score")score=true;
      else if(key=="--qwen27-serial-prefill")serial_prefill=true;
      else if(key=="--qwen27-no-graph")graph_enabled=false;
      else if(key=="--qwen27-mtp")mtp_enabled=true;
      else if(key=="--qwen27-no-mtp")mtp_enabled=false;
      else if(key=="--qwen27-dflash")df_enabled=true;
      else if(key=="--qwen27-no-dflash")df_enabled=false;
      else if(key=="--qwen27-dflash-reduced-vocab")df_full_vocab=false;
      else if(key=="--qwen27-dflash-gamma")df_gamma=std::stoi(next());
      else if(key=="--qwen27-dflash-probe"){df_probe=std::stoi(next());if(df_probe<0||df_probe>=vocab)throw std::runtime_error("invalid DFlash2 probe token");df_enabled=true;}
      else if(key=="--qwen27-mtp-gamma")mtp_gamma=std::stoi(next());
      else if(key=="--qwen27-mtp-probe"){mtp_probe=std::stoi(next());mtp_enabled=true;}
      else if(key=="--qwen27-force-batch")force_batch=true;
      else if(key=="--qwen27-prefill-mode"){auto mode=next();if(mode!="fp32"&&mode!="compensated")throw std::runtime_error("invalid Qwen27 prefill mode");fast_prefill=mode=="compensated";}
      else if(key=="--qwen27-prefill-chunk")prefill_chunk=std::stoi(next());
      else if(key=="--tokens")parse_tokens(next());
      else if(key=="--tokens-file"){std::ifstream f(next());if(!f)throw std::runtime_error("tokens file unreadable");std::string s;while(std::getline(f,s))parse_tokens(s);}
      else if(key=="--maxctx")capacity=std::stoi(next());
      else if(key=="--gen")gen=std::stoi(next());
      else if(key=="--port")port=std::stoi(next());
      else if(key=="--host")host=next();
      else if(key=="--qwen27-trace")trace=next();
      else if(key!="--no-ple")throw std::runtime_error("unsupported Qwen27 argument: "+key);
    }
    if(!serving&&(tokens.empty()||gen<0||uint64_t(tokens.size())+std::max(gen-1,0)>uint64_t(capacity)))
      throw std::runtime_error("invalid tokens/generation/context");
    if(capacity<1||capacity>262144)throw std::runtime_error("Qwen27 context must be 1..262144");
    sockaddr_in address{};
    if(serving&&!engine_net::address(host,port,&address))throw std::runtime_error("invalid engine listen address");
    if(prefill_chunk<1||prefill_chunk>Model::batch_size)throw std::runtime_error("Qwen27 prefill chunk must be 1..512");
    if(mtp_gamma<1||mtp_gamma>3)throw std::runtime_error("Qwen27 MTP gamma must be 1..3");
    if(df_gamma<1||df_gamma>7)throw std::runtime_error("Qwen27 DFlash2 gamma must be 1..7");
    if(df_probe>=vocab)throw std::runtime_error("invalid DFlash2 probe token");
    if(df_enabled&&mtp_probe>=0)throw std::runtime_error("MTP probe cannot be combined with DFlash2");
    if(df_enabled)mtp_enabled=false;
    if(score&&(serving||tokens.size()<2||gen!=0||mtp_enabled||df_enabled||!trace.empty()))throw std::runtime_error("score requires >=2 tokens, --gen 0 and no draft/serve/trace");
    // The final teacher-forced token becomes a readout target, not an input to
    // prefill; validate it here before removing it from the input sequence.
    if(score)for(int token:tokens)if(token<0||token>=vocab)throw std::runtime_error("invalid Qwen27 score token");
    validate_header(argv[1]);
#ifdef _WIN32
    _putenv_s("GDEC_DRAFTER",df_enabled?"dflash2":(mtp_enabled?"mtp":"serial"));
#else
    setenv("GDEC_DRAFTER",df_enabled?"dflash2":(mtp_enabled?"mtp":"serial"),1);
#endif
    check(hipSetDevice(0));
    memstats_capture_hip_baseline();
    Model model(argv[1],capacity,mtp_enabled,df_enabled,df_full_vocab);model.trace=trace;model.serial_prefill=serial_prefill;model.graph_enabled=graph_enabled;model.prefill_chunk=prefill_chunk;model.fast_prefill=fast_prefill;model.force_batch=force_batch;
    model.mtp_gamma=mtp_gamma;model.dflash_gamma=df_gamma;
    if(score){model.score_targets.assign(tokens.begin()+1,tokens.end());tokens.pop_back();}
    if(serving)return listen(model,host,port);
    if(mtp_enabled&&gen>0)fprintf(stderr,"qwen27: CLI --gen uses serial target diagnostics; MTP generation is available with --serve\n");
    auto start=std::chrono::steady_clock::now();
    for(size_t i=0;i<tokens.size();){int count=int(std::min(tokens.size()-i,size_t(model.prefill_chunk)));model.prefill(tokens.data()+i,count);i+=count;}
    check(hipDeviceSynchronize());double prefill=milliseconds(start);
    if(df_probe>=0){auto ids=model.draft_sequence(df_probe,df_gamma);printf("dflash proposals");for(int id:ids)printf(" %d",id);printf("\n");}
    if(mtp_probe>=0){if(mtp_probe>=vocab)throw std::runtime_error("invalid MTP probe token");printf("mtp proposal %d\n",model.draft(mtp_probe));}
    start=std::chrono::steady_clock::now();
    for(int i=0;i<gen;++i){auto values=model.read_logits();int id=int(std::max_element(values.begin(),values.end())-values.begin());
      printf("%s%d",i?",":"",id);fflush(stdout);if(i+1<gen)model.forward(id);}
    printf("\n");fprintf(stderr,"qwen27: prefill %zu tokens %.3f ms; decode %d tokens %.3f ms\n",tokens.size(),prefill,gen,milliseconds(start));
    return 0;
  }catch(const std::exception& e){fprintf(stderr,"qwen27: %s\n",e.what());return 1;}
}
}  // namespace qwen27
