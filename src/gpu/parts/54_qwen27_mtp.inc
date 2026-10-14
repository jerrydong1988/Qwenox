// Included inside namespace qwen27, after the shared text kernels and Layer.
// Qwen MTP consumes the target's final normalized hidden state at t and the
// embedding at t+1. Its KV position is t, following vLLM's shifted input rule.
__global__ void mtp_inputs(const uint16_t* emb,const int* tokens,const float* states,
                           const float* previous,const float* ew,const float* hw,
                           float* out,int skip){
  int t=blockIdx.x+skip,tid=threadIdx.x;
  const float* h=t?states+size_t(t-1)*hidden:previous;
  const uint16_t* e=emb+size_t(tokens[t])*hidden;
  __shared__ float se[256],sh[256];float a=0,b=0;
  for(int i=tid;i<hidden;i+=256){float v=bf(e[i]);a+=v*v;b+=h[i]*h[i];}
  se[tid]=a;sh[tid]=b;__syncthreads();
  for(int d=128;d;d>>=1){if(tid<d){se[tid]+=se[tid+d];sh[tid]+=sh[tid+d];}__syncthreads();}
  a=rsqrtf(se[0]/hidden+1e-6f);b=rsqrtf(sh[0]/hidden+1e-6f);
  out+=size_t(blockIdx.x)*hidden*2;
  for(int i=tid;i<hidden;i+=256){out[i]=bf(e[i])*a*(1+ew[i]);out[hidden+i]=h[i]*b*(1+hw[i]);}
}
struct Mtp {
  Allocations memory;
  Matrix embeddings,head,fc;
  Layer layer;
  float *embedding_norm,*hidden_norm,*finalnorm;
  float *joined,*x,*normed,*qkv,*q,*k,*v,*qgate,*attn,*tmp,*fg,*fu,*normalized,*logits,*partial_attention,*partial_ml;
  int *token_device;
  int capacity;
  Mtp(const hgn::Checkpoint& ck,Matrix emb,Matrix lm,int maxctx):embeddings(emb),head(lm),capacity(maxctx){
    auto mat=[&](const std::string& name,int r,int c){const auto& t=required(ck,name,{uint64_t(r),uint64_t(c)});
      auto* p=memory.get<uint8_t>(t.data_size);check(hipMemcpy(p,t.data,t.data_size,hipMemcpyHostToDevice));
      return Matrix{p,t.dtype==5&&t.qparam==1?51:int(t.dtype),r,c,can_factor(t)};};
    auto vec=[&](const std::string& name,int size){auto h=dequant(required(ck,name,{uint64_t(size)}));auto* p=memory.get<float>(size);check(hipMemcpy(p,h.data(),size*4,hipMemcpyHostToDevice));return p;};
    fc=mat("mtp.fc.weight",hidden,2*hidden);
    embedding_norm=vec("mtp.pre_fc_norm_embedding.weight",hidden);hidden_norm=vec("mtp.pre_fc_norm_hidden.weight",hidden);finalnorm=vec("mtp.norm.weight",hidden);
    std::string p="mtp.layers.0.";
    layer.norm=vec(p+"input_layernorm.weight",hidden);layer.post=vec(p+"post_attention_layernorm.weight",hidden);
    layer.q=mat(p+"self_attn.q_proj.weight",12288,hidden);layer.k=mat(p+"self_attn.k_proj.weight",1024,hidden);layer.v=mat(p+"self_attn.v_proj.weight",1024,hidden);
    layer.out=mat(p+"self_attn.o_proj.weight",hidden,6144);layer.qnorm=vec(p+"self_attn.q_norm.weight",256);layer.knorm=vec(p+"self_attn.k_norm.weight",256);
    layer.gate=mat(p+"mlp.gate_proj.weight",intermediate,hidden);layer.up=mat(p+"mlp.up_proj.weight",intermediate,hidden);layer.down=mat(p+"mlp.down_proj.weight",hidden,intermediate);
    layer.keys=memory.get<float>(size_t(capacity)*1024);layer.values=memory.get<float>(size_t(capacity)*1024);
    joined=memory.get<float>(512*hidden*2);x=memory.get<float>(512*hidden);normed=memory.get<float>(512*hidden);
    k=memory.get<float>(512*1024);v=memory.get<float>(512*1024);token_device=memory.get<int>(1);
    qkv=memory.get<float>(12288);q=memory.get<float>(6144);qgate=memory.get<float>(6144);attn=memory.get<float>(6144);
    tmp=memory.get<float>(hidden);normalized=memory.get<float>(hidden);fg=memory.get<float>(intermediate);fu=memory.get<float>(intermediate);
    logits=memory.get<float>(vocab);partial_attention=memory.get<float>(24*32*256);partial_ml=memory.get<float>(24*32*2);
  }
  static void mv(const Matrix& w,const float* in,float* out){
    dim3 grid((w.rows+3)/4);
    if(w.dtype==0)qwen27_fast::gemv<0,1><<<grid,128>>>(w.data,in,out,w.rows,w.cols);
    else if(w.dtype==51)qwen27_fast::gemv<51,1><<<grid,128>>>(w.data,in,out,w.rows,w.cols);
    else if(w.dtype==6)qwen27_fast::gemv<6,1><<<grid,128>>>(w.data,in,out,w.rows,w.cols);
    else if(w.dtype==5)qwen27_fast::gemv<5,1><<<grid,128>>>(w.data,in,out,w.rows,w.cols);
    else throw std::runtime_error("unsupported MTP dtype");
  }
  template<class MM> void sync(const int* tokens,const float* states,const float* previous,int count,int pos,MM mm){
    int skip=pos==0?1:0,n=count-skip;if(n<=0)return;
    mtp_inputs<<<n,256>>>(reinterpret_cast<const uint16_t*>(embeddings.data),tokens,states,previous,embedding_norm,hidden_norm,joined,skip);
    mm(fc,joined,x,n);k_rmsnorm_zc_grouped<<<n,256>>>(x,layer.norm,normed,hidden,1e-6f,1);
    mm(layer.k,normed,k,n);mm(layer.v,normed,v,n);
    int start=pos+skip-1;
    prepare_qk<<<dim3(4,n),256>>>(k,layer.knorm,layer.keys+size_t(start)*1024,nullptr,256,start);
    check(hipMemcpyAsync(layer.values+size_t(start)*1024,v,size_t(n)*4096,hipMemcpyDeviceToDevice));
  }
  int step(int token,int pos,const float* hidden_state,bool predict){
    if(pos<0||pos>=capacity)throw std::runtime_error("MTP position out of bounds");
    check(hipMemcpy(token_device,&token,sizeof token,hipMemcpyHostToDevice));
    mtp_inputs<<<1,256>>>(reinterpret_cast<const uint16_t*>(embeddings.data),token_device,nullptr,hidden_state,embedding_norm,hidden_norm,joined,0);
    mv(fc,joined,x);k_rmsnorm_zc<<<1,256>>>(x,layer.norm,normed,hidden,1e-6f);
    mv(layer.k,normed,k);mv(layer.v,normed,v);
    prepare_qk<<<4,256>>>(k,layer.knorm,layer.keys+size_t(pos)*1024,nullptr,256,pos);
    check(hipMemcpyAsync(layer.values+size_t(pos)*1024,v,4096,hipMemcpyDeviceToDevice));
    if(!predict)return -1;
    mv(layer.q,normed,qkv);prepare_qk<<<24,256>>>(qkv,layer.qnorm,q,qgate,512,pos);
    check(hipMemcpy(token_device,&pos,sizeof pos,hipMemcpyHostToDevice));
    k_qsa_flash<true,false,float,true><<<dim3(32,4),128>>>(q,layer.keys,layer.values,attn,1,256,4,24,nullptr,0,token_device,partial_attention,partial_ml);
    k_qsa_flash_combine<<<dim3(6,4),64>>>(partial_ml,partial_attention,attn,6,256,token_device);
    k_sigmoid_gate<<<24,256>>>(attn,qgate,attn,6144);
    mv(layer.out,attn,tmp);add_residual<<<20,256>>>(x,tmp,hidden);
    k_rmsnorm_zc<<<1,256>>>(x,layer.post,normed,hidden,1e-6f);
    mv(layer.gate,normed,fg);mv(layer.up,normed,fu);k_silu_mul<<<68,256>>>(fg,fu,fg,intermediate);
    mv(layer.down,fg,tmp);add_residual<<<20,256>>>(x,tmp,hidden);
    k_rmsnorm_zc<<<1,256>>>(x,finalnorm,normalized,hidden,1e-6f);mv(head,normalized,logits);
    std::vector<float> values(vocab);check(hipMemcpy(values.data(),logits,vocab*4,hipMemcpyDeviceToHost));
    for(float f:values)if(!std::isfinite(f))throw std::runtime_error("nonfinite MTP logits");
    return int(std::max_element(values.begin(),values.end())-values.begin());
  }
};
