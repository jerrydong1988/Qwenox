// Qwen27 HGN DFlash2. Ordinary RMSNorm, full 128-dimensional RoPE,
// noncausal draft block + sliding committed context; no target approximation.
// Equations: z-lab/dflash model_mlx.py, revision 07ebd93db9f472af339b644bb70221ad8428328a.
__global__ void df_norm(const float* x,const float* w,float* out,int width){
  __shared__ float sums[256];int tid=threadIdx.x;size_t base=size_t(blockIdx.x)*width;float s=0;
  for(int i=tid;i<width;i+=256)s+=x[base+i]*x[base+i];
  sums[tid]=s;__syncthreads();for(int d=128;d;d>>=1){if(tid<d)sums[tid]+=sums[tid+d];__syncthreads();}
  s=rsqrtf(sums[0]/width+1e-6f);for(int i=tid;i<width;i+=256)out[base+i]=x[base+i]*s*w[i];
}
__global__ void df_tap(const float* x,float* joined,int tap){
  int i=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
  if(i<hidden)joined[size_t(t)*hidden*5+tap*hidden+i]=x[size_t(t)*hidden+i];
}
__global__ void df_conv(const float* x,const float* dynamic,const float* base,float* out,int side){
  int c=blockIdx.x*256+threadIdx.x,t=blockIdx.y;if(c>=hidden)return;
  float y=0;for(int j=0;j<2&&j<=t;++j)y=fmaf(x[size_t(t-j)*hidden+c],base[(side*2+j)*hidden+c]+dynamic[t*1280+(side*2+j)*320+c/16],y);
  out[size_t(t)*hidden+c]=y;
}
__global__ void df_qk(const float* x,const float* weight,float* out,int start){
  int h=blockIdx.x,t=blockIdx.y,i=threadIdx.x,heads=gridDim.x;
  size_t offset=(size_t(t)*heads+h)*128;__shared__ float sum[128],v[128];
  float a=x[offset+i];sum[i]=a*a;__syncthreads();
  for(int d=64;d;d>>=1){if(i<d)sum[i]+=sum[i+d];__syncthreads();}
  v[i]=a*rsqrtf(sum[0]/128+1e-6f)*weight[i];__syncthreads();
  double angle=double(start+t)*pow(1e7,-double(i%64)/64);
  out[offset+i]=v[i]*float(cos(angle))+(i<64?-v[i+64]:v[i-64])*float(sin(angle));
}
__global__ void df_scores(const float* q,const float* k,float* scores,int pos,int count){
  int lane=threadIdx.x&31,t=blockIdx.x*4+threadIdx.x/32,h=blockIdx.y,p=blockIdx.z;
  int begin=max(0,pos+p-2047),end=pos+count;if(t>=end-begin)return;t+=begin;
  float s=0;for(int i=lane;i<128;i+=32)s=fmaf(q[(p*32+h)*128+i],k[size_t(t)*1024+(h/4)*128+i],s);
  for(int d=16;d;d>>=1)s+=__shfl_down(s,d,32);
  if(lane==0)scores[(p*32+h)*2056+t-begin]=s*0.08838834764831844f;
}
__global__ void df_attention(const float* scores,const float* values,float* out,int pos,int count){
  int h=blockIdx.x,p=blockIdx.y,i=threadIdx.x,begin=max(0,pos+p-2047),len=pos+count-begin;
  __shared__ float reduction[128],prob[2056];const float* s=scores+(p*32+h)*2056;
  float m=-INFINITY;for(int t=i;t<len;t+=128)m=fmaxf(m,s[t]);
  reduction[i]=m;__syncthreads();for(int d=64;d;d>>=1){if(i<d)reduction[i]=fmaxf(reduction[i],reduction[i+d]);__syncthreads();}
  m=reduction[0];__syncthreads();float den=0;
  for(int t=i;t<len;t+=128){float a=expf(s[t]-m);prob[t]=a;den+=a;}
  reduction[i]=den;__syncthreads();for(int d=64;d;d>>=1){if(i<d)reduction[i]+=reduction[i+d];__syncthreads();}
  float inv=1/reduction[0],v=0;for(int t=0;t<len;++t)v=fmaf(prob[t]*inv,values[size_t(begin+t)*1024+(h/4)*128+i],v);
  out[(p*32+h)*128+i]=v;
}

struct Dflash {
  static constexpr int block=8;
  int draft_vocab=98304;
  Allocations memory;
  Matrix embeddings,head,fc,selector;
  Layer layers[5];Matrix conv_proj[5][2];float* conv_base[5][2];
  float *hidden_norm,*final_norm,*joined,*ctx,*ctxnorm,*x,*n,*tmp,*cv,*dynamic,*qraw,*q,*k,*v,*attn,*fg,*fu,*logits,*gate,*scores;
  int* tokens;int capacity;
  const hgn::Tensor *predecessor,*successor;
  std::vector<int> mapping;
  Dflash(const hgn::Checkpoint& ck,Matrix emb,Matrix full_head,int maxctx,bool full_vocab):embeddings(emb),capacity(maxctx){
    const auto& config=ck.at("drafter.config");
    const int32_t expected[21]={1,5,5120,17408,128,32,8,8,2,16,256,16,248070,2048,0,5,5,19,33,47,61};
    if(config.dtype!=3||config.qparam||config.ndims!=1||config.dims[0]!=21||config.data_size!=sizeof expected||memcmp(config.data,expected,sizeof expected))throw std::runtime_error("unsupported Qwen27 DFlash2 config");
    const auto& cf=ck.at("drafter.config_f32");const float expected_f[2]={1e7f,1e-6f};
    if(cf.dtype!=1||cf.qparam||cf.ndims!=1||cf.dims[0]!=2||cf.data_size!=8||memcmp(cf.data,expected_f,8))throw std::runtime_error("unsupported DFlash2 float config");
    auto mat=[&](const std::string& name,int r,int c){const auto& t=required(ck,name,{uint64_t(r),uint64_t(c)});
      if(t.dtype!=0&&!(t.dtype==5&&t.qparam==1))throw std::runtime_error("unsupported DFlash2 matrix recipe");
      auto* p=memory.get<uint8_t>(t.data_size);check(hipMemcpy(p,t.data,t.data_size,hipMemcpyHostToDevice));return Matrix{p,t.dtype==5?51:0,r,c,false};};
    auto vec=[&](const std::string& name,std::initializer_list<uint64_t> dims){auto a=dequant(required(ck,name,dims));auto* p=memory.get<float>(a.size());check(hipMemcpy(p,a.data(),a.size()*4,hipMemcpyHostToDevice));return p;};
    fc=mat("drafter.fc.weight",hidden,hidden*5);if(!full_vocab)head=mat("draft_lm_head.weight",draft_vocab,hidden);
    selector=mat("drafter.candidate_selector.hidden_projection.weight",256,hidden);
    predecessor=&required(ck,"drafter.candidate_selector.predecessor_codebook",{vocab,256});
    successor=&required(ck,"drafter.candidate_selector.successor_codebook",{vocab,256});
    const auto& map=ck.at("draft_vocab_map");
    if(map.dtype!=3||map.qparam||map.ndims!=1||map.dims[0]!=draft_vocab||map.data_size!=draft_vocab*4)throw std::runtime_error("invalid DFlash2 vocabulary map");
    mapping.resize(draft_vocab);memcpy(mapping.data(),map.data,map.data_size);std::vector<bool> seen(vocab);
    for(int id:mapping){if(id<0||id>=vocab||seen[id])throw std::runtime_error("DFlash2 map must be unique and in range");seen[id]=true;}
    if(full_vocab){head=full_head;draft_vocab=vocab;mapping.resize(vocab);for(int i=0;i<vocab;++i)mapping[i]=i;}
    hidden_norm=vec("drafter.hidden_norm.weight",{hidden});final_norm=vec("drafter.norm.weight",{hidden});
    for(int l=0;l<5;++l){auto& w=layers[l];std::string p="drafter.layers."+std::to_string(l)+".";
      w.norm=vec(p+"input_layernorm.weight",{hidden});w.post=vec(p+"post_attention_layernorm.weight",{hidden});
      w.q=mat(p+"self_attn.q_proj.weight",4096,hidden);w.k=mat(p+"self_attn.k_proj.weight",1024,hidden);w.v=mat(p+"self_attn.v_proj.weight",1024,hidden);w.out=mat(p+"self_attn.o_proj.weight",hidden,4096);
      w.qnorm=vec(p+"self_attn.q_norm.weight",{128});w.knorm=vec(p+"self_attn.k_norm.weight",{128});
      w.gate=mat(p+"mlp.gate_proj.weight",intermediate,hidden);w.up=mat(p+"mlp.up_proj.weight",intermediate,hidden);w.down=mat(p+"mlp.down_proj.weight",hidden,intermediate);
      for(int side=0;side<2;++side){std::string c=p+(side?"mlp_conv.":"attention_conv.");conv_proj[l][side]=mat(c+"kernel_projection.weight",1280,hidden);conv_base[l][side]=vec(c+"base_kernel",{2,2,hidden});}
      w.keys=memory.get<float>(size_t(maxctx+8)*1024);w.values=memory.get<float>(size_t(maxctx+8)*1024);
    }
    joined=memory.get<float>(512*hidden*5);ctx=memory.get<float>(512*hidden);ctxnorm=memory.get<float>(512*hidden);
    x=memory.get<float>(8*hidden);n=memory.get<float>(8*hidden);tmp=memory.get<float>(8*hidden);cv=memory.get<float>(8*hidden);
    dynamic=memory.get<float>(8*1280);qraw=memory.get<float>(8*4096);q=memory.get<float>(8*4096);
    k=memory.get<float>(512*1024);v=memory.get<float>(512*1024);attn=memory.get<float>(8*4096);
    fg=memory.get<float>(8*intermediate);fu=memory.get<float>(8*intermediate);logits=memory.get<float>(8*draft_vocab);
    gate=memory.get<float>(8*256);scores=memory.get<float>(8*32*2056);tokens=memory.get<int>(8);
  }
  static void mm(const Matrix& w,const float* in,float* out,int count){
    // Bounded FP32 batches preserve the serial FMA order, including NF4.
    for(int offset=0;offset<count;offset+=8){int n=std::min(8,count-offset);const float* a=in+size_t(offset)*w.cols;float* b=out+size_t(offset)*w.rows;
      #define DF_B(D,P) qwen27_fast::batch<D,P><<<(w.rows+3)/4,128>>>(w.data,a,b,w.rows,w.cols)
      #define DF_S(D) switch(n){case 1:DF_B(D,1);break;case 2:DF_B(D,2);break;case 3:DF_B(D,3);break;case 4:DF_B(D,4);break;case 5:DF_B(D,5);break;case 6:DF_B(D,6);break;case 7:DF_B(D,7);break;case 8:DF_B(D,8);break;}
      if(w.dtype==0){DF_S(0);}else if(w.dtype==51){DF_S(51);}else if(w.dtype==6){DF_S(6);}else throw std::runtime_error("DFlash2 matrix dtype unsupported");
      #undef DF_S
      #undef DF_B
    }
  }
  void capture(int layer,const float* x,int count,hipStream_t stream=nullptr){
    constexpr int taps[5]={5,19,33,47,61};for(int i=0;i<5;++i)if(layer==taps[i])df_tap<<<dim3(20,count),256,0,stream>>>(x,joined,i);
  }
  void sync(int count,int pos){
    if(count<1||count>512||pos<0||pos+count>capacity)throw std::runtime_error("DFlash2 sync bounds");
    mm(fc,joined,ctx,count);df_norm<<<count,256>>>(ctx,hidden_norm,ctxnorm,hidden);
    for(auto& w:layers){mm(w.k,ctxnorm,k,count);mm(w.v,ctxnorm,v,count);
      df_qk<<<dim3(8,count),128>>>(k,w.knorm,w.keys+size_t(pos)*1024,pos);
      check(hipMemcpyAsync(w.values+size_t(pos)*1024,v,size_t(count)*4096,hipMemcpyDeviceToDevice));}
  }
  template<class Dump> std::vector<int> propose(int anchor,int pos,int gamma,Dump dump){
    if(anchor<0||anchor>=vocab||pos<1||pos>capacity||gamma<1||gamma>7)throw std::runtime_error("DFlash2 proposal bounds");
    const int count=gamma+1;int ids[8];std::fill(ids,ids+8,248070);ids[0]=anchor;
    check(hipMemcpy(tokens,ids,count*4,hipMemcpyHostToDevice));
    embedding_batch<<<dim3(20,count),256>>>(reinterpret_cast<const uint16_t*>(embeddings.data),tokens,x);
    for(int l=0;l<5;++l){auto& w=layers[l];
      df_norm<<<count,256>>>(x,w.norm,n,hidden);mm(conv_proj[l][0],n,dynamic,count);
      df_conv<<<dim3(20,count),256>>>(n,dynamic,conv_base[l][0],cv,0);
      mm(w.q,cv,qraw,count);mm(w.k,cv,k,count);mm(w.v,cv,v,count);
      df_qk<<<dim3(32,count),128>>>(qraw,w.qnorm,q,pos);df_qk<<<dim3(8,count),128>>>(k,w.knorm,w.keys+size_t(pos)*1024,pos);
      check(hipMemcpyAsync(w.values+size_t(pos)*1024,v,size_t(count)*4096,hipMemcpyDeviceToDevice));
      df_scores<<<dim3((std::min(pos,2047)+count+3)/4,32,count),128>>>(q,w.keys,scores,pos,count);
      df_attention<<<dim3(32,count),128>>>(scores,w.values,attn,pos,count);
      mm(w.out,attn,tmp,count);df_conv<<<dim3(20,count),256>>>(tmp,dynamic,conv_base[l][0],n,1);add_residual<<<20*count,256>>>(x,n,count*hidden);
      df_norm<<<count,256>>>(x,w.post,n,hidden);mm(conv_proj[l][1],n,dynamic,count);
      df_conv<<<dim3(20,count),256>>>(n,dynamic,conv_base[l][1],cv,0);
      mm(w.gate,cv,fg,count);mm(w.up,cv,fu,count);k_silu_mul<<<68*count,256>>>(fg,fu,fg,count*intermediate);
      mm(w.down,fg,tmp,count);df_conv<<<dim3(20,count),256>>>(tmp,dynamic,conv_base[l][1],n,1);add_residual<<<20*count,256>>>(x,n,count*hidden);
      dump("dflash-layer"+std::to_string(l),x,count*hidden);
    }
    df_norm<<<count,256>>>(x,final_norm,n,hidden);mm(head,n,logits,count);mm(selector,n,gate,count);
    dump("dflash-normalized",n,count*hidden);dump("dflash-logits",logits,count*draft_vocab);dump("dflash-gate",gate,count*256);
    std::vector<float> logits_h(size_t(count)*draft_vocab),gate_h(count*256);
    check(hipMemcpy(logits_h.data(),logits,logits_h.size()*4,hipMemcpyDeviceToHost));check(hipMemcpy(gate_h.data(),gate,gate_h.size()*4,hipMemcpyDeviceToHost));
    for(float a:logits_h)if(!std::isfinite(a))throw std::runtime_error("nonfinite DFlash2 logits");
    for(float a:gate_h)if(!std::isfinite(a))throw std::runtime_error("nonfinite DFlash2 selector");
    std::vector<int> order(draft_vocab),result;int pred=anchor;
    for(int t=1;t<count;++t){for(int i=0;i<draft_vocab;++i)order[i]=i;const float* row=logits_h.data()+size_t(t)*draft_vocab;
      std::partial_sort(order.begin(),order.begin()+16,order.end(),[&](int a,int b){return row[a]>row[b]||(row[a]==row[b]&&a<b);});
      float best=-INFINITY;int chosen=-1;
      for(int i=0;i<16;++i){int id=mapping[order[i]];float score=row[order[i]];
        for(int d=0;d<256;++d)score=std::fma(value(*predecessor,size_t(pred)*256+d)*gate_h[t*256+d],value(*successor,size_t(id)*256+d),score);
        if(score>best){best=score;chosen=id;}}
      if(chosen<0||!std::isfinite(best))throw std::runtime_error("invalid DFlash2 selected score");
      result.push_back(chosen);pred=chosen;
    }
    return result;
  }
};
