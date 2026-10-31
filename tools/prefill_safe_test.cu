// Differential checks for order-preserving Prefill backports. No model files.
#define main gdec_engine_main
#include "gdec.cpp"
#undef main
#include <random>

static std::vector<void*> allocations;
template<class T> T* device(const std::vector<T>& values) {
  T* p = nullptr;
  CK(hipMalloc(&p, values.size() * sizeof(T)));
  allocations.push_back(p);
  CK(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
template<class T> std::vector<T> host(const T* p, size_t count) {
  std::vector<T> v(count);
  CK(hipMemcpy(v.data(), p, count * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}
static void require(bool yes, const char* why) {
  if (!yes) throw std::runtime_error(why);
}
static void release() {
  for (void* p : allocations) CK(hipFree(p));
  allocations.clear();
}
static void q6_check(int rows, int cols) {
  std::mt19937 random(rows + cols);
  const int stride = (cols * 13 / 16 + 15) / 16 * 16;
  std::vector<uint8_t> codes((size_t)rows * stride);
  for (auto& byte : codes) byte = (uint8_t)random();
  for (int r = 0; r < rows; ++r) for (int g = 0; g < cols / 64; ++g) {
    __half scale = __float2half((int(random() % 129) - 64) / 128.f);
    __half bias = __float2half((int(random() % 257) - 128) / 32.f);
    memcpy(codes.data() + (size_t)r * stride + cols * 3 / 4 + g * 4, &scale, 2);
    memcpy(codes.data() + (size_t)r * stride + cols * 3 / 4 + g * 4 + 2, &bias, 2);
  }
  auto input = device(codes);
  const size_t n = (size_t)rows * cols;
  auto baseline = device(std::vector<uint16_t>(n + 32, 0xffff));
  auto candidate = device(std::vector<uint16_t>(n + 32, 0xffff));
  k_q6g64_deq_bf16<<<(unsigned)((rows * (cols / 64) + 255) / 256),256>>>(input,baseline,rows,cols,stride);
  k_q6g64_deq_bf16_c8<<<(unsigned)((rows * (cols / 8) + 255) / 256),256>>>(input,candidate,rows,cols,stride);
  CK(hipDeviceSynchronize());
  auto a=host(baseline,n+32),b=host(candidate,n+32);
  require(a==b,"Q6 BF16 output differs");
  for(size_t i=0;i<n;++i) require((b[i]&0x7f80)!=0x7f80,"Q6 output missing/nonfinite");
  for(size_t i=n;i<n+32;++i) require(b[i]==0xffff,"Q6 guard overwritten");
  printf("PASS Q6 %dx%d full BF16 bits + guard\n",rows,cols);
  release();
}
static void identity_check() {
  KvPagePool pool; pool.init(32,1);
  KvSeqTable a,b; a.init(16,pool.guard()); b.init(16,pool.guard());
  require(a.identity && a.reserve(pool,513) && a.identity,"identity reserve");
  auto checkpoint=a.pin(pool,513);
  a.trim(pool,0);
  require(a.identity && a.reserve(pool,257) && !a.identity,"pinned pages prevent identity");
  a.adopt(pool,checkpoint);
  require(a.identity,"adopt identity checkpoint");
  int p=pool.alloc(); a.replace(pool,2,p);
  require(!a.identity,"COW clears identity");
  auto nonidentity=a.pin(pool,513);
  b.adopt(pool,nonidentity);
  require(!b.identity,"adopt nonidentity checkpoint");
  a.trim(pool,0); b.trim(pool,0);
  KvSeqTable::unpin(pool,nonidentity); KvSeqTable::unpin(pool,checkpoint);
  require(pool.nfree()==32 && a.identity && b.identity,"identity cleanup conservation");
  pool.init(32,2);a.init(16,pool.guard());a.reserve(pool,1024);
  require(!a.identity,"scrambled pages cannot elide mapping");
  a.trim(pool,0);
  printf("PASS identity: reserve, pin, adopt, COW, two slots, scrambled, cleanup\n");
}
static void hybrid_check(int d,int mid) {
  const std::vector<int> counts{0,1,63,64,65,127,128,129,191,192,193,255,256,257,511,512};
  const int e=(int)counts.size();
  std::vector<int> offsets(e+1);
  for(int i=0;i<e;++i)offsets[i+1]=offsets[i]+counts[i];
  const int total=offsets.back(), cap=total/64+e;
  std::mt19937 random(901+d+mid);
  auto weight=[&](int rows,int cols) {
    std::vector<uint8_t> w((size_t)rows*cols/2+(size_t)rows*(cols/128)*2);
    for(auto& v:w)v=(uint8_t)random();
    for(size_t i=(size_t)rows*cols/2;i<w.size();i+=2){
      __half scale=__float2half((1+random()%31)/4096.f);memcpy(w.data()+i,&scale,2);
    }
    return device(w);
  };
  auto gu=weight(e*2*mid,d),dn=weight(e*d,mid);
  std::vector<float> codebook(16),sv(2*mid),su(mid);
  for(int i=0;i<16;++i)codebook[i]=(float)(i-8);
  for(auto& v:sv)v=(1+random()%17)/32.f;
  for(auto& v:su)v=(1+random()%17)/32.f;
  auto cb=device(codebook),dsv=device(sv),dsu=device(su);
  auto pgu=upstream_moe::lut_i4r_view(gu,(uint64_t)e*2*mid,d,cb,2*mid,mid);
  auto pdn=upstream_moe::lut_i4r_view(dn,(uint64_t)e*d,mid,cb,d,0);
  std::vector<int> ids(total);for(int i=0;i<total;++i)ids[i]=i;
  auto tokens=device(ids),off=device(offsets);
  auto tiles=device(std::vector<MoeTile>(cap)),tail=device(std::vector<MoeTile>(e));
  auto nt=device(std::vector<int>(1)),na=device(std::vector<int>(1));
  std::vector<__half> x((size_t)total*d);
  for(auto& v:x)v=__float2half((int(random()%2001)-1000)/2048.f);
  auto dx=device(x);
  auto hid=device(std::vector<__half>((size_t)total*mid+16));
  auto pairs=device(std::vector<float>((size_t)total*d+16));
  std::vector<uint16_t> reference_hid;
  std::vector<uint32_t> reference_pairs;
  for(int kind=0;kind<3;++kind) {
    CK(hipMemset(hid,0xff,((size_t)total*mid+16)*2));
    CK(hipMemset(pairs,0xff,((size_t)total*d+16)*4));
    if(kind==2) {
      k_moe_tiles_hyb<<<1,512>>>(off,tiles,nt,tail,na,e);
      auto bc=host(nt,1),ac=host(na,1);
      auto bt=host(tiles,bc[0]),at=host(tail,ac[0]);
      std::vector<int> seen(total);
      for(auto t:bt)for(int i=0;i<t.count;++i)++seen[t.first+i];
      for(auto t:at)for(int i=0;i<t.count;++i)++seen[t.first+i];
      for(int n:seen)require(n==1,"hybrid tiles overlap or miss slots");
      upstream_moe::moe_lut_up_mid_mt(pgu,dx,tokens,tiles,nt,cap,hid,dsv,dsu,mid,d,0);
      upstream_moe::moe_lut_up_mid(pgu,dx,tokens,tail,na,e,hid,dsv,dsu,mid,d,0);
      upstream_moe::moe_lut_down_mt(pdn,hid,tiles,nt,cap,pairs,d,mid,0,false);
      upstream_moe::moe_lut_down(upstream_moe::moelut::kI4R,pdn,hid,tail,na,e,pairs,d,mid,0,false);
    } else {
      k_moe_tiles<<<1,512>>>(off,tiles,nt,e,kind?128:64);
      if(kind) {
        upstream_moe::moe_lut_up_mid_mt(pgu,dx,tokens,tiles,nt,cap,hid,dsv,dsu,mid,d,0);
        upstream_moe::moe_lut_down_mt(pdn,hid,tiles,nt,cap,pairs,d,mid,0,false);
      } else {
        upstream_moe::moe_lut_up_mid(pgu,dx,tokens,tiles,nt,cap,hid,dsv,dsu,mid,d,0);
        upstream_moe::moe_lut_down(upstream_moe::moelut::kI4R,pdn,hid,tiles,nt,cap,pairs,d,mid,0,false);
      }
    }
    CK(hipDeviceSynchronize());
    auto h=host((uint16_t*)hid,(size_t)total*mid+16);
    auto p=host((uint32_t*)pairs,(size_t)total*d+16);
    for(size_t i=0;i<(size_t)total*mid;++i)require(std::isfinite((float)__ushort_as_half(h[i])),"hybrid hid missing/nonfinite");
    for(size_t i=0;i<(size_t)total*d;++i){float f;memcpy(&f,&p[i],4);require(std::isfinite(f),"hybrid pairs missing/nonfinite");}
    for(size_t i=(size_t)total*mid;i<h.size();++i)require(h[i]==0xffff,"hid guard");
    for(size_t i=(size_t)total*d;i<p.size();++i)require(p[i]==0xffffffffu,"pair guard");
    if(!kind){reference_hid=h;reference_pairs=p;}
    else require(h==reference_hid && p==reference_pairs,"MoE BN64/BN128/hybrid bits differ");
  }
  printf("PASS hybrid D=%d mid=%d %d expert-slot rows: full f16 hid, FP32 pairs, guards\n",d,mid,total);
  release();
}
int main() {
  try {
    identity_check();
    for(auto shape : {std::pair<int,int>{3,64},{17,128},{640,2560},{320,10240},{10240,320}})
      q6_check(shape.first,shape.second);
    hybrid_check(256,128);
    hybrid_check(2560,640);
    puts("ALL PREFILL SAFE CHECKS PASS");
    return 0;
  } catch(const std::exception& e) { fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
