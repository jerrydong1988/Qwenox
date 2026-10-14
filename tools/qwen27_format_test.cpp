#include "../src/hgn_qwen27.h"
#include <cassert>
#include <iostream>

int main(int argc,char** argv){
  // Two NVFP4 rows, deliberately distinct scales in adjacent 16-value groups.
  std::vector<uint8_t> data(64+2*(128+16),0);
  float cb[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
  memcpy(data.data(),cb,64);
  for(int row=0;row<2;++row){
    uint8_t* r=data.data()+64+row*144;
    for(int i=0;i<128;++i)r[i]=0x92; // low=1, high=-0.5
    for(int g=0;g<16;++g)r[128+g]=uint8_t(0x38+((g+row)%2)*8); // 1 or 2
  }
  hgn::Tensor t;t.name="test";t.dtype=5;t.ndims=2;t.dims[0]=2;t.dims[1]=256;
  t.data=data.data();t.data_size=data.size();qwen27::validate(t);
  for(int r=0;r<2;++r)for(int c=0;c<256;++c)
    assert(qwen27::value(t,r*256+c)==(c%2?-0.5f:1.f)*(1+((c/16+r)%2)));
  assert(qwen27::can_factor(t));
  cb[3]+=0.125f;memcpy(data.data(),cb,64);assert(!qwen27::can_factor(t));
  cb[3]-=0.125f;memcpy(data.data(),cb,64);
  t.qparam=2;bool rejected=false;try{qwen27::validate(t);}catch(...){rejected=true;}assert(rejected);
  t.qparam=0;t.data_size--;rejected=false;try{qwen27::validate(t);}catch(...){rejected=true;}assert(rejected);
  // FP8 row scale bits are BF16, not IEEE half.
  uint8_t fp[8]={0x38,0x40,0xb8,0xc0,0x80,0x3f,0x00,0x40};
  t.dtype=6;t.dims[1]=2;t.data=fp;t.data_size=8;
  auto decoded=qwen27::dequant(t);
  assert(decoded==std::vector<float>({1,2,-2,-4}));
  // E4M3 finite maximum, subnormal, signed zero, and NaN behavior.
  assert(hgn::fp8e4m3_to_f32(0x7e)==448);
  assert(hgn::fp8e4m3_to_f32(1)==0x1p-9f);
  assert(std::signbit(hgn::fp8e4m3_to_f32(0x80)));
  assert(std::isnan(hgn::fp8e4m3_to_f32(0xff)));
  // Row-interleaved NF4: FP16 scale 1/2 in alternating groups and rows.
  t.dtype=5;t.qparam=1;t.dims[1]=256;t.data=data.data();t.data_size=data.size();
  for(int r=0;r<2;++r)for(int g=0;g<8;++g){
    uint16_t s=uint16_t(0x3c00+((r+g)%2)*0x400);
    memcpy(data.data()+64+r*144+128+g*2,&s,2);
  }
  qwen27::validate(t);assert(!qwen27::can_factor(t));
  for(int r=0;r<2;++r)for(int c=0;c<256;++c)
    assert(qwen27::value(t,r*256+c)==(c%2?-0.5f:1.f)*(1+((c/32+r)%2)));
  if(argc==3){
    hgn::Checkpoint ck(argv[1]);std::ofstream out(argv[2],std::ios::binary);
    for(const char* name:{"layers.0.linear_attn.in_proj_qkv.weight","layers.0.mlp.gate_proj.weight",
                         "layers.28.mlp.up_proj.weight","layers.55.mlp.down_proj.weight",
                         "mtp.layers.0.mlp.gate_proj.weight","mtp.layers.0.self_attn.q_proj.weight"}){
      const auto& w=ck.at(name);qwen27::validate(w);
      for(uint64_t r:{uint64_t(0),w.dims[0]/2,w.dims[0]-1})for(uint64_t c=0;c<w.dims[1];++c){
        float f=qwen27::value(w,r*w.dims[1]+c);assert(std::isfinite(f));out.write((char*)&f,4);
      }
    }
    if(!out)return 1;
  }
  std::cout<<"Qwen27 format tests PASS\n";
}
