// GPU regressions for the production router, image buffers and HGN decode paths.
#define main gdec_real_main
#include "gdec.cpp"
#undef main
#include <random>

// Exercise the actual decode kernels at the HGN hidden size and branch count.
static bool decode_fusion_parity() {
  constexpr int d = 2560, branches = 4;
  std::mt19937 rng(76175);
  std::vector<float> input(d * branches), residual(input.size());
  for (auto& x : input) x = (int(rng() % 2001) - 1000) * 0.03125f;
  for (auto& x : residual) x = (int(rng() % 2001) - 1000) * 0.015625f;
  qwen27::Allocations memory;
  float* g = memory.get<float>(input.size());
  float* r = memory.get<float>(residual.size());
  float* old_out = memory.get<float>(d);
  float* new_out = memory.get<float>(d);
  CK(hipMemcpy(g, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));
  CK(hipMemcpy(r, residual.data(), residual.size() * sizeof(float), hipMemcpyHostToDevice));
  auto same_bits = [&] {
    std::vector<uint32_t> a(d), b(d);
    CK(hipMemcpy(a.data(), old_out, d * sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), new_out, d * sizeof(float), hipMemcpyDeviceToHost));
    return a == b;
  };
  k_gr_combine<true><<<(d + 255) / 256, 256>>>(g, r, new_out, d, branches);
  k_sigmoid_inplace<<<(d * branches + 255) / 256, 256>>>(g, d * branches);
  k_gr_combine<false><<<(d + 255) / 256, 256>>>(g, r, old_out, d, branches);
  if (!same_bits()) return false;
  CK(hipMemcpy(old_out, input.data(), d * sizeof(float), hipMemcpyHostToDevice));
  k_axpy_sg_out<<<(d + 255) / 256, 256>>>(old_out, r, g, new_out, d, d);
  k_axpy_sg<<<(d + 255) / 256, 256>>>(old_out, r, g, d, d);
  return same_bits();
}

static bool checked_greedy_rows() {
  constexpr int rows = 8, n = qwen27::vocab;
  std::vector<float> values((size_t)rows * n, -10.f);
  values[1] = -0.0f;
  values[n - 1] = +0.0f;  // smallest id wins, even across signed zero
  values[n + 65] = values[2 * n - 1] = 1.f;
  values[3 * n - 1] = -std::numeric_limits<float>::infinity();
  values[3 * n + 77] = std::numeric_limits<float>::infinity();
  values[5 * n - 1] = std::numeric_limits<float>::quiet_NaN();
  values[6 * n + 11] = -std::numeric_limits<float>::quiet_NaN();
  values[8 * n - 1] = 5.f;
  const std::array<int, rows> expected{1, 65, -1, -1, -1, 0, -1, n - 1};
  qwen27::Allocations memory;
  float* scores = memory.get<float>(values.size());
  auto* scratch = memory.get<unsigned long long>(rows * ARGMAX_B1);
  int* result = memory.get<int>(rows);
  CK(hipMemcpy(scores, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice));
  argmax_fast_rows<true>(scores, n, rows, scratch, result, nullptr);
  std::array<int, rows> actual{};
  CK(hipMemcpy(actual.data(), result, sizeof(actual), hipMemcpyDeviceToHost));
  return actual == expected;
}

int main() {
  const bool fusion = decode_fusion_parity(), greedy = checked_greedy_rows();
  printf("decode fusion bit parity=%d; checked greedy ties/nonfinite=%d\n", fusion, greedy);
  int failed = !fusion + !greedy, cases = 2;
  for (int count : {1, 16, 17, 20, 32, 33}) {
    std::istringstream request("1 1 0 1 1 0 VIMG " + std::to_string(count));
    auto parsed = serve::parse_gen(request);
    const bool ok = parsed.vimg == count && parsed.reject == (count > 32);
    ++cases; failed += !ok;
    printf("%s VIMG count=%d reject=%d\n", ok ? "PASS" : "FAIL", count, parsed.reject);
  }
  std::mt19937 rng(61006);
  for (int P : {1, 2, 7, 8, 9, 33, 257}) {
    for (int pattern = 0; pattern < 6; ++pattern) {
      std::vector<float> x(P * 512);
      for (size_t i = 0; i < x.size(); ++i) {
        if (pattern == 0) x[i] = std::uniform_real_distribution<float>(-20, 20)(rng);
        if (pattern == 1) x[i] = 0;  // all tied; the old tree is NOT index order
        if (pattern == 2) x[i] = float(rng() % 7) - 3;
        if (pattern == 3) x[i] = (i % 512 == 3 || i % 512 == 32) ? 9.f : -9.f;
        if (pattern == 4) x[i] = (i % 17) ? -1.e20f : 1.e20f;
        if (pattern == 5) x[i] = std::nextafter(1.f, (i % 3) ? 2.f : 0.f);
      }
      float *dx, *wa, *wb; int *ia, *ib;
      CK(hipMalloc(&dx, x.size() * 4));
      CK(hipMalloc(&wa, (P * 16 + 32) * 4)); CK(hipMalloc(&wb, (P * 16 + 32) * 4));
      CK(hipMalloc(&ia, (P * 16 + 32) * 4)); CK(hipMalloc(&ib, (P * 16 + 32) * 4));
      CK(hipMemcpy(dx, x.data(), x.size()*4, hipMemcpyHostToDevice));
      for (void* ptr : {(void*)wa, (void*)wb, (void*)ia, (void*)ib})
        CK(hipMemset(ptr, 0xa5, (P*16+32)*4));
      k_router_topk<<<P,512>>>(dx, ia+16, wa+16, 512, 10);
      k_router_topk_w32<10><<<(P+7)/8,256>>>(dx, ib+16, wb+16, P);
      CK(hipDeviceSynchronize());
      std::vector<uint32_t> a(P*16+32),b(P*16+32);
      int diff=0;
      for (auto pair : {std::make_pair((void*)ia,(void*)ib), std::make_pair((void*)wa,(void*)wb)}) {
        CK(hipMemcpy(a.data(),pair.first,a.size()*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(),pair.second,b.size()*4,hipMemcpyDeviceToHost));
        for (size_t i=0;i<a.size();++i) diff += a[i]!=b[i];
        for (int i=0;i<16;++i) if(a[i]!=0xa5a5a5a5 || b[i]!=0xa5a5a5a5 ||
            a[a.size()-1-i]!=0xa5a5a5a5 || b[b.size()-1-i]!=0xa5a5a5a5) ++diff;
      }
      ++cases; failed += diff!=0;
      printf("%s router P=%d pattern=%d differing_words=%d\n",diff?"FAIL":"PASS",P,pattern,diff);
      CK(hipFree(dx));CK(hipFree(wa));CK(hipFree(wb));CK(hipFree(ia));CK(hipFree(ib));
    }
  }
  uint16_t* images=nullptr;size_t cap=0;
  grow_image_embeddings(images,cap,0,2000,8000);
  std::vector<uint16_t> data(4500*2560),out(data.size());
  for(size_t i=0;i<data.size();++i) data[i]=uint16_t(i*137+19);
  CK(hipMemcpy(images,data.data(),2000*2560*2,hipMemcpyHostToDevice));
  grow_image_embeddings(images,cap,2000,2500,10000);
  CK(hipMemcpy(out.data(),images,2000*2560*2,hipMemcpyDeviceToHost));
  bool first=cap>=4500 && std::equal(out.begin(),out.begin()+2000*2560,data.begin());
  CK(hipMemcpy(images,data.data(),data.size()*2,hipMemcpyHostToDevice));
  grow_image_embeddings(images,cap,4500,17,68);
  CK(hipMemcpy(out.data(),images,out.size()*2,hipMemcpyDeviceToHost));
  bool second=out==data && cap>=4517;
  auto same=images;grow_image_embeddings(images,cap,1,1,4);
  bool no_growth=same==images;
  // Recover in the same process, retaining the embeddings of earlier images.
#ifdef _WIN32
  _putenv_s("QWENOX_VISION_EMBED_OOM", "1");
  _putenv_s("QWENOX_VISION_OOM_AT", "6");
#else
  setenv("QWENOX_VISION_EMBED_OOM", "1", 1);
  setenv("QWENOX_VISION_OOM_AT", "6", 1);
#endif
  const size_t old_cap = cap;
  bool embedding_failed = false;
  try {
    grow_image_embeddings(images, cap, old_cap, 1, 4);
  } catch (const std::runtime_error&) {
    embedding_failed = images == same && cap == old_cap;
  }
  grow_image_embeddings(images, cap, old_cap, 1, 4);
  CK(hipMemcpy(out.data(), images, out.size() * 2, hipMemcpyDeviceToHost));
  const bool embedding_recovered = embedding_failed && cap > old_cap && out == data;
  VisionTower tower;
  bool workspace_failed = false;
  try {
    tower.ensure(4);
  } catch (const std::runtime_error&) {
    workspace_failed = tower.capP == 0 && !tower.d_h && !tower.d_qkv && !tower.d_out;
  }
  tower.ensure(4);
  const bool workspace_recovered = workspace_failed && tower.capP == 4 && tower.d_wts;
  tower.clear_workspace();
  failed += !embedding_recovered + !workspace_recovered;
  cases += 2;
  printf("vision OOM retry: embeddings=%d workspace=%d\n", embedding_recovered, workspace_recovered);
  CK(hipFree(images));
  failed+=!first+!second+!no_growth;cases+=3;
  printf("image_growth first=%d second=%d no_growth=%d\n",first,second,no_growth);
  printf("RESULT %s cases=%d failed=%d\n",failed?"FAIL":"PASS",cases,failed);
  return failed?1:0;
}
