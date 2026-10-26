// Standalone microbenchmark of the grouped MoE kernels on the real dumped routing (no torch): kernel time of every launch variant, GB/s over the
// experts the routing touches, and the share of the memory floor. Weights are random packed FP4 with E8M0 scales in [117, 127], activations random bf16.
//   moe_bench prefill <ids.bin> <M> [iters] [order]       e.g. ids/ctx16384/ids_L02_M1536.bin 1536
//   moe_bench decode  <ids.bin> <M_in_file> <rows> [start] [iters]   rows = decode rows taken from the dumped routing starting at row <start>
//   moe_bench random  <rows> [iters]                       decode rows with synthetic skewed routing
// Build: therock-hip-clang++ -x hip --offload-arch=gfx1151 -O3 -std=c++20 -w moe_bench.cpp -o moe_bench
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <cstdint>

#ifndef MOE_SRC
#error "build with -DMOE_SRC=\"/path/to/sglang/kernels/ops/moe/dsv41_moe_grouped.hip.cpp\""
#endif
#include MOE_SRC

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2); } } while (0)

constexpr int E = 384, H = 5120, I = 576, I2 = 1152;
constexpr double GU_BYTES = (double)I2 * (H / 2 + H / 32), DN_BYTES = (double)H * (I / 2 + I / 32);

__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed, int lo, int hi) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = hi > 0 ? (uint8_t)(lo + (x >> 8) % (uint32_t)(hi - lo + 1)) : (uint8_t)(x >> 5);
  }
}
__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    const float f = ((int)(x & 0xFFFF) - 32768) * (1.0f / 16384.0f);
    p[i] = (uint16_t)(__builtin_bit_cast(unsigned, f) >> 16);
  }
}

static std::vector<int> read_ids(const char* path, int M) {
  std::vector<int> v((size_t)M * 6);
  FILE* f = fopen(path, "rb");
  if (!f || fread(v.data(), 4, v.size(), f) != v.size()) { printf("cannot read %s\n", path); exit(2); }
  fclose(f);
  return v;
}

struct Timer {
  hipEvent_t a, b;
  Timer() { CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); }
  template <class F> double ms(F fn, int iters, int warm = 3) {
    for (int i = 0; i < warm; ++i) fn(i);
    CK(hipDeviceSynchronize());
    std::vector<double> ts;
    for (int r = 0; r < 5; ++r) {
      CK(hipEventRecord(a));
      for (int i = 0; i < iters; ++i) fn(i);
      CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
      float t; CK(hipEventElapsedTime(&t, a, b)); ts.push_back(t / iters);
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
  }
};

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: see the header of moe_bench.cpp\n"); return 1; }
  const std::string mode = argv[1];
  std::vector<int> ids;
  int M = 0, order = 0, iters = 10;
  bool decode = false;
  if (mode == "prefill") {
    M = atoi(argv[3]); ids = read_ids(argv[2], M); if (argc > 4) iters = atoi(argv[4]); if (argc > 5) order = atoi(argv[5]);
  } else if (mode == "decode") {
    const int Mf = atoi(argv[3]); M = atoi(argv[4]); const int start = argc > 5 ? atoi(argv[5]) : 0; iters = argc > 6 ? atoi(argv[6]) : 200;
    std::vector<int> all = read_ids(argv[2], Mf);
    ids.assign(all.begin() + (size_t)start * 6, all.begin() + (size_t)(start + M) * 6);
    decode = true;
  } else {
    M = atoi(argv[2]); iters = argc > 3 ? atoi(argv[3]) : 200; decode = true;
    ids.resize((size_t)M * 6);
    srand(5);
    for (int t = 0; t < M; ++t) {
      std::set<int> pick;
      while (pick.size() < 6) { const double u = rand() / (double)RAND_MAX; pick.insert((int)(E * u * u * 0.999)); }
      int k = 0; for (int e : pick) ids[(size_t)t * 6 + k++] = e;
    }
  }
  const int S = M * 6;
  std::set<int> uniq(ids.begin(), ids.end());
  const double u = (double)uniq.size();
  printf("mode %s rows %d slots %d unique experts %d\n", mode.c_str(), M, S, (int)u);

  // several copies of the expert weights, used in turn, so that a repeated call never finds its experts in the 32 MB infinity cache (a real layer's weights are 68 GB)
  constexpr int NC = 4;
  uint8_t *W13[NC], *S13[NC], *W2[NC], *S2[NC];
  for (int c = 0; c < NC; ++c) {
    CK(hipMalloc(&W13[c], (size_t)E * I2 * (H / 2))); CK(hipMalloc(&S13[c], (size_t)E * I2 * (H / 32)));
    CK(hipMalloc(&W2[c], (size_t)E * H * (I / 2))); CK(hipMalloc(&S2[c], (size_t)E * H * (I / 32)));
    hipLaunchKernelGGL(fill_bytes, dim3(4096), dim3(256), 0, 0, W13[c], (size_t)E * I2 * (H / 2), 1u + 10 * c, 0, 0);
    hipLaunchKernelGGL(fill_bytes, dim3(1024), dim3(256), 0, 0, S13[c], (size_t)E * I2 * (H / 32), 2u + 10 * c, 117, 127);
    hipLaunchKernelGGL(fill_bytes, dim3(4096), dim3(256), 0, 0, W2[c], (size_t)E * H * (I / 2), 3u + 10 * c, 0, 0);
    hipLaunchKernelGGL(fill_bytes, dim3(1024), dim3(256), 0, 0, S2[c], (size_t)E * H * (I / 32), 4u + 10 * c, 117, 127);
  }
  uint16_t *a1, *a2, *gu, *dn;
  CK(hipMalloc(&a1, (size_t)M * H * 2)); CK(hipMalloc(&a2, (size_t)S * I * 2)); CK(hipMalloc(&gu, (size_t)S * I2 * 2)); CK(hipMalloc(&dn, (size_t)S * H * 2));
  hipLaunchKernelGGL(fill_bf16, dim3(1024), dim3(256), 0, 0, a1, (size_t)M * H, 5u);
  hipLaunchKernelGGL(fill_bf16, dim3(1024), dim3(256), 0, 0, a2, (size_t)S * I, 6u);
  int *d_ids, *d_slots, *d_tiles, *d_nt;
  const int cap = dsv41_moe_tiles_capacity(S, E);
  CK(hipMalloc(&d_ids, S * 4)); CK(hipMalloc(&d_slots, S * 4)); CK(hipMalloc(&d_tiles, cap * 12)); CK(hipMalloc(&d_nt, 4));
  CK(hipMemcpy(d_ids, ids.data(), S * 4, hipMemcpyHostToDevice));
  CK(hipDeviceSynchronize());

  Timer T;
  auto report = [&](const char* what, int cfg, double ms, double bytes) {
    const double gbps = bytes / (ms * 1e6);
    printf("  %-8s cfg %2d: %8.1f us  %6.1f GB/s  %5.1f%% of 242  %5.1f%% of 226\n", what, cfg, ms * 1000, gbps, 100 * gbps / 242, 100 * gbps / 226);
  };
  if (!decode) {
    const double tr = T.ms([&](int it) { dsv41_moe_route(d_ids, S, E, d_slots, d_tiles, d_nt, order, nullptr); }, iters);
    printf("  route (order %d): %.1f us\n", order, tr * 1000);
    dsv41_moe_route(d_ids, S, E, d_slots, d_tiles, d_nt, order, nullptr);
    CK(hipDeviceSynchronize());
    int nt; CK(hipMemcpy(&nt, d_nt, 4, hipMemcpyDeviceToHost));
    printf("  tiles: %d\n", nt);
    for (int cfg : {0, 1, 2, 3, 4, 8, 9, 10, 11, 16, 17, 18}) {
      if (dsv41_moe_gemm(a1, H, 6, W13[0], S13[0], d_slots, d_tiles, d_nt, cap, gu, I2, H, cfg, nullptr)) continue;
      const double t = T.ms([&](int it) { dsv41_moe_gemm(a1, H, 6, W13[it % NC], S13[it % NC], d_slots, d_tiles, d_nt, cap, gu, I2, H, cfg, nullptr); }, iters);
      report("gate/up", cfg, t, u * GU_BYTES);
    }
    for (int cfg : {12, 13, 5, 6, 7, 14, 15}) {
      if (dsv41_moe_gemm(a2, I, 1, W2[0], S2[0], d_slots, d_tiles, d_nt, cap, dn, H, I, cfg, nullptr)) continue;
      const double t = T.ms([&](int it) { dsv41_moe_gemm(a2, I, 1, W2[it % NC], S2[it % NC], d_slots, d_tiles, d_nt, cap, dn, H, I, cfg, nullptr); }, iters);
      report("down", cfg, t, u * DN_BYTES);
    }
  } else {
    for (int cfg : {0, 1, 2, 3, 4, 8, 9, 10, 11, 16, 17, 18}) {
      if (dsv41_moe_gemm_decode(a1, H, 6, W13[0], S13[0], d_ids, S, E, gu, I2, H, cfg, nullptr)) continue;
      const double t = T.ms([&](int it) { dsv41_moe_gemm_decode(a1, H, 6, W13[it % NC], S13[it % NC], d_ids, S, E, gu, I2, H, cfg, nullptr); }, iters);
      report("gate/up", cfg, t, u * GU_BYTES);
    }
    for (int cfg : {12, 13, 5, 6, 7, 14, 15}) {
      if (dsv41_moe_gemm_decode(a2, I, 1, W2[0], S2[0], d_ids, S, E, dn, H, I, cfg, nullptr)) continue;
      const double t = T.ms([&](int it) { dsv41_moe_gemm_decode(a2, I, 1, W2[it % NC], S2[it % NC], d_ids, S, E, dn, H, I, cfg, nullptr); }, iters);
      report("down", cfg, t, u * DN_BYTES);
    }
  }
  return 0;
}
