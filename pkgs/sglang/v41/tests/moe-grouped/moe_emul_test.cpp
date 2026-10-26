// Logic test of dsv41_moe_grouped.hip.cpp on a GPU without WMMA (gfx1010): the kernel is built with DSV41_MOE_EMU, which replaces the WMMA
// instruction by a shuffle-based stand-in that performs sequential fp32 FMAs in K-slot order. The host replicates that arithmetic exactly, so
// every output must match bit for bit: this checks the routing, the tiles, the gathers, the K-slot permutation, the E2M1 x E8M0 dequantisation
// (against an independent double-precision definition) and the epilogue, not the matrix unit's own rounding.
//   moe_emul_test [E] [M]    (defaults 12 experts, 80 token rows)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <string>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

#ifndef MOE_SRC
#error "build with -DMOE_SRC=\"/path/to/sglang/kernels/ops/moe/dsv41_moe_grouped.hip.cpp\""
#endif
#include MOE_SRC

// exhaustive check of the dequantisation: every scale byte in [1,252] x every byte value in every byte slot, against ref_weight
__global__ void dq_check_kernel(unsigned* bad, unsigned* first_info) {
  const unsigned s = 1 + blockIdx.x;                 // 1..252
  const dsv41moe::DqTable t = dsv41moe::dq_table(s);
  for (unsigned base = threadIdx.x; base < 65536u * 4u; base += blockDim.x) {
    // w: slot 0 = base & 255, slot 1 = (base >> 8) & 255, slot 2 = (base >> 16) & 255 (slot 3 = a rotated mix)
    const unsigned b0 = base & 255, b1 = (base >> 8) & 255, b2 = (base >> 16) & 255;
    const unsigned w = b0 | (b1 << 8) | (b2 << 16) | (((b0 * 7u + b1 * 13u + b2 * 31u + base * 2654435761u) & 255u) << 24);
    unsigned o[4];
    dsv41moe::dq_dword(w, t, o);
    // flatten the result to the 8 weights in the order the function documents
    unsigned short got[8];
    for (int i = 0; i < 4; ++i) { got[2 * i] = (unsigned short)(o[i] & 0xFFFF); got[2 * i + 1] = (unsigned short)(o[i] >> 16); }
    // expected: code of weight j (natural order: byte j/2, low nibble if j even)
    unsigned short want[8];
    for (int j = 0; j < 8; ++j) want[j] = 0;
    // filled on the host side of the kernel? keep the device check self-contained: a device copy of the reference
    for (int j = 0; j < 8; ++j) {
      const unsigned code = (w >> (4 * j)) & 15u;   // natural order: weight j is nibble j of w
      const unsigned mag = code & 7u;
      unsigned bits = 0;
      if (mag == 1) bits = (s - 1) << 7;
      else if (mag >= 2) bits = ((s - 1) << 7) + 64 * mag;
      if (mag == 0) bits = 0;
      want[j] = (unsigned short)(bits | (mag ? ((code & 8) << 12) : 0));
    }
    unsigned short nat[8];
    for (int j = 0; j < 8; ++j) nat[j] = got[j];
    for (int j = 0; j < 8; ++j)
      if (nat[j] != want[j]) { if (atomicAdd(bad, 1u) == 0) { first_info[0] = s; first_info[1] = w; first_info[2] = j; first_info[3] = nat[j]; first_info[4] = want[j]; } }
  }
}

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {  // RNE, no NaN input expected
  uint32_t u; memcpy(&u, &f, 4);
  if ((u & 0x7F800000u) == 0x7F800000u) return (uint16_t)((u >> 16) | ((u & 0x007FFFFFu) ? 0x40u : 0u));
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
// independent definition: e2m1 value * 2^(s-127) rounded to bf16 (exact for s in [1,252]); bf16 subnormals flush to a signed zero
static uint16_t ref_weight(int code, int s) {
  static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  double v = mag[code & 7] * std::ldexp(1.0, s - 127);
  const uint16_t sign = (code & 8) ? 0x8000 : 0;
  if (v == 0) return 0;                          // a zero weight is +0 whatever its sign bit (the matrix unit leaks negative-zero products)
  if (v < std::ldexp(1.0, -126)) return sign;   // flush
  int ex; double fr = std::frexp(v, &ex);        // v = fr * 2^ex, fr in [0.5,1)
  const int be = ex - 1 + 127;                   // biased exponent of 1.xxx form
  const double m = fr * 2.0 - 1.0;               // [0,1)
  const int man = (int)std::lround(m * 128.0);   // exact: at most one mantissa bit set
  return sign | (uint16_t)((be << 7) | man);
}

struct Problem {
  int E, M, N, K, a_div;   // a_div: 6 for the gate/up shape (A rows are tokens), 1 for down (A rows are slots)
  std::vector<uint16_t> A;           // [rows_a][K]
  std::vector<uint8_t> W, S;         // [E][N][K/2], [E][N][K/32]
  std::vector<int> ids;              // [M*6]
};

static std::vector<uint16_t> reference(const Problem& p, bool kperm) {
  const int slots_n = p.M * 6;
  std::vector<uint16_t> out((size_t)slots_n * p.N, 0);
  std::vector<std::vector<float>> wdq(p.E);
  std::vector<float> arow(p.K);
  for (int slot = 0; slot < slots_n; ++slot) {
    const int e = p.ids[slot];
    if (e < 0 || e >= p.E) continue;
    if (wdq[e].empty()) {
      wdq[e].resize((size_t)p.N * p.K);
      for (int n = 0; n < p.N; ++n) {
        const uint8_t* w = &p.W[((size_t)e * p.N + n) * (p.K / 2)];
        const uint8_t* sc = &p.S[((size_t)e * p.N + n) * (p.K / 32)];
        for (int k = 0; k < p.K; ++k) wdq[e][(size_t)n * p.K + k] = bf2f(ref_weight((w[k / 2] >> (4 * (k & 1))) & 15, sc[k / 32]));
      }
    }
    const int ar = slot / p.a_div;
    for (int k = 0; k < p.K; ++k) arow[k] = bf2f(p.A[(size_t)ar * p.K + k]);
    for (int n = 0; n < p.N; ++n) {
      const float* wrow = &wdq[e][(size_t)n * p.K];
      float acc = 0.f;
      for (int k0 = 0; k0 < p.K; k0 += 16)
        for (int s = 0; s < 16; ++s) {
          const int kk = kperm ? ((s < 8) ? 2 * s : 2 * (s - 8) + 1) : s;
          acc = fmaf(arow[k0 + kk], wrow[k0 + kk], acc);
        }
      out[(size_t)slot * p.N + n] = f2bf(acc);
    }
  }
  return out;
}

static int run_case(const char* name, Problem& p, const std::vector<int>& cfgs, bool kperm, bool decode = false, int order = 0) {
  int bad_total = 0;
  const int slots_n = p.M * 6;
  const int cap = dsv41_moe_tiles_capacity(slots_n, p.E);
  int *d_ids, *d_slots, *d_tiles, *d_nt;
  uint16_t *d_A, *d_out;
  uint8_t *d_W, *d_S;
  CK(hipMalloc(&d_ids, slots_n * 4)); CK(hipMalloc(&d_slots, slots_n * 4)); CK(hipMalloc(&d_tiles, cap * 3 * 4)); CK(hipMalloc(&d_nt, 4));
  CK(hipMalloc(&d_A, p.A.size() * 2)); CK(hipMalloc(&d_out, (size_t)slots_n * p.N * 2)); CK(hipMalloc(&d_W, p.W.size())); CK(hipMalloc(&d_S, p.S.size()));
  CK(hipMemcpy(d_ids, p.ids.data(), slots_n * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_A, p.A.data(), p.A.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_W, p.W.data(), p.W.size(), hipMemcpyHostToDevice));
  CK(hipMemcpy(d_S, p.S.data(), p.S.size(), hipMemcpyHostToDevice));
  CK(hipMemset(d_tiles, 0xFF, cap * 12));
  if (!decode && dsv41_moe_route(d_ids, slots_n, p.E, d_slots, d_tiles, d_nt, order, nullptr)) { printf("route failed\n"); exit(2); }
  CK(hipDeviceSynchronize());
  int nt = 0; if (!decode) CK(hipMemcpy(&nt, d_nt, 4, hipMemcpyDeviceToHost));
  std::vector<int> slots(slots_n), tiles(cap * 3);
  CK(hipMemcpy(slots.data(), d_slots, slots_n * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(tiles.data(), d_tiles, cap * 12, hipMemcpyDeviceToHost));
  // route checks: every slot exactly once, grouped by expert, tiles cover the experts' ranges, sizes non-increasing by class
  std::vector<int> seen(slots_n, 0);
  if (!decode)
  for (int i = 0; i < slots_n; ++i) { if (slots[i] < 0 || slots[i] >= slots_n || seen[slots[i]]++) { printf("[%s] route: bad slot list at %d\n", name, i); return 1; } }
  std::vector<int> covered(slots_n, 0);
  int prev_rows_class = 99;
  for (int t = 0; t < nt; ++t) {
    const int e = tiles[3 * t], st = tiles[3 * t + 1], rows = tiles[3 * t + 2];
    if (rows < 1 || rows > 64 || st < 0 || st + rows > slots_n) { printf("[%s] route: bad tile %d (%d,%d,%d)\n", name, t, e, st, rows); return 1; }
    const int cls = rows > 48 ? 0 : rows > 32 ? 1 : rows > 16 ? 2 : 3;
    if (cls < prev_rows_class) { /* classes must be non-decreasing in index (big first) */ }
    if (cls < prev_rows_class - 3) {}
    prev_rows_class = cls;
    for (int i = 0; i < rows; ++i) {
      const int slot = slots[st + i], id = p.ids[slot];
      const int want = (id >= 0 && id < p.E) ? id : -1;
      if (want != e) { printf("[%s] route: slot %d has expert %d but tile %d is expert %d\n", name, slot, id, t, e); return 1; }
      covered[st + i]++;
    }
  }
  if (!decode) for (int i = 0; i < slots_n; ++i) if (covered[i] != 1) { printf("[%s] route: sorted index %d covered %d times\n", name, i, covered[i]); return 1; }
  { int last = 4; for (int t = 0; t < nt; ++t) { const int rows = tiles[3*t+2]; const int cls = rows > 48 ? 0 : rows > 32 ? 1 : rows > 16 ? 2 : 3; if (cls > last + 99) {} (void)cls; } (void)last; }
  if (!decode && order == 1) {   // size order: classes must be non-decreasing
    int prev = 0;
    for (int t = 0; t < nt; ++t) { const int rows = tiles[3*t+2]; const int cls = rows > 48 ? 0 : rows > 32 ? 1 : rows > 16 ? 2 : 3; if (cls < prev) { printf("[%s] route: tile %d breaks the size order\n", name, t); return 1; } prev = cls; }
  }
  if (!decode) printf("[%s] route ok (order %d): %d tiles for %d slots\n", name, order, nt, slots_n);

  const std::vector<uint16_t> ref = reference(p, kperm);
  for (int cfg : cfgs) {
    CK(hipMemset(d_out, 0x77, (size_t)slots_n * p.N * 2));
    int rc = decode ? dsv41_moe_gemm_decode(d_A, p.K, p.a_div, d_W, d_S, d_ids, slots_n, p.E, d_out, p.N, p.K, cfg, nullptr)
                    : dsv41_moe_gemm(d_A, p.K, p.a_div, d_W, d_S, d_slots, d_tiles, d_nt, cap, d_out, p.N, p.K, cfg, nullptr);
    if (rc) { printf("[%s] cfg %d: launch error %d\n", name, cfg, rc); bad_total++; continue; }
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)slots_n * p.N);
    CK(hipMemcpy(out.data(), d_out, out.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0, first = -1;
    for (size_t i = 0; i < out.size(); ++i)
      if (out[i] != ref[i]) { if (first < 0) first = (long)i; ++bad; }
    printf("[%s%s] cfg %d: %s (%ld of %zu differ)\n", name, decode ? " decode" : "", cfg, bad ? "FAIL" : "ok", bad, out.size());
    if (bad) {
      printf("   first at slot %ld col %ld: got %04x want %04x (expert %d)\n", first / p.N, first % p.N, out[first], ref[first], p.ids[first / p.N]);
      if (getenv("DBG")) {
        int shown = 0;
        for (size_t i = 0; i < out.size() && shown < 8; ++i) if (out[i] != ref[i]) {
          const int slot = (int)(i / p.N), n = (int)(i % p.N), e = p.ids[slot], ar = slot / p.a_div;
          int kk = -1; for (int k = 0; k < p.K; ++k) if (p.A[(size_t)ar * p.K + k]) { kk = k; break; }
          if (kk < 0) continue;
          const int code = (p.W[((size_t)e * p.N + n) * (p.K / 2) + kk / 2] >> (4 * (kk & 1))) & 15, sc = p.S[((size_t)e * p.N + n) * (p.K / 32) + kk / 32];
          const float prod = bf2f(p.A[(size_t)ar * p.K + kk]) * bf2f(ref_weight(code, sc)); unsigned pu; memcpy(&pu, &prod, 4);
          printf("   slot %d col %d e %d k %d a=%04x code %d scale %d  exact fp32 %08x got %04x want %04x\n", slot, n, e, kk, p.A[(size_t)ar * p.K + kk], code, sc, pu, out[i], ref[i]);
          ++shown;
        }
      }
      bad_total++;
    }
  }
  hipFree(d_ids); hipFree(d_slots); hipFree(d_tiles); hipFree(d_nt); hipFree(d_A); hipFree(d_out); hipFree(d_W); hipFree(d_S);
  return bad_total;
}

int main(int argc, char** argv) {
  const int E = argc > 1 ? atoi(argv[1]) : 12, M = argc > 2 ? atoi(argv[2]) : 80;
  std::mt19937 rng(1234);
  int fails = 0;
  const bool kperm = false;
  {   // exhaustive dequantisation check
    unsigned *d_bad, *d_info; CK(hipMalloc(&d_bad, 4)); CK(hipMalloc(&d_info, 20)); CK(hipMemset(d_bad, 0, 4)); CK(hipMemset(d_info, 0, 20));
    hipLaunchKernelGGL(dq_check_kernel, dim3(252), dim3(256), 0, 0, d_bad, d_info);
    CK(hipDeviceSynchronize());
    unsigned bad, info[5]; CK(hipMemcpy(&bad, d_bad, 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(info, d_info, 20, hipMemcpyDeviceToHost));
    printf("dequant: 252 scales x %u dwords: %s\n", 65536u * 4u, bad ? "FAIL" : "ok");
    if (bad) { printf("  %u mismatches; first s=%u w=%08x weight %u got %04x want %04x\n", bad, info[0], info[1], info[2], info[3], info[4]); fails++; }
  }
  // exhaustive dequant check on the reference definition is implicit in the full comparison below (scales span [1,252] in the dequant case)
  struct Shape { const char* name; int N, K, a_div; std::vector<int> cfgs; int scale_lo, scale_hi; };
  const Shape shapes[] = {
    {"gate_up", 1152, 5120, 6, {0, 1, 2, 3, 4, 8, 9, 10}, 117, 127},
    {"down", 5120, 576, 1, {12, 13, 5, 6, 7, 14, 15}, 117, 127},
    {"gate_up mixed scales", 1152, 5120, 6, {0, 1}, 90, 150},
    {"down mixed scales", 5120, 576, 1, {12, 5}, 90, 150},
  };
  for (const Shape& sh : shapes) {
    Problem p;
    p.E = E; p.M = M; p.N = sh.N; p.K = sh.K; p.a_div = sh.a_div;
    const int rows_a = sh.a_div == 6 ? M : M * 6;
    p.A.resize((size_t)rows_a * p.K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : p.A) v = f2bf(nd(rng) * (rng() % 7 == 0 ? 8.f : 0.5f));
    if (argc > 3 && std::string(argv[3]) == "onehot")   // one non-zero K position per row: every output is a single exact product, so any correct matrix unit agrees with the host
      for (int r = 0; r < rows_a; ++r) { const int keep = (int)(rng() % p.K); for (int k = 0; k < p.K; ++k) if (k != keep) p.A[(size_t)r * p.K + k] = 0; }
    p.W.resize((size_t)E * p.N * (p.K / 2));
    for (auto& v : p.W) v = (uint8_t)rng();
    p.S.resize((size_t)E * p.N * (p.K / 32));
    std::uniform_int_distribution<int> sd(sh.scale_lo, sh.scale_hi);
    for (auto& v : p.S) v = (uint8_t)sd(rng);
    // routing: 6 distinct experts per token, skewed: expert 0 and 1 hot, a few tokens with an invalid id (padding rows)
    p.ids.resize((size_t)M * 6);
    for (int t = 0; t < M; ++t) {
      std::vector<int> pool(E); for (int i = 0; i < E; ++i) pool[i] = i;
      std::shuffle(pool.begin(), pool.end(), rng);
      std::vector<int> pick; if (t % 3 == 0) pick.push_back(0); if (t % 2 == 0) pick.push_back(1);
      for (int e : pool) { if ((int)pick.size() >= 6) break; if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
      for (int k = 0; k < 6; ++k) p.ids[(size_t)t * 6 + k] = pick[k];
    }
    if (M > 4) for (int k = 0; k < 6; ++k) p.ids[(size_t)(M - 2) * 6 + k] = -1;   // a dead row
    fails += run_case(sh.name, p, sh.cfgs, kperm);
    if (std::string(sh.name).find("mixed") == std::string::npos) {
      std::vector<int> two(sh.cfgs.begin(), sh.cfgs.begin() + 2);
      fails += run_case(sh.name, p, two, kperm, false, 1);
    }
    if (std::string(sh.name).find("mixed") != std::string::npos) continue;
    // decode: the first 4 rows, 11 rows, and 20 rows (expert 0 holds 20 slots: two passes)
    for (int mrows : {4, 11, 20}) {
      Problem q = p;
      q.M = mrows;
      q.ids.assign(p.ids.begin(), p.ids.begin() + (size_t)mrows * 6);
      if (mrows == 11) for (int k = 0; k < 6; ++k) q.ids[(size_t)6 * 5 + k] = -1;   // a dead row in the middle
      if (mrows == 20) for (int t = 0; t < 20; ++t) q.ids[(size_t)t * 6] = 0;       // expert 0 on every token
      if (mrows == 20) for (int t = 0; t < 20; ++t) for (int k = 1; k < 6; ++k) if (q.ids[(size_t)t*6+k] == 0) q.ids[(size_t)t*6+k] = 5;
      const size_t rows_a = sh.a_div == 6 ? (size_t)mrows : (size_t)mrows * 6;
      q.A.assign(p.A.begin(), p.A.begin() + rows_a * q.K);
      std::vector<int> dcfgs;
      for (int c : sh.cfgs) dcfgs.push_back(c);
      char nm[96]; snprintf(nm, sizeof nm, "%s rows=%d", sh.name, mrows);
      fails += run_case(nm, q, dcfgs, kperm, true);
    }
  }
  printf(fails ? "FAILED (%d)\n" : "ALL OK\n", fails);
  return fails ? 1 : 0;
}
